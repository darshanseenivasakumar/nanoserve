"""M7b: the speculative-decoding cost model again, now with CUDA graphs on every decode-like step.

    uv run python -m bench.runs.m7b_costs

Per token:  plain = T_target(1 row)      speculative = (k * T_draft(1 row) + T_target(k+1 rows)) / tokens_per_step
tokens_per_step measured in M5 (fp32 test / bench): k=4 -> 3.05, k=2 -> 2.28. Batch 1, context ~300.
Each timing includes what the engine does per step: copy inputs into the graph buffers, replay, argmax, .tolist().
"""
from __future__ import annotations

import json
import statistics

import torch

from nanoserve.cuda_graphs import DecodeGraphRunner
from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import Qwen2Model, SeqInput

TOKENS_PER_STEP = {2: 2.28, 4: 3.05}


def graphed_ms(model: Qwen2Model, rows: int, ctx: int = 300, iters: int = 50) -> float:
    pool = PagedKVCache.from_token_budget(model.cfg, 4096, 16, "cuda", model.dtype)
    runner = DecodeGraphRunner(model, pool, 1024)
    table = BlockTable(16)
    table.reserve(ctx + rows, pool.allocator)
    toks = torch.arange(100, 100 + ctx, device="cuda")
    model.forward_batch(toks, torch.arange(ctx, device="cuda"), [SeqInput(ctx)], paged=pool,
                        slot_mapping=torch.tensor(table.slots(ctx), device="cuda"))
    table.commit(ctx)
    args = dict(tokens=list(range(7, 7 + rows)), positions=list(range(ctx, ctx + rows)), slots=table.slots(rows),
                block_tables=[table.blocks] * rows, context_lens=list(range(ctx + 1, ctx + rows + 1)))

    def one():
        runner.run(**args).argmax(-1).tolist()

    for _ in range(5):
        one()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        one()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main():
    target = Qwen2Model.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    draft = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    t1, d1 = graphed_ms(target, 1), graphed_ms(draft, 1)
    verify = {k: graphed_ms(target, k + 1) for k in (2, 4)}
    print(json.dumps({"target_1": round(t1, 2), "draft_1": round(d1, 2),
                      **{f"target_verify_{k + 1}": round(v, 2) for k, v in verify.items()}}))
    print(f"draft/target cost ratio: {d1 / t1:.2f} (was 0.86 without graphs)")
    for k in (2, 4):
        step = k * d1 + verify[k]
        per_token = step / TOKENS_PER_STEP[k]
        print(f"k={k}: spec step {step:.1f} ms / {TOKENS_PER_STEP[k]} tokens = {per_token:.1f} ms/token "
              f"vs plain {t1:.1f} -> PREDICTED {t1 / per_token:.2f}x (break-even needs {step / t1:.2f} tokens/step)")


if __name__ == "__main__":
    main()
