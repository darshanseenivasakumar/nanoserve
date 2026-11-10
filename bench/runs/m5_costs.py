"""M5: why speculative decoding does (not) pay off here. Measures the three costs that decide it.

    uv run python -m bench.runs.m5_costs

Speculative step = k draft forwards + 1 target verification forward, emitting (accepted + 1) tokens.
Plain step = 1 target forward, emitting 1 token. So, per token:
    plain       = T_target_1
    speculative = (k * T_draft_1 + T_target_verify(k+1)) / tokens_per_step
Speedup > 1 only if the draft is MUCH cheaper than the target. Batch size 1, context ~300 tokens.
"""
from __future__ import annotations

import json
import statistics

import torch

from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import Qwen2Model, SeqInput


def step_ms(model: Qwen2Model, n_new: int, ctx: int = 300, iters: int = 30) -> float:
    pool = PagedKVCache.from_token_budget(model.cfg, 2048, 16, "cuda", model.dtype)
    table = BlockTable(16)
    table.reserve(ctx + n_new, pool.allocator)
    toks = torch.arange(100, 100 + ctx, device="cuda")
    model.forward_batch(toks, torch.arange(ctx, device="cuda"), [SeqInput(ctx)], paged=pool,
                        slot_mapping=torch.tensor(table.slots(ctx), device="cuda"))
    table.commit(ctx)
    bt = torch.tensor(table.blocks, device="cuda")
    slots = torch.tensor(table.slots(n_new), device="cuda")
    new = torch.arange(7, 7 + n_new, device="cuda")
    pos = torch.arange(ctx, ctx + n_new, device="cuda")
    rows = torch.arange(n_new, device="cuda")

    def one():
        out = model.forward_batch(new, pos, [SeqInput(n_new, block_table=bt, ctx_len=ctx)],
                                  logits=rows, paged=pool, slot_mapping=slots)
        out.argmax(-1).tolist()      # the engine syncs once per forward too

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
    res = {"target_1": step_ms(target, 1), "draft_1": step_ms(draft, 1)}
    for k in (2, 4):
        res[f"target_verify_{k + 1}"] = step_ms(target, k + 1)
    print(json.dumps({k: round(v, 2) for k, v in res.items()}))
    print(f"draft/target cost ratio: {res['draft_1'] / res['target_1']:.2f} (needs to be << 1)")
    for k, tps in ((2, None), (4, 3.06)):
        verify = res[f"target_verify_{k + 1}"]
        spec_step = k * res["draft_1"] + verify
        breakeven = spec_step / res["target_1"]
        line = f"k={k}: spec step {spec_step:.1f} ms vs plain {res['target_1']:.1f} ms/token -> needs {breakeven:.2f} tokens/step to break even"
        if tps:
            line += f"; measured {tps} tokens/step (fp32 test) -> predicted speedup {tps / breakeven:.2f}x"
        print(line)


if __name__ == "__main__":
    main()
