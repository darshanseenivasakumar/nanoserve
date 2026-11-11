"""M9 probe: time of ONE mixed step (D one-token decodes + one P-token prefill chunk), eager vs piecewise
CUDA graphs, interleaved in the same process. Also: capture time + GPU memory for every bucket up to 2048.

    uv run python -m bench.runs.m9_step_probe
"""
from __future__ import annotations

import statistics
import time

import torch

from nanoserve.cuda_graphs import DecodeGraphRunner, PiecewiseGraphRunner
from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import DecodeMeta, Qwen2Model, SeqInput

CTX = 200          # cached tokens per decoding request


def make_step(model, pool, D: int, P: int):
    tables = []
    for _ in range(D):
        t = BlockTable(16)
        t.reserve(CTX + 1, pool.allocator)
        t.commit(CTX)
        tables.append(t)
    pre = BlockTable(16)
    pre.reserve(P, pool.allocator)
    tokens = [1000 + i for i in range(P)] + [42] * D
    positions = list(range(P)) + [CTX] * D
    slots = pre.slots(P) + [t.slots(1)[0] for t in tables]
    seqs = ([SeqInput(P, ctx_len=0)] if P else []) + [SeqInput(1, ctx_len=CTX, use_kernel=True)] * D
    meta = None
    if D:
        meta = DecodeMeta(token_idx=torch.arange(P, P + D, device="cuda"),
                          block_tables=torch.tensor([t.blocks for t in tables], dtype=torch.int32, device="cuda"),
                          context_lens=torch.full((D,), CTX + 1, dtype=torch.int32, device="cuda"))
    release = lambda: [t.release(pool.allocator) for t in tables + [pre]]   # noqa: E731
    return dict(tokens=tokens, positions=positions, slots=slots, seqs=seqs, decode_meta=meta), release


def eager(model, pool, s):
    # Includes building the input tensors, as the engine's eager path does every step.
    return model.forward_batch(torch.tensor(s["tokens"], device="cuda"), torch.tensor(s["positions"], device="cuda"),
                               s["seqs"], paged=pool, slot_mapping=torch.tensor(s["slots"], device="cuda"),
                               decode_meta=s["decode_meta"])


def timed(fn, reps=15) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def main():
    model = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    pool = PagedKVCache.from_token_budget(model.cfg, 40000, 16, "cuda", model.dtype)
    dec = DecodeGraphRunner(model, pool, max_seq_len=1024, max_batch=64)
    pw = PiecewiseGraphRunner(model, pool, dec.garbage_block, max_tokens=2048)

    torch.cuda.synchronize()
    mem0, alloc0 = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    pw.capture_all()
    torch.cuda.synchronize()
    print(f"captured {len(pw.buckets)} buckets (<= {pw.max_tokens} tokens) x {model.cfg.n_layers + 1} pieces in "
          f"{time.perf_counter() - t0:.1f} s; reserved +{(torch.cuda.memory_reserved() - mem0) / 2**20:.0f} MiB, "
          f"allocated +{(torch.cuda.memory_allocated() - alloc0) / 2**20:.0f} MiB")

    print(f"{'D':>3} {'P':>5} {'T':>5} {'bucket':>6} | {'eager ms':>8} {'piecewise ms':>12} {'speedup':>7} "
          f"{'graphs-only ms':>14}")
    for D, P in [(32, 0), (8, 32), (8, 128), (32, 32), (32, 128), (32, 224), (32, 480)]:
        s, release = make_step(model, pool, D, P)
        T = D + P
        e_runs, p_runs = [], []
        for _ in range(3):                                  # interleave: background load hits both
            e_runs.append(timed(lambda: eager(model, pool, s)))
            p_runs.append(timed(lambda: pw.run(**s)))
        e, p = statistics.median(e_runs), statistics.median(p_runs)
        # Where does the piecewise step's time go? Replay its 25 graphs with no attention in between.
        graphs = pw.graphs[pw.bucket_for(T)]
        g_only = timed(lambda: [g.replay() for g in graphs])
        extra = ""
        if P == 0:
            g = timed(lambda: dec.run(s["tokens"], s["positions"], s["slots"],
                                      [list(r) for r in s["decode_meta"].block_tables.tolist()],
                                      s["decode_meta"].context_lens.tolist()))
            extra = f"   (full decode graph: {g:.1f} ms)"
        print(f"{D:3d} {P:5d} {T:5d} {pw.bucket_for(T):6d} | {e:8.1f} {p:12.1f} {e / p:6.2f}x {g_only:14.1f}{extra}")
        release()


if __name__ == "__main__":
    main()
