"""M10 profile: where does GPU time go now that launches are graphed?

    uv run python -m bench.runs.m10_profile

For a decode step (B = 1, 32) and one 2048-token prefill step: kernel count and GPU time by category
(profiled on the eager step: the graph replays exactly these kernels), plus the graphed step's wall time.
If wall time > summed kernel time, the rest is per-kernel overhead on the GPU (gaps between graph nodes).
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from bench.runs.m9_step_probe import eager, make_step, timed
from nanoserve.cuda_graphs import DecodeGraphRunner
from nanoserve.kv_cache import PagedKVCache
from nanoserve.model import Qwen2Model


def categorize(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "xmma" in n or "cutlass" in n or "cublas" in n or "sm80" in n or "ampere" in n:
        return "matmul (GEMM)"
    if "flash" in n or "fmha" in n or "attention" in n or "_paged_decode" in n or "efficient" in n:
        return "attention"
    if "index" in n or "gather" in n or "scatter" in n or "copy" in n or "cat" in n:
        return "index / copy / cat"
    if "reduce" in n or "norm" in n or "mean" in n:
        return "reductions (norm)"
    return "elementwise"


def breakdown(fn, label: str, wall_graph: float | None = None) -> None:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    by, count = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA":
            by[categorize(e.name)] += e.device_time / 1e3
            count[categorize(e.name)] += 1
    gpu, n = sum(by.values()), sum(count.values())
    extra = f"; graphed step wall {wall_graph:.2f} ms" if wall_graph else ""
    print(f"{label}: {n} kernels, summed GPU time {gpu:.2f} ms{extra}")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"   {k:20s} {v:7.2f} ms  ({100 * v / gpu:4.1f}%, {count[k]:4d} kernels, "
              f"{1e3 * v / count[k]:6.1f} us each)")


def main():
    model = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    pool = PagedKVCache.from_token_budget(model.cfg, 40000, 16, "cuda", model.dtype)
    dec = DecodeGraphRunner(model, pool, max_seq_len=1024, max_batch=64)
    for D in (1, 32):
        s, release = make_step(model, pool, D, 0)
        tables = [list(r) for r in s["decode_meta"].block_tables.tolist()]
        lens = s["decode_meta"].context_lens.tolist()
        g = timed(lambda: dec.run(s["tokens"], s["positions"], s["slots"], tables, lens), reps=30)
        breakdown(lambda: eager(model, pool, s), f"decode B={D}", g)
        release()
    s, release = make_step(model, pool, 0, 2048)
    breakdown(lambda: eager(model, pool, s), "prefill T=2048")
    release()


if __name__ == "__main__":
    main()
