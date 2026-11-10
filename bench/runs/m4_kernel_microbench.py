"""M4 microbenchmark: one layer's decode attention, Triton paged kernel vs M3 gather+SDPA loop.

    uv run python -m bench.runs.m4_kernel_microbench

Qwen2.5-0.5B shapes (14 q heads, 2 kv heads, dim 64), bf16, block size 16. For each batch size B and
context length L, times one call (CUDA events, median of 50 after 10 warmup) and checks the outputs agree.
"""
from __future__ import annotations

import json
import statistics

import torch

from nanoserve.kernels.paged_attention import paged_attention_decode, paged_attention_reference
from tests.test_paged_attention_kernel import make_case


def time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main():
    rows = []
    for L in (128, 512, 1024):
        for B in (1, 4, 8, 16, 32):
            q, k, v, tables, lens = make_case(14, 2, 64, [L] * B, torch.bfloat16, seed=B * 7 + L)
            scale = 64 ** -0.5
            tri = time_ms(lambda: paged_attention_decode(q, k, v, tables, lens, scale))
            ref = time_ms(lambda: paged_attention_reference(q, k, v, tables, lens, scale))
            err = (paged_attention_decode(q, k, v, tables, lens, scale).float()
                   - paged_attention_reference(q.float(), k.float(), v.float(), tables, lens, scale)).abs().max().item()
            rows.append({"ctx": L, "B": B, "triton_ms": round(tri, 4), "gather_sdpa_ms": round(ref, 4),
                         "speedup": round(ref / tri, 1), "max_err": err})
            print(json.dumps(rows[-1]))
    print("per decode step (24 layers) at L=512, B=8: triton %.2f ms vs gather+sdpa %.2f ms" % (
        24 * next(r["triton_ms"] for r in rows if r["ctx"] == 512 and r["B"] == 8),
        24 * next(r["gather_sdpa_ms"] for r in rows if r["ctx"] == 512 and r["B"] == 8)))


if __name__ == "__main__":
    main()
