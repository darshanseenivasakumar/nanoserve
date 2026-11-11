"""M10 probe: one step, unfused vs fused ops, interleaved in one process (same weights, same KV pool).

    uv run python -m bench.runs.m10_step_probe

  decode B = 1, 8, 32   -> M7 full CUDA graph
  mixed 32 + 128        -> M9 piecewise graphs
  prefill 2048 tokens   -> eager (above the piecewise cap)
Plus the kernel count of one decode step (eager profile = what the graph replays).
"""
from __future__ import annotations

import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from bench.runs.m9_step_probe import eager, make_step, timed
from nanoserve.cuda_graphs import DecodeGraphRunner, PiecewiseGraphRunner
from nanoserve.kv_cache import PagedKVCache
from nanoserve.model import Qwen2Model


def n_kernels(fn) -> int:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def main():
    base = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    models = {"unfused": base, "fused": base.with_fused_ops()}
    pool = PagedKVCache.from_token_budget(base.cfg, 40000, 16, "cuda", base.dtype)
    dec = {k: DecodeGraphRunner(m, pool, max_seq_len=1024, max_batch=64) for k, m in models.items()}
    pw = {k: PiecewiseGraphRunner(m, pool, dec[k].garbage_block, max_tokens=512) for k, m in models.items()}

    def compare(label, fns):
        runs = {k: [] for k in fns}
        for _ in range(3):                                   # interleaved: background load hits both
            for k, fn in fns.items():
                runs[k].append(timed(fn, reps=20))
        u, f = statistics.median(runs["unfused"]), statistics.median(runs["fused"])
        print(f"{label:24s} unfused {u:7.2f} ms   fused {f:7.2f} ms   {u / f:5.2f}x   (-{u - f:.2f} ms)")

    s1, rel = make_step(base, pool, 1, 0)
    print(f"kernels per decode step (B=1): unfused {n_kernels(lambda: eager(base, pool, s1))}, "
          f"fused {n_kernels(lambda: eager(models['fused'], pool, s1))}")
    rel()
    for D in (1, 8, 32):
        s, rel = make_step(base, pool, D, 0)
        tables = [list(r) for r in s["decode_meta"].block_tables.tolist()]
        lens = s["decode_meta"].context_lens.tolist()
        compare(f"decode B={D} (graph)", {k: (lambda k=k: dec[k].run(s["tokens"], s["positions"], s["slots"],
                                                                     tables, lens)) for k in models})
        rel()
    s, rel = make_step(base, pool, 32, 128)
    compare("mixed 32+128 (piecewise)", {k: (lambda k=k: pw[k].run(**s)) for k in models})
    rel()
    s, rel = make_step(base, pool, 0, 2048)
    compare("prefill 2048 (eager)", {k: (lambda k=k: eager(models[k], pool, s)) for k in models})
    rel()


if __name__ == "__main__":
    main()
