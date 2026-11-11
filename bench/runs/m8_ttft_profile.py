"""M8 step 1: where does time-to-first-token go under load?

    uv run python -m bench.runs.m8_ttft_profile

A) Synchronous closed loop at concurrency 32 (fixed 128 output tokens, real prompts, graphs on). For each
   request: TTFT = queue wait (arrival -> start of the step that admits it) + service (that step's
   duration). Also: step durations for mixed (prefill-containing) vs decode-only steps, and prefill
   tokens per mixed step.
B) One big prefill step (16 prompts) under torch.profiler: GPU time by kernel category, vs wall time.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from bench.prompts import build_requests, load_prompt_set
from nanoserve.engine import LLMEngine
from nanoserve.model import Qwen2Model

M = "Qwen/Qwen2.5-0.5B-Instruct"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def closed_loop(model, prompts, c=32, n=160):
    eng = LLMEngine(model, [151645], max_seq_len=2048, max_num_seqs=64, kv_cache_tokens=170_000)
    queue = list(prompts[:n])
    arrival, admitted_at, first_at = {}, {}, {}
    now = time.perf_counter
    for _ in range(c):
        sid = eng.add_request(queue.pop(0), 128, ignore_eos=True)
        arrival[sid] = now()
    steps = []   # (duration, n_prefill_seqs, n_prefill_tokens, n_decode)
    while eng.has_unfinished():
        waiting_before = {s.seq_id for s in eng.scheduler.waiting}
        t0 = now()
        outs = eng.step()
        torch.cuda.synchronize()
        t1 = now()
        running_ids = {s.seq_id for s in eng.scheduler.running}
        st = eng.last_step
        if st:
            pre_tokens = st.n_tokens - st.n_decode
            steps.append((t1 - t0, st.n_prefill, pre_tokens, st.n_decode))
        for o in outs:
            if o.seq_id not in first_at:
                first_at[o.seq_id] = t1
                admitted_at[o.seq_id] = t0
        for sid in waiting_before - running_ids:
            pass
        for o in outs:
            if o.finished and queue:
                sid = eng.add_request(queue.pop(0), 128, ignore_eos=True)
                arrival[sid] = now()
    ids = [i for i in first_at if i >= c]          # skip the cold-start wave
    ttft = [first_at[i] - arrival[i] for i in ids]
    wait = [admitted_at[i] - arrival[i] for i in ids]
    service = [first_at[i] - admitted_at[i] for i in ids]
    mixed = [s for s in steps if s[1] > 0]
    decode = [s for s in steps if s[1] == 0]
    print(f"A) closed loop c={c}, {len(ids)} requests measured (cold-start wave excluded)")
    print(f"   TTFT p50/p90 {1e3 * pct(ttft, 50):.0f}/{1e3 * pct(ttft, 90):.0f} ms = queue wait p50 "
          f"{1e3 * pct(wait, 50):.0f} ms + service (admitting step) p50 {1e3 * pct(service, 50):.0f} ms")
    print(f"   steps: {len(decode)} decode-only, median {1e3 * statistics.median(s[0] for s in decode):.1f} ms; "
          f"{len(mixed)} mixed, median {1e3 * statistics.median(s[0] for s in mixed):.1f} ms, "
          f"prefill seqs/step median {statistics.median(s[1] for s in mixed):.0f}, "
          f"prefill tokens/step median {statistics.median(s[2] for s in mixed):.0f} (max {max(s[2] for s in mixed)})")
    big = sorted(mixed, key=lambda s: -s[2])[:3]
    print("   largest mixed steps (ms, prefill seqs, prefill tokens, decodes):",
          [(round(1e3 * d), a, b, c_) for d, a, b, c_ in big])


def categorize(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "xmma" in n or "cutlass" in n or "cublas" in n or "sm80" in n or "ampere" in n:
        return "matmul (GEMM)"
    if "flash" in n or "fmha" in n or "attention" in n or "_paged_decode" in n or "efficient" in n:
        return "attention"
    if "index" in n or "gather" in n or "scatter" in n or "copy" in n or "cat" in n:
        return "index / copy / cat"
    if "reduce" in n or "norm" in n or "mean" in n or "max" in n:
        return "reductions"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n or "silu" in n or "mul" in n or "add" in n:
        return "elementwise"
    return "other"


def prefill_profile(model, prompts, n_seqs=16):
    eng = LLMEngine(model, [151645], max_seq_len=2048, max_num_seqs=64, kv_cache_tokens=60_000)
    batch = prompts[:n_seqs]
    n_tok = sum(len(p) for p in batch)

    def one_prefill():
        for p in batch:
            eng.add_request(p, 1, ignore_eos=True)
        eng.step()                 # admits all and prefills them in ONE step; max_new=1 so they finish

    for _ in range(3):
        one_prefill()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        one_prefill()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / 5 * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        one_prefill()
        torch.cuda.synchronize()
    by = defaultdict(float)
    count = defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA":
            by[categorize(e.name)] += e.device_time / 1e3
            count[categorize(e.name)] += 1
    gpu = sum(by.values())
    print(f"B) one prefill step: {n_seqs} prompts, {n_tok} tokens: wall {wall:.1f} ms, GPU busy {gpu:.1f} ms "
          f"({100 * gpu / wall:.0f}%), {sum(count.values())} kernels")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"   {k:20s} {v:7.1f} ms  ({100 * v / gpu:4.1f}% of GPU time, {count[k]} kernels)")


def main():
    tok = AutoTokenizer.from_pretrained(M)
    reqs = build_requests(tok, load_prompt_set(), 160, 128, seed=0)
    prompts = [tok(r.prompt, add_special_tokens=False)["input_ids"] for r in reqs]
    print(f"prompt tokens: mean {statistics.mean(map(len, prompts)):.0f}, max {max(map(len, prompts))}")
    model = Qwen2Model.from_pretrained(M)
    closed_loop(model, prompts)
    prefill_profile(model, prompts)


if __name__ == "__main__":
    main()
