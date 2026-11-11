"""M11 profile: where does a step's WALL time go under load, and how long is the GPU idle?

    uv run python -m bench.runs.m11_loop_profile

Closed loop, c=32, the benchmark prompts, fixed 128 output tokens (ignore_eos), two ways:
  A) the bare engine loop (eng.step() in a for loop)       -> pure step cost
  B) through AsyncEngine with 32 asyncio clients            -> + thread hand-off and GIL sharing
Per step: CPU phases (schedule, prepare, launch, wait-for-GPU, post) and the GPU busy span (CUDA events
around the forward). GPU idle per step ~= wall - GPU span.
"""
from __future__ import annotations

import asyncio
import time

import torch
from transformers import AutoTokenizer

from bench.prompts import build_requests, load_prompt_set
from nanoserve.async_engine import AsyncEngine
from nanoserve.engine import LLMEngine

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
C, N_REQ, MAX_NEW = 32, 160, 128


def report(label: str, T: dict, wall: float, tokens: int) -> None:
    n = T["steps"]
    phases = ["schedule", "prepare", "launch", "wait", "post"]
    cpu = {p: 1e3 * T.get(p, 0.0) / n for p in phases}
    step_wall = 1e3 * wall / n
    gpu = 1e3 * T["gpu"] / n
    other = step_wall - sum(cpu.values())
    print(f"{label}: {tokens / wall:7.1f} tok/s, {n} steps, wall {step_wall:5.2f} ms/step, GPU busy span "
          f"{gpu:5.2f} ms/step ({100 * gpu / step_wall:3.0f}%)")
    for p in phases:
        print(f"   {p:9s} {cpu[p]:5.2f} ms  ({100 * cpu[p] / step_wall:3.0f}%)")
    print(f"   {'outside step()':9s} {other:5.2f} ms  ({100 * other / step_wall:3.0f}%)")


def run_bare(eng, prompts) -> None:
    queue = list(prompts)
    for _ in range(C):
        eng.add_request(queue.pop(0), MAX_NEW, ignore_eos=True)
    for _ in range(20):                              # warm up before measuring
        eng.step()
    eng.timing, done, tokens = {}, 0, 0
    t0 = time.perf_counter()
    while done < N_REQ:
        for out in eng.step():
            tokens += 1
            if out.finished:
                done += 1
                eng.add_request(queue.pop(0) if queue else prompts[done % len(prompts)], MAX_NEW, ignore_eos=True)
    wall = time.perf_counter() - t0
    T, eng.timing = eng.timing, None
    while eng.has_unfinished():
        eng.step()
    report("A) bare engine loop", T, wall, tokens)


async def run_async(eng, prompts) -> None:
    ae = AsyncEngine(eng)
    queue = list(prompts)
    tokens = 0

    async def client():
        nonlocal tokens
        while queue:
            p = queue.pop(0)
            async for _tok, _t in ae.generate(p, MAX_NEW, ignore_eos=True):
                tokens += 1

    await asyncio.gather(*[client() for _ in range(4)])     # warm-up
    queue.extend(prompts[:N_REQ])
    tokens = 0
    eng.timing = {}
    t0 = time.perf_counter()
    await asyncio.gather(*[client() for _ in range(C)])
    wall = time.perf_counter() - t0
    T, eng.timing = eng.timing, None
    ae.close()
    report("B) AsyncEngine + asyncio clients", T, wall, tokens)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    reqs = build_requests(tok, load_prompt_set(), 64, MAX_NEW, seed=0)
    prompts = [tok(r.prompt, add_special_tokens=False)["input_ids"] for r in reqs]
    prompts = (prompts * 4)[:N_REQ + C]
    eng = LLMEngine.from_pretrained(MODEL, max_seq_len=2048, max_num_seqs=64,
                                    kv_cache_tokens=int(2 * 2**30) // (12 * 1024))
    print(f"captured graphs in {eng.capture_graphs():.1f} s")
    torch.cuda.synchronize()
    run_bare(eng, prompts)
    asyncio.run(run_async(eng, prompts))


if __name__ == "__main__":
    main()
