"""M7 probe: synchronous decode-step time with and without CUDA graphs, same process, interleaved.

    uv run python -m bench.runs.m7_step_probe

Interleaving the two modes (A B A B ...) means background CPU load hits both equally.
"""
from __future__ import annotations

import statistics
import time

import torch

from nanoserve.engine import LLMEngine
from nanoserve.model import Qwen2Model


def step_ms(eng: LLMEngine, B: int, steps: int = 30) -> float:
    prompt = list(range(1000, 1200))
    for _ in range(B):
        eng.add_request(prompt, steps + 12, ignore_eos=True)
    eng.step()                                   # prefill (eager)
    for _ in range(5):                           # warm (captures the bucket's graph if enabled)
        eng.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.step()
    dt = (time.perf_counter() - t0) / steps * 1e3
    while eng.has_unfinished():
        eng.step()
    return dt


def main():
    model = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    engines = {g: LLMEngine(model, [151645], max_seq_len=1024, max_num_seqs=64, kv_cache_tokens=40000,
                            cuda_graphs=g) for g in (False, True)}
    for B in (1, 8, 32):
        res = {False: [], True: []}
        for _ in range(3):
            for g in (False, True):
                res[g].append(step_ms(engines[g], B))
        off, on = statistics.median(res[False]), statistics.median(res[True])
        print(f"B={B:2d}: eager {off:5.1f} ms/step  graphs {on:5.1f} ms/step  -> {off / on:.2f}x  "
              f"({B / on * 1e3:6.0f} tok/s with graphs)")


if __name__ == "__main__":
    main()
