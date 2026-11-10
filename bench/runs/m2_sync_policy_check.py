"""Closed-loop benchmark WITHOUT threads/asyncio: does continuous vs static still differ on fixed lengths?"""
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/home/darshan/nanoserve")
from bench.prompts import build_requests, load_prompt_set  # noqa: E402
from nanoserve.engine import LLMEngine  # noqa: E402

M = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
reqs = build_requests(tok, load_prompt_set(), 64, 128, seed=0)
prompts = [tok(r.prompt, add_special_tokens=False)["input_ids"] for r in reqs]   # tokenized up front
eng = LLMEngine.from_pretrained(M, max_num_seqs=8, batch_wait_s=0.0)


def run(policy, c):
    eng.scheduler.policy = policy
    eng.scheduler.max_num_seqs = c
    queue = list(prompts)
    for _ in range(c):
        eng.add_request(queue.pop(0), 128, ignore_eos=True)
    torch.cuda.synchronize()
    t0, steps, toks, step_ms = time.perf_counter(), 0, 0, []
    while eng.has_unfinished():
        s = time.perf_counter()
        outs = eng.step()
        step_ms.append((time.perf_counter() - s) * 1e3)
        steps += 1
        toks += len(outs)
        for o in outs:
            if o.finished and queue:                       # closed loop: replace immediately
                eng.add_request(queue.pop(0), 128, ignore_eos=True)
    wall = time.perf_counter() - t0
    dec = sorted(step_ms)
    print(f"{policy:10s} c={c}: {toks / wall:6.1f} tok/s  steps={steps}  "
          f"step p50={dec[len(dec) // 2]:.1f} ms  p90={dec[int(len(dec) * .9)]:.1f} ms")


for policy in ("static", "continuous", "static", "continuous"):
    run(policy, 8)
