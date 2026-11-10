"""M3 capacity experiment: how many requests run at once under the SAME explicit KV budget?

    uv run python -m bench.runs.m3_capacity            (from the repo root)

Floods the engine with requests from the fixed prompt set (128 forced output tokens each), steps until the
running count stops growing, and reports peak concurrency and KV utilisation for contiguous vs paged.
Uses the real model on the GPU. The per-step token budget (8192) means admission ramps up over steps.
"""
from __future__ import annotations

import json
import math
import time

import torch
from transformers import AutoTokenizer

from bench.prompts import build_requests, load_prompt_set
from nanoserve.engine import LLMEngine
from nanoserve.model import Qwen2Model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BUDGET_GIB = 2.0
MAX_SEQ_LEN = 2048
N_REQUESTS = 1500


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    reqs = build_requests(tok, load_prompt_set(), N_REQUESTS, 128, seed=0)
    prompts = [tok(r.prompt, add_special_tokens=False)["input_ids"] for r in reqs]
    model = Qwen2Model.from_pretrained(MODEL)
    budget_tokens = int(BUDGET_GIB * 2**30) // model.cfg.kv_bytes_per_token(model.dtype)
    avg_need = sum(len(p) + 128 for p in prompts) / len(prompts)
    print(f"KV budget {BUDGET_GIB} GiB = {budget_tokens:,} tokens; avg request needs {avg_need:.0f} tokens "
          f"(prompt + 128 output)")
    results = {}
    for kind in ("contiguous", "paged"):
        eng = LLMEngine(model, [151645, 151643], max_seq_len=MAX_SEQ_LEN, max_num_seqs=100_000,
                        kv_cache=kind, kv_cache_tokens=budget_tokens)
        for p in prompts:
            eng.add_request(p, 128, ignore_eos=True)
        t0, history, flat = time.perf_counter(), [], 0
        while flat < 3 and len(history) < 200:
            eng.step()
            history.append(len(eng.scheduler.running))
            flat = flat + 1 if len(history) > 1 and history[-1] <= history[-2] else 0
        st = eng.stats
        used, reserved = eng.kv.usage(eng.scheduler.running)
        results[kind] = {"peak_running": st["peak_running"], "steps": len(history),
                         "seconds": round(time.perf_counter() - t0, 1), "preemptions": st["num_preemptions"],
                         "kv_used_tokens_at_end": used, "kv_reserved_tokens_at_end": reserved,
                         "kv_util_at_end": round(used / reserved, 3) if reserved else None,
                         "theoretical_contiguous": budget_tokens // MAX_SEQ_LEN,
                         "theoretical_paged_avg": math.floor(budget_tokens / avg_need)}
        print(kind, json.dumps(results[kind]))
        for sid in list(eng.sequences):
            eng.abort(sid)
        del eng
        torch.cuda.empty_cache()
    print("RESULT", json.dumps(results))


if __name__ == "__main__":
    main()
