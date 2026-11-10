#!/usr/bin/env bash
# M5: plain 1.5B greedy vs speculative (0.5B draft, k=2 and k=4), same session, same 0.25 GiB target KV budget.
# 32 prompts x 128 forced tokens, 2 repeats. DECISION: smaller than the 64x3 default to fit the night;
# the M5 comparison is internal to the 1.5B model, so all three rows use identical settings.
cd ~/nanoserve
H="uv run python -m bench.harness --backend nanoserve --model Qwen/Qwen2.5-1.5B-Instruct --num-prompts 32 --max-tokens 128 --warmup 2 --repeats 2 --kv-cache-gb 0.25"
run() { echo "=== $* ==="; $H "$@" > /tmp/m5_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m5_last.log | head -3; }
run --concurrency 1,4 --tag m5-plain
run --concurrency 1,4 --draft-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 4 --tag m5-spec4
run --concurrency 1,4 --draft-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 2 --tag m5-spec2
echo "=== done ==="
