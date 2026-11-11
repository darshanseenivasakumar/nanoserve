#!/usr/bin/env bash
# M8b: per-step token budget sweep WITH chunked prefill (budgets below 2048 are now real).
cd ~/nanoserve
H="uv run python -m bench.harness --backend nanoserve --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --kv-cache-gb 2"
run() { echo "=== $* ==="; $H "$@" > /tmp/m8b_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m8b_last.log | head -3; }
for B in 8192 2048 1024 512; do
  run --max-num-batched-tokens $B --concurrency 8,32 --tag m8b-b$B
  run --max-num-batched-tokens $B --concurrency 8,32 --output-len-range 16,256 --tag m8b-b$B-var16-256
done
echo "=== done ==="
