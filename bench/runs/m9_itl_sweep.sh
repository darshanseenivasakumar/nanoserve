#!/usr/bin/env bash
# M9b: re-run the c=32 token-budget sweep now that the harness records ITL (every token gap), to test:
# "a smaller budget bounds the WORST token gap (ITL p99) even though each request's MEAN gap (TPOT) can't
# shrink, because the same prefill compute is just cut into more, shorter interruptions".
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --backend nanoserve --kv-cache-gb 2 --concurrency 32"
run() { echo "=== $* ==="; $H "$@" > /tmp/m9b_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m9b_last.log | head -3; }
for b in 2048 1024 512; do
  run --max-num-batched-tokens $b --output-len-range 16,256 --tag m9b-b$b-var
  run --max-num-batched-tokens $b --tag m9b-b$b
done
for b in 2048 512; do
  run --max-num-batched-tokens $b --output-len-range 16,256 --no-piecewise-graphs --tag m9b-eager-b$b-var
done
echo "=== done ==="
