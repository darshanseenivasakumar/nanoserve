#!/usr/bin/env bash
# M3 throughput/latency: paged vs contiguous KV, same session, same 2 GiB KV budget.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --kv-cache-gb 2"
run() { echo "=== $* ==="; $H "$@" > /tmp/m3_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m3_last.log | head -3; }
run --backend nanoserve --kv-cache paged --concurrency 1,4,8,16,32 --tag m3
run --backend nanoserve --kv-cache contiguous --concurrency 4,8,16,32 --tag m3
run --backend nanoserve --kv-cache paged --concurrency 8,16 --output-len-range 16,256 --tag m3-var16-256
run --backend nanoserve --kv-cache contiguous --concurrency 8,16 --output-len-range 16,256 --tag m3-var16-256
echo "=== done ==="
