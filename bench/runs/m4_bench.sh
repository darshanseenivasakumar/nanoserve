#!/usr/bin/env bash
# M4: Triton paged-attention decode kernel vs gather+SDPA, same session, same 2 GiB paged KV budget.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --kv-cache-gb 2 --backend nanoserve --kv-cache paged"
run() { echo "=== $* ==="; $H "$@" > /tmp/m4_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m4_last.log | head -3; }
run --attn-backend sdpa --concurrency 8 --tag m4-control
run --attn-backend triton --concurrency 1,4,8,16,32 --tag m4
run --attn-backend triton --concurrency 8,16 --output-len-range 16,256 --tag m4-var16-256
echo "=== done ==="
