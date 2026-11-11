#!/usr/bin/env bash
# M7b: plain 1.5B greedy (CUDA graphs) vs speculative k=2/k=4 (graphed draft + graphed verification).
# Identical settings to M5 (32 prompts x 128 forced tokens, 2 repeats, 0.25 GiB target KV), same session.
cd ~/nanoserve
H="uv run python -m bench.harness --backend nanoserve --model Qwen/Qwen2.5-1.5B-Instruct --num-prompts 32 --max-tokens 128 --warmup 2 --repeats 2 --kv-cache-gb 0.25"
run() { echo "=== $* ==="; $H "$@" > /tmp/m7b_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m7b_last.log | head -3; }
run --concurrency 1,4 --tag m7b-plain
run --concurrency 1,4 --draft-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 2 --tag m7b-spec2
run --concurrency 1,4 --draft-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 4 --tag m7b-spec4
echo "=== done ==="
