#!/usr/bin/env bash
# M14: the 1.5B model on the laptop (the original spec: "develop on 0.5B, then 1.5B"). Same methodology as
# the 0.5B runs; nanoserve server, vLLM and llama.cpp (8-slot = its best laptop config) in ONE session.
# Memory on the 6 GB GPU: weights ~3.1 GB, so the KV budget is 1 GiB (~37k tokens at 28 KB/token).
cd ~/nanoserve
MODEL=Qwen/Qwen2.5-1.5B-Instruct
H="uv run python -m bench.harness --model $MODEL --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m14_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m14_last.log | head -3; }
wait_ready() { for i in $(seq 1 900); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

# ONLY=nanoserve|vllm|llamacpp runs just that engine (e.g. to finish a paused session).
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

want nanoserve && { uv run python -m nanoserve.server --model $MODEL --port 8002 --kv-cache-gb 1 --max-num-seqs 64 > /tmp/nanoserve_m14.log 2>&1 &
wait_ready 8002 && {
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 1,4,8,16,32 --tag m14-1p5b
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m14-1p5b-var16-256
}
stop_port 8002; }

want vllm && { MODEL=$MODEL GPU_UTIL=0.8 bash bench/servers/vllm.sh > /tmp/vllm_m14.log 2>&1 &
wait_ready 8000 && {
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,4,8,16,32 --tag m14-1p5b
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m14-1p5b-var16-256
}
stop_port 8000; }

want llamacpp && { GGUF=$HOME/models/qwen2.5-1.5b-instruct-fp16.gguf NP=8 bash bench/servers/llamacpp.sh > /tmp/llamacpp_m14.log 2>&1 &
wait_ready 8001 && {
  run --backend openai --engine-name llamacpp-np8 --base-url http://localhost:8001 --concurrency 1,4,8 --tag m14-1p5b
  run --backend openai --engine-name llamacpp-np8 --base-url http://localhost:8001 --concurrency 4,8 --output-len-range 16,256 --tag m14-1p5b-var16-256
}
stop_port 8001; }
echo "=== done ==="
