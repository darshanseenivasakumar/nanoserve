#!/usr/bin/env bash
# M6 final benchmark (laptop): every engine through the SAME OpenAI-compatible HTTP client, same session.
#   nanoserve server (port 8002), vLLM (8000), llama.cpp (8001), plus the in-process HF baseline.
# Workloads: fixed 128 output tokens, and varied 16-256 (seeded). Concurrency 1,4,8,16,32.
# One server at a time on the GPU (6 GB).
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m6_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m6_last.log | head -3; }
wait_ready() { for i in $(seq 1 600); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

C=1,4,8,16,32
CV=4,8,16,32

echo "=== nanoserve server ==="
uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 > /tmp/nanoserve_server.log 2>&1 &
wait_ready 8002 && {
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency $C --tag m6
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency $CV --output-len-range 16,256 --tag m6-var16-256
}
stop_port 8002

echo "=== vllm ==="
bash bench/servers/vllm.sh > /tmp/vllm_m6.log 2>&1 &
wait_ready 8000 && {
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency $C --tag m6
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency $CV --output-len-range 16,256 --tag m6-var16-256
}
stop_port 8000

echo "=== llama.cpp (32 slots) ==="
NP=32 bash bench/servers/llamacpp.sh > /tmp/llama_m6.log 2>&1 &
wait_ready 8001 && {
  run --backend openai --engine-name llamacpp --base-url http://localhost:8001 --concurrency $C --tag m6
  run --backend openai --engine-name llamacpp --base-url http://localhost:8001 --concurrency $CV --output-len-range 16,256 --tag m6-var16-256
}
stop_port 8001

echo "=== HF baseline (in-process; fixed lengths only, c<=8) ==="
run --backend hf --concurrency 1,4,8 --tag m6
echo "=== done ==="
