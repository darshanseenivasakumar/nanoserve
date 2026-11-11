#!/usr/bin/env bash
# M7: CUDA-graph decode. Same session: (1) in-process A/B graphs off vs on, (2) the HTTP server with graphs
# on both workloads (same client as M6), (3) a vLLM control.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m7_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m7_last.log | head -3; }
wait_ready() { for i in $(seq 1 600); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

run --backend nanoserve --kv-cache-gb 2 --no-cuda-graphs --concurrency 1,8,32 --tag m7-ab
run --backend nanoserve --kv-cache-gb 2 --cuda-graphs --concurrency 1,8,32 --tag m7-ab

uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 > /tmp/nanoserve_m7.log 2>&1 &
wait_ready 8002 && {
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 1,4,8,16,32 --tag m7
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m7-var16-256
}
stop_port 8002

bash bench/servers/vllm.sh > /tmp/vllm_m7.log 2>&1 &
wait_ready 8000 && run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,8,32 --tag m7-vllm-control
stop_port 8000
echo "=== done ==="
