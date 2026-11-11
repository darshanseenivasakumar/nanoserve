#!/usr/bin/env bash
# M8 final: nanoserve server (graphs + chunked prefill, budget 2048) on both workloads + vLLM control, same session.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m8_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m8_last.log | head -3; }
wait_ready() { for i in $(seq 1 600); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }


uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 > /tmp/nanoserve_m8.log 2>&1 &
wait_ready 8002 && {
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 1,4,8,16,32 --tag m8
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m8-var16-256
}
stop_port 8002

bash bench/servers/vllm.sh > /tmp/vllm_m8.log 2>&1 &
wait_ready 8000 && run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,8,32 --tag m8-vllm-control
stop_port 8000
echo "=== done ==="
