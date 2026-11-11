#!/usr/bin/env bash
# M11: engine in its own process. Same session: server A/B (engine in a thread = the M10 server, then own process) + vLLM control.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m11_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m11_last.log | head -3; }
wait_ready() { for i in $(seq 1 900); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

for mode in off on; do
  flag=""; [ "$mode" = off ] && flag="--engine-in-thread"
  uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 $flag > /tmp/nanoserve_m11_$mode.log 2>&1 &
  wait_ready 8002 && {
    grep "captured" /tmp/nanoserve_m11_$mode.log
    run --backend openai --engine-name nanoserve-proc-$mode --base-url http://localhost:8002 --concurrency 1,4,8,16,32 --tag m11-proc-$mode
    run --backend openai --engine-name nanoserve-proc-$mode --base-url http://localhost:8002 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m11-proc-$mode-var16-256
  }
  stop_port 8002
done

bash bench/servers/vllm.sh > /tmp/vllm_m11.log 2>&1 &
wait_ready 8000 && {
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,4,8,16,32 --tag m11-vllm-control
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m11-vllm-control-var16-256
}
stop_port 8000
echo "=== done ==="
