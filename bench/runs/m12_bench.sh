#!/usr/bin/env bash
# M12: prefix caching. Workload: the same 1024-token system prompt in front of all 64 prompts (fixed 128
# output tokens). Same session: nanoserve server prefix caching off -> on, vLLM off -> on.
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --shared-prefix-tokens 1024"
run() { echo "=== $* ==="; $H "$@" > /tmp/m12_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m12_last.log | head -3; }
wait_ready() { for i in $(seq 1 900); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

for mode in off on; do
  flag=""; [ "$mode" = on ] && flag="--enable-prefix-caching"
  uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 $flag > /tmp/nanoserve_m12_$mode.log 2>&1 &
  wait_ready 8002 && run --backend openai --engine-name nanoserve-apc-$mode --base-url http://localhost:8002 \
    --concurrency 1,4,8,16,32 --tag m12-nanoserve-apc-$mode
  stop_port 8002
done

for mode in off on; do
  pc=0; [ "$mode" = on ] && pc=1
  PREFIX_CACHING=$pc bash bench/servers/vllm.sh > /tmp/vllm_m12_$mode.log 2>&1 &
  wait_ready 8000 && run --backend openai --engine-name vllm-apc-$mode --base-url http://localhost:8000 \
    --concurrency 1,4,8,16,32 --tag m12-vllm-apc-$mode
  stop_port 8000
done
echo "=== done ==="
