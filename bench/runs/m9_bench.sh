#!/usr/bin/env bash
# M9: piecewise CUDA graphs. Same session: server A/B (piecewise off -> on), vLLM control, then an
# in-process token-budget re-sweep (M8 found smaller budgets made TPOT p99 WORSE because mixed steps ran
# eagerly; with piecewise graphs that should no longer be true).
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" > /tmp/m9_last.log 2>&1 || echo "FAILED: $*"; grep -E "PROBLEM|Traceback" /tmp/m9_last.log | head -3; }
wait_ready() { for i in $(seq 1 900); do curl -sf "localhost:$1/health" >/dev/null && { echo "ready :$1 after ${i}s"; return 0; }; sleep 1; done; echo "NOT READY :$1"; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 8; }

for mode in off on; do
  flag=""; [ "$mode" = off ] && flag="--no-piecewise-graphs"
  uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 $flag > /tmp/nanoserve_m9_$mode.log 2>&1 &
  wait_ready 8002 && {
    grep "captured" /tmp/nanoserve_m9_$mode.log
    run --backend openai --engine-name nanoserve-pw-$mode --base-url http://localhost:8002 --concurrency 1,4,8,16,32 --tag m9-pw-$mode
    run --backend openai --engine-name nanoserve-pw-$mode --base-url http://localhost:8002 --concurrency 4,8,16,32 --output-len-range 16,256 --tag m9-pw-$mode-var16-256
  }
  stop_port 8002
done

bash bench/servers/vllm.sh > /tmp/vllm_m9.log 2>&1 &
wait_ready 8000 && run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,8,32 --tag m9-vllm-control
stop_port 8000

for b in 2048 1024 512; do
  run --backend nanoserve --kv-cache-gb 2 --max-num-batched-tokens $b --concurrency 32 --tag m9-b$b
  run --backend nanoserve --kv-cache-gb 2 --max-num-batched-tokens $b --concurrency 32 --output-len-range 16,256 --tag m9-b$b-var
done
for b in 2048 512; do      # same-session control: the M8 behaviour (mixed steps eager)
  run --backend nanoserve --kv-cache-gb 2 --max-num-batched-tokens $b --concurrency 32 --no-piecewise-graphs --tag m9-eager-b$b
  run --backend nanoserve --kv-cache-gb 2 --max-num-batched-tokens $b --concurrency 32 --output-len-range 16,256 --no-piecewise-graphs --tag m9-eager-b$b-var
done
echo "=== done ==="
