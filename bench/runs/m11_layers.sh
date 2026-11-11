#!/usr/bin/env bash
# M11: same-session throughput at c=32 for the serving layers above the engine: in-process harness
# (AsyncEngine, no HTTP) vs the HTTP server. Pair with bench/runs/m11_loop_profile.py (bare engine loop).
cd ~/nanoserve
TAG=${1:-m11-layers}
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3 --concurrency 32"
wait_ready() { for i in $(seq 1 300); do curl -sf "localhost:$1/health" >/dev/null && return 0; sleep 1; done; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 5; }
shift || true
uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2 --max-num-seqs 64 "$@" > ~/m11_server.log 2>&1 &
wait_ready 8002 && $H --backend openai --engine-name nanoserve --base-url http://localhost:8002 --tag "$TAG-server" 2>&1 \
  | grep -E "nanoserve|PROBLEM" | tail -1
stop_port 8002
$H --backend nanoserve --kv-cache-gb 2 --tag "$TAG-inproc" 2>&1 | grep -E "nanoserve|PROBLEM" | tail -1
