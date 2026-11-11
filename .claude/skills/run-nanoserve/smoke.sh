#!/usr/bin/env bash
# Launch nanoserve's OpenAI-compatible server in the background, wait for /health, drive it over HTTP with
# client.py (greedy, seeded sampling, streaming, error handling), then stop it and check nothing is left.
#
#   bash .claude/skills/run-nanoserve/smoke.sh                 # from the repo root (~/nanoserve)
#
# Env (all optional): MODEL (Qwen/Qwen2.5-0.5B-Instruct), PORT (8002), KV_GB (1),
#   SERVER_ARGS (extra server flags, e.g. "--enable-prefix-caching" or "--engine-in-thread"),
#   KEEP=1 leaves the server running afterwards (stop it: kill the PID listening on $PORT).
# Exit code: 0 = every check passed, 1 = a check failed or the server never came up, 2 = port busy.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."                                   # repo root
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8002}
KV_GB=${KV_GB:-1}
LOG=/tmp/nanoserve_smoke_$PORT.log

listener() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP "pid=\K[0-9]+" | head -1; }
stop() {
  local pid kids
  pid=$(listener)
  [ -z "$pid" ] && return 0
  # Children: the engine process (spawned, owns the GPU) and multiprocessing's resource tracker. On SIGTERM
  # the port closes at once, but they need ~2 s to exit cleanly (server's finally: engine.close()) -- wait.
  kids=$(pgrep -P "$pid")
  kill "$pid"
  for _ in $(seq 1 30); do
    alive=""; for k in $kids; do kill -0 "$k" 2>/dev/null && alive="$alive $k"; done
    [ -z "$(listener)" ] && [ -z "$alive" ] && break
    sleep 0.5
  done
  for k in $kids; do
    if kill -0 "$k" 2>/dev/null; then echo "WARN: child $k still alive 15 s after the server; killing it"; kill "$k"; fi
  done
  echo "stopped server (pid $pid); port $PORT free: $([ -z "$(listener)" ] && echo yes || echo NO)"
}

if [ -n "$(listener)" ]; then echo "port $PORT is busy (pid $(listener)); stop it or set PORT"; exit 2; fi
# shellcheck disable=SC2086
uv run python -m nanoserve.server --model "$MODEL" --port "$PORT" --kv-cache-gb "$KV_GB" ${SERVER_ARGS:-} \
  > "$LOG" 2>&1 &
for i in $(seq 1 300); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
if ! curl -sf "localhost:$PORT/health" >/dev/null; then
  echo "NOT READY after 300 s; last lines of $LOG:"; tail -20 "$LOG"; stop; exit 1
fi
echo "ready after ${i}s -- $(grep -h 'ready\|captured' "$LOG" | tail -1)"

python3 "$HERE/client.py" --base-url "http://localhost:$PORT" --model "$MODEL"
rc=$?
if [ "${KEEP:-0}" = "1" ]; then
  echo "KEEP=1: server left running on :$PORT (pid $(listener)), log $LOG"
else
  stop
fi
exit $rc
