#!/usr/bin/env bash
# M6 addendum: llama.cpp's speed depends strongly on its slot count (-np). The main M6 run used 32 slots
# (needed for c=32); M0 used 8 slots and got far more at c=8. Re-measure the 8-slot config in the same
# session so the comparison shows llama.cpp's best config at each concurrency (c <= 8).
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
NP=8 bash bench/servers/llamacpp.sh > /tmp/llama_np8.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8001/health >/dev/null && break; sleep 1; done
$H --backend openai --engine-name llamacpp-np8 --base-url http://localhost:8001 --concurrency 1,4,8 --tag m6 > /tmp/np8a.log 2>&1 || echo FAILED-a
$H --backend openai --engine-name llamacpp-np8 --base-url http://localhost:8001 --concurrency 4,8 --output-len-range 16,256 --tag m6-var16-256 > /tmp/np8b.log 2>&1 || echo FAILED-b
pid=$(ss -ltnp | grep ":8001 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"
echo done
