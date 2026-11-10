#!/usr/bin/env bash
cd ~/nanoserve
H="uv run python -m bench.harness --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" 2>&1 | grep -E "^(hf-generate|nanoserve|vllm) |PROBLEM|Traceback|Error" | sort -u; }
run --backend hf --concurrency 1,4,8 --tag m2session
run --backend nanoserve --nanoserve-policy continuous --concurrency 1,4,8,16,32
run --backend nanoserve --nanoserve-policy static --concurrency 4,8,16,32
run --backend nanoserve --nanoserve-policy continuous --concurrency 4,8,16 --output-len-range 16,256 --tag var16-256
run --backend nanoserve --nanoserve-policy static --concurrency 4,8,16 --output-len-range 16,256 --tag var16-256
echo "=== vllm start ==="
bash bench/servers/vllm.sh > /tmp/vllm_m2.log 2>&1 &
for i in $(seq 1 600); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
curl -sf localhost:8000/health >/dev/null && echo "vllm ready after ${i}s" || { echo "vllm FAILED"; tail -5 /tmp/vllm_m2.log; }
run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 1,4,8,16,32 --tag m2session
run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency 4,8,16 --output-len-range 16,256 --tag var16-256
pkill -f "[v]llm serve"; sleep 5
echo "=== done ==="
