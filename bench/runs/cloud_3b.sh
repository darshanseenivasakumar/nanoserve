#!/usr/bin/env bash
# M14: the 3B benchmark on a rented Linux GPU, in one command (NOT run yet: needs a cloud GPU).
#
#   git clone https://github.com/darshanseenivasakumar/nanoserve.git ~/nanoserve && cd ~/nanoserve && bash bench/runs/cloud_3b.sh
#
# Assumes: Ubuntu, an NVIDIA GPU with >= 24 GB (L4 / A10 / 4090 / A100 / H100), the CUDA toolkit (nvcc),
# cmake, git, curl, and uv. Set CUDA_ARCH for llama.cpp (89 = L4/4090, 80 = A100, 90 = H100).
# Same methodology as the laptop runs: closed loop, 64 prompts, fixed 128 + varied 16-256 outputs,
# 3 repeats, all three engines in ONE session, prefix caching off everywhere.
set -euo pipefail
cd ~/nanoserve
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CUDA_ARCH="${CUDA_ARCH:-89}"
KV_GB="${KV_GB:-12}"
CONC="${CONC:-1,4,8,16,32,64}"
TAG="${TAG:-cloud3b}"

uv sync
uv run pytest -q tests/test_paged_attention_kernel.py tests/test_fused_ops.py   # kernels on this GPU
if [ ! -x "$HOME/envs/vllm/bin/vllm" ]; then
  uv venv "$HOME/envs/vllm" --python 3.12 && uv pip install --python "$HOME/envs/vllm/bin/python" vllm
fi
CUDA_ARCH=$CUDA_ARCH GGUF_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF \
  GGUF_FILE="qwen2.5-3b-instruct-fp16-00001-of-00002.gguf qwen2.5-3b-instruct-fp16-00002-of-00002.gguf" \
  bash bench/servers/build_llamacpp.sh

H="uv run python -m bench.harness --model $MODEL --num-prompts 64 --max-tokens 128 --warmup 4 --repeats 3"
run() { echo "=== $* ==="; $H "$@" || echo "FAILED: $*"; }
wait_ready() { for i in $(seq 1 1200); do curl -sf "localhost:$1/health" >/dev/null && return 0; sleep 1; done; return 1; }
stop_port() { pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$pid" ] && kill "$pid"; sleep 10; }

uv run python -m nanoserve.server --model "$MODEL" --kv-cache-gb "$KV_GB" --max-num-seqs 128 --port 8002 &
wait_ready 8002 && {
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency "$CONC" --tag "$TAG"
  run --backend openai --engine-name nanoserve --base-url http://localhost:8002 --concurrency "$CONC" \
    --output-len-range 16,256 --tag "$TAG-var16-256"
}
stop_port 8002

MODEL=$MODEL GPU_UTIL=0.85 bash bench/servers/vllm.sh &
wait_ready 8000 && {
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency "$CONC" --tag "$TAG"
  run --backend openai --engine-name vllm --base-url http://localhost:8000 --concurrency "$CONC" \
    --output-len-range 16,256 --tag "$TAG-var16-256"
}
stop_port 8000

GGUF=$HOME/models/qwen2.5-3b-instruct-fp16-00001-of-00002.gguf NP=64 bash bench/servers/llamacpp.sh &
wait_ready 8001 && run --backend openai --engine-name llamacpp --base-url http://localhost:8001 \
  --concurrency "$CONC" --tag "$TAG"
stop_port 8001

uv run python -m bench.plot --tag "$TAG"
echo "results: bench/results/*$TAG*.json, plot: bench/plots/$TAG.png"
