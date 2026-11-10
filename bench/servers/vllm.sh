#!/usr/bin/env bash
# Launch vLLM's OpenAI-compatible server for benchmarking. vLLM lives in its own venv (~/envs/vllm)
# because it pins its own torch version.
#
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8000 bash bench/servers/vllm.sh
#
# Fairness flags (see CLAUDE.md methodology):
#   --no-enable-prefix-caching   our engine has no prefix cache; repeats would otherwise hit vLLM's
#   --generation-config vllm     ignore the model's generation_config.json defaults (repetition_penalty)
#   --dtype bfloat16             same precision as the HF baseline
# Laptop-sized memory flags (6 GB GPU, display uses ~0.6-0.9 GB):
#   --gpu-memory-utilization     fraction of TOTAL VRAM vLLM may claim (weights + KV cache + activations)
#   --max-model-len 2048         longest prompt+output we need; smaller = less KV memory reserved
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.7}"
VENV="${VLLM_VENV:-$HOME/envs/vllm}"
# FlashInfer (vLLM's attention kernels) JIT-compiles CUDA code on first start: it needs `ninja`
# (lives in the venv's bin/) and `nvcc` (CUDA toolkit). We don't activate the venv, so put both on PATH.
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 2048 \
  --no-enable-prefix-caching \
  --generation-config vllm \
  "$@"
