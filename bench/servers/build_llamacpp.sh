#!/usr/bin/env bash
# Build llama.cpp's llama-server with CUDA at a PINNED release tag, and fetch the f16 GGUF model.
# Pinning matters: llama.cpp ships many releases a week, and a benchmark must name the exact build.
#
#   bash bench/servers/build_llamacpp.sh
#
# Needs the CUDA toolkit (nvcc) and cmake: see CLAUDE.md. Arch 86 = RTX 30-series (Ampere);
# override CUDA_ARCH for the cloud GPU (e.g. 89 for L4/4090, 90 for H100).
set -euo pipefail
TAG="${LLAMA_TAG:-b12247}"
SRC="${LLAMA_SRC:-$HOME/src/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-86}"
export PATH="/usr/local/cuda/bin:$PATH"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin tag "$TAG" && git -C "$SRC" checkout -q "$TAG"

cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" --config Release -j"$(nproc)" --target llama-server
"$SRC/build/bin/llama-server" --version

# f16 GGUF of the same model (same precision class as bf16 HF/vLLM; NOT a 4-bit quant).
# GGUF_FILE may list several space-separated parts: bigger GGUFs are split (verified 2026-11-11: the 3B fp16
# is qwen2.5-3b-instruct-fp16-00001-of-00002.gguf + -00002-of-00002.gguf). Point llama-server at part 1.
MODELS="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS"
GGUF_REPO="${GGUF_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
GGUF_FILE="${GGUF_FILE:-qwen2.5-0.5b-instruct-fp16.gguf}"
for f in $GGUF_FILE; do
  if [ ! -f "$MODELS/$f" ]; then
    curl -L --fail -sS -o "$MODELS/$f" "https://huggingface.co/$GGUF_REPO/resolve/main/$f"
  fi
  ls -lh "$MODELS/$f"
done
