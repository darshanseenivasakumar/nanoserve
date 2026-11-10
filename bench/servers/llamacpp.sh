#!/usr/bin/env bash
# Launch llama.cpp's llama-server (OpenAI-compatible) for benchmarking.
#
#   GGUF=~/models/qwen2.5-0.5b-instruct-fp16.gguf PORT=8001 NP=8 bash bench/servers/llamacpp.sh
#
# Flags:
#   -ngl 999          offload every layer to the GPU
#   -np  N            parallel "slots" = max requests decoded together (continuous batching across slots).
#                     Set it to the highest concurrency you benchmark.
#   -c   CTX          TOTAL context shared by all slots: each slot gets CTX/NP tokens. Our longest prompt
#                     + output is ~850 tokens, so 1024 per slot.
#   prompt caching is disabled per request (cache_prompt=false) by the harness, not here.
# Precision: f16 GGUF (HF/vLLM use bf16; llama.cpp's GGUF path is f16) -- a small, flagged mismatch.
set -euo pipefail
GGUF="${GGUF:-$HOME/models/qwen2.5-0.5b-instruct-fp16.gguf}"
PORT="${PORT:-8001}"
NP="${NP:-8}"
CTX="${CTX:-$((NP * 1024))}"
BIN="${LLAMA_SERVER:-$HOME/src/llama.cpp/build/bin/llama-server}"

exec "$BIN" -m "$GGUF" --port "$PORT" -ngl 999 -np "$NP" -c "$CTX" "$@"
