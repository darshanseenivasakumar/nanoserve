# nanoserve

A mini LLM inference engine built from scratch to learn ML infrastructure: continuous batching, a paged KV
cache, speculative decoding and Triton kernels, benchmarked honestly against vLLM and llama.cpp.

**Status:** M0 — benchmark harness. The engine itself starts in M1.

## Milestones
| | Milestone | Status |
|---|---|---|
| M0 | Repo, environment, benchmark harness (HF / vLLM / llama.cpp) | in progress |
| M1 | Naive engine: own loop, contiguous KV cache, greedy; matches HF exactly | |
| M2 | Static → continuous batching scheduler | |
| M3 | Paged KV cache | |
| M4 | Triton paged-attention decode kernel | |
| M5 | Speculative decoding | |
| M6 | OpenAI-style streaming server, full benchmark, plots, gap analysis | |

## Setup (WSL2 Ubuntu, NVIDIA GPU)
```bash
uv sync
uv run pytest -v
```

## Running the benchmark
```bash
# HuggingFace generate baseline (in-process, static batches of N)
uv run python -m bench.harness --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 1,4,8

# Any OpenAI-compatible server (vLLM, llama.cpp, later nanoserve)
bash bench/servers/vllm.sh          # in another terminal
uv run python -m bench.harness --backend openai --engine-name vllm --base-url http://localhost:8000
```
Results land in `bench/results/*.json`. Methodology: see `CLAUDE.md` → Benchmark methodology.
