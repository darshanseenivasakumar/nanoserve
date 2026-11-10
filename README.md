# nanoserve

A mini LLM inference engine built from scratch to learn ML infrastructure: continuous batching, a paged KV
cache, speculative decoding and Triton kernels, benchmarked honestly against vLLM and llama.cpp.

**Status:** M2 done: own Qwen2 forward pass (bitwise-identical to HuggingFace in bf16 for a single
request), contiguous KV cache, and a continuous-batching scheduler (Orca-style selective batching).

### Laptop results (RTX 3050 6GB, Qwen2.5-0.5B-Instruct, 64 prompts, greedy, same session)
Varied output lengths (16–256 tokens per request), the realistic case:

| engine | tok/s @ c=8 | TTFT p50 @ c=8 | tok/s @ c=16 |
|---|---|---|---|
| nanoserve M2, static batching | 104 | 3,364 ms | 146 |
| **nanoserve M2, continuous batching** | **132** | **114 ms** | **186** |
| vLLM 0.33.0 | 765 | 46 ms | 1,289 |

Fixed 128 output tokens:

| engine | tok/s @ c=1 | @ c=8 | @ c=32 | TPOT p50 @ c=1 |
|---|---|---|---|---|
| HF generate (static, padded) | 30 | 184 | - | 33.3 ms |
| nanoserve M1 (batch size 1)¹ | 40 | 41 | - | 24.4 ms |
| nanoserve M2, continuous | 35 | 135 | 246 | 28.1 ms |
| llama.cpp b12247¹ | 104 | 478 | - | 9.3 ms |
| vLLM 0.33.0 | 122 | 804 | 2,213 | 8.0 ms |

¹ measured in an earlier session on a quieter machine (not directly comparable; see LOG.md).
Where we lose and why (so far): every extra sequence costs ~4–5 ms per step because attention runs as a
Python loop over sequences — the target of M3/M4. Full tables, methodology and caveats: `LOG.md`.

## Milestones
| | Milestone | Status |
|---|---|---|
| M0 | Repo, environment, benchmark harness (HF / vLLM / llama.cpp) | done |
| M1 | Naive engine: own loop, contiguous KV cache, greedy; matches HF exactly | done |
| M2 | Static → continuous batching scheduler | done |
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
