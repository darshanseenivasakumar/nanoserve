# nanoserve — instructions for Claude

nanoserve is a mini LLM inference engine built from scratch to learn and showcase ML-infra skills:
continuous batching, paged KV cache, speculative decoding, Triton kernels (paged attention first),
benchmarked against vLLM and llama.cpp. The owner is a 2nd-year B.Tech student new to the area and must be
able to explain every design choice in an interview: **teach as you build**, define jargon on first use with
simple analogies, and add it to GLOSSARY.md.

## Working rules
1. **Correctness gates speed.** Every stage must match HuggingFace greedy outputs token-for-token (or within
   stated tolerance for kernels) before optimizing.
2. **One change, one measurement.** Never bundle optimizations. Record every result in LOG.md: date, change,
   numbers, what failed.
3. **Tests first:** write the correctness test, then the implementation.
4. **Small steps.** One component at a time, stop for review after each.
5. Before coding any component: explain the concept simply, show data structures and the plan, and wait for OK.
6. **Never modify the benchmark harness or tests to make something pass** without saying so explicitly.
7. Commit after each working step, with measured numbers in the message.
8. Run the tests and show the output before saying a step is done.
9. Flag uncertainty and any shortcut taken.
10. Maintain GLOSSARY.md: add every new term with a plain-language definition.

## Milestones (stop after each)
- M0 repo + benchmark harness (HF / vLLM / llama.cpp adapters)
- M1 naive engine (own loop, contiguous KV cache, greedy) — must match HF exactly
- M2 static batching → continuous batching scheduler
- M3 paged KV cache (block allocator, block tables, free list, preemption)
- M4 Triton paged-attention decode kernel
- M5 speculative decoding with a draft model
- M6 OpenAI-style streaming server, full benchmark, plots, README with honest gap analysis

## Environment
- Windows 11 host, **WSL2 distro `Ubuntu`** (the *default* WSL distro is `docker-desktop`, so from Windows
  always use `wsl -d Ubuntu`). User `darshan`, repo at `~/nanoserve`.
- GPU: RTX 3050 6GB Laptop (Ampere, sm_86, bf16 OK). ~0.6–0.9 GB VRAM is used by the display.
- WSL RAM cap is ~11 GB (WSL default). Raise with `memory=` in `C:\Users\Lenovo\.wslconfig` if needed.
- Windows NVIDIA "sysmem fallback": CUDA allocations past VRAM silently spill into shared system RAM
  (~12 GB) instead of raising OOM, then run slowly. Never use "allocate until OOM" as a capacity metric
  here; admit by an explicit VRAM budget. (The Linux cloud GPU OOMs normally.)
- The Ubuntu virtual disk lives at `D:\WSL\Ubuntu\ext4.vhdx` (moved off C: on 2026-11-10 after C: filled
  up and WSL hit I/O errors). Watch free space on D: before big downloads (models, engines).
- Python 3.11 via uv; torch from the PyTorch cu130 index (see pyproject.toml). Triton ships with torch.
- vLLM lives in its **own venv** `~/envs/vllm` (it pins its own torch). llama.cpp is built in `~/src/llama.cpp`.
- Dev models: `Qwen/Qwen2.5-0.5B-Instruct`, then 1.5B. Final 3B benchmarks run on a rented cloud GPU, so
  model name and device are always CLI flags, never hard-coded.

## Commands
```bash
uv sync                                   # install env
uv run pytest -v                          # all tests (gpu tests included)
uv run pytest -v -m "not gpu"             # fast tests only
uv run pytest -m spec                     # speculative-decoding tests (1.5B, ~8 GB RAM), excluded by default
uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2   # OpenAI-compatible server
uv run python -m bench.summarize "*_m6.json"   # one line per result file
uv run python -m bench.plot --tag m6     # plots into bench/plots/
uv run python -m bench.harness --backend hf --concurrency 1,4,8
uv run python -m bench.harness --backend openai --base-url http://localhost:8000 --engine-name vllm ...
uv run python -m bench.harness --backend nanoserve --concurrency 1,4,8   # our engine, in-process
bash bench/servers/build_llamacpp.sh      # build llama-server (pinned tag, CUDA) + fetch f16 GGUF
bash bench/servers/vllm.sh                # start vLLM server (own venv)
bash bench/servers/llamacpp.sh            # start llama-server
```

## Layout
- `nanoserve/` engine (model.py, kv_cache.py, scheduler.py, engine.py, speculative.py, kernels/, server.py — added per milestone)
- `bench/` harness: `harness.py` CLI, `prompts.py` + `prompts.jsonl`, `metrics.py`, `report.py`,
  `backends/` (hf, openai_http), `servers/` launch scripts, `results/` committed JSON evidence
- `tests/` pytest; `gpu` marker for tests needing CUDA + model download
- `LOG.md` measurement log, `GLOSSARY.md` terms, `README.md`

## Benchmark methodology (do not change silently — rule 6)
- Closed-loop concurrency N; warmup requests discarded; `--repeats` runs, median reported with spread.
- Greedy (temperature 0), **fixed output length** (`ignore_eos` / `min_new_tokens`) so every engine does
  identical work. Chat template applied once by us; raw text sent to every engine; engine-reported prompt
  token count is checked against ours.
- Before any benchmark: close background apps (games/launchers/browsers), plug in, max-performance power
  mode, and run a CONTROL in the same session (re-measure the previous milestone at c=1). Decode here is
  CPU/launch-bound, so background CPU load moves TPOT by 40%+ (seen 24 -> 35 ms during M2).
- Metrics: output tok/s, request/s, TTFT, TPOT, E2E latency p50/p99 (with sample counts).
