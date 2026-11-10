# nanoserve

A mini LLM inference engine built from scratch to learn ML infrastructure: **continuous batching**, a
**paged KV cache**, a **Triton paged-attention kernel**, **speculative decoding**, and an
**OpenAI-compatible streaming server**. Benchmarked honestly against vLLM and llama.cpp, with every
stage gated on matching HuggingFace's outputs.

Development hardware: an RTX 3050 6 GB *laptop* GPU under WSL2, Qwen2.5-0.5B-Instruct (and 1.5B for
speculative decoding). Everything is parameterised (`--model`, `--device`, KV budget, CUDA arch) so the
3B run on a cloud GPU is a flag change (see "Running on a cloud GPU").

## Results (final, same session, all engines through the same OpenAI streaming client)

Qwen2.5-0.5B-Instruct, bf16 (llama.cpp: f16 GGUF), 64 prompts, greedy, 3 repeats (median; plots show
min-max), all fairness checks passed (exact output lengths, identical prompt token counts).

**Fixed 128 output tokens — output tokens/s**

| engine | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| HF `generate` (static padded batching, in-process) | 26.9 | 87.9 | 173.0 | – | – |
| **nanoserve** (HTTP server) | 27.8 | 107.8 | 208.4 | 379.9 | 652.1 |
| llama.cpp b12247, 8 slots | 105.1 | 193.8 | 485.6 | – | – |
| llama.cpp b12247, 32 slots | 104.7 | 176.4 | 214.3 | 332.0 | 611.7 |
| vLLM 0.33.0 | 121.1 | 436.3 | 802.4 | 1405.2 | 2208.1 |
| nanoserve / vLLM | 23% | 25% | 26% | 27% | 30% |

**Varied output lengths (16–256 per request, seeded) — the realistic case**

| engine | c=4 | c=8 | c=16 | c=32 | TTFT p50 / p99 @ c=16 |
|---|---|---|---|---|---|
| **nanoserve** | 106.5 | 192.5 | 343.1 | 478.8 | 101 / 304 ms |
| llama.cpp, 8 slots | 170.4 | 391.9 | – | – | – |
| llama.cpp, 32 slots | 115.6 | 129.3 | 177.9 | 330.0 | 220 / 585 ms |
| vLLM 0.33.0 | 433.5 | 761.1 | 1282.8 | 1722.1 | 56 / 229 ms |

![fixed-length results](bench/plots/m6.png)
![varied-length results](bench/plots/m6-var16-256.png)

**In one line:** nanoserve reaches **23–30% of vLLM's throughput**, beats HF `generate` from c=4 up, and
its time per token stays nearly flat as load grows (35 → 44 ms from c=1 to c=32) — but every step costs
~4x more than vLLM's, for one dominant reason (below).

## Architecture

```
 HTTP client ──SSE──▶ nanoserve/server.py (FastAPI)          /v1/completions, /v1/chat/completions
                         │  tokenize, validate (greedy only), incremental detokenize, abort on disconnect
                         ▼
                 nanoserve/async_engine.py   engine loop in a background thread; per-request asyncio
                         │                   queues; works across event loops
                         ▼
                 nanoserve/engine.py  LLMEngine.step()   (one iteration = one forward pass)
                         │
     ┌───────────────────┼─────────────────────────────────────────────┐
     ▼                   ▼                                             ▼
 scheduler.py        kv_manager.py / kv_cache.py                   model.py  forward_batch()
 continuous batching  paged pool [blocks,16,kv_heads,dim]           Qwen2 from scratch (plain tensors)
 FCFS admission by    block tables, free list, slot mapping         selective batching: token-wise layers
 seats + token budget explicit KV budget; contiguous mode kept      over ALL tokens at once; attention per
 + KV blocks          for A/B comparison                            request:
 LIFO preemption with                                                 prefill / multi-token -> SDPA
 recompute                                                            decode -> kernels/paged_attention.py
                                                                      (Triton, 1 launch per layer)
                                                                   greedy argmax, 1 GPU->CPU sync / step
 speculative.py: 0.5B draft proposes k tokens, 1.5B target verifies k+1 in one pass, rewind caches
```

## Milestones and what each one showed

| | Milestone | Key result (laptop) |
|---|---|---|
| M0 | Benchmark harness first: closed-loop load, warmup, repeats, TTFT/TPOT/E2E p50/p99, fairness checks (exact output length, identical prompt tokens), HF/vLLM/llama.cpp adapters | baselines: HF 34 / vLLM 121 / llama.cpp 104 tok/s at c=1 |
| M1 | Own Qwen2 forward, contiguous KV cache, greedy loop | **bf16 logits bitwise equal to HF**; generation token-for-token equal; 1,012 kernel launches/step, GPU 37% busy |
| M2 | Selective batching + scheduler with static and continuous policies | varied lengths: continuous **+27-49% throughput, ~30x lower TTFT** than static; fixed lengths: equal (as theory says) |
| M3 | Paged KV cache, block tables, preemption by recompute | same 2 GiB budget: **85 -> 706 concurrent requests (8.3x)**, KV utilisation 11% -> 97%; costs 4-8% speed (gather) |
| M4 | Triton paged-attention decode kernel | per layer 14x faster at B=8, 35-81x at B=32; end-to-end **c=32: 221 -> 730 tok/s**; TPOT flat in concurrency |
| M5 | Speculative decoding (0.5B draft -> 1.5B target) | output **identical** to plain greedy; acceptance 53-67%; but **0.71-0.82x slower** here (draft costs 0.86x the target: both launch-bound) |
| M6 | OpenAI-compatible streaming server + full benchmark | see results above |

Full numbers, every failure and every judgment call: [`LOG.md`](LOG.md). Every term: [`GLOSSARY.md`](GLOSSARY.md).

## How correctness is gated

- **Bitwise vs HuggingFace (bf16) for a single request**: by performing floating-point operations in exactly
  HF's order (RMSNorm in fp32, cos/sin cast before rotation, SDPA with `enable_gqa`, LM head only on rows
  needed). Generation is token-for-token identical to HF `generate` on 8 prompts / 503 tokens.
- **fp32 strict gates** for everything that changes numerics (batching, paging, preemption, speculative
  decoding): token-for-token identical to the single-request path.
- **bf16 tie rule** where exact equality is impossible: batching changes matmul shapes and therefore
  rounding ("batch invariance"). A bf16 divergence is accepted only at a near-tie (reference top-2 logit
  gap <= 0.25, about one bf16 rounding step at these magnitudes); anything else fails.
- **Kernel**: Triton output vs a PyTorch oracle: fp32 <= 1e-5, bf16 <= 1e-2 (measured <= 4.5e-3), for
  the head layouts of 0.5B, 1.5B and 3B.
- 117 tests (`uv run pytest`, plus `uv run pytest -m spec` for the 1.5B speculative tests).

## Where we lose, and why (honest section)

1. **A fixed ~35 ms per decode step, vs ~8 ms for vLLM and ~9 ms for llama.cpp.** This is most of the gap.
   The M1 profile: one decode step = ~1,012 small GPU kernels launched one by one from Python; the GPU is
   busy only ~37% of the step (9 ms of real work in a 24 ms step on a quiet machine). vLLM records the
   whole decode step once as a **CUDA graph** and replays it with one launch, and fuses small ops
   (torch.compile); llama.cpp is C++ and also uses CUDA graphs. Our memory-bandwidth floor for 0.5B on this
   GPU is ~6 ms/step (estimated: ~0.99 GB of weights / ~168 GB/s, bandwidth not measured), so the rest is overhead.
   Fix (not done): capture decode steps as CUDA graphs per batch-size bucket (static input buffers; the
   Triton kernel and slot-mapped writes are graph-safe), and fuse RMSNorm/RoPE/SiLU.
2. **Load scaling is now fine; the base cost is not.** After the M4 kernel, TPOT grows only 35 → 44 ms from
   1 to 32 requests (vLLM 8 → 12 ms). Before it (M3) it grew 33 → 142 ms because attention was a Python loop
   over requests. So the ratio to vLLM stays ~3.3–4.4x across load: one constant factor, not a scaling flaw.
3. **Prefill.** Prompts go through SDPA one request at a time, and there is no chunked prefill: a long prompt
   joining the batch delays everyone's next token, and TTFT grows under load (p50 618 ms at c=32 vs vLLM 339).
4. **One Python process, two busy threads.** The engine thread and the HTTP/tokenizer thread share the GIL;
   measured as noise in M2 (continuous batching looked 12% slower than static in the async harness, equal in a
   single-thread check). vLLM v1 runs the API server and the engine core as separate processes.
5. **Speculative decoding loses here (0.71x at k=4).** It is lossless (output identical to the target), and
   verifying 5 tokens costs only 1.3x one token — but the 0.5B draft costs 0.86x the 1.5B target per forward,
   because both are launch-bound (point 1). A k=4 step needs 4.7 accepted-plus-bonus tokens to break even and
   gets ~3.05. With a ~6 ms draft (CUDA graphs) the same acceptance would predict ~1.6x.
6. **llama.cpp: depends on its configuration.** We beat llama.cpp configured with 32 slots at c ≥ 16 and on
   varied lengths, but **llama.cpp with 8 slots beats us 2.3x at c=8** (486 vs 208 tok/s). Its per-token cost
   drops when the slot count matches the load; ours doesn't need tuning, but it starts from a higher base (point 1).
7. **Not implemented:** CUDA graphs, op fusion, chunked prefill, prefix caching, swap-to-CPU preemption,
   sampling (greedy only), quantization, tensor parallelism, split-K decode for long contexts. Speculative
   engine: reserves KV for prompt + max_new up front (no preemption) and uses SDPA, not the M4 kernel.
8. **Measurement caveats.** Laptop GPU shared with the Windows display; background Windows load moved our
   TPOT by up to 40% during development (24 → 35 ms), so every comparison above was re-measured in one
   session (controls in `LOG.md`). The NVIDIA Windows driver spills over-budget CUDA allocations into system
   RAM instead of failing, so capacity was measured against explicit KV budgets. llama.cpp runs f16 weights,
   the others bf16.

## Numbers worth remembering

- KV cache per token = layers × 2 (K,V) × kv_heads × head_dim × bytes: **12 KB** for Qwen2.5-0.5B, 28 KB for
  1.5B, 36 KB for 3B (bf16). vLLM's log confirms it: 2.25 GiB ↔ 196,992 tokens.
- Contiguous vs paged, same 2 GiB budget: **85 vs 706 concurrent requests**, KV utilisation 11% vs 97%.
- Decode attention per layer at 8 requests: **36.7 ms → 2.6 ms per step** (gather+SDPA loop → one Triton launch).
- Continuous vs static batching on varied lengths: **TTFT 3.4 s → 114 ms** at c=8.

## Reproduce

```bash
uv sync
uv run pytest                      # 112 tests (GPU + model download)
uv run pytest -m spec              # 5 speculative-decoding tests (1.5B model, ~8 GB RAM)

# in-process engine
uv run python -m bench.harness --backend nanoserve --concurrency 1,4,8,16,32 --kv-cache-gb 2

# HTTP server + any OpenAI-compatible benchmark client
uv run python -m nanoserve.server --port 8002 --kv-cache-gb 2
uv run python -m bench.harness --backend openai --engine-name nanoserve --base-url http://localhost:8002

# reference engines
bash bench/servers/vllm.sh                       # vLLM in its own venv (~/envs/vllm)
bash bench/servers/build_llamacpp.sh && bash bench/servers/llamacpp.sh

bash bench/runs/m6_bench.sh && uv run python -m bench.plot --tag m6
```

## Running on a cloud GPU (the 3B benchmark)

Nothing is hard-coded to the laptop. On a Linux box with a bigger GPU:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct
uv run python -m nanoserve.server --model $MODEL --kv-cache-gb 20 --max-num-seqs 256 --port 8002
MODEL=$MODEL GPU_UTIL=0.9 bash bench/servers/vllm.sh
CUDA_ARCH=90 bash bench/servers/build_llamacpp.sh      # 89 for L4/4090, 90 for H100
GGUF_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF GGUF_FILE=qwen2.5-3b-instruct-fp16.gguf bash bench/servers/build_llamacpp.sh
```
(⚠ not verified: check the exact fp16 GGUF filename in that repo first — larger GGUFs are sometimes split
into parts, in which case point llama-server at the first part.)
then run the harness with `--model $MODEL`. The kernel tests already cover 3B's head layout
(16 query / 2 KV heads, dim 128). Notes: on Linux an over-budget allocation OOMs normally (on this
Windows laptop the driver silently spills to system RAM, so capacity is measured against explicit budgets).
