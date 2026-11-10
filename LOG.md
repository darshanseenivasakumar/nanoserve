# Measurement log

One entry per change: date, change, exact command, numbers, what failed. Newest at the bottom.
Hardware unless noted: RTX 3050 6GB Laptop (WSL2 Ubuntu), Ryzen 5 7325HS, WSL RAM cap ~11 GB.

---

## 2026-11-10 — M0: repo + environment
- Created `~/nanoserve`, git init, uv project (Python 3.11), torch from PyTorch cu130 index.
- Found: WSL default distro is `docker-desktop` (must use `wsl -d Ubuntu`); WSL RAM cap 11 GB; no nvcc.
- Env check: torch 2.14.1+cu130, triton 3.8.0, transformers 5.22.2. RTX 3050 cap (8,6), bf16 OK,
  5.03 GiB VRAM free. bf16 matmul OK; Triton vector-add max err 0.0.
- FAILED: HF backend hung on the 2nd test: its batcher task was bound to the first event loop
  (pytest-asyncio uses a new loop per test). Fix: restart the batcher when called from a new loop.
- FAILED: vLLM install died mid-download: C: hit 0 MB free (the WSL disk lived on C:), and WSL started
  throwing I/O errors. Fix: `wsl --manage Ubuntu --move D:\WSL\Ubuntu`. Tests afterwards: 19 pass,
  4 fail (only the unimplemented percentile) -- same as before the move, so no corruption.
- Percentile: linear interpolation (= numpy / vLLM bench scripts). Tests: 23 passed.

## 2026-11-10 — M0: baselines on the laptop (Qwen2.5-0.5B-Instruct)
Config for every row: 64 prompts (fixed set, seed 0), 128 output tokens forced (ignore_eos), greedy,
warmup 4, 3 repeats -> median run shown, (min-max) tok/s across repeats. All sanity checks OK
(exact 128 tokens, engine prompt-token count == ours). ~1.0 token per streamed chunk for every engine.

| engine | conc | out tok/s (min-max) | TTFT p50/p99 ms | TPOT p50/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| HF generate (bf16, static batch=conc) | 1 | 34.0 (34-34)    | 33 / 64   | 29.2 / 30.8 | 3.75 / 3.96 |
|                                       | 4 | 97.5 (97-105)   | 188 / 215 | 40.0 / 41.5 | 5.27 / 5.46 |
|                                       | 8 | 185.9 (183-189) | 400 / 406 | 40.4 / 41.6 | 5.49 / 5.68 |
| vLLM 0.33.0 (bf16)                    | 1 | 120.7 (119-121) | 29 / 57   | 8.1 / 8.2   | 1.05 / 1.09 |
|                                       | 4 | 435.4 (435-435) | 64 / 140  | 8.6 / 9.4   | 1.17 / 1.25 |
|                                       | 8 | 801.2 (799-802) | 122 / 169 | 9.0 / 10.0  | 1.26 / 1.34 |
| llama.cpp b12247 (f16 GGUF, 8 slots)  | 1 | 104.3 (104-105) | 35 / 87   | 9.3 / 9.4   | 1.22 / 1.27 |
|                                       | 4 | 184.2 (184-190) | 156 / 247 | 24.3 / 35.2 | 3.24 / 4.72 |
|                                       | 8 | 478.3 (455-479) | 280 / 346 | 14.6 / 15.0 | 2.13 / 2.21 |

Commands: `python -m bench.harness --backend hf --concurrency 1,4,8`;
`--backend openai --engine-name vllm --base-url http://localhost:8000 --tag smoke` (server: bench/servers/vllm.sh);
`--backend openai --engine-name llamacpp --base-url http://localhost:8001 --tag smoke` (server: bench/servers/llamacpp.sh).
Results: bench/results/20261110-150258_* (HF), 20261110-152626_* (vLLM), 20261110-153157_* (llama.cpp).

Observations:
- HF single-stream TPOT 29 ms vs vLLM 8 ms: the GPU is idle most of each HF step; Python + per-layer kernel
  launch overhead dominates for a 0.5B model. vLLM replays CUDA graphs (whole step = one launch).
- HF TTFT grows with conc (33 -> 400 ms) and p50≈p99: static batching gives every request in a batch the
  same first-token time = batch_wait + prefill of the whole left-padded batch.
- HF TPOT rises 29 -> 40 ms from c=1 to c=4, then is flat to c=8: still overhead-bound, so a bigger batch
  is nearly free -> throughput scales ~linearly with batch size. This is the headroom batching exploits.
- vLLM at c=8 is 4.3x HF at c=8 and 6.6x its own c=1; TPOT barely moves (8.1 -> 9.0 ms): decode is
  memory-bandwidth bound, and reading the weights once serves 8 requests.
- ANOMALY (not investigated): llama.cpp c=4 TPOT (24 ms) is worse than c=8 (14.6 ms), and its c=4
  throughput spread is the widest. Suspect a different CUDA matmul path for batch 4 with f16 weights.
  Revisit in M6, not M0.
- vLLM KV cache on this GPU (gpu-memory-utilization 0.7): 2.25 GiB = 196,992 tokens.

Failed along the way:
- vLLM start: FlashInfer JIT needs `ninja` + `nvcc` on PATH -> fixed in bench/servers/vllm.sh.
- First HF baseline run aborted deliberately (CUDA toolkit apt install would have contended for CPU).

Caveats: laptop GPU, display shares the GPU; llama.cpp is f16 vs bf16 for the others; vLLM/llama.cpp
rows are smoke-level comparisons (full comparison is M6, on the cloud GPU too).

## 2026-11-10 — M1: naive engine (own forward, contiguous KV cache, greedy, batch size 1)
Correctness (all before any measurement):
- Forward pass vs HF: fp32 (CPU) max |diff| < 1e-3; **bf16 (GPU) bitwise equal** on 4 prompts (43-612 tok).
- Prefill + cached decode == cache-free forward (fp32, identical argmax).
- Greedy generate vs HF pure-greedy generate: **token-for-token identical**, 8 prompts, 503 tokens
  (7 hit the 64 cap, 1 stopped at EOS). Tests: 55 passed.

Benchmark (same config as the M0 rows): `python -m bench.harness --backend nanoserve --concurrency 1,4,8`
Results: bench/results/20261110-155214_*

| engine | conc | out tok/s (min-max) | TTFT p50/p99 ms | TPOT p50/p99 ms | E2E p50/p99 s |
|---|---|---|---|---|---|
| nanoserve-m1 (bf16, batch 1) | 1 | 40.4 (40-41) | 26 / 51          | 24.4 / 29.5 | 3.13 / 3.78 |
|                              | 4 | 39.3 (38-41) | 9,496 / 12,040   | 24.7 / 33.5 | 12.62 / 15.38 |
|                              | 8 | 40.6 (39-41) | 21,988 / 22,765  | 24.4 / 26.5 | 25.08 / 25.87 |

vs M0 at c=1: HF 34.0 tok/s (TPOT 29.2 ms) -> nanoserve 40.4 (+19%, TPOT 24.4 ms); vLLM 120.7 (8.1 ms).
vs M0 at c=8: HF static batching 185.9 tok/s, vLLM 801.2; nanoserve 40.6 (4.6x behind HF, 20x behind vLLM).

Observations:
- Throughput is flat across concurrency (~40 tok/s): batch size 1, requests serialized. All extra
  concurrency becomes queueing: TTFT p50 goes 26 ms -> 9.5 s -> 22 s. This is the M2 target.
- Faster than HF at c=1 (24.4 vs 29.2 ms TPOT): no generate() machinery, logits processors, or growing
  DynamicCache concatenation; our cache writes in place.
- Profile of one decode step (torch.profiler, 50 steps after warmup, cache at ~220 tokens):
  GPU kernel time 9.0 ms/step, ~1,012 kernel launches/step. Against the real (unprofiled) 24.4 ms TPOT
  the GPU is busy ~37%; the rest is CPU-side launch/Python overhead (~15 ms / 1,012 ~= 15 us per launch).
  Profiler inflates wall time (115 ms/step), so only its GPU-side numbers are used.
- Roofline estimate: ~0.99 GB of weights / ~168 GB/s (assumed laptop 3050 bandwidth, not measured)
  = ~5.9 ms. GPU-side 9.0 ms is ~65% of that bound; wall 24.4 ms is ~24%.
- Memory: each request reserves 2048 x 12 KB = 24 MB of KV cache regardless of length; a 128-token
  benchmark request (~170-720 tokens total) uses 8-35% of it.

What failed: nothing in M1 itself. (Fixed beforehand: repetition_penalty comment said 1.05, real 1.1.)
