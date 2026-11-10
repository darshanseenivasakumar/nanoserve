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

## 2026-11-10 — M2 in progress: environment drift found before benchmarking
- Synchronous LLMEngine.step() timing (200-token prompts, decode): B=1 35.4 ms, B=2 40.3, B=4 48.7,
  B=8 63.2 ms/step (28 / 50 / 82 / 127 tok/s).
- B=1 looked like a regression vs M1 (24.4 ms TPOT), so ran an A/B in one session: M1 code (f8aa4b1)
  now 34-38 ms, M2 code 35-41 ms. => NOT the refactor: the machine got slower since the M1 benchmark.
- Likely cause: Windows background load 28% CPU (Epic Games Launcher + EOS overlay renderer, Opera),
  power plan "Balanced". Our decode is CPU/launch-bound, so it tracks CPU availability directly.
  GPU was at full clock (2025 MHz, P0), so not the GPU.
- Action: methodology now requires a same-session control (CLAUDE.md). M2 numbers wait for a quiet machine.

## 2026-11-10/11 — M2: static vs continuous batching (selective batching, contiguous KV)
Correctness (before measuring): fp32 batched == alone token-for-token (12 staggered requests, both
policies); bf16 10/12 identical, 2 diverge at near-ties (gaps 0.062, 0.250; approved rule <= 0.25).
Tests: 78 passed.

Control: background apps closed; M1 code vs M2 code A/B at batch 1: 29.5-35 ms vs 29.8-33 ms TPOT
(equal). Machine still ~20% slower than during M0/M1, so HF and vLLM were RE-MEASURED in this session.
All rows below: same session, 64 prompts, warmup 4, median of 3, all sanity checks OK.
Script: bench/runs/m2_bench.sh (sync check: bench/runs/m2_sync_policy_check.py; capacity: bench/runs/m2_max_contiguous_caches.py) (commands below). Summaries: `python -m bench.summarize "*m2session*" "*nanoserve-m2*" "*var16-256*"`.

Workload A: fixed 128 output tokens
| engine | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| HF generate (static, padded)  tok/s | 29.9 | 98.8 | 183.7 | - | - |
| nanoserve M2 static           tok/s | (=cont.) | 99.5 | 151.0 | 184.2 | 221.3 |
| nanoserve M2 continuous       tok/s | 35.4 | 104.2 | 135.4 (108-140) | 161.4 | 246.4 |
| vLLM 0.33.0                   tok/s | 121.9 | 437.6 | 803.8 | 1422.2 | 2212.6 |
| TPOT p50 ms: HF / static / cont / vLLM | 33.3 / - / 28.1 / 8.0 | 39.4 / 38.8 / 38.0 / 8.7 | 40.7 / 49.8 / 57.5 / 8.9 | - / 83.0 / 98.1 / 9.5 | - / 126.3 / 126.5 / 11.6 |
| TTFT p50 ms: HF / static / cont / vLLM | 39 / - / 30 / 29 | 191 / 78 / 91 / 48 | 384 / 156 / 191 / 127 | - / 318 / 325 / 195 | - / 575 / 571 / 336 |

Workload B: varied output length, uniform 16-256 per request (seeded), `--output-len-range 16,256`
| engine | c=4 tok/s | c=8 tok/s | c=16 tok/s | TTFT p50/p99 ms @c=8 | TTFT p50/p99 ms @c=16 |
|---|---|---|---|---|---|
| nanoserve M2 static     | 61.4 | 103.8 | 146.0 | 3,364 / 10,489 | 3,415 / 14,298 |
| nanoserve M2 continuous | 91.5 | 131.8 | 185.6 | 114 / 157      | 161 / 268 |
| vLLM 0.33.0             | 439.7 | 765.4 | 1289.3 | 46 / 136      | 59 / 233 |
(HF not run on B: its backend needs equal lengths per static batch; our static policy shows the effect.)

Results files: bench/results/20261110-165115_* (HF), -171202_* (cont), -173557_* (static),
*_var16-256.json (B), 20261110-18*_vllm_* (vLLM). Same-session M0 comparisons: HF c=1 34.0 -> 29.9,
vLLM c=8 801.2 -> 803.8 (vLLM barely moved: CUDA graphs make it far less CPU-sensitive).

Findings:
1. Continuous batching wins where it should, on VARIED lengths: +49% / +27% / +27% throughput at
   c=4/8/16, and TTFT p50 ~30x lower (3.4 s -> 114 ms at c=8; p99 10.5 s -> 157 ms). Static batches wait
   for their longest member while finished seats sit idle, and new arrivals wait for the whole batch.
2. On FIXED lengths the policies are equal (as predicted): all requests in a batch finish together.
   The harness showed static > continuous at c=8 (151 vs 135, continuous spread 108-140), but a
   synchronous single-thread closed loop (no asyncio) gives static 142-146 vs continuous 147 tok/s with
   identical ~53 ms steps. => the gap is the async layer + noise, not the policy. Suspect GIL contention
   (harness thread tokenizes/submits while the engine thread launches kernels) -- NOT proven.
3. Each extra sequence costs ~4-5 ms per step (TPOT 28 ms @1 -> 57 @8 -> 98 @16 -> 126 @32): the
   per-sequence attention loop (SDPA + 2 cache writes + slicing per sequence per layer). This is why we
   LOSE to HF's padded static batch at c=8 on fixed lengths (151 vs 184 tok/s): HF runs ONE attention call
   per layer for the whole batch. vLLM's TPOT barely grows (8.0 -> 11.6 ms at c=32).
4. vs vLLM: 17% of its throughput at c=8 (fixed), 11% at c=32; 17% at c=8 on varied lengths.
5. Memory: contiguous reservation = 24 MiB/request (2048 x 12,288 B). VRAM after model load 4.06 GiB
   -> at most 173 requests (upper bound; activations need room too).
   SURPRISE: allocating reservations until OOM got 642 (15 GiB) -- the Windows NVIDIA driver's
   "sysmem fallback" spills CUDA allocations into shared system RAM (~12 GB) instead of failing.
   On this laptop "max concurrency before OOM" is therefore not a clean metric: M3 must admit by an
   explicit VRAM budget (like vLLM's --gpu-memory-utilization). The Linux cloud GPU will OOM normally.

What failed / changed along the way:
- Test criterion changed (rule 6): bf16 batched-vs-alone logits check (torch default bf16 tolerance)
  replaced by "no added error vs fp32 (<= 1.5x alone's) + same argmax unless top-2 gap <= 0.25" --
  bf16 alone already fails the default tolerance vs fp32 (0.26-0.41 max |diff|). fp32 gate unchanged.
- Test changed (rule 6): M1's "requests are serialized" test replaced by "requests overlap" (M2 intent).
- My benchmark-log grep missed nanoserve rows (engine name has a hyphen); all results were saved,
  read back with the new bench/summarize.py.
