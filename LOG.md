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

## 2026-11-11 — Autonomous mode
User, 00:11: "im going to sleepp dont ask anything, u finish the whole project". From here on, each
milestone plan is written here and treated as approved (rule 5 waived by this instruction). Tests-first,
one change per measurement, and commit-with-numbers still apply. Judgment calls are marked DECISION.

## M3 plan (auto-approved) — paged KV cache
- Pool per layer [num_blocks, 16, kv_heads, head_dim], free list, per-request block table, slot mapping
  (one indexed write per layer for all requests). Block size 16 (vLLM default): average waste half a
  block per request; smaller blocks mean longer tables and more gather work.
- Admission by an EXPLICIT token budget (not by OOM: Windows sysmem fallback hides OOM). Contiguous mode
  gets the same budget (capacity = budget / max_seq_len) so the comparison is fair.
- Out of blocks during decode: preempt the most recently admitted request (LIFO), free its blocks,
  put it back at the FRONT of the queue, and later recompute its KV by re-prefilling prompt + generated
  tokens (vLLM's default "recompute" preemption; the alternative, swapping KV to CPU, isn't built).
  Requests that could never fit even alone are rejected on arrival.
- Attention before M4: gather each request's blocks into a dense tensor, then SDPA (an extra copy;
  expected to cost a little speed until the Triton kernel in M4).
- Measures: (1) max concurrent requests under the same 2 GiB budget, contiguous vs paged;
  (2) KV utilisation (used / reserved); (3) throughput/latency vs M2 in the same session;
  (4) preemption count and its effect on correctness.
- Correctness: paged == contiguous token-for-token (fp32 strict; bf16 tie rule); preempted+recomputed
  requests == never-preempted (fp32 strict; bf16 tie rule); no leaked blocks after a run.

## 2026-11-11 — M3 results: paged KV cache
Correctness (tests: 97 passed at M3 steps 2-3):
- One mixed batched step, paged vs contiguous: fp32 < 1e-5; **bf16 bitwise identical** (max |diff| 0).
- Engine paged == contiguous token-for-token (fp32 and bf16), 12 staggered requests.
- Forced preemption (700-token pool, 8 growing requests): 2 preemptions; fp32 identical; bf16 5/8
  identical + 3 near-ties (gaps 0.000, 0.125, 0.125 <= 0.25); zero leaked blocks.
- Multi-token step on an existing context (offset-causal mask) == token-by-token decode (fp32 < 1e-4).

Capacity (bench/runs/m3_capacity.py; same explicit 2 GiB KV budget = 174,762 tokens; 1,500 requests of
the prompt set with 128 forced output tokens; average request needs 354 tokens):
| KV cache | peak concurrent requests | KV utilisation (used/reserved) |
|---|---|---|
| contiguous (2048-token reservation each) | 85 | 11% |
| paged (16-token blocks) | **706 (8.3x)** | **97%** |
Caveat: 706 is the peak while requests are young (prompt + few tokens). When all reach full length the
pool holds ~494 (= 174,762 / 354), still 5.8x, and preemption keeps it there (1 preemption seen).

Speed (bench/runs/m3_bench.sh; same session; both with the 2 GiB budget; continuous batching):
| tok/s | c=1 | c=4 | c=8 | c=16 | c=32 | var c=8 | var c=16 |
|---|---|---|---|---|---|---|---|
| contiguous | - | 94.2 | 143.0 | 194.5 | 236.0 | 139.3 | 186.0 |
| paged (gather + SDPA) | 30.3 | 89.5 | 137.5 | 182.6 | 221.0 | 129.1 | 171.6 |
| TPOT p50 ms contiguous / paged | - / 33.0 | 41.5 / 43.4 | 55.0 / 57.1 | 80.2 / 85.9 | 131.9 / 142.2 | 54.6 / 59.1 | 81.0 / 86.4 |
=> Paged costs 4-8% throughput here: every step copies each request's scattered blocks into a dense
tensor before SDPA (one gather per request per layer). That copy is exactly what M4's kernel removes.
Machine note: c=1 TPOT 33 ms again (the laptop stayed ~20-35% slower than during M1).

DECISION/flag: these M3 runs used the working tree with the M4 kernel code present but attention pinned
to SDPA ("TEMP during M3 benchmark" line in engine.py), i.e. the M3 code path. A first attempt at this
benchmark died silently (background job killed when its wsl.exe call returned; nothing recorded).

## M4 plan (auto-approved) — Triton paged-attention decode kernel
- One launch per layer for ALL one-token decodes; grid (request, kv_head); each program handles the
  GROUP query heads sharing a kv head (pads to 16 for tl.dot); walks the block table, reads K/V tiles in
  place, online softmax. Prefills and multi-token steps stay on SDPA.
- Gates: op-level vs gather+SDPA oracle: fp32 (IEEE dot) <= 1e-5, bf16 <= 1e-2 vs fp32 oracle, shapes of
  0.5B/1.5B/3B, partial blocks, fragmented tables. Engine-level bf16 tie rule vs the SDPA engine.
- Process flag (rule 3): the kernel was written before its test (both while the M3 benchmark ran); the
  test ran and passed before any integration or measurement.

## 2026-11-11 — M4 results
Correctness: kernel fp32 max|diff| <= 1e-5 (all shapes); bf16 3.8e-3 / 4.5e-3 / 4.1e-3 vs fp32 oracle
(0.5B / 1.5B / 3B shapes). Engine triton vs sdpa (bf16, 12 staggered requests): 10/12 identical, 2
diverge at near-ties (gaps 0.062, 0.125). Full suite: 106 passed. M1's single-request Engine stays on
SDPA on purpose (it is the reference that must equal HF bitwise).

Microbenchmark (bench/runs/m4_kernel_microbench.py; one layer, bf16, 0.5B shapes, CUDA events, median of 50):
| ctx \ B | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 128  triton / gather+SDPA ms | 0.079 / 0.331 | 0.102 / 0.756 | 0.096 / 1.425 | 0.073 / 2.741 | 0.068 / 5.538 |
| 512  | 0.098 / 0.381 | 0.086 / 1.023 | 0.109 / 1.528 | 0.087 / 2.906 | 0.113 / 5.768 |
| 1024 | 0.103 / 0.346 | 0.118 / 0.809 | 0.119 / 1.724 | 0.124 / 2.926 | 0.156 / 5.510 |
Speedup 3.4-4.2x at B=1, ~14x at B=8, 35-81x at B=32: the kernel's cost is ~flat in B (one launch);
the loop pays ~0.17 ms per request per layer. Per decode step (24 layers, B=8, ctx 512): 2.6 vs 36.7 ms.
FAILED first: my first microbench reference did int(context_lens[b]) inside its loop = one host-device
sync per request, which the engine never does -> speedups overstated (up to 115x). Fixed the reference to
sync once (.tolist()) and re-ran; numbers above are the fair ones.

End-to-end (bench/runs/m4_bench.sh; same session; paged, 2 GiB budget, continuous batching):
| tok/s | c=1 | c=4 | c=8 | c=16 | c=32 | var c=8 | var c=16 |
|---|---|---|---|---|---|---|---|
| M3 paged gather+SDPA (c=8: same-session control; rest from M3 run) | 30.3 | 89.5 | 130.8 | 182.6 | 221.0 | 129.1 | 171.6 |
| **M4 paged + Triton kernel** | 28.6 | 110.0 | **215.2** | **402.9** | **730.5** | 205.9 | 368.3 |
| TPOT p50 ms, M4 | 34.8 | 35.8 | 36.1 | 37.8 | 39.9 | 35.8 | 37.7 |
- TPOT is now nearly FLAT in concurrency (35.8 -> 39.9 ms from c=4 to c=32; M3: 43 -> 142 ms). The
  per-request cost of attention is gone; what remains is the fixed per-step cost.
- c=1 doesn't improve (34.8 vs 33.0 ms): with one request the step is launch/Python-bound, and the
  kernel path launches about as many ops as it removes (index, kernel, buffer, index_copy).
- vs same-session-ish references: HF static c=8 183.7 -> we now beat it (215); vLLM c=8 803.8 / c=32
  2212.6 -> we are at 27% / 33%. Remaining gap = the flat ~35 ms base step vs vLLM's ~8 ms: ~1,000
  Python-launched kernels per step (vLLM replays CUDA graphs) -- see README "where we lose".

## M5 plan (auto-approved) — speculative decoding
- Target Qwen2.5-1.5B-Instruct (downloaded, 3.09 GB), draft Qwen2.5-0.5B-Instruct (same vocab). Greedy,
  fixed lookahead k. Draft proposes k (first step catches up on unseen tokens), target verifies k+1
  positions in one forward (logits only for those rows), accept longest matching prefix + target token,
  rewind both paged caches. Blocks reserved per request at admission (no preemption); SDPA for both models.
- Gates: output == plain greedy target, fp32 strict; bf16 tie rule; self-speculation sanity check.
- Tests run separately (pytest -m spec): 1.5B fp32 on CPU needs ~6 GB RAM.

## 2026-11-11 — M5 results
Correctness (pytest -m spec: 5 passed): self-speculation acceptance 1.000 + identical; **0.5B -> 1.5B fp32
output token-for-token identical to plain greedy 1.5B** (acceptance 0.625, 3.06 tokens/target forward);
batched == one-at-a-time; bf16 acceptance 0.603, 3/4 identical + 1 near-tie (0.125).

Benchmark (bench/runs/m5_bench.sh; 1.5B target, 32 prompts x 128 forced tokens, 2 repeats, 0.25 GiB
target KV, same session; peak VRAM 4,980 MiB, no spill):
| 1.5B | c=1 tok/s | c=1 TPOT p50 | c=4 tok/s | c=4 TPOT p50 | acceptance |
|---|---|---|---|---|---|
| plain greedy (paged + Triton) | 23.2 | 42.9 ms | 87.6 | 44.3 ms | - |
| speculative k=2 | 19.1 (0.82x) | 51.6 ms | 45.4 (0.52x) | 83.6 ms | 0.66-0.67 |
| speculative k=4 | 16.5 (0.71x) | 61.5 ms | 40.7 (0.46x) | 88.7 ms | 0.53 |
(tokens_per_target_forward in the result JSON counts all requests of a batch: c=4 k=4 11.05 = ~2.8/request.)

WHY it loses here (bench/runs/m5_costs.py, batch 1, ctx ~300, CUDA events):
  target 1.5B 1-token forward 36.5 ms | draft 0.5B 1-token forward 31.5 ms | target verify 5 tokens 47.3 ms
- Verification IS nearly free (5 tokens cost 1.3x one token) -- the premise of speculative decoding holds.
- But the draft is NOT cheap: 0.86x the target's cost, because both forwards are dominated by ~1,000
  Python-launched kernels (launch-bound), not by weight reads. A k=4 step costs 4 x 31.5 + 47.3 = 173 ms
  and must emit >= 4.7 tokens to break even; it emits ~3.05. Predicted 0.64x vs measured 0.71x (c=1).
- At c=4 it is worse still: plain decoding already shares each step across 4 requests (TPOT flat), while
  the speculative engine runs k+1 serial forwards per step on the SDPA path (no Triton kernel there).
- What would make it win: a draft forward of ~6 ms (CUDA graphs; memory floor for 0.5B on this GPU)
  predicts (4 x 6 + 47) / 3.05 = 23 ms/token vs 36.5 -> ~1.6x at c=1. Or a free draft (prompt-lookup /
  n-gram). Classic result reproduced: speculative decoding is a low-load LATENCY tool, and only when the
  draft is much cheaper than the target.

## M6 plan (auto-approved) — server + final benchmark
- FastAPI server, OpenAI completions + chat, SSE streaming with include_usage, greedy only (reject anything
  that would change outputs), incremental detokenizer, abort on disconnect. Tested through the SAME client
  the benchmark uses for vLLM/llama.cpp (httpx ASGI transport).
- Final benchmark: every engine over HTTP through one client, same session, fixed + varied workloads,
  c = 1..32; HF in-process baseline; plots; README with an honest gap section.

## 2026-11-11 — M6 results (final)
Tests: 112 passed (default suite) + 5 passed (pytest -m spec). Server: 6 tests incl. benchmark client
over ASGI, output == engine greedy, disconnect aborts and frees KV. Live smoke over real uvicorn OK.
HTTP overhead: nanoserve server vs in-process M4: c=1 27.8 vs 28.6, c=4 107.8 vs 110.0 tok/s (~2%).

Scripts: bench/runs/m6_bench.sh (+ bench/runs/m6_llamacpp_np8.sh), plots: `python -m bench.plot --tag m6`.
Fixed 128 tokens, tok/s (TPOT p50 ms):
| engine | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| HF generate (in-process) | 26.9 (36.8) | 87.9 (43.6) | 173.0 (43.4) | - | - |
| nanoserve server | 27.8 (35.4) | 107.8 (36.6) | 208.4 (37.3) | 379.9 (39.3) | 652.1 (44.0) |
| llama.cpp 8 slots | 105.1 (9.3) | 193.8 (23.6) | 485.6 (14.3) | - | - |
| llama.cpp 32 slots | 104.7 (9.3) | 176.4 (23.8) | 214.3 (37.6) | 332.0 (47.7) | 611.7 (45.8) |
| vLLM 0.33.0 | 121.1 (8.1) | 436.3 (8.7) | 802.4 (9.0) | 1405.2 (9.7) | 2208.1 (11.6) |
Varied 16-256, tok/s: nanoserve 106.5 / 192.5 / 343.1 / 478.8 (c=4/8/16/32); llama.cpp-np8 170.4 / 391.9;
llama.cpp-np32 115.6 / 129.3 / 177.9 / 330.0; vLLM 433.5 / 761.1 / 1282.8 / 1722.1.
TTFT p50 @ varied c=16: nanoserve 101 ms, llama.cpp-np32 220 ms, vLLM 56 ms.

Findings:
- nanoserve = 23-30% of vLLM throughput; the gap is a ~constant ~4x per-step cost (launch overhead).
- Beats HF generate from c=4; beats llama.cpp-np32 at c >= 16 and on varied lengths at c >= 8.
- FAILED/CORRECTED claim: the 32-slot llama.cpp run made nanoserve look competitive at c=8 (208 vs 214).
  M0 had shown llama.cpp at 478 with 8 slots, so I re-ran the 8-slot config in the same session: 485.6
  at c=8 (2.3x nanoserve). llama.cpp's speed depends on matching -np to the load; the README reports both.
- llama.cpp c=4 anomaly from M0 persists (TPOT 23.6 ms at c=4 vs 14.3 at c=8 with 8 slots); not investigated.
- openai_http: engine names like "llamacpp-np8" now map to the llamacpp fairness flags (prefix match).

Where we lose and why: README.md, "Where we lose, and why".

## 2026-11-11 — M7: CUDA-graph decode steps (user: "Yes, add CUDA graphs")
Plan: record each PURE-decode step (every request one token, Triton kernel path) as a CUDA graph per
batch-size bucket (1,2,4,8,16,24,32,48,64,96,128), captured lazily, shared memory pool. Inputs in static
device buffers overwritten before replay; batches between buckets padded; padding rows write/read only a
reserved "garbage" KV block. Steps with any prefill stay eager. On by default with paged + triton.

Correctness (tests/test_cuda_graphs.py, 5 passed; full suite 117 passed):
- graph step vs eager step at an exact bucket size (B=4): BITWISE equal (max |diff| 0).
- padded batch (3 real rows in the 4-graph): within the tie rule.
- engine with graphs vs without, 12 staggered requests: 12/12 token-identical (33 graph / 8 eager steps).
- padding only touched the garbage block; only it stays allocated after a run.

Synchronous step probe (bench/runs/m7_step_probe.py; same process, interleaved A/B):
| batch | eager ms/step | graphs ms/step | speedup |
|---|---|---|---|
| 1 | 33.9 | 9.8 | 3.48x |
| 8 | 33.8 | 10.6 | 3.19x |
| 32 | 32.9 | 12.3 | 2.67x |

Benchmark (bench/runs/m7_bench.sh; same session; Windows load checked: Steam open but idle, no measurable CPU):
In-process A/B (paged, triton, 2 GiB): tok/s c=1/8/32 eager 29.2 / 224.7 / 802.0 -> graphs 88.5 / 558.7 /
1517.1 (3.0x / 2.5x / 1.9x). TPOT p50 33.6/33.9/35.5 -> 11.0/12.9/16.0 ms.
HTTP server with graphs (same client as M6):
  fixed tok/s c=1/4/8/16/32: 94.5 / 314.1 / 553.8 / 943.4 / 1440.4 (TPOT p50 10.3 / 12.1 / 13.2 / 14.2 / 17.4 ms)
  varied 16-256 c=4/8/16/32: 304.1 / 504.3 / 830.2 / 1116.6 (TTFT p50 54 / 59 / 77 / 113 ms)
vLLM same-session control c=1/8/32: 121.1 / 798.5 / 2175.9 (M6: 121.1 / 802.4 / 2208.1, within 2%).
=> nanoserve = 78% / 69% / 66% of vLLM (c=1/8/32), 65-70% on varied lengths (vs 23-30% in M6).
=> now beats llama.cpp's best measured config at c=8 (554 vs 486 fixed; 504 vs 392 varied).
Remaining gaps: TTFT under load (prefill eager, per-request SDPA, no chunked prefill: p50 606 vs 335 ms at
c=32); per-step Python work outside the graph (batch building, H2D copies, argmax + tolist, GIL); no op
fusion. Speculative engine does not use graphs (not re-measured).
Plots: `python -m bench.plot --tag m7,m6` (plot.py now merges tags; nanoserve M6 shown as "nanoserve [m6]").
Process note: /tmp in WSL was cleared by a WSL restart mid-session; patch scripts now live in the session
scratchpad.
