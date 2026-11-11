---
name: run-nanoserve
description: Start, run, smoke-test and drive nanoserve, the from-scratch LLM inference engine with an OpenAI-compatible server (lives in WSL Ubuntu at ~/nanoserve). Use when asked to run or start nanoserve, check the server works end to end, send it chat/completion requests, try sampling or streaming, exercise an engine change in-process, or run its tests.
---

nanoserve is a GPU inference server for Qwen2.5 models. Drive it with **`.claude/skills/run-nanoserve/smoke.sh`**
(launches the server in the background, waits for `/health`, runs 13 HTTP checks via `client.py`, stops
it; exit code = verdict) or, for changes to engine internals, **`engine_direct.py`** (same engine,
in-process, no HTTP). All paths below are relative to the repo root, `~/nanoserve`, inside WSL Ubuntu.

## Prerequisites

Already present on this machine: WSL2 Ubuntu, an NVIDIA GPU (RTX 3050 6 GB laptop), `uv`, CUDA 13 PyTorch
in `.venv`. From Windows, always name the distro (the default one is docker-desktop):

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd ~/nanoserve && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

## Setup

```bash
cd ~/nanoserve
uv sync
uv run python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
# -> 2.14.1+cu130 3.8.0 True
```

Model weights download from Hugging Face on first use (cached afterwards). No build step: Triton kernels
compile on first launch.

## Run (agent path)

**HTTP server, end to end** (~45 s; ~20 s of it is startup):

```bash
bash .claude/skills/run-nanoserve/smoke.sh
```

It prints `ready after 19s -- engine process ready; captured CUDA graphs in 4.6 s`, then one `[PASS]`/`[FAIL]`
line per check (models list, greedy + determinism, seeded sampling: same seed = same text, different seed
= different text, SSE streaming with `[DONE]`/finish_reason/usage and TTFT, raw `/v1/completions`, four
invalid parameters rejected with HTTP 400), then `ALL CHECKS PASSED` and `stopped server ... port 8002
free: yes`. Exit 0 = all passed, 1 = a check failed or no `/health` within 300 s (log tail printed),
2 = port busy.

| env var | default | |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | |
| `PORT` | `8002` | |
| `KV_GB` | `1` | KV-cache budget in GiB |
| `SERVER_ARGS` | – | extra server flags, e.g. `--enable-prefix-caching`, `--engine-in-thread`, `--no-cuda-graphs` |
| `KEEP` | `0` | `1` leaves the server running after the checks |

Server log: `/tmp/nanoserve_smoke_<PORT>.log`. To poke a kept server yourself:

```bash
KEEP=1 SERVER_ARGS="--enable-prefix-caching" bash .claude/skills/run-nanoserve/smoke.sh
python3 .claude/skills/run-nanoserve/client.py --base-url http://localhost:8002
kill "$(ss -ltnp | grep ':8002 ' | grep -oP 'pid=\K[0-9]+' | head -1)"
```

(`client.py` is stdlib-only, so plain `python3` works; it never starts or stops anything.)

**Engine in-process** (for edits to `engine.py`, `scheduler.py`, `kv_*.py`, `model.py`, `sampling.py`,
`cuda_graphs.py`, `kernels/`): builds the engine exactly like the server (`engine_process.build_engine`),
batches 3 requests (2 greedy, 1 seeded sample), prints answers and step counters. Stack traces land in your
terminal instead of a child process's log.

```bash
uv run python .claude/skills/run-nanoserve/engine_direct.py
# -> 50 tokens in 0.52 s; steps 30 (graph 29, piecewise 1); fused_ops True; prefix_caching False
uv run python .claude/skills/run-nanoserve/engine_direct.py --no-graphs --prefix-caching
# -> same text; 50 tokens in 2.03 s; steps 30 (graph 0, piecewise 0); ...; prefix_caching True
```

Flags: `--model`, `--max-new`, `--no-graphs`, `--no-fused-ops`, `--prefix-caching`.

## Run (human path)

```bash
uv run python -m nanoserve.server --port 8002 --kv-cache-gb 1   # -> serves on :8002; Ctrl-C to stop
```

## Test

```bash
uv run pytest -q            # 175 passed (~10 min)
uv run pytest -m spec -q    # 8 passed (~4 min): 1.5B model, speculative decoding + fast path
```

## Gotchas

- **Never measure or debug two GPU servers at once.** The GPU has 6 GB: a running vLLM (port 8000) or
  another nanoserve next to this one runs out of memory or skews timings. `smoke.sh` refuses a busy port
  (exit 2), but it can't see a server on another port.
- **Startup is ~20 s and `/health` only answers after it.** The engine process loads the model and
  captures ~600 CUDA graphs before the server accepts traffic. Poll `/health`; don't sleep a fixed time.
- **On SIGTERM the port closes at once, but the engine child and multiprocessing's resource tracker take
  ~2 s to exit** (the server's `finally: engine.close()`). Children still alive in that window are not a
  leak; `smoke.sh` waits up to 15 s and only then kills and warns.
- **An omitted `temperature` means greedy** (OpenAI's default would be 1.0). Sampling needs an explicit
  `temperature`; add `seed` for reproducible text.
- **Wrong facts or a sudden switch to Chinese are the 0.5B model, not the engine.** Seen in this run:
  "Mercury ... visited by humans"; a temperature-0.9 poem containing "烦恼". Judge engine changes by the
  tests (bitwise / tie-rule gates), not by answer quality.
- **`--no-graphs` is ~4x slower** (2.03 vs 0.52 s for the same 50 tokens). Fine for debugging; never quote
  its timings.
- **From Windows Git Bash, `$VARS`, `${PIPESTATUS[0]}` and nested quotes inside
  `wsl.exe -d Ubuntu -- bash -lc '...'` arrive empty or broken.** Put multi-line logic in a script file and
  run `wsl.exe -d Ubuntu -- bash -lc 'bash /mnt/c/<path>/script.sh'`. Calling
  `wsl.exe -d Ubuntu -- bash /mnt/c/...` without `-lc` failed with exit 127 here.
- **WSL is capped at 11 GB of RAM and `pytest -m spec` peaks at ~11.4 GB** (fp32 1.5B on CPU). It has
  crashed the whole WSL VM twice. Raising `memory=` in `C:\Users\Lenovo\.wslconfig` is the user's call.

## Troubleshooting

- **A background job ends with no output, and `uptime` says `up 0 min`**: the WSL VM restarted (out of
  memory). Re-run; for the spec tests, run them alone.
- **`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB ... 1.84 GiB is free`**: two
  copies of the 1.5B model on the GPU (seen when two test modules each loaded one). Load one at a time.
- **`python3: can't open file '/patch_....py'`**: a `$VAR` path came through `wsl.exe ... bash -lc '...'`
  empty. Use a script file (see Gotchas).
- **`port 8002 is busy (pid N); stop it or set PORT`**: something already listens there. Kill that PID or
  run with `PORT=8003`.
- **`WARN: child N still alive 15 s after the server; killing it`**: the engine process did not shut down
  on its own. That is a real hang (normally ~2 s); check `/tmp/nanoserve_smoke_<PORT>.log`.
