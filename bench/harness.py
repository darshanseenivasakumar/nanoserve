"""Benchmark harness.

    uv run python -m bench.harness --backend hf --concurrency 1,4,8
    uv run python -m bench.harness --backend openai --engine-name vllm --base-url http://localhost:8000

Load model: CLOSED LOOP. `concurrency` workers each send a request, wait for it to finish, then send the
next one, so exactly N requests are in flight until the queue runs dry. For each concurrency level we
run warmup requests (discarded), then `repeats` measured runs, and report the median run plus spread.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench.backends.base import Backend, Request, RequestRecord
from bench.metrics import summarize
from bench.prompts import DEFAULT_PROMPTS_PATH, build_requests, load_prompt_set, shared_prefix_text
from bench.report import format_table

RESULTS_DIR = Path(__file__).with_name("results")


async def run_closed_loop(backend: Backend, requests: list[Request], concurrency: int):
    """Run all requests with `concurrency` in flight. Returns (records, wall_time_s)."""
    queue: asyncio.Queue[Request] = asyncio.Queue()
    for r in requests:
        queue.put_nowait(r)
    records: list[RequestRecord] = []

    async def worker():
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            records.append(await backend.generate(req))

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(requests)))))
    wall = time.perf_counter() - t0
    return sorted(records, key=lambda r: r.id), wall


def check_records(records: list[RequestRecord], requests: list[Request]) -> list[str]:
    """Fairness/sanity checks. Any problem here means the numbers can't be trusted."""
    by_id = {r.id: r for r in requests}
    problems = []
    for rec in records:
        req = by_id[rec.id]
        if rec.error:
            problems.append(f"id={rec.id}: error: {rec.error}")
            continue
        if rec.num_output_tokens != req.max_tokens:
            problems.append(f"id={rec.id}: produced {rec.num_output_tokens} tokens, expected {req.max_tokens}")
        if rec.prompt_tokens_reported is not None and rec.prompt_tokens_reported != req.prompt_tokens:
            problems.append(f"id={rec.id}: engine saw {rec.prompt_tokens_reported} prompt tokens, we sent {req.prompt_tokens}")
    return problems


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def env_info() -> dict:
    root = Path(__file__).resolve().parent.parent
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "git_commit": _cmd(["git", "-C", str(root), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "-C", str(root), "status", "--porcelain"])),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _per_request(records: list[RequestRecord], requests: list[Request]) -> list[dict]:
    by_id = {r.id: r for r in requests}
    return [{"id": r.id, "prompt_tokens": by_id[r.id].prompt_tokens, "output_tokens": r.num_output_tokens,
             "ttft": r.ttft, "tpot": r.tpot, "e2e": r.e2e, "chunks": len(r.t_chunks), "error": r.error}
            for r in records]


def make_backend(args) -> Backend:
    if args.backend == "hf":
        from bench.backends.hf import HFBackend
        return HFBackend(args.model, device=args.device, dtype=args.dtype)
    if args.backend == "nanoserve":
        from bench.backends.nanoserve_inproc import NanoserveBackend
        return NanoserveBackend(args.model, device=args.device, dtype=args.dtype, max_seq_len=args.max_seq_len,
                                policy=args.nanoserve_policy, max_num_batched_tokens=args.max_num_batched_tokens,
                                kv_cache=args.kv_cache, block_size=args.block_size, attn_backend=args.attn_backend,
                                draft_model=args.draft_model, num_speculative_tokens=args.num_speculative_tokens,
                                cuda_graphs=args.cuda_graphs, piecewise_graphs=args.piecewise_graphs,
                                fused_ops=args.fused_ops, prefix_caching=args.enable_prefix_caching,
                                kv_cache_tokens=(int(args.kv_cache_gb * 2**30) // _kv_bytes_per_token(args)
                                                 if args.kv_cache_gb else None))
    from bench.backends.openai_http import OpenAIBackend
    return OpenAIBackend(args.base_url, model=args.served_model_name or args.model, engine_name=args.engine_name)


def _kv_bytes_per_token(args) -> int:
    from nanoserve.model import ModelConfig
    import torch
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    return ModelConfig.from_pretrained(args.model).kv_bytes_per_token(dtype)


async def bench_level(backend, args, concurrency: int, requests, warmup_requests) -> dict:
    if hasattr(backend, "batch_size"):  # HF static batching: batch size == concurrency
        backend.batch_size = concurrency
    if warmup_requests:
        await run_closed_loop(backend, warmup_requests, concurrency)
    if hasattr(backend, "reset_engine_stats"):
        backend.reset_engine_stats()
    runs = []
    for i in range(args.repeats):
        records, wall = await run_closed_loop(backend, requests, concurrency)
        runs.append({"repeat": i, "summary": summarize(records, wall),
                     "problems": check_records(records, requests),
                     "per_request": _per_request(records, requests),
                     "sample_outputs": [r.output_text for r in records[:2]]})
    tps = [r["summary"]["output_tok_per_s"] for r in runs]
    median_run = sorted(runs, key=lambda r: r["summary"]["output_tok_per_s"])[len(runs) // 2]
    return {"concurrency": concurrency, "runs": runs, "median_repeat": median_run["repeat"],
            "summary": median_run["summary"],
            "spread": {"output_tok_per_s_min": min(tps), "output_tok_per_s_max": max(tps),
                       "output_tok_per_s_stdev": statistics.stdev(tps) if len(tps) > 1 else 0.0},
            "problems": sorted({p for r in runs for p in r["problems"]}),
            "engine_stats": backend.engine_stats() if hasattr(backend, "engine_stats") else None}


async def amain(args) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompts = load_prompt_set(args.prompts)
    rng = args.output_len_range
    # M12: optional shared system prompt (the prefix-caching workload). Warm-up requests carry it too, so
    # an engine with a prefix cache starts the measured run with the prefix cached (steady state).
    shared = shared_prefix_text(tokenizer, prompts, args.shared_prefix_tokens) if args.shared_prefix_tokens else None
    requests = build_requests(tokenizer, prompts, args.num_prompts, args.max_tokens, args.seed, max_tokens_range=rng,
                              shared_prefix=shared)
    # Warmup uses a differently-shuffled draw so it doesn't pre-run the exact measured order.
    warmup = (build_requests(tokenizer, prompts, args.warmup, args.max_tokens, args.seed + 1, max_tokens_range=rng,
                             shared_prefix=shared)
              if args.warmup else [])

    backend = make_backend(args)
    info = backend.info()
    if hasattr(backend, "server_version"):
        info["server_version"] = await backend.server_version()
    env = env_info()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, any_problem = [], False
    try:
        for c in args.concurrency:
            level = await bench_level(backend, args, c, requests, warmup)
            result = {"timestamp": stamp, "args": vars(args), "backend": info, "env": env,
                      "prompt_tokens_total": sum(r.prompt_tokens for r in requests), **level}
            name = f"{stamp}_{info['engine']}_{args.model.split('/')[-1]}_c{c}{('_' + args.tag) if args.tag else ''}.json"
            (out_dir / name).write_text(json.dumps(result, indent=1))
            rows.append({"engine": info["engine"], "concurrency": c, "summary": level["summary"],
                         "spread": level["spread"], "problems": level["problems"]})
            any_problem |= bool(level["problems"])
            print(format_table(rows[-1:], header=(len(rows) == 1)), flush=True)
            for p in level["problems"][:5]:
                print(f"  PROBLEM: {p}")
    finally:
        await backend.close()

    print()
    print(f"model={args.model} prompts={args.num_prompts} max_tokens={args.max_tokens} "
          f"warmup={args.warmup} repeats={args.repeats} seed={args.seed} gpu={env.get('gpu')}")
    print(format_table(rows))
    print(f"results -> {out_dir}/{stamp}_*.json")
    if any_problem:
        print("SANITY CHECKS FAILED: numbers above are not trustworthy.", file=sys.stderr)
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="nanoserve benchmark harness")
    p.add_argument("--backend", choices=["hf", "openai", "nanoserve"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-seq-len", type=int, default=2048, help="nanoserve: per-request KV reservation")
    p.add_argument("--nanoserve-policy", choices=["continuous", "static"], default="continuous")
    p.add_argument("--kv-cache", choices=["paged", "contiguous"], default="paged", help="nanoserve KV cache")
    p.add_argument("--kv-cache-gb", type=float, default=None, help="nanoserve: explicit KV budget in GiB")
    p.add_argument("--block-size", type=int, default=16, help="nanoserve paged KV block size")
    p.add_argument("--draft-model", default=None, help="nanoserve: enable speculative decoding with this draft")
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--cuda-graphs", dest="cuda_graphs", action="store_true", default=None,
                   help="nanoserve: force CUDA-graph decode on (default: on with paged + triton)")
    p.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false")
    p.add_argument("--no-piecewise-graphs", dest="piecewise_graphs", action="store_false", default=None,
                   help="nanoserve: run mixed prefill+decode steps eagerly (M9 default: piecewise graphs)")
    p.add_argument("--no-fused-ops", dest="fused_ops", action="store_false", default=None,
                   help="nanoserve: unfused PyTorch ops (M10 default: fused Triton kernels)")
    p.add_argument("--enable-prefix-caching", action="store_true",
                   help="nanoserve: reuse cached KV blocks of shared prefixes (M12; off by default for fairness)")
    p.add_argument("--shared-prefix-tokens", type=int, default=0,
                   help="put the same N-token system prompt in front of every request (prefix-caching workload)")
    p.add_argument("--attn-backend", choices=["triton", "sdpa"], default=None,
                   help="nanoserve attention (default: triton kernel on GPU for the paged cache)")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048, help="nanoserve: per-step token budget (M8 default)")
    p.add_argument("--output-len-range", default=None,
                   type=lambda s: tuple(int(x) for x in s.split(",")),
                   help="LO,HI: per-request output length drawn (seeded) from [LO, HI] instead of --max-tokens")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--engine-name", default="openai", help="vllm | llamacpp | nanoserve (selects per-engine fairness flags)")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8])
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--tag", default="", help="free-form label added to result filenames")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
