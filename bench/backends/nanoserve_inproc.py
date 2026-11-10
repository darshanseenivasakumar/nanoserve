"""Harness backend that runs the nanoserve engine in-process.

M2: requests go through AsyncEngine (engine loop in a background thread), so concurrent requests are
batched by the scheduler: `--nanoserve-policy continuous` (default) or `static`.

The harness sets `batch_size` = concurrency for each level (same convention as the HF backend); here it
caps the scheduler's max_num_seqs. Timestamps come from the engine thread right after each step, i.e.
when the token exists on the CPU (like the HF backend's CUDA-synced timestamps).
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from bench.backends.base import Request, RequestRecord
from nanoserve.async_engine import AsyncEngine
from nanoserve.engine import LLMEngine

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class NanoserveBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16", max_seq_len: int = 2048,
                 policy: str = "continuous", max_num_batched_tokens: int = 8192, kv_cache: str = "paged",
                 kv_cache_tokens: int | None = None, block_size: int = 16, attn_backend: str | None = None,
                 engine: LLMEngine | None = None):
        self.model_name = model_name
        self.llm = engine or LLMEngine.from_pretrained(model_name, device=device, dtype=DTYPES[dtype],
                                                       max_seq_len=max_seq_len, policy=policy,
                                                       max_num_batched_tokens=max_num_batched_tokens,
                                                       kv_cache=kv_cache, kv_cache_tokens=kv_cache_tokens,
                                                       block_size=block_size, attn_backend=attn_backend)
        self.engine = AsyncEngine(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def batch_size(self) -> int:
        return self.llm.scheduler.max_num_seqs

    @batch_size.setter
    def batch_size(self, n: int) -> None:
        self.llm.scheduler.max_num_seqs = n

    def info(self) -> dict:
        m, s = self.llm.model, self.llm.scheduler
        kv = self.llm.kv
        name = f"nanoserve-{kv.kind}-{s.policy}" + ("-triton" if self.llm.attn_backend == "triton" else "")
        return {"backend": "nanoserve", "engine": name, "model": self.model_name,
                "attn_backend": self.llm.attn_backend,
                "device": str(m.device), "dtype": str(m.dtype).replace("torch.", ""),
                "batching": s.policy, "max_num_batched_tokens": s.max_num_batched_tokens,
                "batch_wait_s": s.batch_wait_s, "kv_cache": kv.kind,
                "kv_cache_tokens": (kv.pool.num_blocks * kv.block_size if kv.kind == "paged"
                                    else kv.capacity * kv.max_seq_len),
                "max_seq_len": self.llm.max_seq_len, "torch": torch.__version__}

    def reset_engine_stats(self) -> None:
        self.llm.reset_stats()

    def engine_stats(self) -> dict:
        return self.llm.stats

    async def close(self) -> None:
        self.engine.close()

    async def generate(self, req: Request) -> RequestRecord:
        t_submit = time.perf_counter()
        ids = self.tokenizer(req.prompt, add_special_tokens=False)["input_ids"]
        rec = RequestRecord(id=req.id, t_submit=t_submit, prompt_tokens_reported=len(ids))
        tokens = []
        try:
            async for tok, t in self.engine.generate(ids, req.max_tokens, ignore_eos=True):
                rec.t_chunks.append(t)
                tokens.append(tok)
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
        rec.num_output_tokens = len(tokens)
        rec.output_token_ids = tokens
        rec.output_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return rec
