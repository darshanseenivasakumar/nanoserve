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
                 policy: str = "continuous", max_num_batched_tokens: int = 8192, engine: LLMEngine | None = None):
        self.model_name = model_name
        self.llm = engine or LLMEngine.from_pretrained(model_name, device=device, dtype=DTYPES[dtype],
                                                       max_seq_len=max_seq_len, policy=policy,
                                                       max_num_batched_tokens=max_num_batched_tokens)
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
        return {"backend": "nanoserve", "engine": f"nanoserve-m2-{s.policy}", "model": self.model_name,
                "device": str(m.device), "dtype": str(m.dtype).replace("torch.", ""),
                "batching": s.policy, "max_num_batched_tokens": s.max_num_batched_tokens,
                "batch_wait_s": s.batch_wait_s, "kv_cache": "contiguous per request",
                "max_seq_len": self.llm.max_seq_len, "torch": torch.__version__}

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
