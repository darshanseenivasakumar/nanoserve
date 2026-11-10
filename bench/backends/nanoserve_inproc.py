"""Harness backend that runs the nanoserve engine in-process.

M1 engine has batch size 1, so concurrent requests are SERIALIZED with a lock: request #2 waits until
#1 finishes. That waiting is real queueing delay and is counted in TTFT, exactly as a user would feel it.
(M2's continuous batching replaces this lock with a scheduler.)

Timestamps: the engine yields a token only after `int(argmax)` has copied it to the CPU, which waits
for the GPU, so each timestamp marks a token that really exists.
"""
from __future__ import annotations

import asyncio
import time

import torch
from transformers import AutoTokenizer

from bench.backends.base import Request, RequestRecord
from nanoserve.engine import Engine

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class NanoserveBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 max_seq_len: int = 2048, engine: Engine | None = None):
        self.model_name = model_name
        self.engine = engine or Engine.from_pretrained(model_name, device=device, dtype=DTYPES[dtype],
                                                       max_seq_len=max_seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def info(self) -> dict:
        m = self.engine.model
        return {"backend": "nanoserve", "engine": "nanoserve-m1", "model": self.model_name,
                "device": str(m.device), "dtype": str(m.dtype).replace("torch.", ""),
                "batching": "none (batch size 1, requests serialized)", "kv_cache": "contiguous per request",
                "max_seq_len": self.engine.max_seq_len, "torch": torch.__version__}

    async def close(self) -> None:
        pass

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:  # an asyncio.Lock belongs to one event loop
            self._lock, self._loop = asyncio.Lock(), loop
        return self._lock

    async def generate(self, req: Request) -> RequestRecord:
        t_submit = time.perf_counter()
        async with self._get_lock():
            try:
                return await asyncio.to_thread(self._run, req, t_submit)
            except Exception as e:
                return RequestRecord(id=req.id, t_submit=t_submit, error=f"{type(e).__name__}: {e}")

    def _run(self, req: Request, t_submit: float) -> RequestRecord:
        ids = self.tokenizer(req.prompt, add_special_tokens=False)["input_ids"]
        rec = RequestRecord(id=req.id, t_submit=t_submit, prompt_tokens_reported=len(ids))
        tokens = []
        for tok in self.engine.generate(ids, req.max_tokens, ignore_eos=True):
            rec.t_chunks.append(time.perf_counter())
            tokens.append(tok)
        rec.num_output_tokens = len(tokens)
        rec.output_token_ids = tokens
        rec.output_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return rec
