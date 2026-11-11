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
                 draft_model: str | None = None, num_speculative_tokens: int = 4, cuda_graphs: bool | None = None,
                 engine: LLMEngine | None = None):
        self.model_name = model_name
        self.spec = draft_model is not None
        if self.spec:
            # M5: speculative decoding (target = model_name, draft = draft_model).
            from nanoserve.engine import load_eos_token_ids
            from nanoserve.model import Qwen2Model
            from nanoserve.speculative import SpeculativeEngine
            target = Qwen2Model.from_pretrained(model_name, device=device, dtype=DTYPES[dtype])
            draft = Qwen2Model.from_pretrained(draft_model, device=device, dtype=DTYPES[dtype])
            self.llm = SpeculativeEngine(target, draft, load_eos_token_ids(model_name),
                                         num_speculative_tokens=num_speculative_tokens, max_seq_len=max_seq_len,
                                         max_num_batched_tokens=max_num_batched_tokens,
                                         target_kv_tokens=kv_cache_tokens, block_size=block_size)
            self.draft_model = draft_model
        else:
            self.llm = engine or LLMEngine.from_pretrained(model_name, device=device, dtype=DTYPES[dtype],
                                                           max_seq_len=max_seq_len, policy=policy,
                                                           max_num_batched_tokens=max_num_batched_tokens,
                                                           kv_cache=kv_cache, kv_cache_tokens=kv_cache_tokens,
                                                           block_size=block_size, attn_backend=attn_backend,
                                                           cuda_graphs=cuda_graphs)
        self.engine = AsyncEngine(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def batch_size(self) -> int:
        return self.llm.max_num_seqs if self.spec else self.llm.scheduler.max_num_seqs

    @batch_size.setter
    def batch_size(self, n: int) -> None:
        if self.spec:
            self.llm.max_num_seqs = n
        else:
            self.llm.scheduler.max_num_seqs = n

    def info(self) -> dict:
        if self.spec:
            e = self.llm
            return {"backend": "nanoserve", "engine": f"nanoserve-spec-k{e.k}", "model": self.model_name,
                    "draft_model": self.draft_model, "num_speculative_tokens": e.k,
                    "dtype": str(e.target.dtype).replace("torch.", ""), "kv_cache": "paged (reserved per request)",
                    "target_kv_tokens": e.tpool.num_blocks * e.block_size,
                    "attn_backend": "triton (graphed decode rows)" if e.t_graph else "sdpa", "cuda_graphs": e.t_graph is not None,
                    "max_seq_len": e.max_seq_len, "torch": torch.__version__}
        m, s = self.llm.model, self.llm.scheduler
        kv = self.llm.kv
        name = (f"nanoserve-{kv.kind}-{s.policy}" + ("-triton" if self.llm.attn_backend == "triton" else "")
                + ("-graphs" if self.llm.graph_runner is not None else ""))
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
