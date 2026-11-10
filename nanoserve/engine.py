"""M1 engine: one request at a time, contiguous KV cache, greedy decoding.

    prompt ids ──▶ PREFILL: forward(all prompt tokens) ─▶ logits of last position ─▶ argmax = token 1
                   DECODE:  forward(token k, position P+k-1) ─▶ argmax = token k+1   (repeat)

Every decode step processes ONE token and reads every cached K/V. The loop stops at an EOS token
(unless ignore_eos) or after max_new_tokens.

Known costs we accept in M1 (they are the M2/M4 targets):
  * batch size 1: the GPU reads all ~1 GB of weights to produce a single token per step.
  * `int(argmax)` copies one number to the CPU each step, which forces the CPU to wait for the GPU
    (a "sync"). Needed anyway to check EOS and stream the token.
  * Python launches ~350 small GPU kernels per step; for a 0.5B model the GPU often waits on Python.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import torch

from nanoserve.kv_cache import ContiguousKVCache
from nanoserve.model import Qwen2Model, _snapshot


def load_eos_token_ids(name: str) -> list[int]:
    """Stop tokens as HF generate sees them: generation_config.json's eos_token_id (may be a list),
    falling back to config.json."""
    folder = _snapshot(name)
    for fname in ("generation_config.json", "config.json"):
        path = folder / fname
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return eos if isinstance(eos, list) else [eos]
    raise ValueError(f"no eos_token_id found for {name}")


class Engine:
    def __init__(self, model: Qwen2Model, eos_token_ids: list[int], max_seq_len: int = 2048):
        self.model = model
        self.eos_token_ids = list(eos_token_ids)
        self.max_seq_len = max_seq_len

    @classmethod
    def from_pretrained(cls, name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        max_seq_len: int = 2048) -> "Engine":
        return cls(Qwen2Model.from_pretrained(name, device=device, dtype=dtype),
                   eos_token_ids=load_eos_token_ids(name), max_seq_len=max_seq_len)

    def generate(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False) -> Iterator[int]:
        """Yield generated token ids one at a time (greedy). The EOS token, if produced, is yielded too."""
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                             f"exceeds max_seq_len ({self.max_seq_len})")
        m = self.model
        # Contiguous reservation: the full max_seq_len, whatever this request actually needs (see kv_cache.py).
        cache = ContiguousKVCache(m.cfg, self.max_seq_len, device=m.device, dtype=m.dtype)
        eos = set(self.eos_token_ids)

        ids = torch.tensor(prompt_ids, device=m.device)
        logits = m.forward(ids, torch.arange(len(prompt_ids), device=m.device), cache=cache, logits="last")
        pos = len(prompt_ids)
        for step in range(max_new_tokens):
            token = int(logits[-1].argmax())        # greedy; .item() syncs GPU -> CPU
            yield token
            if (token in eos and not ignore_eos) or step == max_new_tokens - 1:
                return
            logits = m.forward(torch.tensor([token], device=m.device), torch.tensor([pos], device=m.device),
                               cache=cache, logits="last")
            pos += 1
