"""KV cache, M1 version: one CONTIGUOUS block per request.

Each request reserves room for `max_seq_len` tokens up front:

    k, v : [n_layers, n_kv_heads, max_seq_len, head_dim]
                                  ├── length ──┤ (filled)  ...unused reservation...

Layout choice: [heads, seq, dim] per layer is exactly what scaled_dot_product_attention wants
([batch, heads, seq, dim]), so serving the cache to attention is a zero-copy slice.

KNOWN WASTE (deliberate, fixed in M3): a request that only ever uses 300 of its 2048 slots still holds
all 2048. With 12 KB/token that is 24 MB reserved for ~3.6 MB used. This "internal fragmentation" is
why vLLM invented the paged KV cache.
"""
from __future__ import annotations

import torch

from nanoserve.model import ModelConfig


class ContiguousKVCache:
    def __init__(self, cfg: ModelConfig, max_seq_len: int, device="cuda", dtype=torch.bfloat16):
        shape = (cfg.n_layers, cfg.n_kv_heads, max_seq_len, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.length = 0  # tokens fully written (all layers)
        self.reserved_bytes = max_seq_len * cfg.kv_bytes_per_token(dtype)

    def write(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store K/V for the current step's T tokens ([1, n_kv_heads, T, head_dim]) at slots
        [length, length+T) of `layer`. Returns K/V for all tokens so far, [1, n_kv_heads, length+T, head_dim],
        as views into the cache (no copy).

        `length` is NOT advanced here: every layer writes the same slots during one forward step, and the
        model calls advance() once at the end of the step.
        """
        T = k_new.shape[2]
        end = self.length + T
        if end > self.max_seq_len:
            raise ValueError(f"KV cache full: need {end} slots, reserved {self.max_seq_len}")
        self.k[layer, :, self.length:end] = k_new[0]
        self.v[layer, :, self.length:end] = v_new[0]
        return self.k[layer, :, :end][None], self.v[layer, :, :end][None]

    def advance(self, T: int) -> None:
        self.length += T

    @property
    def used_bytes(self) -> int:
        return self.reserved_bytes * self.length // self.max_seq_len
