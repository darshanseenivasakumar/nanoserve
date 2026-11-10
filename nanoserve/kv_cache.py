"""KV caches.

M3 adds the PAGED cache (bottom of this file); the M1/M2 contiguous cache stays for A/B comparison.

------------------------------------------------------------------------------------------------------
Contiguous (M1/M2): one block per request. Each request reserves room for `max_seq_len` tokens up front:

    k, v : [n_layers, n_kv_heads, max_seq_len, head_dim]
                                  ├── length ──┤ (filled)  ...unused reservation...

Layout choice: [heads, seq, dim] per layer is exactly what scaled_dot_product_attention wants
([batch, heads, seq, dim]), so serving the cache to attention is a zero-copy slice.

KNOWN WASTE (deliberate, fixed in M3): a request that only ever uses 300 of its 2048 slots still holds
all 2048. With 12 KB/token that is 24 MB reserved for ~3.6 MB used. This "internal fragmentation" is
why vLLM invented the paged KV cache.
"""
from __future__ import annotations

import math

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


# ------------------------------------------------------------------------------------------------------
# Paged KV cache (M3), the PagedAttention idea from vLLM (SOSP 2023), borrowed from OS virtual memory:
#
#   pool k[layer] : [num_blocks, block_size, n_kv_heads, head_dim]     one big preallocated array
#   free list     : block ids not owned by anyone
#   block table   : per request, which blocks hold its tokens, in order ("page table")
#
#       request A table: [7, 2, 9]   -> tokens 0-15 in block 7, 16-31 in block 2, 32-40 in block 9
#       request B table: [4, 11]
#       slot of token t = table[t // block_size] * block_size + t % block_size
#
# A request only ever holds ceil(tokens / block_size) blocks, so the waste is < 1 block per request
# (vs. max_seq_len - length for the contiguous cache). Blocks need not be adjacent: that's the point.
# ------------------------------------------------------------------------------------------------------
class NoFreeBlocks(Exception):
    """The pool is exhausted: the scheduler must preempt someone or wait."""


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))   # pop() from the end is O(1)
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise NoFreeBlocks()
        block = self._free.pop()
        self._used.add(block)
        return block

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._used:
                raise ValueError(f"block {b} freed twice (or never allocated)")
            self._used.remove(b)
            self._free.append(b)


class BlockTable:
    """One request's view of the pool: its blocks and how many of its tokens have KV stored."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: list[int] = []
        self.num_tokens = 0          # tokens whose K/V are written (committed)

    def blocks_needed(self, n_new: int) -> int:
        return max(0, math.ceil((self.num_tokens + n_new) / self.block_size) - len(self.blocks))

    def reserve(self, n_new: int, allocator: BlockAllocator) -> None:
        """Make sure there is room for n_new more tokens (allocates whole blocks)."""
        for _ in range(self.blocks_needed(n_new)):
            self.blocks.append(allocator.allocate())

    def slots(self, n_new: int) -> list[int]:
        """Flat pool positions for the next n_new tokens (call reserve() first)."""
        bs = self.block_size
        return [self.blocks[p // bs] * bs + p % bs for p in range(self.num_tokens, self.num_tokens + n_new)]

    def commit(self, n: int) -> None:
        self.num_tokens += n

    def truncate(self, n_tokens: int, allocator: BlockAllocator) -> None:
        """Forget tokens past n_tokens (speculative-decoding rollback); free blocks no longer needed."""
        keep = math.ceil(n_tokens / self.block_size)
        allocator.free(self.blocks[keep:])
        self.blocks = self.blocks[:keep]
        self.num_tokens = n_tokens

    def release(self, allocator: BlockAllocator) -> None:
        allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0


class PagedKVCache:
    """The shared block pool for all requests (allocated once, at engine start)."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int = 16, device="cuda",
                 dtype=torch.bfloat16):
        shape = (cfg.n_layers, num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.reserved_bytes = num_blocks * block_size * cfg.kv_bytes_per_token(dtype)

    @classmethod
    def from_token_budget(cls, cfg: ModelConfig, num_tokens: int, block_size: int = 16, device="cuda",
                          dtype=torch.bfloat16) -> "PagedKVCache":
        return cls(cfg, math.ceil(num_tokens / block_size), block_size, device, dtype)

    @staticmethod
    def tokens_for_bytes(cfg: ModelConfig, nbytes: int, dtype=torch.bfloat16) -> int:
        return nbytes // cfg.kv_bytes_per_token(dtype)

    def write(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Store K/V for every token of this step ([N, n_kv_heads, head_dim]) at their slots: ONE indexed
        copy for all requests together (the contiguous cache needed one copy per request)."""
        kvh, hd = self.cfg.n_kv_heads, self.cfg.head_dim
        self.k[layer].view(-1, kvh, hd).index_copy_(0, slot_mapping, k_new)
        self.v[layer].view(-1, kvh, hd).index_copy_(0, slot_mapping, v_new)

    def gather(self, layer: int, block_table: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy one request's first `length` tokens of K/V out of its scattered blocks into a dense
        [1, n_kv_heads, length, head_dim] tensor for SDPA. (This copy is the cost M4's kernel removes:
        a paged-attention kernel reads the blocks in place.)"""
        kvh, hd = self.cfg.n_kv_heads, self.cfg.head_dim
        k = self.k[layer][block_table].reshape(-1, kvh, hd)[:length]
        v = self.v[layer][block_table].reshape(-1, kvh, hd)[:length]
        return k.transpose(0, 1)[None], v.transpose(0, 1)[None]
