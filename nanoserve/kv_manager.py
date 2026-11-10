"""KV managers: how the scheduler and engine get KV memory for a request, behind one interface.

    ContiguousKVManager (M1/M2 behaviour): each admitted request reserves max_seq_len tokens up front.
                         Capacity = budget // max_seq_len requests. Never needs preemption.
    PagedKVManager (M3):  one shared block pool; a request holds ceil(tokens / block_size) blocks and
                         grows one block at a time. Capacity depends on ACTUAL lengths. May run out
                         mid-decode, so the scheduler preempts.

Both take an explicit token budget (DECISION: never "allocate until OOM" -- on this Windows laptop the
NVIDIA driver spills to system RAM instead of failing).
"""
from __future__ import annotations

import math

import torch

from nanoserve.kv_cache import BlockTable, ContiguousKVCache, PagedKVCache
from nanoserve.model import ModelConfig, SeqInput


class ContiguousKVManager:
    kind = "contiguous"

    def __init__(self, cfg: ModelConfig, max_seq_len: int, budget_tokens: int, device="cuda",
                 dtype=torch.bfloat16):
        self.cfg, self.max_seq_len, self.device, self.dtype = cfg, max_seq_len, device, dtype
        self.capacity = budget_tokens // max_seq_len
        if self.capacity < 1:
            raise ValueError(f"budget of {budget_tokens} tokens can't hold one {max_seq_len}-token reservation")
        self.in_use = 0

    # --- scheduler interface ---
    def can_ever_fit(self, n_tokens: int) -> bool:
        return n_tokens <= self.max_seq_len

    def can_admit(self, seq) -> bool:
        return self.in_use < self.capacity

    def admit(self, seq) -> None:
        seq.cache = ContiguousKVCache(self.cfg, self.max_seq_len, device=self.device, dtype=self.dtype)
        self.in_use += 1

    def try_append(self, seq) -> bool:
        return True            # the reservation already covers max_seq_len

    def free(self, seq) -> None:
        if seq.cache is not None:
            seq.cache = None
            self.in_use -= 1

    # --- engine interface ---
    def begin_step(self) -> None:
        pass

    def seq_input(self, seq, n_tokens: int) -> SeqInput:
        return SeqInput(n_tokens, cache=seq.cache)

    def forward_kwargs(self) -> dict:
        return {}

    def commit(self, seq, n_tokens: int) -> None:
        pass                    # ContiguousKVCache.advance() is called inside the model

    def usage(self, running) -> tuple[int, int]:
        """(tokens actually stored, tokens reserved)."""
        return sum(s.num_cached for s in running), self.in_use * self.max_seq_len


class PagedKVManager:
    kind = "paged"

    def __init__(self, cfg: ModelConfig, budget_tokens: int, block_size: int = 16, device="cuda",
                 dtype=torch.bfloat16, watermark: float = 0.01):
        self.pool = PagedKVCache.from_token_budget(cfg, budget_tokens, block_size, device, dtype)
        self.block_size = block_size
        self.device = device
        # Keep a few blocks free at admission so a just-admitted request doesn't trigger an immediate
        # preemption on its first decode step (vLLM does the same).
        self.watermark_blocks = math.floor(watermark * self.pool.num_blocks)
        self._slots: list[int] = []

    @property
    def num_free_blocks(self) -> int:
        return self.pool.allocator.num_free

    def _blocks_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    # --- scheduler interface ---
    def can_ever_fit(self, n_tokens: int) -> bool:
        return self._blocks_for(n_tokens) <= self.pool.num_blocks

    def can_admit(self, seq) -> bool:
        need = self._blocks_for(seq.step_cost)
        return need <= self.num_free_blocks - self.watermark_blocks

    def admit(self, seq) -> None:
        seq.block_table = BlockTable(self.block_size)
        seq.block_table.reserve(seq.step_cost, self.pool.allocator)

    def try_append(self, seq) -> bool:
        """Reserve room for one more token (a decode step); False if the pool is out of blocks."""
        if seq.block_table.blocks_needed(1) > self.num_free_blocks:
            return False
        seq.block_table.reserve(1, self.pool.allocator)
        return True

    def free(self, seq) -> None:
        if seq.block_table is not None:
            seq.block_table.release(self.pool.allocator)
            seq.block_table = None

    # --- engine interface ---
    def begin_step(self) -> None:
        self._slots = []

    def seq_input(self, seq, n_tokens: int) -> SeqInput:
        table = seq.block_table
        self._slots += table.slots(n_tokens)
        return SeqInput(n_tokens, block_table=torch.tensor(table.blocks, device=self.device),
                        ctx_len=table.num_tokens)

    def forward_kwargs(self) -> dict:
        return {"paged": self.pool, "slot_mapping": torch.tensor(self._slots, device=self.device)}

    def commit(self, seq, n_tokens: int) -> None:
        seq.block_table.commit(n_tokens)

    def usage(self, running) -> tuple[int, int]:
        """(tokens actually stored, tokens reserved = blocks held x block_size)."""
        used = sum(s.block_table.num_tokens for s in running if s.block_table)
        held = sum(len(s.block_table.blocks) for s in running if s.block_table)
        return used, held * self.block_size
