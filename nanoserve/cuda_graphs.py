"""M7: CUDA graphs for decode steps.

Why: a decode step is ~1,000 small kernels launched one by one from Python (M1 profile). For a 0.5B model
the GPU finishes each kernel faster than Python can launch the next, so it idles most of the step.
A CUDA GRAPH records the whole sequence of launches once; replaying it is ONE launch from the CPU.

Constraints, and how we meet them:
  * Fixed shapes      -> one graph per batch-size bucket (1, 2, 4, 8, 16, 24, 32, ...). A batch of 5
                         replays the 8-graph with 3 padding rows.
  * Fixed addresses   -> inputs live in STATIC device buffers that we overwrite before each replay;
                         outputs are read from the tensor captured inside the graph.
  * No CPU syncs or Python decisions inside -> only pure-decode steps (every request feeds one token
                         and uses the M4 Triton kernel). Steps containing a prefill run eagerly.
  * Padding must be harmless -> padding rows write their K/V into one reserved "garbage" block and
                         attend only to it; their logits are ignored.
Graphs are captured lazily, the first time a bucket is needed, and share one memory pool.
"""
from __future__ import annotations

import math

import torch

from nanoserve.kv_cache import PagedKVCache
from nanoserve.model import DecodeMeta, Qwen2Model, SeqInput

BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


class DecodeGraphRunner:
    def __init__(self, model: Qwen2Model, pool: PagedKVCache, max_seq_len: int, max_batch: int = BUCKETS[-1]):
        self.model, self.pool = model, pool
        self.buckets = [b for b in BUCKETS if b <= max_batch]
        self.max_bucket = self.buckets[-1]
        self.max_blocks = math.ceil(max_seq_len / pool.block_size)
        self.garbage_block = pool.allocator.allocate()     # padding rows read/write only this block
        dev, mb = model.device, self.max_bucket
        self.tokens = torch.zeros(mb, dtype=torch.long, device=dev)
        self.positions = torch.zeros(mb, dtype=torch.long, device=dev)
        self.slots = torch.zeros(mb, dtype=torch.long, device=dev)
        self.block_tables = torch.zeros(mb, self.max_blocks, dtype=torch.int32, device=dev)
        self.context_lens = torch.ones(mb, dtype=torch.int32, device=dev)
        self.token_idx = torch.arange(mb, device=dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.mempool = None
        self._pad_row = [self.garbage_block] + [0] * (self.max_blocks - 1)
        self._reset_to_padding()

    @property
    def captured_buckets(self) -> list[int]:
        return sorted(self.graphs)

    def bucket_for(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def _reset_to_padding(self) -> None:
        self.tokens.zero_()
        self.positions.zero_()
        self.slots.fill_(self.garbage_block * self.pool.block_size)
        self.block_tables.zero_()
        self.block_tables[:, 0] = self.garbage_block
        self.context_lens.fill_(1)

    def _forward(self, B: int) -> torch.Tensor:
        meta = DecodeMeta(self.token_idx[:B], self.block_tables[:B], self.context_lens[:B])
        seqs = [SeqInput(1, ctx_len=1, use_kernel=True)] * B
        return self.model.forward_batch(self.tokens[:B], self.positions[:B], seqs, logits="all", paged=self.pool,
                                        slot_mapping=self.slots[:B], decode_meta=meta)

    def _capture(self, B: int) -> None:
        # Warm-up and capture run on PADDING inputs only, so they can't overwrite real K/V.
        self._reset_to_padding()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):        # compile Triton, let the allocator settle (PyTorch's recipe)
            for _ in range(2):
                self._forward(B)
        torch.cuda.current_stream().wait_stream(side)
        if self.mempool is None:
            self.mempool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.mempool):
            out = self._forward(B)
        self.graphs[B], self.outputs[B] = graph, out

    def run(self, tokens: list[int], positions: list[int], slots: list[int], block_tables: list[list[int]],
            context_lens: list[int]) -> torch.Tensor:
        """One decode step for n requests via the bucket's graph. Returns logits [n, vocab] -- a view of
        the graph's output buffer, valid until the next replay of that bucket."""
        n = len(tokens)
        B = self.bucket_for(n)
        if B is None:
            raise ValueError(f"batch of {n} exceeds the largest graph bucket ({self.max_bucket})")
        if B not in self.graphs:
            self._capture(B)
        pad = B - n
        W = self.max_blocks
        gslot = self.garbage_block * self.pool.block_size
        self.tokens[:B].copy_(torch.tensor(tokens + [0] * pad))
        self.positions[:B].copy_(torch.tensor(positions + [0] * pad))
        self.slots[:B].copy_(torch.tensor(slots + [gslot] * pad))
        rows = [t + [0] * (W - len(t)) for t in block_tables] + [self._pad_row] * pad
        self.block_tables[:B].copy_(torch.tensor(rows, dtype=torch.int32))
        self.context_lens[:B].copy_(torch.tensor(context_lens + [1] * pad, dtype=torch.int32))
        self.graphs[B].replay()
        return self.outputs[B][:n]
