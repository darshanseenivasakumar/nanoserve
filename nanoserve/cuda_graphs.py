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
                         and uses the M4 Triton kernel). Steps containing a prefill use the M9
                         piecewise runner at the bottom of this file.
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

_SIDE_STREAM = None


def _capture_stream() -> torch.cuda.Stream:
    """ONE warm-up stream for every capture in the process. PyTorch caches a cuBLAS workspace per stream,
    so a fresh stream per bucket (what M7 did) kept ~8 MiB of workspace alive per captured bucket (found in M9)."""
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


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
        side = _capture_stream()
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

    def capture_all(self, max_batch: int) -> None:
        """Capture every bucket a batch of up to max_batch requests can use (server/benchmark startup)."""
        top = self.bucket_for(min(max_batch, self.max_bucket))
        for B in self.buckets:
            if B <= top and B not in self.graphs:
                self._capture(B)


# ---------------------------------------------------------------------------------------------------------
# M9: piecewise graphs for MIXED steps (prefill chunks + decodes).
#
# A mixed step can't be one graph: attention's work depends on every request's chunk and cached length
# (a Python loop over sequences with changing SDPA shapes). But everything else in a layer is token-wise:
# its shapes depend only on the TOTAL token count T. So we cut the forward at attention:
#
#   piece 0  = embed + RoPE tables + layer 0 _pre (norm, QKV, RoPE, KV write)        -> graph
#   attend 0                                                                          -> eager
#   piece j  = layer j-1 _post (o_proj, residual, MLP) + layer j _pre                 -> graph
#   ...
#   piece L  = layer L-1 _post + final norm                                           -> graph
#   LM head on the sampled rows only                                                  -> eager
#
# One set of L+1 graphs per token bucket. Pieces hand data to each other through STATIC buffers sized for
# the largest bucket, so graph-internal temporaries are dead after each piece and every graph of every bucket
# can share one memory pool. Padding rows (bucket > T) use token 0 / position 0 and write K/V only to the
# garbage block; every piece is row-wise, so they can't touch real rows.
# ---------------------------------------------------------------------------------------------------------
# Measured (bench/runs/m9_step_probe.py): piecewise beats eager 2.7x at T<=64 and 1.5x at T~256, but
# 1.06x at 512 and 0.99x at 1024/2048 -- big steps are compute-bound, launch overhead is hidden. So, like
# vLLM's default, graphs stop at 512 tokens; bigger steps run eagerly (saves capture time and memory).
PIECEWISE_CAP = 512


def piecewise_buckets(max_tokens: int, cap: int = PIECEWISE_CAP) -> list[int]:
    """Token-count buckets for mixed steps: every 16 up to 256, then every 32. Steps above `cap` run eagerly."""
    top = min(max_tokens, cap)
    grid = list(range(16, 256, 16)) + list(range(256, cap + 1, 32))
    return [b for b in grid if b < top] + [top]


class PiecewiseGraphRunner:
    def __init__(self, model: Qwen2Model, pool: PagedKVCache, garbage_block: int, max_tokens: int = PIECEWISE_CAP,
                 buckets: list[int] | None = None):
        self.model, self.pool, self.garbage_block = model, pool, garbage_block
        self.buckets = sorted(buckets) if buckets else piecewise_buckets(max_tokens)
        self.max_tokens = self.buckets[-1]
        cfg, dev, dt, Tm = model.cfg, model.device, model.dtype, self.max_tokens
        self.gslot = garbage_block * pool.block_size
        self.tokens = torch.zeros(Tm, dtype=torch.long, device=dev)
        self.positions = torch.zeros(Tm, dtype=torch.long, device=dev)
        self.slots = torch.full((Tm,), self.gslot, dtype=torch.long, device=dev)
        z = lambda *shape: torch.zeros(1, Tm, *shape, dtype=dt, device=dev)   # noqa: E731
        self.x = z(cfg.hidden_size)                          # residual stream between pieces
        self.q, self.k, self.v = z(cfg.n_heads, cfg.head_dim), z(cfg.n_kv_heads, cfg.head_dim), z(cfg.n_kv_heads, cfg.head_dim)
        self.o = z(cfg.n_heads * cfg.head_dim)               # attention output (written by the eager part)
        self.cos, self.sin = z(cfg.head_dim), z(cfg.head_dim)
        self.graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self.mempool = None

    @property
    def captured_buckets(self) -> list[int]:
        return sorted(self.graphs)

    def bucket_for(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def _piece(self, j: int, B: int) -> None:
        """Piece j over the first B rows of the static buffers (see the diagram above)."""
        m, L = self.model, self.model.cfg.n_layers
        if j == 0:
            x = torch.nn.functional.embedding(self.tokens[:B], m.embed)[None]
            cos, sin = m.rope(self.positions[:B])
            self.cos[:, :B].copy_(cos)
            self.sin[:, :B].copy_(sin)
        else:
            x = m._post(j - 1, self.x[:, :B], self.o[:, :B])
        if j < L:
            q, k, v = m._pre(j, x, self.cos[:, :B], self.sin[:, :B], self.pool, self.slots[:B])
            self.q[:, :B].copy_(q)
            self.k[:, :B].copy_(k)
            self.v[:, :B].copy_(v)
            self.x[:, :B].copy_(x)
        else:
            self.x[:, :B].copy_(m.final_norm(x))

    def _capture(self, B: int) -> None:
        # Warm-up writes K/V through self.slots: point every row at the garbage block first, or stale slots
        # from the previous real step would overwrite live K/V.
        self.tokens.zero_()
        self.positions.zero_()
        self.slots.fill_(self.gslot)
        L = self.model.cfg.n_layers
        side = _capture_stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                for j in range(L + 1):
                    self._piece(j, B)
        torch.cuda.current_stream().wait_stream(side)
        if self.mempool is None:
            self.mempool = torch.cuda.graph_pool_handle()
        graphs = []
        for j in range(L + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.mempool):
                self._piece(j, B)
            graphs.append(g)
        self.graphs[B] = graphs

    @torch.inference_mode()
    def capture_all(self) -> None:
        # LARGEST first (vLLM does the same): the big bucket's temporaries carve the pool's segments once,
        # and every smaller bucket's temporaries then fit in those freed blocks instead of new segments.
        for B in reversed(self.buckets):
            if B not in self.graphs:
                self._capture(B)

    @torch.inference_mode()
    def run(self, tokens: list[int], positions: list[int], slots: list[int], seqs: list[SeqInput],
            decode_meta: DecodeMeta | None, rows: torch.Tensor | None = None) -> torch.Tensor:
        """One mixed step of T = len(tokens) tokens. Returns logits for each sequence's last token, or for
        `rows` (flattened indices) if given."""
        T = len(tokens)
        B = self.bucket_for(T)
        if B is None:
            raise ValueError(f"step of {T} tokens exceeds the largest piecewise bucket ({self.max_tokens})")
        if B not in self.graphs:
            self._capture(B)
        pad = B - T
        self.tokens[:B].copy_(torch.tensor(tokens + [0] * pad))
        self.positions[:B].copy_(torch.tensor(positions + [0] * pad))
        self.slots[:B].copy_(torch.tensor(slots + [self.gslot] * pad))
        m, graphs = self.model, self.graphs[B]
        q, k, v, o = self.q[:, :T], self.k[:, :T], self.v[:, :T], self.o[:, :T]   # attention sees real rows only
        for i in range(m.cfg.n_layers):
            graphs[i].replay()
            m._attend(i, q, k, v, seqs, self.pool, decode_meta, out=o)
        graphs[-1].replay()
        return m.lm_head_rows(self.x[:, :T], seqs, "last" if rows is None else rows)
