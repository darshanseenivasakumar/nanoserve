"""M4: paged-attention DECODE kernel in Triton.

Problem: during decode each request has ONE new query token per head, and its keys/values live in
scattered blocks of the paged pool. M3 copied ("gathered") every request's blocks into a dense tensor and
called SDPA once per request, per layer: ~B x 4 kernel launches per layer plus a full copy of the context.

This kernel does all decoding requests of a layer in ONE launch and reads K/V in place:

    grid = (num_decode_requests, n_kv_heads)
    program (b, g):  q_tile = the GROUP query heads that share kv head g          [GROUP_PAD, HEAD_DIM]
                     for each block j of request b (via its block table):
                         K_j, V_j = pool[block_table[b, j], :, g, :]                [BLOCK_SIZE, HEAD_DIM]
                         s = q_tile @ K_j^T * scale  (masked past the context length)
                         online softmax: m = running max, l = running sum, acc = running weighted V
                     out = acc / l

Online softmax (the FlashAttention trick): instead of materialising all scores, keep a running max m and
rescale the partial sums whenever m grows: exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new).

GQA: query heads h = g*GROUP .. g*GROUP+GROUP-1 share kv head g, so one program loads each K/V tile once
for all of them. GROUP is padded to 16 because tl.dot needs every dimension >= 16.

Not done (flagged): splitting long contexts across programs ("flash-decoding"/split-K). With few requests
and long contexts this kernel under-uses the GPU (B x n_kv_heads programs only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr, o_ptr,
    stride_qb, stride_qh,                       # q: [B, n_heads, HEAD_DIM]
    stride_kblk, stride_kpos, stride_khead,     # pool: [num_blocks, BLOCK_SIZE, n_kv_heads, HEAD_DIM]
    stride_btb,                                 # block tables: [B, max_blocks]
    stride_ob, stride_oh,                       # out: [B, n_heads, HEAD_DIM]
    scale,
    GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PRECISION: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    ctx_len = tl.load(ctx_ptr + b)

    rows = tl.arange(0, GROUP_PAD)              # query heads within this kv group (padded)
    dims = tl.arange(0, HEAD_DIM)
    pos = tl.arange(0, BLOCK_SIZE)
    row_ok = rows < GROUP
    heads = g * GROUP + rows

    q = tl.load(q_ptr + b * stride_qb + heads[:, None] * stride_qh + dims[None, :],
                mask=row_ok[:, None], other=0.0)                                    # [GROUP_PAD, HEAD_DIM]

    m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)

    n_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    for j in range(0, n_blocks):
        block_id = tl.load(bt_ptr + b * stride_btb + j)
        tok_ok = (j * BLOCK_SIZE + pos) < ctx_len                                   # last block may be partial
        kv_off = block_id * stride_kblk + pos[:, None] * stride_kpos + g * stride_khead + dims[None, :]
        k = tl.load(k_ptr + kv_off, mask=tok_ok[:, None], other=0.0)                # [BLOCK_SIZE, HEAD_DIM]
        v = tl.load(v_ptr + kv_off, mask=tok_ok[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale              # [GROUP_PAD, BLOCK_SIZE]
        s = tl.where(tok_ok[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)             # rescale what we accumulated under the old max
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(o_ptr + b * stride_ob + heads[:, None] * stride_oh + dims[None, :],
             out.to(o_ptr.dtype.element_ty), mask=row_ok[:, None])


def paged_attention_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float) -> torch.Tensor:
    """q: [B, n_heads, head_dim]; k_pool/v_pool: ONE layer's pool [num_blocks, block_size, n_kv_heads, head_dim];
    block_tables: int32 [B, max_blocks] (padding entries unused); context_lens: int32 [B] (includes the new
    token, whose K/V must already be written). Returns [B, n_heads, head_dim]."""
    B, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_pool.shape
    group = n_heads // n_kv_heads
    assert q.stride(-1) == 1 and k_pool.stride(-1) == 1 and v_pool.stride() == k_pool.stride()
    out = torch.empty_like(q)
    _paged_decode_kernel[(B, n_kv_heads)](
        q, k_pool, v_pool, block_tables, context_lens, out,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        scale,
        GROUP=group, GROUP_PAD=max(16, triton.next_power_of_2(group)), HEAD_DIM=head_dim, BLOCK_SIZE=block_size,
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
    )
    return out


def paged_attention_reference(q, k_pool, v_pool, block_tables, context_lens, scale):
    """Plain PyTorch: gather each request's blocks, then SDPA (the M3 path). Used as the test oracle."""
    outs = []
    block_size, n_kv_heads, head_dim = k_pool.shape[1:]
    for b in range(q.shape[0]):
        L = int(context_lens[b])
        nb = -(-L // block_size)
        blocks = block_tables[b, :nb].long()
        k = k_pool[blocks].reshape(-1, n_kv_heads, head_dim)[:L].transpose(0, 1)[None]
        v = v_pool[blocks].reshape(-1, n_kv_heads, head_dim)[:L].transpose(0, 1)[None]
        o = F.scaled_dot_product_attention(q[b][None, :, None, :], k, v, scale=scale, enable_gqa=True)
        outs.append(o[0, :, 0, :])
    return torch.stack(outs)
