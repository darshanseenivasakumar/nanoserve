"""M10: fused Triton kernels for the token-wise parts of a layer.

Why: with CUDA graphs the CPU no longer pays per kernel, but the GPU still does. Unfused, one layer is ~42
kernels; most are tiny elementwise ops that each read and write the whole activation:

    RMSNorm   x.float() -> pow(2) -> mean -> +eps -> rsqrt -> mul -> .to(bf16) -> * weight   (8 kernels)
    RoPE      q*cos, rotate_half (neg + cat), *sin, +   ... for q and again for k            (~10 kernels)
    SwiGLU    silu(gate) -> write -> read -> * up                                              (2 kernels)

M10 profile: in a 2048-token prefill these moved ~7.4 GB per step (46 ms of 168); in a decode step the
~830 small kernels cost ~2 us each plus a gap per graph node. Each kernel below reads its inputs once and
writes its outputs once.

NUMERICS: every kernel rounds to the model dtype exactly where PyTorch's unfused ops round (after each bf16
op), so bf16 results match the unfused path bit for bit except where RMSNorm's fp32 sum is accumulated in
a different order (then the output can differ by 1 bf16 ulp).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------------------------- RMSNorm
@triton.jit
def _rms_norm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, N, eps,
                     HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    if HAS_RES:
        # residual + x, rounded to the model dtype (PyTorch's `x + a` in bf16), stored for the next layer.
        r = tl.load(res_ptr + row * N + cols, mask=mask, other=0.0)
        x = (r.to(tl.float32) + x.to(tl.float32)).to(r.dtype)
        tl.store(res_out_ptr + row * N + cols, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N                    # fp32, like HF's x.float().pow(2).mean(-1)
    y = (xf * tl.rsqrt(var + eps)).to(x.dtype)            # HF casts back BEFORE the weight multiply
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * N + cols, (w.to(tl.float32) * y.to(tl.float32)).to(x.dtype), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None):
    """RMSNorm over the last dim in ONE kernel. With `residual`: computes r = residual + x first and returns
    (rms_norm(r), r) -- the residual add fused into the norm (vLLM's fused_add_rms_norm, out of place)."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    assert x2.is_contiguous()
    out = torch.empty_like(x2)
    res_out = torch.empty_like(x2) if residual is not None else out
    res = residual.reshape(-1, N) if residual is not None else x2
    BLOCK = triton.next_power_of_2(N)
    _rms_norm_kernel[(x2.shape[0],)](x2, res, weight, out, res_out, N, eps, HAS_RES=residual is not None,
                                     BLOCK=BLOCK, num_warps=4 if BLOCK <= 2048 else 8)
    if residual is None:
        return out.view(x.shape)
    return out.view(x.shape), res_out.view(x.shape)


# ---------------------------------------------------------------------------------------------- SwiGLU
@triton.jit
def _silu_mul_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    g = tl.load(x_ptr + row * 2 * N + cols, mask=mask, other=0.0)
    u = tl.load(x_ptr + row * 2 * N + N + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    s = (gf / (1.0 + tl.exp(-gf))).to(g.dtype)           # F.silu: computed in fp32, rounded to bf16
    tl.store(out_ptr + row * N + cols, (s.to(tl.float32) * u.to(tl.float32)).to(g.dtype), mask=mask)


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x = [..., 2N] holding [gate | up] (the merged gate/up projection) -> silu(gate) * up, [..., N]."""
    N = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * N)
    assert x2.is_contiguous()
    out = torch.empty(x2.shape[0], N, device=x.device, dtype=x.dtype)
    BLOCK = 1024
    _silu_mul_kernel[(x2.shape[0], triton.cdiv(N, BLOCK))](x2, out, N, BLOCK=BLOCK, num_warps=4)
    return out.view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------------------- RoPE + cache
@triton.jit
def _round(x, ROUND_BF16: tl.constexpr):
    """fp32 -> the nearest bf16 value (ties to even), kept as fp32. Done with INTEGER ops on purpose:
    written as x.to(bf16).to(fp32), the rounding of a product feeding an add/sub was not preserved in the
    compiled kernel (M10: only 84% of RoPE outputs matched PyTorch; with this, all do)."""
    if ROUND_BF16:
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        x = b.to(tl.float32, bitcast=True)
    return x


@triton.jit
def _rotate(x_lo, x_hi, c_lo, c_hi, s_lo, s_hi, ROUND_BF16: tl.constexpr):
    xl, xh = x_lo.to(tl.float32), x_hi.to(tl.float32)
    lo = _round(_round(xl * c_lo, ROUND_BF16) - _round(xh * s_lo, ROUND_BF16), ROUND_BF16)
    hi = _round(_round(xh * c_hi, ROUND_BF16) + _round(xl * s_hi, ROUND_BF16), ROUND_BF16)
    return lo.to(x_lo.dtype), hi.to(x_lo.dtype)


@triton.jit
def _rope_qkv_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, slot_ptr,
                     NH: tl.constexpr, NKV: tl.constexpr, NH_PAD: tl.constexpr, NKV_PAD: tl.constexpr,
                     HD: tl.constexpr, WRITE_CACHE: tl.constexpr, ROUND_BF16: tl.constexpr):
    """One program = one token: rotate its NH query heads and NKV key heads, copy its value heads, and (paged
    cache) store K and V at the token's slot. rotate_half(x) = [-x_hi, x_lo], so with lo/hi halves:
        out_lo = x_lo*cos_lo - x_hi*sin_lo        out_hi = x_hi*cos_hi + x_lo*sin_hi
    with a bf16 rounding after each product and after the sum, exactly where PyTorch rounds."""
    t = tl.program_id(0).to(tl.int64)
    HALF: tl.constexpr = HD // 2
    d = tl.arange(0, HALF)
    W: tl.constexpr = (NH + 2 * NKV) * HD                  # one token's row of the merged QKV output
    row = qkv_ptr + t * W
    c_lo = tl.load(cos_ptr + t * HD + d).to(tl.float32)
    c_hi = tl.load(cos_ptr + t * HD + HALF + d).to(tl.float32)
    s_lo = tl.load(sin_ptr + t * HD + d).to(tl.float32)
    s_hi = tl.load(sin_ptr + t * HD + HALF + d).to(tl.float32)

    hq = tl.arange(0, NH_PAD)
    mq = (hq < NH)[:, None]
    offs = hq[:, None] * HD + d[None, :]
    x_lo = tl.load(row + offs, mask=mq, other=0.0)
    x_hi = tl.load(row + offs + HALF, mask=mq, other=0.0)
    lo, hi = _rotate(x_lo, x_hi, c_lo, c_hi, s_lo, s_hi, ROUND_BF16)
    tl.store(q_ptr + t * NH * HD + offs, lo, mask=mq)
    tl.store(q_ptr + t * NH * HD + offs + HALF, hi, mask=mq)

    hk = tl.arange(0, NKV_PAD)
    mk = (hk < NKV)[:, None]
    koffs = hk[:, None] * HD + d[None, :]
    k_lo = tl.load(row + NH * HD + koffs, mask=mk, other=0.0)
    k_hi = tl.load(row + NH * HD + koffs + HALF, mask=mk, other=0.0)
    klo, khi = _rotate(k_lo, k_hi, c_lo, c_hi, s_lo, s_hi, ROUND_BF16)
    v_lo = tl.load(row + (NH + NKV) * HD + koffs, mask=mk, other=0.0)
    v_hi = tl.load(row + (NH + NKV) * HD + koffs + HALF, mask=mk, other=0.0)
    kv_base = t * NKV * HD
    tl.store(k_ptr + kv_base + koffs, klo, mask=mk)
    tl.store(k_ptr + kv_base + koffs + HALF, khi, mask=mk)
    tl.store(v_ptr + kv_base + koffs, v_lo, mask=mk)
    tl.store(v_ptr + kv_base + koffs + HALF, v_hi, mask=mk)
    if WRITE_CACHE:
        slot = tl.load(slot_ptr + t).to(tl.int64)
        cbase = slot * NKV * HD                             # pool layout [blocks, block_size, NKV, HD]
        tl.store(kc_ptr + cbase + koffs, klo, mask=mk)
        tl.store(kc_ptr + cbase + koffs + HALF, khi, mask=mk)
        tl.store(vc_ptr + cbase + koffs, v_lo, mask=mk)
        tl.store(vc_ptr + cbase + koffs + HALF, v_hi, mask=mk)


def rope_qkv(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_heads: int, n_kv_heads: int,
             head_dim: int, k_cache: torch.Tensor | None = None, v_cache: torch.Tensor | None = None,
             slots: torch.Tensor | None = None):
    """qkv: [N, (n_heads + 2*n_kv_heads) * head_dim] = merged Q/K/V projection output (bias included).
    cos/sin: [N, head_dim]. Returns rotated q [N, n_heads, hd], rotated k and plain v [N, n_kv_heads, hd];
    if a paged cache layer (k_cache/v_cache [blocks, block_size, n_kv_heads, hd]) and slots are given, also
    stores k and v at those slots -- the M3 index_copy_, fused in."""
    N = qkv.shape[0]
    assert qkv.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert qkv.dtype in (torch.bfloat16, torch.float32), "rounding is emulated for bf16 / exact for fp32"
    q = torch.empty(N, n_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(N, n_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty_like(k)
    write = k_cache is not None
    _rope_qkv_kernel[(N,)](qkv, cos, sin, q, k, v, k_cache if write else k, v_cache if write else v,
                           slots if write else qkv,
                           NH=n_heads, NKV=n_kv_heads, NH_PAD=triton.next_power_of_2(n_heads),
                           NKV_PAD=max(triton.next_power_of_2(n_kv_heads), 1), HD=head_dim,
                           WRITE_CACHE=write, ROUND_BF16=qkv.dtype == torch.bfloat16, num_warps=4)
    return q, k, v
