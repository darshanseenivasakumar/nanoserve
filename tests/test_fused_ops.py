"""M10: fused Triton kernels (RMSNorm [+ residual add], SiLU*mul, RoPE + paged KV write) and merged weights.

Each kernel keeps PyTorch's rounding points, so in bf16 it should match the unfused PyTorch ops bit for bit
almost everywhere; the only reordering is the fp32 sum inside RMSNorm. Gates:
  * kernels vs PyTorch references: fp32 within 1e-5; bf16 >= 99% identical and the rest within 1 bf16 ulp;
  * the RoPE kernel writes rotated K and plain V to exactly the given slots, nothing else;
  * merged Q/K/V and gate/up weights: the per-matrix names are views of the merged tensors (no copies);
  * model and engine, fused vs unfused: the approved bf16 tie rule; the M1 reference engine stays unfused.
"""
import pytest
import torch
import torch.nn.functional as F

from nanoserve.engine import Engine, LLMEngine, load_eos_token_ids
from nanoserve.model import rms_norm as rms_norm_ref, rotate_half
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from nanoserve.kernels.fused import rms_norm, rope_qkv, silu_mul  # noqa: E402

DT = [torch.float32, torch.bfloat16]


def check(out, ref, dtype):
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        return
    same = (out == ref).float().mean().item()
    assert same >= 0.99, f"only {same:.4f} bitwise identical"
    torch.testing.assert_close(out.float(), ref.float(), rtol=2 ** -7, atol=1e-6)   # <= 1 bf16 ulp


@pytest.mark.parametrize("dtype", DT)
def test_rms_norm(dtype):
    torch.manual_seed(0)
    x = torch.randn(37, 896, device="cuda").to(dtype) * 3
    w = (1 + 0.1 * torch.randn(896, device="cuda")).to(dtype)
    check(rms_norm(x, w, 1e-6), rms_norm_ref(x, w, 1e-6), dtype)


@pytest.mark.parametrize("dtype", DT)
def test_rms_norm_with_residual_add(dtype):
    torch.manual_seed(1)
    x, a = (torch.randn(1, 37, 896, device="cuda").to(dtype) for _ in range(2))
    w = (1 + 0.1 * torch.randn(896, device="cuda")).to(dtype)
    out, res = rms_norm(a, w, 1e-6, residual=x)
    ref_res = x + a
    assert torch.equal(res, ref_res)                          # the add is exact: same rounding as PyTorch
    check(out, rms_norm_ref(ref_res, w, 1e-6), dtype)


@pytest.mark.parametrize("dtype", DT)
def test_silu_mul(dtype):
    torch.manual_seed(2)
    x = torch.randn(1, 37, 2 * 4864, device="cuda").to(dtype) * 2
    g, u = x[..., :4864], x[..., 4864:]
    check(silu_mul(x), F.silu(g) * u, dtype)


def rope_tables(positions, hd, dtype, theta=1_000_000.0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
    freqs = positions[:, None].float().cpu() * inv
    emb = torch.cat((freqs, freqs), dim=-1).cuda()
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.parametrize("dtype", DT)
@pytest.mark.parametrize("nh,nkv,hd", [(14, 2, 64), (12, 2, 128)])      # 0.5B and 1.5B/3B head layouts
def test_rope_qkv_and_cache_write(dtype, nh, nkv, hd):
    torch.manual_seed(3)
    N = 37
    qkv = torch.randn(N, (nh + 2 * nkv) * hd, device="cuda").to(dtype)
    cos, sin = rope_tables(torch.randint(0, 2000, (N,)), hd, dtype)
    k_cache = torch.zeros(8, 16, nkv, hd, device="cuda", dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.randperm(8 * 16, device="cuda")[:N]
    q, k, v = rope_qkv(qkv, cos, sin, nh, nkv, hd, k_cache, v_cache, slots)

    c, s = cos[:, None, :], sin[:, None, :]
    q0 = qkv[:, :nh * hd].view(N, nh, hd)
    k0 = qkv[:, nh * hd:(nh + nkv) * hd].view(N, nkv, hd)
    v0 = qkv[:, (nh + nkv) * hd:].view(N, nkv, hd)
    check(q, q0 * c + rotate_half(q0) * s, dtype)
    check(k, k0 * c + rotate_half(k0) * s, dtype)
    assert torch.equal(v, v0)
    flat_k, flat_v = k_cache.view(-1, nkv, hd), v_cache.view(-1, nkv, hd)
    assert torch.equal(flat_k[slots], k) and torch.equal(flat_v[slots], v)
    untouched = torch.ones(8 * 16, dtype=torch.bool, device="cuda")
    untouched[slots] = False
    assert not flat_k[untouched].any() and not flat_v[untouched].any()


def test_merged_weights_are_views(ours_bf16_gpu):
    ours_bf16_gpu.with_fused_ops()                       # merging is lazy: done for the fused path only
    layer = ours_bf16_gpu.layers[0]
    nq = layer.q_w.shape[0]
    assert layer.q_w.data_ptr() == layer.qkv_w.data_ptr()
    assert layer.k_w.data_ptr() == layer.qkv_w[nq:].data_ptr()
    assert layer.up_w.data_ptr() == layer.gate_up_w[layer.gate_w.shape[0]:].data_ptr()
    assert layer.qkv_b.shape[0] == layer.qkv_w.shape[0]


def last_logits(model, ids):
    return model.forward(torch.tensor(ids, device="cuda"), torch.arange(len(ids), device="cuda"))[0].float()


def test_fused_model_matches_unfused_up_to_ties(ours_bf16_gpu, encode):
    fused = ours_bf16_gpu.with_fused_ops()
    assert fused.fused and not ours_bf16_gpu.fused and fused.layers is ours_bf16_gpu.layers
    for pid in ["short-00", "medium-04", "long-00"]:
        ids = encode(pick_prompts([pid])[0])
        ref, out = last_logits(ours_bf16_gpu, ids), last_logits(fused, ids)
        top2 = ref.topk(2).values
        print(f"{pid}: max|logit diff| {(out - ref).abs().max().item():.3f}, top-2 gap {(top2[0] - top2[1]).item():.3f}")
        if (top2[0] - top2[1]).item() > TIE_GAP:
            assert out.argmax() == ref.argmax()


def make(model, fused, budget=64):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=6, batch_wait_s=0.0,
                     max_num_batched_tokens=budget, kv_cache="paged", kv_cache_tokens=16384,
                     attn_backend="triton", fused_ops=fused)


def test_engine_fused_matches_unfused_up_to_ties(ours_bf16_gpu, encode):
    eng = make(ours_bf16_gpu, True)
    assert eng.model.fused and eng.graph_runner.model.fused and eng.piecewise_runner.model.fused
    fused_out, _ = run_staggered(eng, encode, REQUESTS)
    plain_out, _ = run_staggered(make(ours_bf16_gpu, False), encode, REQUESTS)
    st = eng.stats
    print(f"fused engine: {st['graph_steps']} graph + {st['piecewise_steps']} piecewise of {st['steps']} steps; "
          f"identical {sum(a == b for a, b in zip(fused_out, plain_out))}/12")
    for i, (a, b) in enumerate(zip(fused_out, plain_out)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        top2 = last_logits(ours_bf16_gpu, encode(pick_prompts([REQUESTS[i][0]])[0]) + b[:j]).topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks - 1


def test_fusion_defaults(ours_bf16_gpu):
    assert LLMEngine(ours_bf16_gpu, [151645], max_seq_len=256, kv_cache_tokens=1024).model.fused
    assert not LLMEngine(ours_bf16_gpu, [151645], max_seq_len=256, kv_cache_tokens=1024,
                         attn_backend="sdpa").model.fused
    ref = Engine(ours_bf16_gpu, [151645], max_seq_len=256)
    list(ref.generate([1, 2, 3], 2))
    assert not ref._llm.model.fused                          # the HF-exact reference path stays unfused
