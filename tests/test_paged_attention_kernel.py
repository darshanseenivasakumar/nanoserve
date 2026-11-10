"""M4: Triton paged-attention decode kernel vs the PyTorch oracle (gather + SDPA).

Tolerances (rule 1, stated up front):
  fp32 inputs (tl.dot in IEEE mode): max |diff| <= 1e-5.
  bf16 inputs: compare against the oracle run in fp32 on the SAME bf16 inputs; max |diff| <= 1e-2
               (the kernel feeds bf16 probabilities into the P@V matmul, like FlashAttention; values
               are O(1), bf16 has ~3 significant digits).
Shapes cover Qwen2.5-0.5B (14 q / 2 kv heads, dim 64), 1.5B (12/2, 128) and 3B (16/2, 128); partial last
blocks; context length 1; and deliberately fragmented block tables.
"""
import pytest
import torch

from nanoserve.kernels.paged_attention import paged_attention_decode, paged_attention_reference

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

CONFIGS = [  # (n_heads, n_kv_heads, head_dim)
    (14, 2, 64),     # Qwen2.5-0.5B
    (12, 2, 128),    # Qwen2.5-1.5B
    (16, 2, 128),    # Qwen2.5-3B
]
CTX_LENS = [1, 15, 16, 17, 100, 333, 700]


def make_case(n_heads, n_kv, hd, ctx_lens, dtype, block_size=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_blocks = sum(-(-L // block_size) for L in ctx_lens) + 5
    k = torch.randn(num_blocks, block_size, n_kv, hd, device="cuda", generator=g).to(dtype)
    v = torch.randn(num_blocks, block_size, n_kv, hd, device="cuda", generator=g).to(dtype)
    q = torch.randn(len(ctx_lens), n_heads, hd, device="cuda", generator=g).to(dtype)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)          # fragmented allocation
    max_blocks = max(-(-L // block_size) for L in ctx_lens)
    tables = torch.zeros(len(ctx_lens), max_blocks, dtype=torch.int32, device="cuda")
    used = 0
    for i, L in enumerate(ctx_lens):
        nb = -(-L // block_size)
        tables[i, :nb] = perm[used:used + nb].int()
        used += nb
    lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    return q, k, v, tables, lens


@pytest.mark.parametrize("n_heads,n_kv,hd", CONFIGS)
def test_fp32_matches_reference(n_heads, n_kv, hd):
    q, k, v, tables, lens = make_case(n_heads, n_kv, hd, CTX_LENS, torch.float32)
    scale = hd ** -0.5
    out = paged_attention_decode(q, k, v, tables, lens, scale)
    ref = paged_attention_reference(q, k, v, tables, lens, scale)
    assert (out - ref).abs().max().item() <= 1e-5


@pytest.mark.parametrize("n_heads,n_kv,hd", CONFIGS)
def test_bf16_matches_reference(n_heads, n_kv, hd):
    q, k, v, tables, lens = make_case(n_heads, n_kv, hd, CTX_LENS, torch.bfloat16)
    scale = hd ** -0.5
    out = paged_attention_decode(q, k, v, tables, lens, scale).float()
    ref = paged_attention_reference(q.float(), k.float(), v.float(), tables, lens, scale)
    err = (out - ref).abs().max().item()
    print(f"bf16 ({n_heads},{n_kv},{hd}) max|diff| vs fp32 oracle = {err:.2e}")
    assert err <= 1e-2


def test_single_request_single_token():
    q, k, v, tables, lens = make_case(14, 2, 64, [1], torch.float32)
    out = paged_attention_decode(q, k, v, tables, lens, 0.125)
    # one key => softmax weight 1 => output is exactly that key's value, per kv group
    blk = tables[0, 0].long()
    expected = v[blk, 0].repeat_interleave(7, dim=0)
    assert torch.allclose(out[0], expected, atol=1e-6)
