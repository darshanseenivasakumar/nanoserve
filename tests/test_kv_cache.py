"""M1 step 2: contiguous KV cache.

Key property: prefill(prompt) + one-token decode steps must produce the same logits as a single
cache-free forward over the whole sequence. That proves the cache stores/serves K and V correctly,
independent of HF. Checked in fp32 on CPU: max |diff| < 1e-3 and identical argmax at every position.
"""
import pytest
import torch

from nanoserve.kv_cache import ContiguousKVCache
from nanoserve.model import ModelConfig
from tests.conftest import pick_prompts

TINY = ModelConfig(vocab_size=10, hidden_size=8, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=2,
                   intermediate_size=16, rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)


def kv(T, fill):
    return torch.full((1, TINY.n_kv_heads, T, TINY.head_dim), float(fill))


def test_preallocates_the_whole_max_length():
    c = ContiguousKVCache(TINY, max_seq_len=16, device="cpu", dtype=torch.float32)
    assert c.length == 0
    assert c.k.shape == (TINY.n_layers, TINY.n_kv_heads, 16, TINY.head_dim) == c.v.shape
    assert c.reserved_bytes == 16 * TINY.kv_bytes_per_token(torch.float32)


def test_write_returns_all_tokens_so_far_and_advance_moves_the_cursor():
    c = ContiguousKVCache(TINY, max_seq_len=16, device="cpu", dtype=torch.float32)
    for layer in range(TINY.n_layers):          # prefill: 3 tokens into every layer
        k, v = c.write(layer, kv(3, layer + 1), kv(3, -(layer + 1)))
        assert k.shape == (1, TINY.n_kv_heads, 3, TINY.head_dim)
    c.advance(3)
    assert c.length == 3
    k, v = c.write(1, kv(1, 9), kv(1, -9))     # decode: 1 token
    assert k.shape[2] == 4
    assert torch.equal(k[0, :, :3], kv(3, 2)[0]) and torch.equal(k[0, :, 3:], kv(1, 9)[0])
    assert torch.equal(v[0, :, 3:], kv(1, -9)[0])
    assert c.length == 3                         # cursor only moves on advance()


def test_overflow_raises():
    c = ContiguousKVCache(TINY, max_seq_len=4, device="cpu", dtype=torch.float32)
    c.write(0, kv(4, 1), kv(4, 1))
    c.advance(4)
    with pytest.raises(ValueError):
        c.write(0, kv(1, 1), kv(1, 1))


@pytest.mark.gpu  # needs the model download
def test_prefill_plus_decode_equals_full_forward(ours_fp32_cpu, encode, tokenizer):
    prompt = encode(pick_prompts(["short-02"])[0])
    continuation = tokenizer(" def is_palindrome(s):\n    return s == s[::-1]", add_special_tokens=False)["input_ids"]
    seq = torch.tensor(prompt + continuation)
    P, T = len(prompt), len(seq)

    full = ours_fp32_cpu.forward(seq, torch.arange(T), cache=None, logits="all")

    cache = ContiguousKVCache(ours_fp32_cpu.cfg, max_seq_len=T, device="cpu", dtype=torch.float32)
    rows = [ours_fp32_cpu.forward(seq[:P], torch.arange(P), cache=cache, logits="all")]
    for t in range(P, T):
        rows.append(ours_fp32_cpu.forward(seq[t:t + 1], torch.tensor([t]), cache=cache, logits="last"))
    cached = torch.cat(rows)

    assert cache.length == T
    assert torch.equal(cached.argmax(-1), full.argmax(-1))
    assert (cached - full).abs().max().item() < 1e-3


@pytest.mark.gpu
def test_chunked_prefill_is_rejected(ours_fp32_cpu):
    cache = ContiguousKVCache(ours_fp32_cpu.cfg, max_seq_len=8, device="cpu", dtype=torch.float32)
    ours_fp32_cpu.forward(torch.tensor([1, 2]), torch.arange(2), cache=cache)
    with pytest.raises(NotImplementedError):
        ours_fp32_cpu.forward(torch.tensor([3, 4]), torch.tensor([2, 3]), cache=cache)
