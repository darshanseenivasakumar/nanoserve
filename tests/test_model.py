"""M1 step 1: our Qwen2 forward pass (no KV cache) must reproduce HF's logits.

Tolerances, stated up front (rule 1):
  fp32 (CPU): max |our logit - HF logit| < 1e-3. Same math in full precision; only summation-order noise.
  bf16 (GPU): identical argmax at every position AND torch.testing.assert_close with PyTorch's default
              bf16 tolerance (rtol 1.6e-2, atol 1e-5). We also report whether they are bitwise equal.
"""
import pytest
import torch

from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.gpu  # needs the model download (bf16 tests also need CUDA)

PROMPT_IDS = ["short-00", "short-02", "medium-04", "long-00"]


def test_config_from_pretrained():
    from nanoserve.model import ModelConfig
    cfg = ModelConfig.from_pretrained(MODEL)
    assert (cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (24, 896, 14, 2, 64)
    assert cfg.vocab_size == 151936 and cfg.intermediate_size == 4864
    assert cfg.rope_theta == 1_000_000.0 and cfg.tie_word_embeddings
    assert cfg.kv_bytes_per_token(torch.bfloat16) == 24 * 2 * 2 * 64 * 2  # 12,288 bytes


def test_weights_load_identically(ours_bf16_gpu, hf_bf16_gpu):
    hf = dict(hf_bf16_gpu.state_dict())
    ours = ours_bf16_gpu.named_weights()
    assert set(ours) == {k for k in hf if k != "lm_head.weight"}  # lm_head is tied to embed_tokens
    for name, w in ours.items():
        assert torch.equal(w, hf[name]), name
    assert ours_bf16_gpu.lm_head.data_ptr() == ours_bf16_gpu.embed.data_ptr()  # tied, not copied


def test_rope_matches_hf(ours_fp32_cpu, hf_fp32_cpu):
    positions = torch.arange(0, 3000, 7)
    cos, sin = ours_fp32_cpu.rope(positions)
    dummy = torch.zeros(1, dtype=torch.float32)
    hf_cos, hf_sin = hf_fp32_cpu.model.rotary_emb(dummy, positions[None])
    assert torch.equal(cos, hf_cos) and torch.equal(sin, hf_sin)


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_fp32_logits_match_hf(prompt_id, ours_fp32_cpu, hf_fp32_cpu, encode):
    ids = torch.tensor(encode(pick_prompts([prompt_id])[0]))
    ours = ours_fp32_cpu.forward(ids, torch.arange(len(ids)), logits="all")
    with torch.no_grad():
        ref = hf_fp32_cpu(ids[None]).logits[0]
    assert ours.shape == ref.shape
    diff = (ours - ref).abs().max().item()
    assert diff < 1e-3, f"max |diff| = {diff}"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_bf16_logits_match_hf(prompt_id, ours_bf16_gpu, hf_bf16_gpu, encode):
    ids = torch.tensor(encode(pick_prompts([prompt_id])[0]), device="cuda")
    ours = ours_bf16_gpu.forward(ids, torch.arange(len(ids), device="cuda"), logits="all")
    with torch.no_grad():
        ref = hf_bf16_gpu(ids[None]).logits[0]
    print(f"{prompt_id}: T={len(ids)} bitwise_equal={torch.equal(ours, ref)} "
          f"max|diff|={(ours.float() - ref.float()).abs().max().item():.4g}")
    assert torch.equal(ours.argmax(-1), ref.argmax(-1))
    torch.testing.assert_close(ours, ref)


def test_last_only_logits_equal_last_row_shape(ours_bf16_gpu, encode):
    ids = torch.tensor(encode(pick_prompts(["short-00"])[0]), device="cuda")
    last = ours_bf16_gpu.forward(ids, torch.arange(len(ids), device="cuda"), logits="last")
    assert last.shape == (1, ours_bf16_gpu.cfg.vocab_size)
