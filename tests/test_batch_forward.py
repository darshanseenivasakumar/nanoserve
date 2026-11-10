"""M2 step 1: forward_batch runs several sequences (any mix of prefill and decode) in ONE forward pass.

Selective batching: token-wise layers (projections, MLP, norms) run once over the flattened tokens of
all sequences; attention runs per sequence against that sequence's own KV cache.

Reference = running each sequence ALONE through the same model (the M1 path).
  fp32 (CPU): max |diff| < 1e-3 and identical argmax.  <- the strict correctness gate
  bf16 (GPU): batching may change matmul shapes, hence rounding ("batch invariance"), so bitwise equality
              is not expected. Criterion (changed 2026-11-10, see LOG.md -- the original torch default bf16
              tolerance is failed even by bf16-vs-fp32 itself over 24 layers):
                1. per row, max|batched - fp32 truth| <= 1.5 x max|alone - fp32 truth|
                   (batching adds no error beyond bf16's own; a real bug gives errors of several units)
                2. argmax(batched) == argmax(alone) unless alone's top-2 gap <= 0.25 (approved tie rule)
"""
import pytest
import torch

from nanoserve.kv_cache import ContiguousKVCache
from nanoserve.model import SeqInput
from tests.conftest import pick_prompts

pytestmark = pytest.mark.gpu  # needs the model

PREFILL_IDS = ["short-00", "medium-04"]          # A, B: fresh requests, whole prompt this step
DECODE_IDS = ["short-02", "short-07", "long-00"]  # C, D, E: already prefilled, one new token this step


def make_state(model, encode, device):
    """Fresh caches for A,B; prefilled caches for C,D,E (run alone). Returns (seqs, caches)."""
    seqs = []
    for pid in PREFILL_IDS:
        ids = encode(pick_prompts([pid])[0])
        cache = ContiguousKVCache(model.cfg, 1024, device=device, dtype=model.dtype)
        seqs.append({"tokens": ids, "positions": list(range(len(ids))), "cache": cache})
    for pid in DECODE_IDS:
        ids = encode(pick_prompts([pid])[0])
        cache = ContiguousKVCache(model.cfg, 1024, device=device, dtype=model.dtype)
        model.forward(torch.tensor(ids, device=device), torch.arange(len(ids), device=device), cache=cache)
        next_tok = 42 + len(seqs)  # any token works: we compare batched vs alone on identical inputs
        seqs.append({"tokens": [next_tok], "positions": [len(ids)], "cache": cache})
    return seqs


def run_alone(model, seqs, device):
    rows = [model.forward(torch.tensor(s["tokens"], device=device), torch.tensor(s["positions"], device=device),
                          cache=s["cache"], logits="last") for s in seqs]
    return torch.cat(rows)


def run_batched(model, seqs, device):
    tokens = torch.tensor([t for s in seqs for t in s["tokens"]], device=device)
    positions = torch.tensor([p for s in seqs for p in s["positions"]], device=device)
    return model.forward_batch(tokens, positions, [SeqInput(len(s["tokens"]), s["cache"]) for s in seqs])


def test_fp32_batched_equals_alone(ours_fp32_cpu, encode):
    ref_state = make_state(ours_fp32_cpu, encode, "cpu")
    bat_state = make_state(ours_fp32_cpu, encode, "cpu")
    ref = run_alone(ours_fp32_cpu, ref_state, "cpu")
    out = run_batched(ours_fp32_cpu, bat_state, "cpu")
    assert out.shape == (5, ours_fp32_cpu.cfg.vocab_size)
    assert torch.equal(out.argmax(-1), ref.argmax(-1))
    assert (out - ref).abs().max().item() < 1e-3
    for r, b in zip(ref_state, bat_state):  # caches advanced identically and hold the same K/V
        assert r["cache"].length == b["cache"].length
        n = r["cache"].length
        assert (r["cache"].k[:, :, :n] - b["cache"].k[:, :, :n]).abs().max().item() < 1e-3


TIE_GAP = 0.25        # approved in the M2 plan: a top-2 gap this small is a rounding tie
NOISE_FACTOR = 1.5


def test_bf16_batching_adds_no_error_beyond_bf16(ours_bf16_gpu, ours_fp32_cpu, encode):
    truth = run_alone(ours_fp32_cpu, make_state(ours_fp32_cpu, encode, "cpu"), "cpu")
    alone = run_alone(ours_bf16_gpu, make_state(ours_bf16_gpu, encode, "cuda"), "cuda").float().cpu()
    batched = run_batched(ours_bf16_gpu, make_state(ours_bf16_gpu, encode, "cuda"), "cuda").float().cpu()
    print(f"bf16 batched vs alone: bitwise_equal={torch.equal(batched, alone)} "
          f"max|diff|={(batched - alone).abs().max().item():.4g}")
    for i in range(len(truth)):
        err_alone = (alone[i] - truth[i]).abs().max().item()
        err_batched = (batched[i] - truth[i]).abs().max().item()
        print(f"  row {i}: |alone-fp32|={err_alone:.3f} |batched-fp32|={err_batched:.3f}")
        assert err_batched <= NOISE_FACTOR * err_alone, f"row {i}: batching added error"
        top2 = alone[i].topk(2).values
        if (top2[0] - top2[1]).item() > TIE_GAP:
            assert batched[i].argmax() == alone[i].argmax(), f"row {i}: different token, not a tie"


def test_all_logits_mode_returns_every_token(ours_bf16_gpu, encode):
    seqs = make_state(ours_bf16_gpu, encode, "cuda")
    tokens = torch.tensor([t for s in seqs for t in s["tokens"]], device="cuda")
    positions = torch.tensor([p for s in seqs for p in s["positions"]], device="cuda")
    out = ours_bf16_gpu.forward_batch(tokens, positions, [SeqInput(len(s["tokens"]), s["cache"]) for s in seqs],
                                      logits="all")
    assert out.shape == (len(tokens), ours_bf16_gpu.cfg.vocab_size)
