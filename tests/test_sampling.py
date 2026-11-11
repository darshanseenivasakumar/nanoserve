"""M13: sampling. Sampler units run on CPU with exact expected probabilities; engine tests check that seeded
sampling is reproducible across runs, batch compositions and preemption, and that temperature 0 is greedy."""
from collections import Counter

import pytest
import torch

from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.sampling import SamplingParams, sample
from tests.conftest import MODEL, pick_prompts

LOGITS = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])
N = 20000


def freqs(params, seed=7):
    toks = sample(LOGITS[None].repeat(N, 1), [params] * N, [seed] * N, list(range(N)))   # N independent steps
    c = Counter(toks)
    return torch.tensor([c[j] / N for j in range(len(LOGITS))])


def test_temperature_draws_match_softmax():
    for t in (0.5, 1.0, 2.0):
        want = (LOGITS / t).softmax(-1)
        got = freqs(SamplingParams(temperature=t))
        assert (got - want).abs().max() < 0.015, (t, got, want)


def test_top_k_keeps_only_the_k_best_renormalised():
    got = freqs(SamplingParams(temperature=1.0, top_k=2))
    want = torch.zeros(5)
    want[:2] = LOGITS[:2].softmax(-1)
    assert got[2:].sum() == 0 and (got - want).abs().max() < 0.015


def test_top_p_keeps_the_smallest_set_reaching_p():
    p = LOGITS.softmax(-1)                     # [0.54, 0.20, 0.12, 0.07, 0.03]: 0.54 < 0.7 <= 0.54 + 0.20
    got = freqs(SamplingParams(temperature=1.0, top_p=0.7))
    want = torch.zeros(5)
    want[:2] = p[:2] / p[:2].sum()
    assert got[2:].sum() == 0 and (got - want).abs().max() < 0.015


def test_greedy_rows_stay_argmax_in_a_mixed_batch():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    params = [None, SamplingParams(temperature=1.0), SamplingParams(), SamplingParams(temperature=0.8, top_k=1),
              SamplingParams(temperature=2.0), None]
    out = sample(logits, params, [1] * 6, [0] * 6)
    for i in (0, 2, 3, 5):
        assert out[i] == logits[i].argmax().item()
    assert sample(logits, [None] * 6, [1] * 6, [0] * 6) == logits.argmax(-1).tolist()


def test_same_seed_and_step_give_the_same_token_whatever_the_batch():
    torch.manual_seed(1)
    row = torch.randn(1, 100)
    p = SamplingParams(temperature=1.0, top_p=0.9)
    for step in range(30):
        alone = sample(row, [p], [42], [step])[0]
        others = torch.randn(5, 100)
        batch = torch.cat([others[:2], row, others[2:]])
        mixed = sample(batch, [p, None, p, SamplingParams(temperature=3.0), p], [1, 2, 42, 3, 4],
                       [0, 0, step, 7, 9])[2]
        assert alone == mixed


def test_different_seeds_give_different_sequences():
    torch.manual_seed(2)
    logits = torch.randn(1, 100)
    p = SamplingParams(temperature=1.0)
    a = [sample(logits, [p], [1], [s])[0] for s in range(50)]
    b = [sample(logits, [p], [2], [s])[0] for s in range(50)]
    assert a != b


def test_invalid_params_are_rejected():
    for kw in ({"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -2}):
        with pytest.raises(ValueError):
            SamplingParams(**kw)


# ---- engine (fp32 CPU: logits barely depend on the batch, so seeded sampling must reproduce exactly) ----
PROMPTS = ["short-00", "short-03", "medium-02", "short-05"]


def run(model, encode, sampling, max_num_seqs=4, kv_tokens=16384, budget=2048, one_at_a_time=False):
    eng = LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=max_num_seqs,
                    batch_wait_s=0.0, kv_cache_tokens=kv_tokens, max_num_batched_tokens=budget)
    outs = {}
    groups = [[i] for i in range(len(PROMPTS))] if one_at_a_time else [list(range(len(PROMPTS)))]
    for group in groups:
        ids = {eng.add_request(encode(pick_prompts([PROMPTS[i]])[0]), 24, sampling=sampling[i]): i for i in group}
        while eng.has_unfinished():
            for out in eng.step():
                outs.setdefault(ids[out.seq_id], []).append(out.token)
    return [outs[i] for i in range(len(PROMPTS)) if i in outs], eng


SEEDED = [SamplingParams(temperature=1.0, top_p=0.95, seed=100 + i) for i in range(len(PROMPTS))]


@pytest.mark.gpu
def test_seeded_sampling_is_reproducible_and_batch_independent(ours_fp32_cpu, encode):
    batched, _ = run(ours_fp32_cpu, encode, SEEDED)
    again, _ = run(ours_fp32_cpu, encode, SEEDED)
    alone, _ = run(ours_fp32_cpu, encode, SEEDED, one_at_a_time=True)
    assert batched == again == alone
    greedy, _ = run(ours_fp32_cpu, encode, [None] * 4)
    assert batched != greedy                                  # it really sampled


@pytest.mark.gpu
def test_seeded_sampling_survives_preemption(ours_fp32_cpu, encode):
    roomy, _ = run(ours_fp32_cpu, encode, SEEDED)
    # prompts need 3+3+12+3 = 21 blocks, finished requests up to 28: a 22-block pool admits all, then must preempt
    tight, eng = run(ours_fp32_cpu, encode, SEEDED, kv_tokens=352, budget=64)
    assert eng.stats["num_preemptions"] > 0
    assert tight == roomy                                     # same step index -> same draw after recompute


@pytest.mark.gpu
def test_temperature_zero_is_greedy(ours_fp32_cpu, encode):
    a, _ = run(ours_fp32_cpu, encode, [SamplingParams(temperature=0.0, seed=5)] * 4)
    b, _ = run(ours_fp32_cpu, encode, [None] * 4)
    assert a == b


@pytest.mark.gpu
def test_different_seeds_change_the_text(ours_fp32_cpu, encode):
    a, _ = run(ours_fp32_cpu, encode, SEEDED)
    b, _ = run(ours_fp32_cpu, encode, [SamplingParams(temperature=1.0, top_p=0.95, seed=900 + i) for i in range(4)])
    assert a != b
