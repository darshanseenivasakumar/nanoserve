"""M2 step 3: LLMEngine (scheduler + batched forward) must produce the same tokens as running each request
alone through the M1 single-request path.

  fp32 (CPU): token-for-token identical for every request.            <- strict gate
  bf16 (GPU): approved tie rule. Report the exact-match rate; where a request diverges, the tokens before
              are identical by definition, and the reference's top-2 logit gap at the divergence point
              must be <= 0.25 (a rounding tie, not a bug).
"""
import pytest
import torch

from nanoserve.engine import Engine, LLMEngine, load_eos_token_ids
from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.gpu

TIE_GAP = 0.25
# (prompt id, max_new_tokens, ignore_eos): varied lengths so requests finish at different steps
REQUESTS = [("short-00", 24, False), ("short-03", 12, True), ("medium-02", 20, False), ("short-05", 8, False),
            ("short-08", 30, True), ("medium-09", 16, False), ("long-02", 10, True), ("short-11", 28, False),
            ("short-14", 6, True), ("medium-12", 22, False), ("short-19", 18, False), ("long-07", 14, True)]


def run_staggered(engine, encode, requests, waves=((0, 4), (3, 8), (8, 12))):
    """Add requests in waves while stepping: wave k's requests are added after `waves[k][0]` steps."""
    outputs, id_to_idx, steps, mixed_steps = {}, {}, 0, 0
    pending = list(waves)
    while pending or engine.has_unfinished():
        while pending and steps >= pending[0][0]:
            _, end = pending.pop(0)
            start = len(id_to_idx)
            for idx in range(start, end):
                pid, max_new, ignore_eos = requests[idx]
                sid = engine.add_request(encode(pick_prompts([pid])[0]), max_new, ignore_eos=ignore_eos)
                id_to_idx[sid] = idx
                outputs[idx] = []
        for out in engine.step():
            outputs[id_to_idx[out.seq_id]].append(out.token)
        stats = engine.last_step
        if stats and stats.n_prefill and stats.n_decode:
            mixed_steps += 1
        steps += 1
    return [outputs[i] for i in range(len(requests))], mixed_steps


def run_alone(model, encode, requests):
    single = Engine(model, eos_token_ids=load_eos_token_ids(MODEL), max_seq_len=1024)
    return [list(single.generate(encode(pick_prompts([pid])[0]), max_new, ignore_eos=ign))
            for pid, max_new, ign in requests]


def make_engine(model, policy, max_num_seqs=6):
    return LLMEngine(model, eos_token_ids=load_eos_token_ids(MODEL), max_seq_len=1024, policy=policy,
                     max_num_seqs=max_num_seqs, batch_wait_s=0.0)


def test_fp32_continuous_equals_alone(ours_fp32_cpu, encode):
    batched, mixed = run_staggered(make_engine(ours_fp32_cpu, "continuous"), encode, REQUESTS)
    ref = run_alone(ours_fp32_cpu, encode, REQUESTS)
    assert mixed > 0, "expected at least one step mixing prefill and decode"
    for i, (b, r) in enumerate(zip(batched, ref)):
        assert b == r, f"request {i} ({REQUESTS[i][0]}) differs"


def test_fp32_static_equals_alone(ours_fp32_cpu, encode):
    reqs = REQUESTS[:6]
    batched, _ = run_staggered(make_engine(ours_fp32_cpu, "static", max_num_seqs=4), encode, reqs,
                               waves=((0, 3), (2, 6)))
    assert batched == run_alone(ours_fp32_cpu, encode, reqs)


def test_bf16_continuous_matches_alone_up_to_ties(ours_bf16_gpu, encode):
    batched, _ = run_staggered(make_engine(ours_bf16_gpu, "continuous"), encode, REQUESTS)
    ref = run_alone(ours_bf16_gpu, encode, REQUESTS)
    exact = sum(b == r for b, r in zip(batched, ref))
    print(f"bf16 continuous vs alone: {exact}/{len(REQUESTS)} requests token-for-token identical")
    for i, (b, r) in enumerate(zip(batched, ref)):
        if b == r:
            continue
        j = next(k for k in range(min(len(b), len(r)) + 1) if k == min(len(b), len(r)) or b[k] != r[k])
        prompt = encode(pick_prompts([REQUESTS[i][0]])[0])
        seq = torch.tensor(prompt + r[:j], device="cuda")
        logits = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float()
        top2 = logits.topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i} ({REQUESTS[i][0]}): diverges at token {j}, reference top-2 gap {gap:.3f}")
        assert gap <= TIE_GAP, f"request {i}: divergence at a non-tie (gap {gap:.3f})"


def test_finished_requests_release_their_cache(ours_bf16_gpu, encode):
    eng = make_engine(ours_bf16_gpu, "continuous")
    sid = eng.add_request(encode("Say hi."), 3, ignore_eos=True)
    seq = eng.sequences[sid]
    while eng.has_unfinished():
        eng.step()
    assert seq.cache is None and sid not in eng.sequences


def test_rejects_requests_longer_than_max_seq_len(ours_bf16_gpu):
    eng = make_engine(ours_bf16_gpu, "continuous")
    with pytest.raises(ValueError):
        eng.add_request(list(range(1000)), 100)
