"""M1 step 3: the greedy generation loop must reproduce HF greedy `generate` TOKEN FOR TOKEN (bf16, GPU).

Reference: HF generate with pure greedy settings (generation_config sampling defaults and
repetition_penalty overridden), natural EOS stopping, max_new_tokens=64.
"""
import pytest
import torch

from nanoserve.engine import Engine, load_eos_token_ids
from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.gpu

PROMPT_IDS = ["short-00", "short-03", "short-10", "medium-01", "medium-07", "medium-15", "long-03", "long-10"]
MAX_NEW = 64


@pytest.fixture(scope="module")
def engine(ours_bf16_gpu):
    return Engine(ours_bf16_gpu, eos_token_ids=load_eos_token_ids(MODEL), max_seq_len=2048)


def hf_greedy(hf_model, prompt_ids, max_new):
    ids = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        out = hf_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    return out[0, len(prompt_ids):].tolist()


def divergence_report(model, prompt_ids, ours, ref):
    """Where do the sequences first differ, and was it a near-tie between the top two candidates?"""
    i = next(j for j in range(min(len(ours), len(ref)) + 1) if j == min(len(ours), len(ref)) or ours[j] != ref[j])
    seq = torch.tensor(prompt_ids + ref[:i], device="cuda")
    logits = model.forward(seq, torch.arange(len(seq), device="cuda"), logits="last")[0].float()
    top2 = logits.topk(2)
    return (f"first divergence at generated token {i}: ours={ours[i:i + 3]} ref={ref[i:i + 3]}; "
            f"top-2 logit gap there = {(top2.values[0] - top2.values[1]).item():.4f}")


def test_eos_ids_come_from_generation_config():
    assert load_eos_token_ids(MODEL) == [151645, 151643]


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_matches_hf_greedy_generate(prompt_id, engine, hf_bf16_gpu, ours_bf16_gpu, encode):
    prompt_ids = encode(pick_prompts([prompt_id])[0])
    ours = list(engine.generate(prompt_ids, max_new_tokens=MAX_NEW))
    ref = hf_greedy(hf_bf16_gpu, prompt_ids, MAX_NEW)
    assert ours == ref, divergence_report(ours_bf16_gpu, prompt_ids, ours, ref)


def test_ignore_eos_generates_exactly_max_new_tokens(engine, encode):
    prompt_ids = encode("Reply with only the single word: yes")
    natural = list(engine.generate(prompt_ids, max_new_tokens=40))
    forced = list(engine.generate(prompt_ids, max_new_tokens=40, ignore_eos=True))
    assert len(natural) < 40 and natural[-1] in engine.eos_token_ids  # the model does want to stop early
    assert len(forced) == 40
    assert forced[:len(natural)] == natural  # ignore_eos keeps going AFTER eos; it doesn't change earlier tokens


def test_streams_tokens_one_at_a_time(engine, encode):
    gen = engine.generate(encode("Count from one to twenty."), max_new_tokens=8)
    first = next(gen)
    assert isinstance(first, int)
    assert len([first, *gen]) <= 8


def test_rejects_requests_longer_than_the_reservation(ours_bf16_gpu):
    small = Engine(ours_bf16_gpu, eos_token_ids=[151645], max_seq_len=16)
    with pytest.raises(ValueError):
        list(small.generate(list(range(10)), max_new_tokens=10))
