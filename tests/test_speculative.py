"""M5: greedy speculative decoding must produce EXACTLY the target model's plain greedy output.

Run separately (needs the 1.5B model and ~8 GB RAM for the fp32 checks):
    uv run pytest -m spec -v

Gates:
  * self-speculation (draft == target == 0.5B, fp32 CPU): acceptance >= 95% (would be 100% in exact
    arithmetic; multi-token verification vs one-token drafting round differently) and output identical.
  * 0.5B draft -> 1.5B target, fp32 CPU: output token-for-token identical to plain greedy 1.5B.   <- strict
  * same in bf16 on the GPU: approved tie rule (divergence only at reference top-2 gap <= 0.25).
  * batched: several requests at once give the same tokens as one at a time.
"""
import pytest
import torch

from nanoserve.engine import Engine, load_eos_token_ids
from nanoserve.model import Qwen2Model
from nanoserve.speculative import SpeculativeEngine
from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.spec

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = ["short-00", "short-10", "medium-04", "short-17"]
MAX_NEW = 24
TIE_GAP = 0.25


@pytest.fixture(scope="module")
def target_fp32_cpu():
    return Qwen2Model.from_pretrained(TARGET, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def target_bf16_gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return Qwen2Model.from_pretrained(TARGET, device="cuda", dtype=torch.bfloat16)


def spec_run(target, draft, encode, prompts, k=4, max_num_seqs=1, cuda_graphs=None):
    eng = SpeculativeEngine(target, draft, load_eos_token_ids(MODEL), num_speculative_tokens=k,
                            max_seq_len=1024, max_num_seqs=max_num_seqs, cuda_graphs=cuda_graphs)
    ids = {eng.add_request(encode(pick_prompts([p])[0]), MAX_NEW): i for i, p in enumerate(prompts)}
    outs = {i: [] for i in range(len(prompts))}
    while eng.has_unfinished():
        for o in eng.step():
            outs[ids[o.seq_id]].append(o.token)
    return [outs[i] for i in range(len(prompts))], eng.stats


def plain_greedy(model, encode, prompts):
    eng = Engine(model, load_eos_token_ids(MODEL), max_seq_len=1024)
    return [list(eng.generate(encode(pick_prompts([p])[0]), MAX_NEW)) for p in prompts]


def test_self_speculation_accepts_almost_everything(ours_fp32_cpu, encode):
    out, st = spec_run(ours_fp32_cpu, ours_fp32_cpu, encode, PROMPTS[:2])
    print(f"self-spec acceptance {st['acceptance_rate']:.3f}")
    assert st["acceptance_rate"] >= 0.95
    assert out == plain_greedy(ours_fp32_cpu, encode, PROMPTS[:2])


def test_fp32_gpu_graphed_self_speculation_is_identical(encode):
    # M7b strict gate for the GRAPHED path: draft steps and verification rows replayed as CUDA graphs
    # (Triton kernel, fp32 IEEE) must still give exactly plain greedy. Loaded here and freed after, so the
    # 2 GB fp32 copy is not resident together with the 1.5B bf16 model used by later tests.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = Qwen2Model.from_pretrained(MODEL, device="cuda", dtype=torch.float32)
    try:
        out, st = spec_run(m, m, encode, PROMPTS[:2], cuda_graphs=True)
        print(f"graphed self-spec fp32: acceptance {st['acceptance_rate']:.3f}, draft graph steps "
              f"{st['graph_draft_steps']}, verify graph steps {st['graph_verify_steps']}")
        assert st["graph_draft_steps"] > 0 and st["graph_verify_steps"] > 0
        assert st["acceptance_rate"] >= 0.95
        assert out == plain_greedy(m, encode, PROMPTS[:2])
    finally:
        del m
        torch.cuda.empty_cache()


def test_fp32_spec_equals_plain_greedy_target(target_fp32_cpu, ours_fp32_cpu, encode):
    out, st = spec_run(target_fp32_cpu, ours_fp32_cpu, encode, PROMPTS)
    print(f"0.5B->1.5B fp32 acceptance {st['acceptance_rate']:.3f}, "
          f"tokens per target forward {st['tokens_per_target_forward']:.2f}")
    assert out == plain_greedy(target_fp32_cpu, encode, PROMPTS)
    assert 0 < st["acceptance_rate"] <= 1


def test_fp32_batched_spec_equals_one_at_a_time(target_fp32_cpu, ours_fp32_cpu, encode):
    batched, _ = spec_run(target_fp32_cpu, ours_fp32_cpu, encode, PROMPTS, max_num_seqs=4)
    single, _ = spec_run(target_fp32_cpu, ours_fp32_cpu, encode, PROMPTS, max_num_seqs=1)
    assert batched == single


@pytest.mark.parametrize("graphs", [False, True])
def test_bf16_spec_matches_plain_greedy_up_to_ties(graphs, target_bf16_gpu, ours_bf16_gpu, encode):
    out, st = spec_run(target_bf16_gpu, ours_bf16_gpu, encode, PROMPTS, cuda_graphs=graphs)
    ref = plain_greedy(target_bf16_gpu, encode, PROMPTS)
    print(f"bf16 graphs={graphs} acceptance {st['acceptance_rate']:.3f}; "
          f"identical {sum(a == b for a, b in zip(out, ref))}/{len(ref)}")
    if graphs:
        assert st["graph_draft_steps"] > 0 and st["graph_verify_steps"] > 0
    for i, (a, b) in enumerate(zip(out, ref)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        seq = torch.tensor(encode(pick_prompts([PROMPTS[i]])[0]) + b[:j], device="cuda")
        top2 = target_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP


def test_rejects_requests_that_cannot_fit(ours_fp32_cpu):
    eng = SpeculativeEngine(ours_fp32_cpu, ours_fp32_cpu, [151645], max_seq_len=64)
    with pytest.raises(ValueError):
        eng.add_request(list(range(60)), 10)
