"""M14: the 1.5B model on the full fast path (fused Triton ops + CUDA graphs + piecewise graphs + Triton
attention) vs the unfused single-request reference (bitwise-HF path), approved bf16 tie rule. Different
head layout from 0.5B (12 query / 2 KV heads, head_dim 128, 28 layers), so it exercises the kernels'
other shapes end to end. Marked `spec` with the other 1.5B tests: `uv run pytest -m spec`."""
import gc

import pytest
import torch

from nanoserve.engine import Engine, LLMEngine, load_eos_token_ids
from tests.conftest import pick_prompts
from tests.test_llm_engine import TIE_GAP

pytestmark = [pytest.mark.gpu, pytest.mark.spec]
MODEL_1P5B = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = ["short-00", "medium-04", "short-07", "long-00"]

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)


def test_1p5b_fast_path_matches_reference_up_to_ties(target_bf16_gpu, encode):
    m1p5 = target_bf16_gpu                                  # the shared session copy (see conftest.py)
    _check(m1p5, encode)
    gc.collect()                                            # free this test's KV pools and CUDA graphs
    torch.cuda.empty_cache()


def _check(m1p5, encode):
    eos = load_eos_token_ids(MODEL_1P5B)
    fast = LLMEngine(m1p5, eos, max_seq_len=1024, max_num_seqs=4, kv_cache_tokens=8192)
    assert fast.model.fused and fast.graph_runner is not None and fast.piecewise_runner is not None
    ref = Engine(m1p5, eos, max_seq_len=1024)
    prompts = [encode(pick_prompts([p])[0]) for p in PROMPTS]
    ids = {fast.add_request(p, 32): i for i, p in enumerate(prompts)}       # batched together
    got = {i: [] for i in range(len(prompts))}
    while fast.has_unfinished():
        for out in fast.step():
            got[ids[out.seq_id]].append(out.token)
    identical = 0
    for i, p in enumerate(prompts):
        want = list(ref.generate(p, 32))
        if got[i] == want:
            identical += 1
            continue
        j = next(k for k in range(min(len(got[i]), len(want)) + 1)
                 if k == min(len(got[i]), len(want)) or got[i][k] != want[k])
        seq = torch.tensor(p + want[:j], device="cuda")
        top2 = m1p5.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  {PROMPTS[i]}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
    print(f"1.5B fast path vs reference: {identical}/{len(prompts)} identical")
