"""M4: the engine with the Triton paged-attention decode kernel vs the same engine on gather+SDPA.

bf16 on the GPU (the kernel needs CUDA). The kernel sums in a different order than SDPA, so this is the
approved tie rule: report the exact-match rate; any divergence must be at a reference top-2 gap <= 0.25.
(The strict fp32 gate for the kernel is at the op level: test_paged_attention_kernel.py, <= 1e-5.)
"""
import pytest
import torch

from nanoserve.engine import LLMEngine, load_eos_token_ids
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

pytestmark = pytest.mark.gpu


def make(model, backend):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=6, batch_wait_s=0.0,
                     kv_cache="paged", kv_cache_tokens=16384, attn_backend=backend)


def test_default_backend_on_gpu_is_triton(ours_bf16_gpu):
    assert make(ours_bf16_gpu, None).attn_backend == "triton"


def test_triton_engine_matches_sdpa_engine_up_to_ties(ours_bf16_gpu, encode):
    tri, mixed = run_staggered(make(ours_bf16_gpu, "triton"), encode, REQUESTS)
    ref, _ = run_staggered(make(ours_bf16_gpu, "sdpa"), encode, REQUESTS)
    assert mixed > 0                                   # kernel decodes coexisted with SDPA prefills
    exact = sum(a == b for a, b in zip(tri, ref))
    print(f"triton vs sdpa engine: {exact}/{len(REQUESTS)} requests identical")
    for i, (a, b) in enumerate(zip(tri, ref)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        prompt = encode(pick_prompts([REQUESTS[i][0]])[0])
        seq = torch.tensor(prompt + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
