"""M8b: chunked prefill. A long prompt (or a preempted request's recompute) is processed over several
steps in chunks that fit the per-step token budget; a token is sampled only when a chunk reaches the end
of the known tokens.

Gates: scheduler behaviour (CPU, no model); engine with a tiny budget (every prompt chunked) gives the same
tokens as an unchunked engine: fp32 strict, bf16 approved tie rule; chunking + preemption still exact in fp32.
"""
import pytest
import torch

from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.kv_manager import PagedKVManager
from nanoserve.model import ModelConfig
from nanoserve.scheduler import Scheduler, Sequence
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

TINY = ModelConfig(vocab_size=10, hidden_size=8, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=2,
                   intermediate_size=8, rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)


def sched(budget, pool_tokens=256):
    kv = PagedKVManager(TINY, budget_tokens=pool_tokens, block_size=4, device="cpu", dtype=torch.float32)
    return Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=budget, kv=kv, chunked_prefill=True)


def seq(i, prompt_len, max_new=4):
    return Sequence(seq_id=i, prompt_ids=list(range(prompt_len)), max_new_tokens=max_new)


def fake_step(s):
    """Pretend the engine ran the step: commit each chunk; sample only when a chunk reaches the end."""
    out = s.schedule(now=0.0)
    sampled = []
    for x, n in zip(out.scheduled, out.chunks):
        x.block_table.commit(n)
        x.num_cached += n
        if x.num_cached == x.num_tokens:
            x.output_ids.append(7)
            sampled.append(x.seq_id)
    return out, sampled


def test_long_prompt_is_prefilled_in_budget_sized_chunks():
    s = sched(budget=10)
    a = seq(0, 25)
    s.add(a)                                    # 25 > budget: accepted because chunking is on
    chunks = []
    for _ in range(3):
        out, sampled = fake_step(s)
        chunks.append(out.chunks[0])
        assert sampled == ([0] if len(chunks) == 3 else [])
    assert chunks == [10, 10, 5]
    assert a.step_tokens(1) == ([7], [25])       # now decoding: newest token at position 25


def test_decodes_get_the_budget_first():
    s = sched(budget=10)
    a = seq(0, 4)
    s.add(a)
    fake_step(s)                                 # a prefilled, now decoding
    b = seq(1, 25)
    s.add(b)
    out, _ = fake_step(s)
    assert dict(zip([x.seq_id for x in out.scheduled], out.chunks)) == {0: 1, 1: 9}


def test_without_chunking_long_prompts_are_still_rejected():
    kv = PagedKVManager(TINY, budget_tokens=256, block_size=4, device="cpu", dtype=torch.float32)
    s = Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=10, kv=kv, chunked_prefill=False)
    with pytest.raises(ValueError):
        s.add(seq(0, 25))


# ---- with the real model ------------------------------------------------------------------------------
def make(model, budget, kv_tokens=16384, max_num_seqs=6, graphs=None):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=max_num_seqs,
                     max_num_batched_tokens=budget, batch_wait_s=0.0, kv_cache="paged",
                     kv_cache_tokens=kv_tokens, cuda_graphs=graphs)


@pytest.mark.gpu
def test_fp32_tiny_budget_equals_unchunked(ours_fp32_cpu, encode):
    eng = make(ours_fp32_cpu, budget=64)        # every prompt (43-600 tokens) gets chunked
    chunked, _ = run_staggered(eng, encode, REQUESTS)
    whole, _ = run_staggered(make(ours_fp32_cpu, budget=8192), encode, REQUESTS)
    print(f"partial prefill chunks: {eng.stats['prefill_chunks']}")
    assert eng.stats["prefill_chunks"] > 0
    assert chunked == whole


@pytest.mark.gpu
def test_bf16_tiny_budget_matches_unchunked_up_to_ties(ours_bf16_gpu, encode):
    chunked, _ = run_staggered(make(ours_bf16_gpu, budget=64), encode, REQUESTS)
    whole, _ = run_staggered(make(ours_bf16_gpu, budget=8192), encode, REQUESTS)
    print(f"bf16 chunked vs whole: {sum(a == b for a, b in zip(chunked, whole))}/12 identical")
    for i, (a, b) in enumerate(zip(chunked, whole)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        s = torch.tensor(encode(pick_prompts([REQUESTS[i][0]])[0]) + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(s, torch.arange(len(s), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP


@pytest.mark.gpu
def test_fp32_chunking_plus_preemption_is_exact(ours_fp32_cpu, encode):
    from tests.test_paged_engine import PREEMPT_REQUESTS, PREEMPT_WAVES
    eng = make(ours_fp32_cpu, budget=48, kv_tokens=700, max_num_seqs=8)
    tight, _ = run_staggered(eng, encode, PREEMPT_REQUESTS, PREEMPT_WAVES)
    roomy, _ = run_staggered(make(ours_fp32_cpu, budget=8192, max_num_seqs=8), encode, PREEMPT_REQUESTS,
                             PREEMPT_WAVES)
    print(f"preemptions {eng.stats['num_preemptions']}, partial chunks {eng.stats['prefill_chunks']}")
    assert eng.stats["num_preemptions"] > 0 and eng.stats["prefill_chunks"] > 0
    assert tight == roomy
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks
