"""M3 steps 2-3: paged attention path and the engine on the paged cache.

Gates:
  * one mixed batched step (2 prefill + 3 decode): paged == contiguous. fp32 max|diff| < 1e-5;
    bf16 expected BITWISE equal (identical K/V values feed identical kernels; only memory layout differs).
  * multi-token step on a non-empty paged context (needed by M5 verification) == token-by-token decode
    (fp32 < 1e-4, same argmax).
  * engine: paged == contiguous token-for-token (fp32 and bf16).
  * preemption: with a pool too small for everyone, preempted+recomputed requests still produce the same
    tokens (fp32 strict; bf16 approved tie rule), preemptions actually happened, no blocks leak.
"""
import random

import pytest
import torch

from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.kv_cache import BlockTable, ContiguousKVCache, PagedKVCache
from nanoserve.model import SeqInput
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

pytestmark = pytest.mark.gpu

PREFILL_IDS = ["short-00", "medium-04"]
DECODE_IDS = ["short-02", "short-07", "long-00"]


def mixed_step(model, encode, device, paged: bool):
    """Prefill DECODE_IDS alone first, then run one batched step: 2 prefills + 3 one-token decodes."""
    pool = PagedKVCache.from_token_budget(model.cfg, 4096, 16, device, model.dtype) if paged else None
    seqs = []
    for pid in PREFILL_IDS + DECODE_IDS:
        ids = encode(pick_prompts([pid])[0])
        seqs.append({"ids": ids, "table": BlockTable(16) if paged else None,
                     "cache": None if paged else ContiguousKVCache(model.cfg, 1024, device, model.dtype)})
    if paged:
        random.Random(0).shuffle(pool.allocator._free)   # fragmented blocks: indirection must work

    def run(batch):  # batch: list of (seq, tokens, positions)
        toks = torch.tensor([t for _, ts, _ in batch for t in ts], device=device)
        pos = torch.tensor([p for _, _, ps in batch for p in ps], device=device)
        inputs, slots = [], []
        for s, ts, _ in batch:
            if paged:
                s["table"].reserve(len(ts), pool.allocator)
                slots += s["table"].slots(len(ts))
                inputs.append(SeqInput(len(ts), block_table=torch.tensor(s["table"].blocks, device=device),
                                       ctx_len=s["table"].num_tokens))
            else:
                inputs.append(SeqInput(len(ts), cache=s["cache"]))
        kw = {"paged": pool, "slot_mapping": torch.tensor(slots, device=device)} if paged else {}
        out = model.forward_batch(toks, pos, inputs, **kw)
        if paged:
            for s, ts, _ in batch:
                s["table"].commit(len(ts))
        return out

    for s in seqs[2:]:
        run([(s, s["ids"], list(range(len(s["ids"]))))])
    batch = [(s, s["ids"], list(range(len(s["ids"])))) for s in seqs[:2]]
    batch += [(s, [42 + i], [len(s["ids"])]) for i, s in enumerate(seqs[2:])]
    return run(batch)


def test_fp32_paged_step_equals_contiguous(ours_fp32_cpu, encode):
    a = mixed_step(ours_fp32_cpu, encode, "cpu", paged=True)
    b = mixed_step(ours_fp32_cpu, encode, "cpu", paged=False)
    assert (a - b).abs().max().item() < 1e-5


def test_bf16_paged_step_bitwise_equals_contiguous(ours_bf16_gpu, encode):
    a = mixed_step(ours_bf16_gpu, encode, "cuda", paged=True)
    b = mixed_step(ours_bf16_gpu, encode, "cuda", paged=False)
    print(f"bf16 paged vs contiguous: max|diff|={(a.float() - b.float()).abs().max().item():.4g}")
    assert torch.equal(a, b)


def test_multi_token_step_on_existing_context_equals_token_by_token(ours_fp32_cpu, encode):
    m = ours_fp32_cpu
    ids = encode(pick_prompts(["short-02"])[0])
    extra = [11, 22, 33, 44]

    def fresh():
        pool = PagedKVCache.from_token_budget(m.cfg, 1024, 16, "cpu", m.dtype)
        table = BlockTable(16)
        return pool, table

    def step(pool, table, toks, start):
        table.reserve(len(toks), pool.allocator)
        slots = torch.tensor(table.slots(len(toks)))
        out = m.forward_batch(torch.tensor(toks), torch.arange(start, start + len(toks)),
                              [SeqInput(len(toks), block_table=torch.tensor(table.blocks), ctx_len=table.num_tokens)],
                              logits="all", paged=pool, slot_mapping=slots)
        table.commit(len(toks))
        return out

    pool, table = fresh()
    step(pool, table, ids, 0)
    together = step(pool, table, extra, len(ids))                     # 4 tokens on a non-empty context
    pool, table = fresh()
    step(pool, table, ids, 0)
    one_by_one = torch.cat([step(pool, table, [t], len(ids) + i) for i, t in enumerate(extra)])
    assert torch.equal(together.argmax(-1), one_by_one.argmax(-1))
    assert (together - one_by_one).abs().max().item() < 1e-4


def engine(model, kv_cache, kv_cache_tokens=16384, max_num_seqs=6):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=max_num_seqs,
                     batch_wait_s=0.0, kv_cache=kv_cache, kv_cache_tokens=kv_cache_tokens)


def test_fp32_engine_paged_equals_contiguous(ours_fp32_cpu, encode):
    paged, _ = run_staggered(engine(ours_fp32_cpu, "paged"), encode, REQUESTS)
    contig, _ = run_staggered(engine(ours_fp32_cpu, "contiguous", kv_cache_tokens=6 * 1024), encode, REQUESTS)
    assert paged == contig


def test_bf16_engine_paged_equals_contiguous(ours_bf16_gpu, encode):
    paged, _ = run_staggered(engine(ours_bf16_gpu, "paged"), encode, REQUESTS)
    contig, _ = run_staggered(engine(ours_bf16_gpu, "contiguous", kv_cache_tokens=6 * 1024), encode, REQUESTS)
    assert paged == contig


# Preemption workload: 8 requests that all grow by 40 tokens (ignore_eos) in a 700-token pool. At
# admission they fill the pool; as they decode they need more blocks than exist -> preemption is certain.
PREEMPT_REQUESTS = [("medium-01", 40, True), ("short-01", 40, True), ("medium-03", 40, True),
                    ("short-04", 40, True), ("medium-05", 40, True), ("short-06", 40, True),
                    ("medium-08", 40, True), ("short-09", 40, True)]
PREEMPT_WAVES = ((0, 4), (1, 8))


def tight_engine(model):
    return engine(model, "paged", kv_cache_tokens=700, max_num_seqs=8)


def test_fp32_preemption_preserves_outputs(ours_fp32_cpu, encode):
    eng = tight_engine(ours_fp32_cpu)
    tight, _ = run_staggered(eng, encode, PREEMPT_REQUESTS, PREEMPT_WAVES)
    roomy, _ = run_staggered(engine(ours_fp32_cpu, "paged", max_num_seqs=8), encode, PREEMPT_REQUESTS, PREEMPT_WAVES)
    print(f"preemptions: {eng.stats['num_preemptions']}")
    assert eng.stats["num_preemptions"] > 0
    assert tight == roomy
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks      # nothing leaked


def test_bf16_preemption_preserves_outputs_up_to_ties(ours_bf16_gpu, encode):
    eng = tight_engine(ours_bf16_gpu)
    tight, _ = run_staggered(eng, encode, PREEMPT_REQUESTS, PREEMPT_WAVES)
    roomy, _ = run_staggered(engine(ours_bf16_gpu, "paged", max_num_seqs=8), encode, PREEMPT_REQUESTS, PREEMPT_WAVES)
    print(f"preemptions: {eng.stats['num_preemptions']}; identical: {sum(a == b for a, b in zip(tight, roomy))}/{len(tight)}")
    assert eng.stats["num_preemptions"] > 0
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks
    for i, (a, b) in enumerate(zip(tight, roomy)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        prompt = encode(pick_prompts([PREEMPT_REQUESTS[i][0]])[0])
        seq = torch.tensor(prompt + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
