"""M12: automatic prefix caching.

Allocator units (no model): reference counts, exact-prefix lookup, LRU eviction of cached blocks,
duplicate registration, double-free detection. Engine: prefix caching on vs off must give the SAME tokens
(fp32 strict, also with eviction + preemption, no leaked blocks); a repeated prompt hits all of its full
blocks but one token; the bf16 GPU fast path follows the approved tie rule.
"""
import pytest
import torch

from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.kv_cache import BlockAllocator, BlockTable, NoFreeBlocks, PrefixCachingAllocator
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import TIE_GAP

BS = 4


def table_with(alloc, tokens):
    """A block table holding `tokens` (KV 'written'), registered in the cache like the engine does."""
    t = BlockTable(BS)
    t.reserve(len(tokens), alloc)
    t.commit(len(tokens))
    alloc.register(t, tokens)
    return t


# ---- allocator units ------------------------------------------------------------------------------------
def test_full_blocks_are_cached_and_found_by_exact_prefix():
    alloc = PrefixCachingAllocator(BlockAllocator(16), BS)
    toks = list(range(10))                                  # 2 full blocks + 2 tokens
    t = table_with(alloc, toks)
    assert alloc.lookup(toks) == t.blocks[:2]               # the partial third block is never shared
    assert alloc.lookup(toks[:8] + [99, 98]) == t.blocks[:2]
    assert alloc.lookup([0, 1, 2, 3, 9, 9, 9, 9]) == t.blocks[:1]   # stops at the first different block
    assert alloc.lookup([5, 1, 2, 3]) == []
    assert alloc.lookup(toks, max_blocks=1) == t.blocks[:1]


def test_same_tokens_after_a_different_prefix_do_not_match():
    alloc = PrefixCachingAllocator(BlockAllocator(16), BS)
    table_with(alloc, [1, 1, 1, 1, 7, 7, 7, 7])
    assert alloc.lookup([2, 2, 2, 2, 7, 7, 7, 7]) == []      # block 2 equal, but its prefix differs
    assert len(alloc.lookup([1, 1, 1, 1, 7, 7, 7, 7])) == 2


def test_freed_cached_blocks_are_evictable_and_reusable():
    alloc = PrefixCachingAllocator(BlockAllocator(4), BS)
    t = table_with(alloc, list(range(8)))                   # 2 cached blocks
    assert alloc.num_free == 2
    t.release(alloc)
    assert alloc.num_free == 4 and alloc.num_evictable == 2  # still cached, but counted as free capacity
    hits = alloc.lookup(list(range(8)))
    alloc.pin(hits)                                         # a new request takes them over
    assert alloc.num_evictable == 0 and alloc.num_free == 2
    alloc.free(hits)
    assert alloc.num_evictable == 2


def test_allocation_evicts_least_recently_freed_cached_block():
    alloc = PrefixCachingAllocator(BlockAllocator(2), BS)
    a = table_with(alloc, [1, 1, 1, 1])
    b = table_with(alloc, [2, 2, 2, 2])
    a_block = a.blocks[0]
    a.release(alloc)                                        # a's block freed first -> evicted first
    b.release(alloc)
    assert alloc.allocate() == a_block
    assert alloc.lookup([1, 1, 1, 1]) == [] and len(alloc.lookup([2, 2, 2, 2])) == 1
    alloc.allocate()
    assert alloc.lookup([2, 2, 2, 2]) == []
    with pytest.raises(NoFreeBlocks):
        alloc.allocate()


def test_duplicate_prefix_computed_twice_is_registered_once():
    alloc = PrefixCachingAllocator(BlockAllocator(8), BS)
    a = table_with(alloc, list(range(8)))
    b = table_with(alloc, list(range(8)))                   # same prompt, computed concurrently (no hit)
    assert alloc.lookup(list(range(8))) == a.blocks
    a.release(alloc)
    b.release(alloc)                                        # b's blocks were never registered: truly free
    assert alloc.num_free == 8 and alloc.num_evictable == 2


def test_double_free_is_detected():
    alloc = PrefixCachingAllocator(BlockAllocator(4), BS)
    blk = alloc.allocate()
    alloc.free([blk])
    with pytest.raises(ValueError):
        alloc.free([blk])


# ---- engine ---------------------------------------------------------------------------------------------
SYSTEM = pick_prompts(["long-00"])[0]                     # a long pasted paragraph, used as a shared prefix
USERS = ["short-00", "short-01", "short-03", "short-04", "short-05", "medium-02"]


def shared_prefix_prompts(tokenizer):
    out = []
    for pid in USERS:
        text = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": pick_prompts([pid])[0]}],
                                             tokenize=False, add_generation_prompt=True)
        out.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    return out


def run_waves(eng, prompts, max_new=16, waves=((0, 2), (6, 4), (12, 6))):
    """Add prompts in waves (later waves arrive after earlier prefixes are cached); collect tokens."""
    outs, idx, steps, pending = {}, {}, 0, list(waves)
    while pending or eng.has_unfinished():
        while pending and steps >= pending[0][0]:
            _, end = pending.pop(0)
            for i in range(len(idx), end):
                idx[eng.add_request(prompts[i], max_new)] = i
                outs[i] = []
        for out in eng.step():
            outs[idx[out.seq_id]].append(out.token)
        steps += 1
    return [outs[i] for i in range(len(prompts))]


def make(model, prefix, kv_tokens=16384, budget=2048, **kw):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=6, batch_wait_s=0.0,
                     kv_cache="paged", kv_cache_tokens=kv_tokens, max_num_batched_tokens=budget,
                     prefix_caching=prefix, **kw)


@pytest.mark.gpu
def test_fp32_prefix_caching_gives_identical_tokens(ours_fp32_cpu, tokenizer):
    prompts = shared_prefix_prompts(tokenizer)
    eng = make(ours_fp32_cpu, True)
    cached = run_waves(eng, prompts)
    plain = run_waves(make(ours_fp32_cpu, False), prompts)
    st = eng.stats
    print(f"prefix hits: {st['prefix_hit_tokens']} of {st['prefix_query_tokens']} prompt tokens")
    assert st["prefix_hit_tokens"] >= 0.3 * st["prefix_query_tokens"]   # waves 2-3 reuse the shared prefix
    assert cached == plain
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks


@pytest.mark.gpu
def test_fp32_eviction_and_preemption_stay_exact(ours_fp32_cpu, tokenizer):
    prompts = shared_prefix_prompts(tokenizer)
    n = max(len(p) for p in prompts)
    eng = make(ours_fp32_cpu, True, kv_tokens=n + 128, budget=128)   # tight: forces evictions/preemption
    tight = run_waves(eng, prompts)
    roomy = run_waves(make(ours_fp32_cpu, False), prompts)
    print(f"preemptions {eng.stats['num_preemptions']}, hits {eng.stats['prefix_hit_tokens']}")
    assert eng.stats["num_preemptions"] > 0
    assert tight == roomy
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks


@pytest.mark.gpu
def test_repeated_prompt_hits_all_full_blocks_but_one_token(ours_fp32_cpu, encode):
    eng = make(ours_fp32_cpu, True)
    p = encode(pick_prompts(["medium-04"])[0])
    first = run_waves(eng, [p], waves=((0, 1),))
    eng.reset_stats()
    second = run_waves(eng, [p], waves=((0, 1),))
    assert first == second
    assert eng.stats["prefix_hit_tokens"] == (len(p) - 1) // 16 * 16


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_fast_path_with_prefix_caching_up_to_ties(ours_bf16_gpu, tokenizer):
    prompts = shared_prefix_prompts(tokenizer)
    eng = make(ours_bf16_gpu, True)
    assert eng.model.fused and eng.graph_runner is not None and eng.piecewise_runner is not None
    cached = run_waves(eng, prompts)
    plain = run_waves(make(ours_bf16_gpu, False), prompts)
    print(f"bf16: identical {sum(a == b for a, b in zip(cached, plain))}/{len(prompts)}, "
          f"hits {eng.stats['prefix_hit_tokens']}")
    for i, (a, b) in enumerate(zip(cached, plain)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        seq = torch.tensor(prompts[i] + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks - 1          # only the graphs' garbage block


def test_prefix_caching_needs_the_paged_cache(ours_fp32_cpu):
    with pytest.raises(ValueError):
        LLMEngine(ours_fp32_cpu, [151645], max_seq_len=256, kv_cache="contiguous", kv_cache_tokens=1024,
                  prefix_caching=True)
