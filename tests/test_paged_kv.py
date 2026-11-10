"""M3 step 1: block allocator, block tables and the paged KV pool (CPU, no model)."""
import random

import pytest
import torch

from nanoserve.kv_cache import BlockAllocator, BlockTable, NoFreeBlocks, PagedKVCache
from nanoserve.model import ModelConfig

TINY = ModelConfig(vocab_size=10, hidden_size=8, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=3,
                   intermediate_size=16, rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)


def test_allocator_hands_out_every_block_once_then_raises():
    a = BlockAllocator(4)
    got = [a.allocate() for _ in range(4)]
    assert sorted(got) == [0, 1, 2, 3] and a.num_free == 0
    with pytest.raises(NoFreeBlocks):
        a.allocate()
    a.free([got[1]])
    assert a.num_free == 1 and a.allocate() == got[1]


def test_allocator_rejects_double_free():
    a = BlockAllocator(2)
    b = a.allocate()
    a.free([b])
    with pytest.raises(ValueError):
        a.free([b])


def test_block_table_grows_one_block_per_block_size_tokens():
    a = BlockAllocator(10)
    t = BlockTable(block_size=4)
    assert t.blocks_needed(5) == 2
    t.reserve(5, a)
    assert len(t.blocks) == 2
    slots = t.slots(5)
    assert slots == [t.blocks[0] * 4 + i for i in range(4)] + [t.blocks[1] * 4]
    t.commit(5)
    assert t.blocks_needed(3) == 0          # tokens 5,6,7 still fit in block 2
    t.reserve(3, a)
    t.commit(3)
    assert t.blocks_needed(1) == 1          # token 8 starts block 3
    t.reserve(1, a)
    assert t.slots(1) == [t.blocks[2] * 4]
    t.commit(1)
    used = a.num_free
    t.release(a)
    assert a.num_free == used + 3 and t.blocks == [] and t.num_tokens == 0


def test_truncate_frees_blocks_past_the_new_length():
    a = BlockAllocator(10)
    t = BlockTable(block_size=4)
    t.reserve(10, a)
    t.commit(10)
    t.truncate(5, a)                        # speculative-decoding rollback
    assert t.num_tokens == 5 and len(t.blocks) == 2 and a.num_free == 8


def test_pool_write_and_gather_round_trip_with_fragmented_blocks():
    torch.manual_seed(0)
    pool = PagedKVCache(TINY, num_blocks=12, block_size=4, device="cpu", dtype=torch.float32)
    random.seed(0)
    random.shuffle(pool.allocator._free)    # scatter blocks so indirection really matters
    seqs = {name: (BlockTable(4), torch.randn(n, TINY.n_kv_heads, TINY.head_dim), torch.randn(n, 2, 3))
            for name, n in (("A", 7), ("B", 10))}
    # write in interleaved chunks, the way steps would: A 3, B 10, A 4 (one batched write per step)
    for chunk in ((("A", 3), ("B", 10)), (("A", 4),)):
        slots, ks, vs = [], [], []
        for name, n in chunk:
            table, k, v = seqs[name]
            table.reserve(n, pool.allocator)
            slots += table.slots(n)
            ks.append(k[table.num_tokens:table.num_tokens + n])
            vs.append(v[table.num_tokens:table.num_tokens + n])
            table.commit(n)
        for layer in range(TINY.n_layers):
            pool.write(layer, torch.cat(ks), torch.cat(vs), torch.tensor(slots))
    for name, (table, k, v) in seqs.items():
        gk, gv = pool.gather(1, torch.tensor(table.blocks), table.num_tokens)
        assert gk.shape == (1, TINY.n_kv_heads, table.num_tokens, TINY.head_dim)
        assert torch.equal(gk[0].transpose(0, 1), k) and torch.equal(gv[0].transpose(0, 1), v)


def test_pool_sizing_from_a_token_budget():
    pool = PagedKVCache.from_token_budget(TINY, num_tokens=33, block_size=16, device="cpu", dtype=torch.float32)
    assert pool.num_blocks == 3                                  # rounds up
    assert pool.reserved_bytes == 3 * 16 * TINY.kv_bytes_per_token(torch.float32)
    assert PagedKVCache.tokens_for_bytes(TINY, pool.reserved_bytes, torch.float32) == 48
