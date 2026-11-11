"""M7: CUDA-graph decode steps.

A graph replays EXACTLY the kernels of one eager decode step, with inputs in static buffers. Gates:
  * at an exact bucket size (no padding): graph logits BITWISE equal to the eager step;
  * padded batch (3 real rows in a 4-row graph): matmul shapes differ -> approved tie rule;
  * engine with graphs vs without (12 staggered requests): approved tie rule; graphs actually used;
  * padding rows only ever touch the reserved garbage block; no blocks leak.
"""
import pytest
import torch

from nanoserve.cuda_graphs import DecodeGraphRunner
from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import DecodeMeta, SeqInput
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

PROMPTS = ["short-00", "medium-04", "short-07", "long-00"]


def setup(model, encode, n):
    pool = PagedKVCache.from_token_budget(model.cfg, 8192, 16, "cuda", model.dtype)
    runner = DecodeGraphRunner(model, pool, max_seq_len=1024, max_batch=8)
    prompts = [encode(pick_prompts([p])[0]) for p in PROMPTS[:n]]
    tables = []
    for ids in prompts:
        t = BlockTable(16)
        t.reserve(len(ids) + 8, pool.allocator)
        model.forward_batch(torch.tensor(ids, device="cuda"), torch.arange(len(ids), device="cuda"),
                            [SeqInput(len(ids))], paged=pool,
                            slot_mapping=torch.tensor(t.slots(len(ids)), device="cuda"))
        t.commit(len(ids))
        tables.append(t)
    inputs = dict(tokens=[42 + i for i in range(n)], positions=[len(p) for p in prompts],
                  slots=[t.slots(1)[0] for t in tables], block_tables=[t.blocks for t in tables],
                  context_lens=[t.num_tokens + 1 for t in tables])
    return pool, runner, tables, inputs


def eager(model, pool, tables, inp):
    width = max(len(b) for b in inp["block_tables"])
    meta = DecodeMeta(token_idx=torch.arange(len(tables), device="cuda"),
                      block_tables=torch.tensor([b + [0] * (width - len(b)) for b in inp["block_tables"]],
                                                dtype=torch.int32, device="cuda"),
                      context_lens=torch.tensor(inp["context_lens"], dtype=torch.int32, device="cuda"))
    return model.forward_batch(torch.tensor(inp["tokens"], device="cuda"), torch.tensor(inp["positions"], device="cuda"),
                               [SeqInput(1, ctx_len=t.num_tokens, use_kernel=True) for t in tables],
                               logits="all", paged=pool,
                               slot_mapping=torch.tensor(inp["slots"], device="cuda"), decode_meta=meta)


def test_graph_step_bitwise_equals_eager_at_bucket_size(ours_bf16_gpu, encode):
    pool, runner, tables, inp = setup(ours_bf16_gpu, encode, 4)
    ref = eager(ours_bf16_gpu, pool, tables, inp).clone()
    out = runner.run(**inp).clone()          # writes the same K/V to the same slots, then same kernels
    print(f"graph vs eager (B=4): max|diff|={(out.float() - ref.float()).abs().max().item():.3g}")
    assert out.shape == ref.shape and torch.equal(out, ref)
    assert runner.captured_buckets == [4]


def test_padded_batch_matches_eager_up_to_ties(ours_bf16_gpu, encode):
    pool, runner, tables, inp = setup(ours_bf16_gpu, encode, 3)
    ref = eager(ours_bf16_gpu, pool, tables, inp).float()
    out = runner.run(**inp).float()          # 3 real rows inside the 4-row graph
    assert out.shape == ref.shape and runner.captured_buckets == [4]
    for i in range(3):
        top2 = ref[i].topk(2).values
        if (top2[0] - top2[1]).item() > TIE_GAP:
            assert out[i].argmax() == ref[i].argmax()


def test_padding_only_touches_the_garbage_block(ours_bf16_gpu, encode):
    pool, runner, tables, inp = setup(ours_bf16_gpu, encode, 3)
    real_blocks = {b for t in tables for b in t.blocks}
    assert runner.garbage_block not in real_blocks
    k_before = pool.k[:, sorted(real_blocks)].clone()
    runner.run(**inp)
    # Only the 3 real slots (one per sequence) may change in real blocks.
    changed = (pool.k[:, sorted(real_blocks)] != k_before).any(dim=(0, 3, 4)).nonzero().tolist()
    assert len(changed) <= 3


def make(model, graphs):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=6, batch_wait_s=0.0,
                     kv_cache="paged", kv_cache_tokens=16384, attn_backend="triton", cuda_graphs=graphs)


def test_engine_with_graphs_matches_without_up_to_ties(ours_bf16_gpu, encode):
    eng = make(ours_bf16_gpu, True)
    with_g, _ = run_staggered(eng, encode, REQUESTS)
    without, _ = run_staggered(make(ours_bf16_gpu, False), encode, REQUESTS)
    st = eng.stats
    print(f"graph steps {st['graph_steps']}, eager steps {st['steps'] - st['graph_steps']}, "
          f"buckets {eng.graph_runner.captured_buckets}; identical {sum(a == b for a, b in zip(with_g, without))}/12")
    assert st["graph_steps"] > 0 and st["graph_steps"] < st["steps"]   # decode-only steps used graphs
    for i, (a, b) in enumerate(zip(with_g, without)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        seq = torch.tensor(encode(pick_prompts([REQUESTS[i][0]])[0]) + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks - 1     # only the garbage block is held


def test_graphs_off_by_default_without_triton(ours_bf16_gpu):
    eng = LLMEngine(ours_bf16_gpu, [151645], max_seq_len=256, kv_cache_tokens=1024, attn_backend="sdpa")
    assert eng.graph_runner is None
