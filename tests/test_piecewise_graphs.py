"""M9: piecewise CUDA graphs for MIXED steps (prefill chunks + decodes in one step).

Everything per-token (norms, projections, RoPE, KV write, MLP) is replayed from per-bucket graphs; only
attention runs eagerly between them. Gates:
  * one mixed batch (fresh prefill + chunk continuation on a cache + 2 kernel decodes) exactly filling a
    bucket: piecewise logits BITWISE equal the eager forward;
  * the same batch padded into a bigger bucket: approved tie rule;
  * padding rows only ever write the reserved garbage block;
  * engine with piecewise graphs vs without (budget 64, so most steps are mixed): approved tie rule,
    piecewise steps actually used, no leaked blocks.
"""
import pytest
import torch

from nanoserve.cuda_graphs import PiecewiseGraphRunner, piecewise_buckets
from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.kv_cache import BlockTable, PagedKVCache
from nanoserve.model import DecodeMeta, SeqInput
from tests.conftest import MODEL, pick_prompts
from tests.test_llm_engine import REQUESTS, TIE_GAP, run_staggered

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

A_NEW, B_CACHED, B_NEW = 40, 30, 22          # T = 40 + 22 + 1 + 1 = 64


def prefill(model, pool, table, ids, start):
    """Eagerly store K/V of ids at positions start.. (no attention output needed)."""
    model.forward_batch(torch.tensor(ids, device="cuda"), torch.arange(start, start + len(ids), device="cuda"),
                        [SeqInput(len(ids), block_table=torch.tensor(table.blocks, device="cuda") if start else None,
                                  ctx_len=start)],
                        paged=pool, slot_mapping=torch.tensor(table.slots(len(ids)), device="cuda"))
    table.commit(len(ids))


def setup(model, encode, buckets):
    """A 64-token mixed step: A = fresh 40-token prefill, B = 22-token chunk on top of 30 cached tokens,
    C and D = one-token decodes through the Triton kernel."""
    pool = PagedKVCache.from_token_budget(model.cfg, 8192, 16, "cuda", model.dtype)
    garbage = pool.allocator.allocate()
    runner = PiecewiseGraphRunner(model, pool, garbage, buckets=buckets)
    a_ids = encode(pick_prompts(["medium-04"])[0])[:A_NEW]
    b_ids = encode(pick_prompts(["long-00"])[0])[:B_CACHED + B_NEW]
    c_ids, d_ids = (encode(pick_prompts([p])[0]) for p in ("short-00", "short-07"))
    ta, tb, tc, td = (BlockTable(16) for _ in range(4))
    ta.reserve(A_NEW, pool.allocator)
    tb.reserve(B_CACHED + B_NEW, pool.allocator)
    prefill(model, pool, tb, b_ids[:B_CACHED], 0)
    for t, ids in ((tc, c_ids), (td, d_ids)):
        t.reserve(len(ids) + 1, pool.allocator)
        prefill(model, pool, t, ids, 0)

    tokens = a_ids + b_ids[B_CACHED:] + [42, 43]
    positions = list(range(A_NEW)) + list(range(B_CACHED, B_CACHED + B_NEW)) + [tc.num_tokens, td.num_tokens]
    slots = ta.slots(A_NEW) + tb.slots(B_NEW) + tc.slots(1) + td.slots(1)
    seqs = [SeqInput(A_NEW, ctx_len=0),
            SeqInput(B_NEW, block_table=torch.tensor(tb.blocks, device="cuda"), ctx_len=B_CACHED),
            SeqInput(1, ctx_len=tc.num_tokens, use_kernel=True),
            SeqInput(1, ctx_len=td.num_tokens, use_kernel=True)]
    width = max(len(tc.blocks), len(td.blocks))
    meta = DecodeMeta(token_idx=torch.tensor([62, 63], device="cuda"),
                      block_tables=torch.tensor([t.blocks + [0] * (width - len(t.blocks)) for t in (tc, td)],
                                                dtype=torch.int32, device="cuda"),
                      context_lens=torch.tensor([tc.num_tokens + 1, td.num_tokens + 1], dtype=torch.int32,
                                                device="cuda"))
    rows = torch.tensor([A_NEW - 1, A_NEW + B_NEW - 1, 62, 63], device="cuda")
    step = dict(tokens=tokens, positions=positions, slots=slots, seqs=seqs, decode_meta=meta, rows=rows)
    return pool, runner, step, (ta, tb, tc, td)


def eager(model, pool, step):
    return model.forward_batch(torch.tensor(step["tokens"], device="cuda"),
                               torch.tensor(step["positions"], device="cuda"), step["seqs"], logits=step["rows"],
                               paged=pool, slot_mapping=torch.tensor(step["slots"], device="cuda"),
                               decode_meta=step["decode_meta"])


def test_buckets_cover_the_budget():
    b = piecewise_buckets(2048)
    assert b == sorted(set(b)) and b[0] >= 1 and b[-1] == 512   # bigger steps run eagerly (compute-bound)
    assert piecewise_buckets(100)[-1] == 100            # a budget below the cap is itself a bucket


def test_mixed_step_bitwise_equals_eager_at_bucket_size(ours_bf16_gpu, encode):
    pool, runner, step, _ = setup(ours_bf16_gpu, encode, buckets=[64])
    ref = eager(ours_bf16_gpu, pool, step).clone()
    out = runner.run(**step).clone()          # same K/V to the same slots, same kernels, same shapes
    print(f"piecewise vs eager (T=64): max|diff|={(out.float() - ref.float()).abs().max().item():.3g}")
    assert out.shape == ref.shape == (4, ours_bf16_gpu.cfg.vocab_size)
    assert torch.equal(out, ref)
    assert runner.captured_buckets == [64]


def test_padded_mixed_step_matches_eager_up_to_ties(ours_bf16_gpu, encode):
    pool, runner, step, _ = setup(ours_bf16_gpu, encode, buckets=[48, 80])
    ref = eager(ours_bf16_gpu, pool, step).float()
    out = runner.run(**step).float()          # 64 real rows inside the 80-row graphs
    assert out.shape == ref.shape and runner.captured_buckets == [80]
    for i in range(4):
        top2 = ref[i].topk(2).values
        if (top2[0] - top2[1]).item() > TIE_GAP:
            assert out[i].argmax() == ref[i].argmax()


def test_padding_only_writes_the_garbage_block(ours_bf16_gpu, encode):
    pool, runner, step, tables = setup(ours_bf16_gpu, encode, buckets=[80])
    bs = pool.block_size
    k_before, v_before = pool.k.clone(), pool.v.clone()
    runner.run(**step)
    changed = ((pool.k != k_before) | (pool.v != v_before)).any(dim=(0, 3, 4))      # [blocks, block_size]
    flat = {b * bs + o for b, o in changed.nonzero().tolist() if b != runner.garbage_block}
    assert flat <= set(step["slots"]), "a padding row wrote outside the garbage block"
    assert len(flat) > 0


def make(model, piecewise, budget=64):
    return LLMEngine(model, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=6, batch_wait_s=0.0,
                     max_num_batched_tokens=budget, kv_cache="paged", kv_cache_tokens=16384,
                     attn_backend="triton", cuda_graphs=True, piecewise_graphs=piecewise)


def test_engine_with_piecewise_matches_without_up_to_ties(ours_bf16_gpu, encode):
    eng = make(ours_bf16_gpu, True)
    with_p, _ = run_staggered(eng, encode, REQUESTS)
    without, _ = run_staggered(make(ours_bf16_gpu, False), encode, REQUESTS)
    st = eng.stats
    print(f"steps {st['steps']}: full-graph {st['graph_steps']}, piecewise {st['piecewise_steps']}, "
          f"buckets {st['piecewise_buckets']}; identical {sum(a == b for a, b in zip(with_p, without))}/12")
    assert st["piecewise_steps"] > 0 and st["graph_steps"] > 0
    assert st["piecewise_steps"] + st["graph_steps"] == st["steps"]    # budget 64 <= largest bucket: no eager
    for i, (a, b) in enumerate(zip(with_p, without)):
        if a == b:
            continue
        j = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
        seq = torch.tensor(encode(pick_prompts([REQUESTS[i][0]])[0]) + b[:j], device="cuda")
        top2 = ours_bf16_gpu.forward(seq, torch.arange(len(seq), device="cuda"))[0].float().topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"  request {i}: diverges at token {j}, gap {gap:.3f}")
        assert gap <= TIE_GAP
    assert eng.kv.num_free_blocks == eng.kv.pool.num_blocks - 1     # only the shared garbage block is held


def test_piecewise_follows_cuda_graphs_flag(ours_bf16_gpu):
    on = LLMEngine(ours_bf16_gpu, [151645], max_seq_len=256, kv_cache_tokens=1024, max_num_batched_tokens=256)
    assert on.piecewise_runner is not None and on.piecewise_runner.buckets[-1] == 256
    off = LLMEngine(ours_bf16_gpu, [151645], max_seq_len=256, kv_cache_tokens=1024, cuda_graphs=False)
    assert off.piecewise_runner is None
