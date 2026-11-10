"""M2 step 2: scheduler policies, tested without a model (CPU only)."""
import pytest

from nanoserve.scheduler import Scheduler, Sequence, SeqStatus


def seq(i, prompt_len=4, max_new=8, t=0.0):
    return Sequence(seq_id=i, prompt_ids=list(range(prompt_len)), max_new_tokens=max_new, arrival_time=t)


def ids(seqs):
    return [s.seq_id for s in seqs]


def test_continuous_admits_up_to_max_num_seqs():
    s = Scheduler("continuous", max_num_seqs=2, max_num_batched_tokens=100)
    for i in range(3):
        s.add(seq(i))
    out = s.schedule(now=0.0)
    assert ids(out.scheduled) == [0, 1] and ids(out.admitted) == [0, 1]
    assert [x.status for x in out.scheduled] == [SeqStatus.RUNNING] * 2
    assert ids(s.waiting) == [2]


def test_continuous_fills_a_freed_seat_on_the_very_next_step():
    s = Scheduler("continuous", max_num_seqs=2, max_num_batched_tokens=100)
    a, b, c = seq(0), seq(1), seq(2)
    for x in (a, b, c):
        s.add(x)
    s.schedule(now=0.0)
    a.output_ids.append(7); a.num_cached = 4   # M3: decoding means KV is stored
    b.output_ids.append(7); b.num_cached = 4
    s.finish(a)
    out = s.schedule(now=0.0)
    assert ids(out.scheduled) == [1, 2] and ids(out.admitted) == [2]   # b decodes while c prefills: mixed step
    assert a.status == SeqStatus.FINISHED


def test_token_budget_limits_prefill_admission():
    s = Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=10)
    s.add(seq(0, prompt_len=6))
    s.add(seq(1, prompt_len=6))
    out = s.schedule(now=0.0)
    assert ids(out.admitted) == [0]                       # 6 + 6 > 10
    out.scheduled[0].output_ids.append(1); out.scheduled[0].num_cached = 6
    out = s.schedule(now=0.0)                             # running decode costs 1 token: 1 + 6 <= 10
    assert ids(out.admitted) == [1] and ids(out.scheduled) == [0, 1]


def test_first_come_first_served_never_skips_the_head_of_the_queue():
    s = Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=10)
    s.add(seq(0, prompt_len=6))
    s.add(seq(1, prompt_len=9))
    s.add(seq(2, prompt_len=2))   # would fit, but #1 is ahead of it
    out = s.schedule(now=0.0)
    assert ids(out.admitted) == [0]


def test_prompt_longer_than_the_token_budget_is_rejected_on_add():
    s = Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=10)
    with pytest.raises(ValueError):
        s.add(seq(0, prompt_len=11))


def test_static_never_admits_while_a_batch_is_running():
    s = Scheduler("static", max_num_seqs=2, max_num_batched_tokens=100, batch_wait_s=0.0)
    a, b, c = seq(0), seq(1), seq(2)
    for x in (a, b, c):
        s.add(x)
    assert ids(s.schedule(now=0.0).admitted) == [0, 1]
    a.output_ids.append(1); a.num_cached = 4
    b.output_ids.append(1); b.num_cached = 4
    s.finish(a)                                            # a seat is free...
    out = s.schedule(now=1.0)
    assert ids(out.scheduled) == [1] and out.admitted == []   # ...but static waits for the whole batch
    s.finish(b)
    assert ids(s.schedule(now=2.0).admitted) == [2]


def test_static_waits_for_a_full_batch_or_the_timeout():
    s = Scheduler("static", max_num_seqs=3, max_num_batched_tokens=100, batch_wait_s=0.02)
    s.add(seq(0, t=1.0))
    s.add(seq(1, t=1.005))
    assert s.schedule(now=1.01).scheduled == []            # 2 of 3 seats, waited 10 ms: keep waiting
    assert ids(s.schedule(now=1.03).admitted) == [0, 1]    # waited 30 ms > 20 ms: go with what we have


def test_step_inputs_prefill_then_decode():
    # M3: "prefill" now means "no KV cached" (num_cached == 0), not "no outputs yet", because a
    # preempted request has outputs but must recompute its KV. (Test updated for M3, flagged in LOG.md.)
    x = seq(0, prompt_len=3)
    assert x.is_prefill and x.step_tokens() == ([0, 1, 2], [0, 1, 2])
    x.num_cached = 3
    x.output_ids.append(99)
    assert not x.is_prefill and x.step_tokens() == ([99], [3])
    x.num_cached = 4
    x.output_ids.append(98)
    assert x.step_tokens() == ([98], [4])


def test_preempted_request_recomputes_prompt_plus_outputs():
    x = seq(0, prompt_len=3)
    x.output_ids += [7, 8]
    x.num_cached = 0                           # its KV was thrown away
    assert x.is_prefill and x.step_tokens() == ([0, 1, 2, 7, 8], [0, 1, 2, 3, 4])
    assert x.step_cost == 5


# ---- M3: block-aware admission and preemption (tiny CPU pool, no model) ------------------------------
import torch  # noqa: E402

from nanoserve.kv_manager import PagedKVManager  # noqa: E402
from nanoserve.model import ModelConfig  # noqa: E402

TINY = ModelConfig(vocab_size=10, hidden_size=8, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=2,
                   intermediate_size=8, rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)


def paged_scheduler(num_tokens, block_size=4, **kw):
    kv = PagedKVManager(TINY, budget_tokens=num_tokens, block_size=block_size, device="cpu", dtype=torch.float32)
    return Scheduler("continuous", max_num_seqs=8, max_num_batched_tokens=100, kv=kv, **kw), kv


def run_step(s, now=0.0):
    """Pretend the engine ran the scheduled step: commit KV and append one output token each."""
    out = s.schedule(now=now)
    for x in out.scheduled:
        n = len(x.step_tokens()[0])
        x.block_table.commit(n)
        x.num_cached += n
        x.output_ids.append(1)
    return out


def test_admission_waits_for_free_blocks():
    s, kv = paged_scheduler(16)                 # 4 blocks of 4
    s.add(seq(0, prompt_len=10, max_new=2))    # needs 3 blocks
    s.add(seq(1, prompt_len=6, max_new=2))     # needs 2 blocks: only 1 left
    out = run_step(s)
    assert ids(out.admitted) == [0] and kv.num_free_blocks == 1


def test_decode_takes_a_new_block_only_at_a_block_boundary():
    s, kv = paged_scheduler(32)
    a = seq(0, prompt_len=7)
    s.add(a)
    run_step(s)                                 # prefill 7 tokens: 2 blocks
    assert len(a.block_table.blocks) == 2
    run_step(s)                                 # token 8 of 8: still 2 blocks
    assert len(a.block_table.blocks) == 2
    run_step(s)                                 # token 9: third block
    assert len(a.block_table.blocks) == 3


def test_preempts_the_most_recent_request_when_blocks_run_out():
    s, kv = paged_scheduler(16)                 # 4 blocks
    a, b = seq(0, prompt_len=8), seq(1, prompt_len=8)
    s.add(a)
    s.add(b)
    run_step(s)                                 # both admitted: 4 blocks used, each holds 8 tokens
    assert kv.num_free_blocks == 0
    out = run_step(s)                           # both need a 3rd block -> preempt b (newest), a continues
    assert ids(out.scheduled) == [0] and s.num_preemptions == 1
    assert b.status == SeqStatus.WAITING and s.waiting[0] is b
    assert b.num_cached == 0 and b.block_table is None and b.output_ids == [1]   # outputs kept
    assert len(a.block_table.blocks) == 3


def test_request_that_can_never_fit_is_rejected():
    s, kv = paged_scheduler(16)
    with pytest.raises(ValueError):
        s.add(seq(0, prompt_len=10, max_new=10))   # 20 tokens > 16-token pool


def test_finishing_returns_all_blocks():
    s, kv = paged_scheduler(32)
    a = seq(0, prompt_len=9)
    s.add(a)
    run_step(s)
    s.finish(a)
    assert kv.num_free_blocks == 8 and a.block_table is None


def test_has_unfinished():
    s = Scheduler("continuous", max_num_seqs=1, max_num_batched_tokens=100)
    assert not s.has_unfinished()
    x = seq(0)
    s.add(x)
    assert s.has_unfinished()
    s.schedule(now=0.0)
    s.finish(x)
    assert not s.has_unfinished()
