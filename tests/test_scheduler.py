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
    a.output_ids.append(7)
    b.output_ids.append(7)
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
    out.scheduled[0].output_ids.append(1)
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
    a.output_ids.append(1)
    b.output_ids.append(1)
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
    x = seq(0, prompt_len=3)
    assert x.is_prefill and x.step_tokens() == ([0, 1, 2], [0, 1, 2])
    x.output_ids.append(99)
    assert not x.is_prefill and x.step_tokens() == ([99], [3])
    x.output_ids.append(98)
    assert x.step_tokens() == ([98], [4])


def test_has_unfinished():
    s = Scheduler("continuous", max_num_seqs=1, max_num_batched_tokens=100)
    assert not s.has_unfinished()
    x = seq(0)
    s.add(x)
    assert s.has_unfinished()
    s.schedule(now=0.0)
    s.finish(x)
    assert not s.has_unfinished()
