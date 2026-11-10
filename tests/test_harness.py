"""Closed-loop load generator and sanity checks, using a fake backend (no GPU)."""
import asyncio

from bench.backends.base import Request, RequestRecord
from bench.harness import check_records, run_closed_loop


class FakeBackend:
    def __init__(self, delay=0.01):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.seen = []

    async def generate(self, req):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        self.seen.append(req.id)
        t1 = loop.time()
        return RequestRecord(id=req.id, t_submit=t0, t_chunks=[t1], num_output_tokens=req.max_tokens,
                             prompt_tokens_reported=req.prompt_tokens)


def reqs(n, max_tokens=4):
    return [Request(id=i, prompt=f"p{i}", prompt_tokens=3, max_tokens=max_tokens) for i in range(n)]


async def test_closed_loop_keeps_exactly_n_in_flight():
    backend = FakeBackend()
    records, wall = await run_closed_loop(backend, reqs(20), concurrency=4)
    assert backend.max_in_flight == 4
    assert sorted(r.id for r in records) == list(range(20))
    assert wall > 0


async def test_closed_loop_with_more_workers_than_requests():
    backend = FakeBackend()
    records, _ = await run_closed_loop(backend, reqs(3), concurrency=8)
    assert len(records) == 3
    assert backend.max_in_flight == 3


def test_check_records_passes_good_records():
    rs = [RequestRecord(id=0, t_submit=0, t_chunks=[1], num_output_tokens=4, prompt_tokens_reported=3)]
    assert check_records(rs, reqs(1)) == []


def test_check_records_flags_problems():
    rs = [
        RequestRecord(id=0, t_submit=0, t_chunks=[1], num_output_tokens=2, prompt_tokens_reported=3),  # short output
        RequestRecord(id=1, t_submit=0, t_chunks=[1], num_output_tokens=4, prompt_tokens_reported=5),  # prompt mismatch
        RequestRecord(id=2, t_submit=0, t_chunks=[], num_output_tokens=0, error="boom"),               # error
        RequestRecord(id=3, t_submit=0, t_chunks=[1], num_output_tokens=4, prompt_tokens_reported=None),  # ok: not reported
    ]
    problems = check_records(rs, reqs(4))
    assert len(problems) == 3
    joined = " ".join(problems)
    assert "id=0" in joined and "id=1" in joined and "id=2" in joined
