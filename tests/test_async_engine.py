"""M2 step 4: AsyncEngine runs LLMEngine in a background thread and streams tokens to asyncio callers."""
import asyncio

import pytest

from nanoserve.async_engine import AsyncEngine
from nanoserve.engine import LLMEngine, load_eos_token_ids
from tests.conftest import MODEL

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def aengine(ours_bf16_gpu):
    eng = AsyncEngine(LLMEngine(ours_bf16_gpu, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=8))
    yield eng
    eng.close()


async def collect(aengine, prompt, n):
    return [item async for item in aengine.generate(prompt, n, ignore_eos=True)]


async def test_concurrent_requests_are_batched_together(aengine, encode):
    prompts = [encode(p) for p in ("Name a color.", "Name a fruit.", "Name a planet.")]
    results = await asyncio.gather(*(collect(aengine, p, 12) for p in prompts))
    assert [len(r) for r in results] == [12, 12, 12]
    spans = [(r[0][1], r[-1][1]) for r in results]
    starts, ends = [s for s, _ in spans], [e for _, e in spans]
    assert max(starts) < min(ends), "requests should be in flight at the same time"


async def test_timestamps_increase(aengine, encode):
    out = await collect(aengine, encode("Count to five."), 6)
    ts = [t for _, t in out]
    assert ts == sorted(ts)


def test_works_across_event_loops(aengine, encode):
    # One engine thread serves callers from different asyncio loops (each asyncio.run is a new loop).
    a = asyncio.run(collect(aengine, encode("Hi."), 3))
    b = asyncio.run(collect(aengine, encode("Hello."), 3))
    assert len(a) == len(b) == 3


async def test_errors_reach_the_caller(aengine):
    with pytest.raises(ValueError):
        await collect(aengine, list(range(2000)), 10)   # longer than max_seq_len
