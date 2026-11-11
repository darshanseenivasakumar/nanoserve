"""M11: the engine loop in its own PROCESS (EngineProcess), same interface as AsyncEngine.

Gates: same kernels + same inputs => tokens IDENTICAL to the in-thread engine (not just tie-rule close);
requests batch together; timestamps increase and are comparable with this process's clock; callers from
different event loops; errors and rejected requests reach the caller; a client that stops listening gets
its request aborted and its KV freed IN THE CHILD; close() stops the child; the HTTP server works on top.
"""
import asyncio
import time

import httpx
import pytest

from nanoserve.engine import LLMEngine, load_eos_token_ids
from nanoserve.engine_process import EngineProcess, EngineSpec
from nanoserve.server import create_app
from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.gpu

SPEC = dict(model=MODEL, max_seq_len=1024, max_num_seqs=8, kv_cache_tokens=16384, capture_graphs=False)


@pytest.fixture(scope="module")
def proc():
    ep = EngineProcess(EngineSpec(**SPEC))
    yield ep
    ep.close()


async def collect(ep, prompt, n, ignore_eos=True):
    return [item async for item in ep.generate(prompt, n, ignore_eos=ignore_eos)]


def test_tokens_identical_to_the_in_thread_engine(proc, ours_bf16_gpu, encode):
    # One request at a time, so both engines run identical batches (concurrent arrivals could split
    # differently across steps, and bf16 results depend on batch composition).
    ref = LLMEngine(ours_bf16_gpu, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=8,
                    kv_cache_tokens=16384)
    for pid in ("short-00", "medium-04", "short-07"):
        prompt = encode(pick_prompts([pid])[0])
        ref.add_request(prompt, 40)
        want = []
        while ref.has_unfinished():
            want += [out.token for out in ref.step()]
        got = [tok for tok, _ in asyncio.run(collect(proc, prompt, 40, ignore_eos=False))]
        assert got == want, pid


async def test_concurrent_requests_are_batched_together(proc, encode):
    prompts = [encode(p) for p in ("Name a color.", "Name a fruit.", "Name a planet.")]
    results = await asyncio.gather(*(collect(proc, p, 12) for p in prompts))
    assert [len(r) for r in results] == [12, 12, 12]
    starts, ends = [r[0][1] for r in results], [r[-1][1] for r in results]
    assert max(starts) < min(ends), "requests should be in flight at the same time"


async def test_timestamps_increase_on_this_process_clock(proc, encode):
    t0 = time.perf_counter()
    out = await collect(proc, encode("Count to five."), 6)
    ts = [t for _, t in out]
    assert ts == sorted(ts) and t0 < ts[0] <= time.perf_counter()   # child stamps, same monotonic clock


def test_works_across_event_loops(proc, encode):
    a = asyncio.run(collect(proc, encode("Hi."), 3))
    b = asyncio.run(collect(proc, encode("Hello."), 3))
    assert len(a) == len(b) == 3


async def test_rejected_request_raises_in_the_caller(proc):
    with pytest.raises(ValueError):
        await collect(proc, list(range(2000)), 10)          # longer than max_seq_len


async def test_disconnect_aborts_and_frees_kv_in_the_child(proc, encode):
    before = proc.stats()
    gen = proc.generate(encode("Write a long story about a lighthouse."), 500, ignore_eos=True)
    async for _ in gen:
        break                                               # one token, then "disconnect"
    await gen.aclose()
    for _ in range(200):
        st = proc.stats()
        if not st["unfinished"] and st["num_free_blocks"] == before["num_free_blocks"]:
            break
        await asyncio.sleep(0.01)
    assert not st["unfinished"] and st["num_free_blocks"] == before["num_free_blocks"]
    assert st["num_aborted"] == before["num_aborted"] + 1


async def test_server_on_top_of_the_engine_process(proc, tokenizer):
    app = create_app(proc, tokenizer, MODEL)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=120) as c:
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "Say hi.", "max_tokens": 8,
                                                  "temperature": 0})
    assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] <= 8


def test_close_stops_the_child(proc):
    # Last test of the module: closes the shared child (the fixture's close() afterwards is a no-op).
    # Spawning a SECOND child here failed once inside the full suite (two CUDA contexts + models next to
    # every other fixture's GPU memory) while passing alone.
    assert proc.is_alive()
    proc.close()
    assert not proc.is_alive()
    proc.close()                                            # idempotent
