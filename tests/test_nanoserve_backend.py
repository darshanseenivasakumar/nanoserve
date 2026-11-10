"""M1 step 4: in-process harness backend for our engine."""
import asyncio

import pytest

from bench.harness import parse_args
from bench.prompts import build_requests
from nanoserve.engine import Engine, load_eos_token_ids
from tests.conftest import MODEL

pytestmark = pytest.mark.gpu

PROMPTS = [{"id": "a", "length": "short", "prompt": "Name three primary colors."},
           {"id": "b", "length": "short", "prompt": "Write a haiku about GPUs."}]


@pytest.fixture(scope="module")
def backend(ours_bf16_gpu):
    from bench.backends.nanoserve_inproc import NanoserveBackend
    return NanoserveBackend(MODEL, engine=Engine(ours_bf16_gpu, load_eos_token_ids(MODEL)))


@pytest.fixture(scope="module")
def requests(tokenizer):
    return build_requests(tokenizer, PROMPTS, num_prompts=2, max_tokens=16, seed=0)


async def test_forced_length_and_timestamps(backend, requests):
    r = await backend.generate(requests[0])
    assert r.error is None
    assert r.num_output_tokens == 16 == len(r.t_chunks) == len(r.output_token_ids)
    assert r.t_chunks == sorted(r.t_chunks) and r.t_chunks[0] > r.t_submit
    assert r.prompt_tokens_reported == requests[0].prompt_tokens


async def test_concurrent_requests_run_one_at_a_time(backend, requests):
    a, b = await asyncio.gather(*(backend.generate(r) for r in requests))
    first, second = sorted([a, b], key=lambda r: r.t_chunks[0])
    assert first.t_chunks[-1] <= second.t_chunks[0], "M1 has batch size 1: generations must not overlap"


def test_harness_accepts_the_backend():
    assert parse_args(["--backend", "nanoserve"]).backend == "nanoserve"
