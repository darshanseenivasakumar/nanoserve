"""In-process harness backend for our engine (M2: continuous/static batching via AsyncEngine)."""
import asyncio

import pytest

from bench.harness import parse_args
from bench.prompts import build_requests
from nanoserve.engine import LLMEngine, load_eos_token_ids
from tests.conftest import MODEL

pytestmark = pytest.mark.gpu

PROMPTS = [{"id": "a", "length": "short", "prompt": "Name three primary colors."},
           {"id": "b", "length": "short", "prompt": "Write a haiku about GPUs."}]


@pytest.fixture(scope="module")
def backend(ours_bf16_gpu):
    from bench.backends.nanoserve_inproc import NanoserveBackend
    b = NanoserveBackend(MODEL, engine=LLMEngine(ours_bf16_gpu, load_eos_token_ids(MODEL), max_seq_len=1024))
    yield b
    asyncio.run(b.close())


@pytest.fixture(scope="module")
def requests(tokenizer):
    return build_requests(tokenizer, PROMPTS, num_prompts=2, max_tokens=16, seed=0)


async def test_forced_length_and_timestamps(backend, requests):
    r = await backend.generate(requests[0])
    assert r.error is None
    assert r.num_output_tokens == 16 == len(r.t_chunks) == len(r.output_token_ids)
    assert r.t_chunks == sorted(r.t_chunks) and r.t_chunks[0] > r.t_submit
    assert r.prompt_tokens_reported == requests[0].prompt_tokens


async def test_concurrent_requests_overlap(backend, requests):
    # M1 asserted the opposite (batch size 1). M2 batches concurrent requests together.
    a, b = await asyncio.gather(*(backend.generate(r) for r in requests))
    assert max(a.t_chunks[0], b.t_chunks[0]) < min(a.t_chunks[-1], b.t_chunks[-1])


def test_batch_size_sets_max_num_seqs(backend):
    backend.batch_size = 5
    assert backend.llm.scheduler.max_num_seqs == 5


def test_harness_accepts_the_backend_and_policy():
    args = parse_args(["--backend", "nanoserve", "--nanoserve-policy", "static", "--output-len-range", "16,256"])
    assert args.backend == "nanoserve" and args.nanoserve_policy == "static"
    assert args.output_len_range == (16, 256)
