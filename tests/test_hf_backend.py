"""HF baseline backend on the real model (GPU). Checks token counts and that outputs equal plain greedy generate."""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from transformers import AutoTokenizer  # noqa: E402

from bench.backends.hf import HFBackend  # noqa: E402
from bench.prompts import build_requests  # noqa: E402

MODEL = os.environ.get("NANOSERVE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS = 16
PROMPTS = [
    {"id": "a", "length": "short", "prompt": "Name three primary colors."},
    {"id": "b", "length": "short", "prompt": "Write a haiku about GPUs and the smell of warm silicon at night."},
]


@pytest.fixture(scope="module")
def backend():
    b = HFBackend(MODEL, device="cuda", dtype="bfloat16", batch_size=1)
    yield b


@pytest.fixture(scope="module")
def requests(backend):
    return build_requests(backend.tokenizer, PROMPTS, num_prompts=2, max_tokens=MAX_TOKENS, seed=0)


async def test_generates_exactly_max_tokens(backend, requests):
    for req in requests:
        r = await backend.generate(req)
        assert r.error is None
        assert r.num_output_tokens == MAX_TOKENS
        assert len(r.t_chunks) == MAX_TOKENS
        assert r.t_chunks == sorted(r.t_chunks) and r.t_chunks[0] > r.t_submit
        assert r.prompt_tokens_reported == req.prompt_tokens


async def test_matches_plain_hf_greedy_generate(backend, requests):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = backend.model  # same weights; reference path is plain generate with no hooks
    for req in requests:
        r = await backend.generate(req)
        ids = tok(req.prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            # Pure greedy reference: override Qwen's generation_config defaults (incl. repetition_penalty=1.1)
            out = model.generate(**ids, max_new_tokens=MAX_TOKENS, min_new_tokens=MAX_TOKENS,
                                 do_sample=False, temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0)
        expected = out[0, ids["input_ids"].shape[1]:].tolist()
        assert r.output_token_ids == expected


def test_uses_requested_device_and_dtype(backend):
    p = next(backend.model.parameters())
    assert p.device.type == "cuda" and p.dtype == torch.bfloat16
