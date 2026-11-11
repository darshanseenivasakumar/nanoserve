"""M6: OpenAI-compatible server. Tested in-process through httpx's ASGI transport, using the SAME streaming
client the benchmark uses for vLLM/llama.cpp (bench/backends/openai_http.py) -- so the server is proven
compatible with the harness, not just with a hand-written test client."""
import asyncio
import json

import httpx
import pytest

from bench.backends.base import Request
from bench.backends.openai_http import OpenAIBackend
from nanoserve.async_engine import AsyncEngine
from nanoserve.engine import Engine, LLMEngine, load_eos_token_ids
from nanoserve.server import create_app
from tests.conftest import MODEL, pick_prompts

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def served(ours_bf16_gpu, tokenizer):
    llm = LLMEngine(ours_bf16_gpu, load_eos_token_ids(MODEL), max_seq_len=1024, max_num_seqs=8,
                    kv_cache_tokens=16384, attn_backend="sdpa")
    aeng = AsyncEngine(llm)
    app = create_app(aeng, tokenizer, MODEL)
    yield app, aeng, llm
    aeng.close()


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=120)


async def test_health_and_models(served):
    app, _, _ = served
    async with client(app) as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        assert (await c.get("/v1/models")).json()["data"][0]["id"] == MODEL


async def test_completion_text_equals_engine_greedy_output(served, tokenizer, ours_bf16_gpu):
    app, _, _ = served
    text_prompt = tokenizer.apply_chat_template([{"role": "user", "content": pick_prompts(["short-00"])[0]}],
                                                tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    expected = list(Engine(ours_bf16_gpu, load_eos_token_ids(MODEL), max_seq_len=1024).generate(ids, 20))
    async with client(app) as c:
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": text_prompt, "max_tokens": 20,
                                                  "temperature": 0})
    body = r.json()
    assert r.status_code == 200
    assert body["choices"][0]["text"] == tokenizer.decode(expected, skip_special_tokens=True)
    assert body["usage"] == {"prompt_tokens": len(ids), "completion_tokens": len(expected),
                             "total_tokens": len(ids) + len(expected)}


async def test_streaming_through_the_benchmark_client(served, tokenizer):
    app, _, _ = served
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Name three rivers."}],
                                           tokenize=False, add_generation_prompt=True)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    backend = OpenAIBackend("http://test", model=MODEL, engine_name="nanoserve",
                            transport=httpx.ASGITransport(app=app))
    rec = await backend.generate(Request(id=0, prompt=prompt, prompt_tokens=n_prompt, max_tokens=24))
    await backend.close()
    assert rec.error is None
    assert rec.num_output_tokens == 24            # ignore_eos honoured
    assert rec.prompt_tokens_reported == n_prompt
    assert rec.output_text and len(rec.t_chunks) >= 1


async def test_chat_streaming_format(served):
    app, _, _ = served
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8,
            "stream": True, "stream_options": {"include_usage": True}})
    lines = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    events = [json.loads(l) for l in lines[:-1]]
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] <= 8
    assert events[-2]["choices"][0]["finish_reason"] in ("stop", "length")
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:-2])
    assert content.strip()


async def test_sampling_is_honoured_and_unsupported_params_rejected(served):
    # M13 (changed from M6's "temperature 0.7 -> 400", which held while nanoserve was greedy-only).
    app, _, _ = served
    body = {"model": MODEL, "prompt": "Write a short poem about rain.", "max_tokens": 24, "temperature": 1.0,
            "top_p": 0.95, "seed": 7}
    async with client(app) as c:
        text = lambda r: r.json()["choices"][0]["text"]                       # noqa: E731
        a = text(await c.post("/v1/completions", json=body))
        b = text(await c.post("/v1/completions", json=body))
        greedy = text(await c.post("/v1/completions", json={**body, "temperature": 0}))
        other = text(await c.post("/v1/completions", json={**body, "seed": 8}))
        assert a == b                              # same seed -> same text
        assert a != greedy or a != other           # ... and it really sampled
        for bad in ({"temperature": -1}, {"top_p": 0}, {"top_k": -5}, {"presence_penalty": 0.5}, {"n": 2}):
            r = await c.post("/v1/completions", json={**body, **bad})
            assert r.status_code == 400, bad
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "hi", "presence_penalty": 0})
        assert r.status_code == 200                # neutral defaults are fine


async def test_client_disconnect_aborts_request_and_frees_kv(served, encode):
    _, aeng, llm = served
    before = llm.kv.num_free_blocks
    gen = aeng.generate(encode("Write a long story about a lighthouse."), 500, ignore_eos=True)
    async for _ in gen:
        break                                      # read one token, then "disconnect"
    await gen.aclose()
    for _ in range(200):
        if llm.kv.num_free_blocks == before and not llm.has_unfinished():
            break
        await asyncio.sleep(0.01)
    assert llm.kv.num_free_blocks == before and aeng.num_aborted >= 1
