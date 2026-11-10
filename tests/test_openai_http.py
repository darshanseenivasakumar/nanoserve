"""OpenAI-compatible streaming client, tested against a fake server (httpx.MockTransport)."""
import json

import httpx

from bench.backends.base import Request
from bench.backends.openai_http import OpenAIBackend


def sse(*events):
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def text_chunk(text):
    return {"choices": [{"index": 0, "text": text, "finish_reason": None}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
REQ = Request(id=7, prompt="<templated prompt>", prompt_tokens=11, max_tokens=3)


def make_backend(handler, engine_name="vllm"):
    return OpenAIBackend(
        base_url="http://fake", model="m", engine_name=engine_name,
        transport=httpx.MockTransport(handler),
    )


async def test_streams_tokens_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = sse(text_chunk("Hel"), text_chunk("lo"), text_chunk("!"), USAGE)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = make_backend(handler)
    r = await backend.generate(REQ)
    await backend.close()

    assert r.error is None
    assert r.id == 7
    assert r.output_text == "Hello!"
    assert len(r.t_chunks) == 3
    assert r.t_chunks == sorted(r.t_chunks) and r.t_chunks[0] >= r.t_submit
    assert r.num_output_tokens == 3
    assert r.prompt_tokens_reported == 11

    body = seen["body"]
    assert seen["url"] == "http://fake/v1/completions"
    assert body["prompt"] == REQ.prompt
    assert body["max_tokens"] == 3
    assert body["temperature"] == 0
    assert body["stream"] is True
    assert body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_missing_usage_falls_back_to_chunk_count():
    def handler(request):
        return httpx.Response(200, content=sse(text_chunk("a"), text_chunk("b")))

    backend = make_backend(handler)
    r = await backend.generate(REQ)
    assert r.num_output_tokens == 2
    assert r.prompt_tokens_reported is None


async def test_http_error_is_recorded_not_raised():
    def handler(request):
        return httpx.Response(500, content=b"kaboom")

    backend = make_backend(handler)
    r = await backend.generate(REQ)
    assert r.error is not None and "500" in r.error
    assert r.num_output_tokens == 0


async def test_llamacpp_disables_prompt_cache():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(text_chunk("x"), USAGE))

    backend = make_backend(handler, engine_name="llamacpp")
    await backend.generate(REQ)
    assert seen["body"]["cache_prompt"] is False
