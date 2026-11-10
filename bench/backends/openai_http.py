"""Streaming client for any OpenAI-compatible /v1/completions server (vLLM, llama.cpp, later nanoserve).

We use the *completions* endpoint (raw text in) rather than chat, because we already applied the chat
template ourselves -- every engine gets byte-identical input.

Each Server-Sent-Events chunk is timestamped on arrival. Token counts come from the server's `usage`
block when present, because a server may pack several tokens into one chunk (vLLM's stream_interval,
llama.cpp under load); the per-chunk timestamps still give honest TTFT and E2E.
"""
from __future__ import annotations

import json
import time

import httpx

from bench.backends.base import Request, RequestRecord

# Per-engine request fields needed for a fair comparison. Documented in CLAUDE.md (methodology).
ENGINE_EXTRA_BODY: dict[str, dict] = {
    # vLLM applies the model's generation_config.json defaults; Qwen's has repetition_penalty=1.05.
    # Pure greedy means no penalty. (Also launched with --generation-config vllm, belt and braces.)
    "vllm": {"repetition_penalty": 1.0},
    # llama-server reuses a slot's KV cache for a shared prompt prefix by default. Our engine has no
    # prefix cache, and repeats would hit it, so turn it off. (vLLM's equivalent is a server flag,
    # --no-enable-prefix-caching, set in bench/servers/vllm.sh.)
    "llamacpp": {"cache_prompt": False, "repeat_penalty": 1.0},
}


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, engine_name: str = "openai",
                 timeout_s: float = 600.0, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.engine_name = engine_name
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
        self.client = httpx.AsyncClient(timeout=timeout_s, limits=limits, transport=transport)

    def info(self) -> dict:
        return {"backend": "openai", "engine": self.engine_name, "base_url": self.base_url, "model": self.model,
                "extra_body": ENGINE_EXTRA_BODY.get(self.engine_name, {})}

    async def server_version(self) -> str | None:
        """Best-effort engine version string for the results file."""
        for path in ("/version", "/props"):
            try:
                r = await self.client.get(self.base_url + path, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    return str(data.get("version") or data.get("build_info") or data)[:200]
            except Exception:
                pass
        return None

    async def close(self) -> None:
        await self.client.aclose()

    def _payload(self, req: Request) -> dict:
        body = {
            "model": self.model,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        body.update(ENGINE_EXTRA_BODY.get(self.engine_name, {}))
        return body

    async def generate(self, req: Request) -> RequestRecord:
        rec = RequestRecord(id=req.id, t_submit=time.perf_counter())
        text_parts: list[str] = []
        usage = None
        try:
            async with self.client.stream("POST", f"{self.base_url}/v1/completions", json=self._payload(req)) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    rec.error = f"HTTP {resp.status_code}: {body}"
                    return rec
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators, SSE comments/pings
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    now = time.perf_counter()
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        text = choice.get("text") or ""
                        if text:
                            rec.t_chunks.append(now)
                            text_parts.append(text)
        except Exception as e:  # network errors must not kill the whole benchmark run
            rec.error = f"{type(e).__name__}: {e}"
            return rec

        rec.output_text = "".join(text_parts)
        if usage:
            rec.num_output_tokens = int(usage.get("completion_tokens", len(rec.t_chunks)))
            rec.prompt_tokens_reported = usage.get("prompt_tokens")
        else:
            rec.num_output_tokens = len(rec.t_chunks)  # assumption: 1 token per chunk (flagged in results)
        return rec
