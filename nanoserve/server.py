"""M6: minimal OpenAI-compatible HTTP server with streaming.

    uv run python -m nanoserve.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8002 --kv-cache-gb 2

Endpoints: GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions.
Streaming uses Server-Sent Events ("data: {json}\\n\\n" ... "data: [DONE]"), like OpenAI/vLLM, so the same
benchmark client measures all engines. Supported: greedy only (temperature must be 0 or omitted),
max_tokens, stream, stream_options.include_usage, ignore_eos (vLLM extension). Anything else that would
change the output (sampling, stop strings, n>1, logprobs) is rejected with HTTP 400 instead of silently
ignored. A client that disconnects mid-stream has its request aborted and its KV blocks freed.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanoserve.async_engine import AsyncEngine

# Parameters that would change the output if set to anything but their neutral default -> reject them.
NEUTRAL = {"n": (None, 1), "best_of": (None, 1), "logprobs": (None, False, 0), "top_logprobs": (None, 0),
           "stop": (None, [], ""), "presence_penalty": (None, 0, 0.0), "frequency_penalty": (None, 0, 0.0),
           "repetition_penalty": (None, 1, 1.0), "logit_bias": (None, {}), "suffix": (None, ""),
           "echo": (None, False), "tools": (None, []), "response_format": (None, {"type": "text"})}


class IncrementalDetokenizer:
    """Turn a growing list of token ids into text deltas. Decoding token by token can split a multi-byte
    character (emoji, Hindi, Chinese) across tokens; hold text back while it ends in the replacement
    character U+FFFD and emit it once the character is complete."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.sent = 0

    def add(self, token: int) -> str:
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def text(self) -> str:
        return self.tok.decode(self.ids, skip_special_tokens=True)


def _validate(body: dict) -> tuple[int, bool]:
    temp = body.get("temperature")
    if temp not in (None, 0, 0.0):
        raise HTTPException(400, "nanoserve supports greedy decoding only: temperature must be 0")
    for key, neutral in NEUTRAL.items():
        if body.get(key) not in neutral:
            raise HTTPException(400, f"parameter {key!r}={body.get(key)!r} is not supported by nanoserve")
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise HTTPException(400, "max_tokens must be a positive integer")
    return max_tokens, bool(body.get("ignore_eos", False))


def create_app(engine: AsyncEngine, tokenizer, model_name: str) -> FastAPI:
    app = FastAPI(title="nanoserve")
    eos = set(engine.engine.eos_token_ids)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nanoserve"}]}

    async def _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(delta_text=None, finish=None, first=False):
            if chat:
                delta = {"role": "assistant", "content": ""} if first else ({"content": delta_text} if delta_text else {})
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": delta_text or "", "finish_reason": finish}
            return {"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice]}

        def finish_reason(tokens):
            return "stop" if tokens and tokens[-1] in eos and not ignore_eos else "length"

        def usage(n_out):
            return {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out,
                    "total_tokens": len(prompt_ids) + n_out}

        if not stream:
            detok, tokens = IncrementalDetokenizer(tokenizer), []
            async for tok, _ in engine.generate(prompt_ids, max_tokens, ignore_eos=ignore_eos):
                tokens.append(tok)
                detok.add(tok)
            text = detok.text()
            choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                       "finish_reason": finish_reason(tokens)} if chat else
                      {"index": 0, "text": text, "finish_reason": finish_reason(tokens)})
            return JSONResponse({"id": rid, "object": "chat.completion" if chat else "text_completion",
                                 "created": created, "model": model_name, "choices": [choice],
                                 "usage": usage(len(tokens))})

        async def events():
            detok, tokens = IncrementalDetokenizer(tokenizer), []
            if chat:
                yield f"data: {json.dumps(chunk(first=True))}\n\n"
            gen = engine.generate(prompt_ids, max_tokens, ignore_eos=ignore_eos)
            try:
                async for tok, _ in gen:
                    tokens.append(tok)
                    delta = detok.add(tok)
                    if delta:
                        yield f"data: {json.dumps(chunk(delta))}\n\n"
            finally:
                await gen.aclose()     # client gone -> AsyncEngine aborts the request, frees its KV
            yield f"data: {json.dumps(chunk(finish=finish_reason(tokens)))}\n\n"
            if include_usage:
                yield f"data: {json.dumps({'id': rid, 'object': obj, 'created': created, 'model': model_name, 'choices': [], 'usage': usage(len(tokens))})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        max_tokens, ignore_eos = _validate(body)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_ids = prompt
        elif isinstance(prompt, str):
            # The prompt is used exactly as given (no chat template, no BOS), like vLLM's completions API.
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            raise HTTPException(400, "prompt must be a string or a list of token ids")
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        try:
            return await _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat=False)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        max_tokens, ignore_eos = _validate(body)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        try:
            return await _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat=True)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def main(argv=None) -> None:
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from nanoserve.engine import LLMEngine, load_eos_token_ids
    from nanoserve.model import Qwen2Model

    p = argparse.ArgumentParser(description="nanoserve OpenAI-compatible server")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--kv-cache-gb", type=float, default=2.0, help="explicit KV budget (GiB)")
    p.add_argument("--attn-backend", choices=["triton", "sdpa"], default=None)
    p.add_argument("--no-cuda-graphs", action="store_true", help="disable CUDA-graph decode steps")
    p.add_argument("--draft-model", default=None, help="enable speculative decoding with this draft model")
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = Qwen2Model.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    kv_tokens = int(args.kv_cache_gb * 2**30) // target.cfg.kv_bytes_per_token(torch.bfloat16)
    eos = load_eos_token_ids(args.model)
    if args.draft_model:
        from nanoserve.speculative import SpeculativeEngine
        draft = Qwen2Model.from_pretrained(args.draft_model, device="cuda", dtype=torch.bfloat16)
        llm = SpeculativeEngine(target, draft, eos, num_speculative_tokens=args.num_speculative_tokens,
                                max_seq_len=args.max_seq_len, max_num_seqs=args.max_num_seqs,
                                max_num_batched_tokens=args.max_num_batched_tokens, target_kv_tokens=kv_tokens)
    else:
        llm = LLMEngine(target, eos, max_seq_len=args.max_seq_len, max_num_seqs=args.max_num_seqs,
                        max_num_batched_tokens=args.max_num_batched_tokens, kv_cache="paged",
                        kv_cache_tokens=kv_tokens, attn_backend=args.attn_backend,
                        cuda_graphs=False if args.no_cuda_graphs else None)
    app = create_app(AsyncEngine(llm), tokenizer, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
