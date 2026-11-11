"""M6: minimal OpenAI-compatible HTTP server with streaming.

    uv run python -m nanoserve.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8002 --kv-cache-gb 2

Endpoints: GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions.
Streaming uses Server-Sent Events ("data: {json}\\n\\n" ... "data: [DONE]"), like OpenAI/vLLM, so the same
benchmark client measures all engines. Supported: temperature / top_p / top_k / seed (M13; an omitted
temperature means greedy), max_tokens, stream, stream_options.include_usage, ignore_eos (vLLM extension).
Anything else that would change the output (stop strings, penalties, n>1, logprobs) is rejected with HTTP
400 instead of silently ignored. A client that disconnects mid-stream has its request aborted and its KV
blocks freed.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanoserve.async_engine import AsyncEngine
from nanoserve.sampling import SamplingParams

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


def _sampling(body: dict) -> SamplingParams | None:
    """M13: temperature / top_p / top_k (vLLM extension; -1 or 0 = off) / seed. An omitted temperature means
    GREEDY here (deterministic by default; OpenAI's default would be 1.0 -- a deliberate, documented choice)."""
    temp, top_p, top_k, seed = (body.get(k) for k in ("temperature", "top_p", "top_k", "seed"))
    try:
        params = SamplingParams(temperature=float(temp or 0.0), top_p=float(1.0 if top_p is None else top_p),
                                top_k=0 if top_k in (None, -1) else int(top_k),
                                seed=None if seed is None else int(seed))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"invalid sampling parameters: {e}")
    return None if params.greedy else params


def _validate(body: dict) -> tuple[int, bool, SamplingParams | None]:
    for key, neutral in NEUTRAL.items():
        if body.get(key) not in neutral:
            raise HTTPException(400, f"parameter {key!r}={body.get(key)!r} is not supported by nanoserve")
    sampling = _sampling(body)
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise HTTPException(400, "max_tokens must be a positive integer")
    return max_tokens, bool(body.get("ignore_eos", False)), sampling


def create_app(engine: AsyncEngine, tokenizer, model_name: str) -> FastAPI:
    app = FastAPI(title="nanoserve")
    eos = set(engine.eos_token_ids)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nanoserve"}]}

    async def _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat, sampling=None):
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
            async for tok, _ in engine.generate(prompt_ids, max_tokens, ignore_eos=ignore_eos, sampling=sampling):
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
            gen = engine.generate(prompt_ids, max_tokens, ignore_eos=ignore_eos, sampling=sampling)
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
        max_tokens, ignore_eos, sampling = _validate(body)
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
            return await _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat=False, sampling=sampling)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        max_tokens, ignore_eos, sampling = _validate(body)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        try:
            return await _run(prompt_ids, max_tokens, ignore_eos, stream, include_usage, chat=True, sampling=sampling)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app


def main(argv=None) -> None:
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from nanoserve.engine_process import EngineProcess, EngineSpec, build_engine
    from nanoserve.model import ModelConfig

    p = argparse.ArgumentParser(description="nanoserve OpenAI-compatible server")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048, help="per-step token budget (vLLM default on <70 GB GPUs)")
    p.add_argument("--kv-cache-gb", type=float, default=2.0, help="explicit KV budget (GiB)")
    p.add_argument("--attn-backend", choices=["triton", "sdpa"], default=None)
    p.add_argument("--no-cuda-graphs", action="store_true", help="disable all CUDA graphs")
    p.add_argument("--no-piecewise-graphs", action="store_true",
                   help="run mixed prefill+decode steps eagerly (M9 default: piecewise CUDA graphs)")
    p.add_argument("--no-fused-ops", action="store_true",
                   help="unfused PyTorch ops instead of M10's fused Triton kernels")
    p.add_argument("--draft-model", default=None, help="enable speculative decoding with this draft model")
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--enable-prefix-caching", action="store_true",
                   help="reuse cached KV blocks of shared prompt prefixes across requests (M12)")
    p.add_argument("--engine-in-thread", action="store_true",
                   help="run the engine as a thread of this process (M6-M10) instead of its own process (M11)")
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kv_tokens = int(args.kv_cache_gb * 2**30) // ModelConfig.from_pretrained(args.model).kv_bytes_per_token(torch.bfloat16)
    spec = EngineSpec(model=args.model, max_seq_len=args.max_seq_len, max_num_seqs=args.max_num_seqs,
                      max_num_batched_tokens=args.max_num_batched_tokens, kv_cache_tokens=kv_tokens,
                      attn_backend=args.attn_backend, cuda_graphs=False if args.no_cuda_graphs else None,
                      piecewise_graphs=False if (args.no_cuda_graphs or args.no_piecewise_graphs) else None,
                      fused_ops=False if args.no_fused_ops else None,
                      prefix_caching=args.enable_prefix_caching, draft_model=args.draft_model,
                      num_speculative_tokens=args.num_speculative_tokens)
    # Graphs are captured before accepting traffic (vLLM does the same), so no request pays for one.
    if args.engine_in_thread:
        llm = build_engine(spec)
        if hasattr(llm, "capture_graphs"):
            print(f"captured CUDA graphs in {llm.capture_graphs():.1f} s", flush=True)
        engine = AsyncEngine(llm)
    else:
        engine = EngineProcess(spec)        # M11: engine loop in its own process (own GIL)
        print(f"engine process ready; captured CUDA graphs in {engine.capture_s:.1f} s", flush=True)
    app = create_app(engine, tokenizer, args.model)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
