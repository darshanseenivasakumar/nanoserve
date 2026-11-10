"""HuggingFace `generate` baseline, in-process.

`model.generate` has no concurrency of its own, so we give it the simplest possible batching:
STATIC BATCHES. Incoming requests queue up; the batcher takes up to `batch_size` of them (waiting at
most `batch_wait_s` for the batch to fill), runs one left-padded `generate` call, and only then takes the
next batch. A request arriving mid-batch waits for the whole batch to finish. That is deliberately naive
-- it's the gap continuous batching (M2) closes.

Per-token timestamps: a StoppingCriteria hook is called by HF once per generated token. It calls
torch.cuda.synchronize() first, so the timestamp marks when the GPU actually produced the token rather
than when Python queued the work.
"""
from __future__ import annotations

import asyncio
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from bench.backends.base import Request, RequestRecord

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Pure greedy. Qwen's generation_config.json ships sampling defaults (temperature/top_p/top_k) AND
# repetition_penalty=1.05, which still applies when do_sample=False. We override all of them so the
# baseline is "always pick the argmax token" -- the definition every engine is held to.
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)


class _StepTimer(StoppingCriteria):
    def __init__(self, device: torch.device):
        self.device = device
        self.stamps: list[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stamps.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # never stop


class HFBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 1, batch_wait_s: float = 0.02):
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_s
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPES[dtype]).to(self.device).eval()
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def info(self) -> dict:
        return {
            "backend": "hf", "engine": "hf-generate", "model": self.model_name, "device": str(self.device),
            "dtype": self.dtype, "batch_size": self.batch_size, "batching": "static",
            "transformers": transformers.__version__, "torch": torch.__version__,
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
        }

    async def close(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker = None
            self._queue = None

    async def generate(self, req: Request) -> RequestRecord:
        t_submit = time.perf_counter()
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            # The batcher task belongs to one event loop. If we're called from a new loop (each
            # asyncio.run(), each pytest-asyncio test), the old worker is dead: start a fresh one.
            self._loop = loop
            self._queue = asyncio.Queue()
            self._worker = loop.create_task(self._batcher())
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((req, t_submit, fut))
        return await fut

    async def _batcher(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                records = await asyncio.to_thread(self._run_batch, batch)
                for (_, _, fut), rec in zip(batch, records):
                    fut.set_result(rec)
            except Exception as e:  # report on every request in the batch, keep serving
                for req, t_submit, fut in batch:
                    fut.set_result(RequestRecord(id=req.id, t_submit=t_submit, error=f"{type(e).__name__}: {e}"))

    @torch.inference_mode()
    def _run_batch(self, batch) -> list[RequestRecord]:
        reqs = [b[0] for b in batch]
        n_new = reqs[0].max_tokens
        if any(r.max_tokens != n_new for r in reqs):
            # SHORTCUT: the harness always uses one max_tokens per run, so mixed lengths aren't supported.
            raise ValueError("static batch requires equal max_tokens")
        enc = self.tokenizer([r.prompt for r in reqs], return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.device)
        timer = _StepTimer(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=n_new, min_new_tokens=n_new, **GREEDY,
            pad_token_id=self.tokenizer.pad_token_id, stopping_criteria=StoppingCriteriaList([timer]),
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:].tolist()
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        records = []
        for (req, t_submit, _), ids, plen in zip(batch, new_tokens, prompt_lens):
            records.append(RequestRecord(
                id=req.id, t_submit=t_submit, t_chunks=list(timer.stamps), num_output_tokens=len(ids),
                prompt_tokens_reported=int(plen), output_token_ids=ids,
                output_text=self.tokenizer.decode(ids, skip_special_tokens=True),
            ))
        return records
