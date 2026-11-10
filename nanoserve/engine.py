"""Inference engine: scheduler + batched forward + greedy sampling.

One engine STEP:
    1. scheduler.schedule()   -> which sequences run now (running decodes + newly admitted prefills)
    2. allocate a KV cache for each newly admitted sequence
    3. model.forward_batch()  -> one forward pass over all their tokens (selective batching)
    4. argmax per sequence    -> ONE GPU->CPU copy for the whole batch
    5. append tokens, finish sequences that hit EOS / max_new_tokens, free their caches

`Engine` (the M1 single-request API) is kept as a thin wrapper: one request through an LLMEngine.

Known costs accepted for now:
  * attention is a Python loop over sequences (M4 replaces it with one kernel)
  * each sequence reserves a contiguous max_seq_len KV cache (M3 replaces it with paging)
  * ~1,000 kernel launches per step from Python (CUDA graphs would remove most of that)
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from nanoserve.kv_manager import ContiguousKVManager, PagedKVManager
from nanoserve.model import Qwen2Model, _snapshot
from nanoserve.scheduler import Scheduler, Sequence


def load_eos_token_ids(name: str) -> list[int]:
    """Stop tokens as HF generate sees them: generation_config.json's eos_token_id (may be a list),
    falling back to config.json."""
    folder = _snapshot(name)
    for fname in ("generation_config.json", "config.json"):
        path = folder / fname
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return eos if isinstance(eos, list) else [eos]
    raise ValueError(f"no eos_token_id found for {name}")


@dataclass
class StepOutput:
    seq_id: int
    token: int
    finished: bool


@dataclass
class StepStats:
    n_prefill: int      # sequences prefilling this step
    n_decode: int       # sequences decoding this step
    n_tokens: int       # total tokens through the model this step


class LLMEngine:
    def __init__(self, model: Qwen2Model, eos_token_ids: list[int], max_seq_len: int = 2048,
                 policy: str = "continuous", max_num_seqs: int = 8, max_num_batched_tokens: int = 8192,
                 batch_wait_s: float = 0.02, kv_cache: str = "paged", kv_cache_tokens: int | None = None,
                 block_size: int = 16, attn_backend: str | None = None):
        """kv_cache: "paged" (M3, default) or "contiguous" (M1/M2). kv_cache_tokens: the explicit KV budget
        in tokens (default 32 x max_seq_len)."""
        self.model = model
        self.eos_token_ids = list(eos_token_ids)
        self._eos = set(self.eos_token_ids)
        self.max_seq_len = max_seq_len
        budget = kv_cache_tokens if kv_cache_tokens is not None else 32 * max_seq_len
        if attn_backend is None:   # M4 default: the Triton kernel wherever there is a GPU
            attn_backend = "sdpa"   # TEMP during M3 benchmark; M4 flips this to triton on CUDA
        self.attn_backend = attn_backend
        if kv_cache == "paged":
            self.kv = PagedKVManager(model.cfg, budget, block_size, device=model.device, dtype=model.dtype,
                                     attn_backend=attn_backend)
            # A preempted request re-prefills prompt + outputs (up to max_seq_len tokens) in ONE step.
            max_num_batched_tokens = max(max_num_batched_tokens, max_seq_len)
        elif kv_cache == "contiguous":
            self.kv = ContiguousKVManager(model.cfg, max_seq_len, budget, device=model.device, dtype=model.dtype)
        else:
            raise ValueError(f"unknown kv_cache {kv_cache!r}")
        self.scheduler = Scheduler(policy, max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                                   batch_wait_s=batch_wait_s, kv=self.kv)
        self.sequences: dict[int, Sequence] = {}   # unfinished only
        self.last_step: StepStats | None = None
        self._next_id = 0
        self._stats = {"steps": 0, "peak_running": 0, "kv_used_tokens_peak": 0, "kv_reserved_tokens_peak": 0,
                      "kv_util_sum": 0.0, "kv_util_samples": 0}

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0.0 if isinstance(self._stats[k], float) else 0
        self.scheduler.num_preemptions = 0

    @property
    def stats(self) -> dict:
        """Counters for the benchmark report: steps, peak running, preemptions, KV utilisation."""
        s = dict(self._stats)
        s["num_preemptions"] = self.scheduler.num_preemptions
        s["kv_util_mean"] = s["kv_util_sum"] / s["kv_util_samples"] if s["kv_util_samples"] else None
        s["kv_cache"] = self.kv.kind
        s["attn_backend"] = self.attn_backend
        return s

    @classmethod
    def from_pretrained(cls, name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        **kwargs) -> "LLMEngine":
        return cls(Qwen2Model.from_pretrained(name, device=device, dtype=dtype),
                   eos_token_ids=load_eos_token_ids(name), **kwargs)

    def add_request(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False,
                    arrival_time: float | None = None) -> int:
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                             f"exceeds max_seq_len ({self.max_seq_len})")
        seq = Sequence(seq_id=self._next_id, prompt_ids=list(prompt_ids), max_new_tokens=max_new_tokens,
                       ignore_eos=ignore_eos,
                       arrival_time=time.perf_counter() if arrival_time is None else arrival_time)
        self.scheduler.add(seq)   # may raise (prompt longer than the token budget)
        self.sequences[seq.seq_id] = seq
        self._next_id += 1
        return seq.seq_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort(self, seq_id: int) -> None:
        """Stop a request early (e.g. the client disconnected) and free its KV."""
        seq = self.sequences.pop(seq_id, None)
        if seq is not None:
            self.scheduler.abort(seq)

    def step(self) -> list[StepOutput]:
        plan = self.scheduler.schedule(now=time.perf_counter())   # admits (allocating KV) and may preempt
        if not plan.scheduled:
            self.last_step = None
            return []
        m = self.model
        self.kv.begin_step()
        tokens, positions, inputs, counts = [], [], [], []
        for seq in plan.scheduled:
            t, p = seq.step_tokens()
            tokens += t
            positions += p
            inputs.append(self.kv.seq_input(seq, len(t)))
            counts.append(len(t))
        n_prefill = sum(seq.is_prefill for seq in plan.scheduled)
        self.last_step = StepStats(n_prefill=n_prefill, n_decode=len(plan.scheduled) - n_prefill,
                                   n_tokens=len(tokens))

        logits = m.forward_batch(torch.tensor(tokens, device=m.device), torch.tensor(positions, device=m.device),
                                 inputs, logits="last", **self.kv.forward_kwargs())
        next_tokens = logits.argmax(-1).tolist()      # greedy; one sync for the whole batch

        for seq, n in zip(plan.scheduled, counts):
            self.kv.commit(seq, n)
            seq.num_cached += n
        self._record_stats(plan.scheduled)

        outputs = []
        for seq, tok in zip(plan.scheduled, next_tokens):
            seq.output_ids.append(tok)
            finished = (tok in self._eos and not seq.ignore_eos) or len(seq.output_ids) >= seq.max_new_tokens
            if finished:
                self.scheduler.finish(seq)             # frees its KV
                del self.sequences[seq.seq_id]
            outputs.append(StepOutput(seq.seq_id, tok, finished))
        return outputs

    def _record_stats(self, running: list[Sequence]) -> None:
        st = self._stats
        st["steps"] += 1
        st["peak_running"] = max(st["peak_running"], len(running))
        used, reserved = self.kv.usage(running)
        st["kv_used_tokens_peak"] = max(st["kv_used_tokens_peak"], used)
        st["kv_reserved_tokens_peak"] = max(st["kv_reserved_tokens_peak"], reserved)
        if reserved:
            st["kv_util_sum"] += used / reserved
            st["kv_util_samples"] += 1


class Engine:
    """The M1 single-request API, now a wrapper: one request through an LLMEngine."""

    def __init__(self, model: Qwen2Model, eos_token_ids: list[int], max_seq_len: int = 2048):
        self.model = model
        self.eos_token_ids = list(eos_token_ids)
        self.max_seq_len = max_seq_len
        self._llm: LLMEngine | None = None   # one persistent engine (its KV pool is allocated once)

    @classmethod
    def from_pretrained(cls, name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        max_seq_len: int = 2048) -> "Engine":
        return cls(Qwen2Model.from_pretrained(name, device=device, dtype=dtype),
                   eos_token_ids=load_eos_token_ids(name), max_seq_len=max_seq_len)

    def generate(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False) -> Iterator[int]:
        """Yield generated token ids one at a time (greedy). The EOS token, if produced, is yielded too."""
        if self._llm is None:
            # SDPA on purpose: this single-request path is the reference that must match HF bitwise.
            self._llm = LLMEngine(self.model, self.eos_token_ids, max_seq_len=self.max_seq_len, max_num_seqs=1,
                                  max_num_batched_tokens=self.max_seq_len, batch_wait_s=0.0,
                                  kv_cache_tokens=self.max_seq_len, attn_backend="sdpa")
        llm = self._llm
        sid = llm.add_request(prompt_ids, max_new_tokens, ignore_eos=ignore_eos)
        try:
            while llm.has_unfinished():
                for out in llm.step():
                    yield out.token
        finally:
            llm.abort(sid)   # no-op if it finished; frees KV if the caller stopped iterating early
