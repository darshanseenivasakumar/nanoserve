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
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from nanoserve.kv_manager import ContiguousKVManager, PagedKVManager
from nanoserve.model import Qwen2Model, _snapshot
from nanoserve.sampling import SamplingParams, sample
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
                 policy: str = "continuous", max_num_seqs: int = 8, max_num_batched_tokens: int = 2048,
                 batch_wait_s: float = 0.02, kv_cache: str = "paged", kv_cache_tokens: int | None = None,
                 block_size: int = 16, attn_backend: str | None = None, cuda_graphs: bool | None = None,
                 chunked_prefill: bool | None = None, piecewise_graphs: bool | None = None,
                 fused_ops: bool | None = None, prefix_caching: bool = False):
        """kv_cache: "paged" (M3, default) or "contiguous" (M1/M2). kv_cache_tokens: the explicit KV budget
        in tokens (default 32 x max_seq_len)."""
        self.eos_token_ids = list(eos_token_ids)
        self._eos = set(self.eos_token_ids)
        self.max_seq_len = max_seq_len
        budget = kv_cache_tokens if kv_cache_tokens is not None else 32 * max_seq_len
        if attn_backend is None:   # M4 default: the Triton kernel wherever there is a GPU
            attn_backend = "triton" if (kv_cache == "paged" and model.device.type == "cuda") else "sdpa"
        self.attn_backend = attn_backend
        # M10: fused Triton kernels for the token-wise ops; part of the same fast-path bundle as the kernel.
        if fused_ops is None:
            fused_ops = attn_backend == "triton"
        if fused_ops:
            if model.device.type != "cuda":
                raise ValueError("fused ops are Triton kernels: they need a GPU")
            model = model.with_fused_ops()
        self.model = model
        if kv_cache == "paged":
            # M12: prefix caching is opt-in (the main benchmark keeps it off for every engine).
            self.kv = PagedKVManager(model.cfg, budget, block_size, device=model.device, dtype=model.dtype,
                                     attn_backend=attn_backend, prefix_caching=prefix_caching)
            if chunked_prefill is None:
                chunked_prefill = True               # M8 default for the paged cache
            if not chunked_prefill:
                # Without chunking a preempted request re-prefills prompt + outputs (up to max_seq_len
                # tokens) in ONE step, so the budget can't be smaller than that.
                max_num_batched_tokens = max(max_num_batched_tokens, max_seq_len)
        elif kv_cache == "contiguous":
            if prefix_caching:
                raise ValueError("prefix caching shares paged blocks: it needs kv_cache='paged'")
            self.kv = ContiguousKVManager(model.cfg, max_seq_len, budget, device=model.device, dtype=model.dtype)
        else:
            raise ValueError(f"unknown kv_cache {kv_cache!r}")
        self.chunked_prefill = bool(chunked_prefill) and kv_cache == "paged"   # contiguous can't chunk
        self.scheduler = Scheduler(policy, max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                                   batch_wait_s=batch_wait_s, kv=self.kv, chunked_prefill=self.chunked_prefill)
        # M7: CUDA graphs for pure-decode steps (needs the paged cache + Triton kernel; on by default there).
        if cuda_graphs is None:
            cuda_graphs = kv_cache == "paged" and attn_backend == "triton"
        self.graph_runner = None
        if cuda_graphs:
            from nanoserve.cuda_graphs import DecodeGraphRunner
            self.graph_runner = DecodeGraphRunner(model, self.kv.pool, max_seq_len)
        # M9: piecewise graphs for steps containing prefill chunks (on wherever decode graphs are).
        if piecewise_graphs is None:
            piecewise_graphs = bool(cuda_graphs)
        self.piecewise_runner = None
        if piecewise_graphs:
            if kv_cache != "paged" or model.device.type != "cuda":
                raise ValueError("piecewise CUDA graphs need the paged KV cache on a GPU")
            from nanoserve.cuda_graphs import PiecewiseGraphRunner
            garbage = (self.graph_runner.garbage_block if self.graph_runner is not None
                       else self.kv.pool.allocator.allocate())
            self.piecewise_runner = PiecewiseGraphRunner(model, self.kv.pool, garbage,
                                                         max_tokens=max_num_batched_tokens)
        self.sequences: dict[int, Sequence] = {}   # unfinished only
        self.last_step: StepStats | None = None
        self._next_id = 0
        self._stats = {"steps": 0, "graph_steps": 0, "piecewise_steps": 0, "prefill_chunks": 0, "peak_running": 0,
                       "kv_used_tokens_peak": 0,
                       "kv_reserved_tokens_peak": 0, "kv_util_sum": 0.0, "kv_util_samples": 0}

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0.0 if isinstance(self._stats[k], float) else 0
        self.scheduler.num_preemptions = 0
        if hasattr(self.kv, "reset_stats"):
            self.kv.reset_stats()

    @property
    def stats(self) -> dict:
        """Counters for the benchmark report: steps, peak running, preemptions, KV utilisation."""
        s = dict(self._stats)
        s["num_preemptions"] = self.scheduler.num_preemptions
        s["kv_util_mean"] = s["kv_util_sum"] / s["kv_util_samples"] if s["kv_util_samples"] else None
        s["kv_cache"] = self.kv.kind
        s["attn_backend"] = self.attn_backend
        s["cuda_graphs"] = self.graph_runner is not None
        s["graph_buckets"] = self.graph_runner.captured_buckets if self.graph_runner else []
        s["piecewise_graphs"] = self.piecewise_runner is not None
        s["fused_ops"] = self.model.fused
        s["prefix_caching"] = getattr(self.kv, "prefix_caching", False)
        s["prefix_query_tokens"] = getattr(self.kv, "prefix_query_tokens", 0)
        s["prefix_hit_tokens"] = getattr(self.kv, "prefix_hit_tokens", 0)
        s["piecewise_buckets"] = self.piecewise_runner.captured_buckets if self.piecewise_runner else []
        return s

    def capture_graphs(self) -> float:
        """Capture every CUDA graph this engine can use NOW (server / benchmark startup), so no request pays
        for a capture mid-run. Returns seconds taken."""
        t0 = time.perf_counter()
        if self.graph_runner is not None:
            self.graph_runner.capture_all(self.scheduler.max_num_seqs)
        if self.piecewise_runner is not None:
            self.piecewise_runner.capture_all()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    @classmethod
    def from_pretrained(cls, name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        **kwargs) -> "LLMEngine":
        return cls(Qwen2Model.from_pretrained(name, device=device, dtype=dtype),
                   eos_token_ids=load_eos_token_ids(name), **kwargs)

    def add_request(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False,
                    arrival_time: float | None = None, sampling: SamplingParams | None = None) -> int:
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                             f"exceeds max_seq_len ({self.max_seq_len})")
        seed = sampling.seed if (sampling is not None and sampling.seed is not None) else random.getrandbits(62)
        seq = Sequence(seq_id=self._next_id, prompt_ids=list(prompt_ids), max_new_tokens=max_new_tokens,
                       ignore_eos=ignore_eos,
                       arrival_time=time.perf_counter() if arrival_time is None else arrival_time,
                       sampling=sampling, seed=seed)
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
        if self.timing is not None:
            self.timing["_step_start"] = time.perf_counter()
        plan = self.scheduler.schedule(now=time.perf_counter())   # admits (allocating KV) and may preempt
        if not plan.scheduled:
            self.last_step = None
            return []
        self._mark("schedule")
        m = self.model
        self.kv.begin_step()
        tokens, positions, inputs, counts = [], [], [], []
        for seq, n in zip(plan.scheduled, plan.chunks):
            t, p = seq.step_tokens(n)
            tokens += t
            positions += p
            inputs.append(self.kv.seq_input(seq, n))
            counts.append(n)
        n_prefill = sum(seq.is_prefill for seq in plan.scheduled)
        self.last_step = StepStats(n_prefill=n_prefill, n_decode=len(plan.scheduled) - n_prefill,
                                   n_tokens=len(tokens))
        self._mark("prepare", gpu_start=True)

        runner = self.graph_runner
        if (runner is not None and n_prefill == 0 and len(inputs) <= runner.max_bucket
                and all(i.use_kernel for i in inputs)):
            # M7: pure-decode step -> replay the captured CUDA graph (one launch instead of ~1,000).
            slots, tables, lens = self.kv.graph_inputs()
            logits = runner.run(tokens, positions, slots, tables, lens)
            self._stats["graph_steps"] += 1
        elif self.piecewise_runner is not None and len(tokens) <= self.piecewise_runner.max_tokens:
            # M9: mixed step -> token-wise pieces from graphs, attention eagerly in between.
            logits = self.piecewise_runner.run(tokens, positions, self.kv.graph_inputs()[0], inputs,
                                               self.kv.decode_meta())
            self._stats["piecewise_steps"] += 1
        else:
            logits = m.forward_batch(torch.tensor(tokens, device=m.device), torch.tensor(positions, device=m.device),
                                     inputs, logits="last", **self.kv.forward_kwargs())
        self._mark("launch", gpu_end=True)
        if any(s.sampling is not None and not s.sampling.greedy for s in plan.scheduled):
            # M13: draw n of request r uses mix(seed_r, n), n = tokens generated so far -> reproducible
            # whatever the batch, and across a preemption + recompute.
            next_tokens = sample(logits, [s.sampling for s in plan.scheduled], [s.seed for s in plan.scheduled],
                                 [len(s.output_ids) for s in plan.scheduled])
        else:
            next_tokens = logits.argmax(-1).tolist()      # greedy; one sync for the whole batch
        self._mark("wait")

        for seq, n in zip(plan.scheduled, counts):
            self.kv.commit(seq, n)
            seq.num_cached += n
        self._record_stats(plan.scheduled)

        outputs = []
        for seq, tok in zip(plan.scheduled, next_tokens):
            if seq.num_cached < seq.num_tokens:
                # M8: a non-final prefill chunk -- its last logit is mid-prompt, nothing to sample.
                self._stats["prefill_chunks"] += 1
                continue
            seq.output_ids.append(tok)
            finished = (tok in self._eos and not seq.ignore_eos) or len(seq.output_ids) >= seq.max_new_tokens
            if finished:
                self.scheduler.finish(seq)             # frees its KV
                del self.sequences[seq.seq_id]
            outputs.append(StepOutput(seq.seq_id, tok, finished))
        self._mark("post")
        return outputs

    # ---- M11: optional per-phase timing (off unless `engine.timing = {}`) ----------------------------------
    timing: dict | None = None

    def _mark(self, phase: str, gpu_start: bool = False, gpu_end: bool = False) -> None:
        """Add the CPU time since the previous mark to `phase`. With gpu_start/gpu_end, also record CUDA
        events around the forward pass; their elapsed time (GPU busy span) is summed into "gpu" once the
        step has synchronised. Costs nothing when timing is off."""
        T = self.timing
        if T is None:
            return
        now = time.perf_counter()
        last = T.get("_last")
        if last is not None and phase != "schedule":
            T[phase] = T.get(phase, 0.0) + now - last
        elif phase == "schedule" and "_step_start" in T:
            T[phase] = T.get(phase, 0.0) + now - T["_step_start"]
        T["_last"] = now
        if gpu_start:
            T["_ev0"] = torch.cuda.Event(enable_timing=True)
            T["_ev0"].record()
        if gpu_end:
            T["_ev1"] = torch.cuda.Event(enable_timing=True)
            T["_ev1"].record()
        if phase == "post":
            T["gpu"] = T.get("gpu", 0.0) + T["_ev0"].elapsed_time(T["_ev1"]) / 1e3
            T["steps"] = T.get("steps", 0) + 1

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
