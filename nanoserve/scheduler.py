"""Decides, every engine step, WHICH requests run.

Two policies (one flag, so M2 can measure exactly one change):

  static      Admit a new batch only when the previous batch has completely finished (the tour bus).
              Waits up to `batch_wait_s` for the batch to fill, like the HF baseline backend.
  continuous  Admit waiting requests at EVERY step whenever a seat and token budget are free
              (iteration-level scheduling, Orca OSDI'22 / vLLM). Finished requests leave immediately.

Limits per step:
  max_num_seqs            how many sequences may be running (batch size cap)
  max_num_batched_tokens  token budget: each running decode costs 1, a new request costs its whole prompt
                          (prefill). Bounds how long one step can take.

Admission is strictly first-come-first-served: if the head of the queue doesn't fit this step, nobody
behind it jumps ahead (simple, and no request can starve).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    """One request inside the engine."""
    seq_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    ignore_eos: bool = False
    arrival_time: float = 0.0
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    cache: object | None = None        # contiguous KV cache (M1/M2), set on admission
    block_table: object | None = None  # paged KV (M3), set on admission
    num_cached: int = 0                # tokens whose K/V are stored. 0 => this sequence must (re)prefill.

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def remaining(self) -> int:
        """Known tokens whose K/V is not stored yet."""
        return self.num_tokens - self.num_cached

    @property
    def is_prefill(self) -> bool:
        """Prefilling: a new request, a PREEMPTED one recomputing its KV (M3), or one part-way through a
        CHUNKED prefill (M8). Decoding = exactly one uncached token (the newest output)."""
        return self.num_cached == 0 or self.remaining > 1

    def step_tokens(self, n: int | None = None) -> tuple[list[int], list[int]]:
        """(token ids, positions) for the next n uncached tokens (default: all of them).

        One rule covers everything: feed tokens[num_cached : num_cached + n]; if that reaches the end of the
        known tokens, the last logit predicts the next token. Decode = n is 1 and only the newest output is
        uncached. Prefill = the whole prompt (+ outputs kept across a preemption), possibly in chunks.
        """
        n = self.remaining if n is None else n
        if n == 1 and self.remaining == 1 and self.output_ids:      # fast path: plain decode
            return [self.output_ids[-1]], [self.num_tokens - 1]
        toks = (self.prompt_ids + self.output_ids)[self.num_cached:self.num_cached + n]
        return toks, list(range(self.num_cached, self.num_cached + n))

    @property
    def step_cost(self) -> int:
        return self.remaining


@dataclass
class SchedulerOutput:
    scheduled: list[Sequence]   # everyone running this step (decodes + prefill chunks), in batch order
    admitted: list[Sequence]    # the subset newly admitted this step (KV reserved for them)
    chunks: list[int] = field(default_factory=list)   # tokens each scheduled sequence processes this step


class Scheduler:
    def __init__(self, policy: str = "continuous", max_num_seqs: int = 8, max_num_batched_tokens: int = 8192,
                 batch_wait_s: float = 0.02, kv=None, chunked_prefill: bool = False):
        if policy not in ("static", "continuous"):
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.batch_wait_s = batch_wait_s
        self.kv = kv                      # KV manager (nanoserve/kv_manager.py) or None = unlimited (M2 tests)
        # M8: split prefills into budget-sized chunks across steps (needs the paged cache: later chunks
        # attend over an existing context). Off = a prefill must fit the per-step budget in one piece.
        self.chunked_prefill = chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        if not self.chunked_prefill and len(seq.prompt_ids) > self.max_num_batched_tokens:
            raise ValueError(f"prompt of {len(seq.prompt_ids)} tokens exceeds the per-step token budget "
                             f"({self.max_num_batched_tokens}) and chunked prefill is off")
        if self.kv is not None and not self.kv.can_ever_fit(len(seq.prompt_ids) + seq.max_new_tokens):
            raise ValueError(f"request needs up to {len(seq.prompt_ids) + seq.max_new_tokens} tokens of KV, "
                             f"more than the whole KV budget: it could never finish")
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def finish(self, seq: Sequence) -> None:
        seq.status = SeqStatus.FINISHED
        self.running.remove(seq)
        if self.kv is not None:
            self.kv.free(seq)

    def abort(self, seq: Sequence) -> None:
        """Drop a request (client went away) wherever it is."""
        if seq in self.running:
            self.finish(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
            seq.status = SeqStatus.FINISHED

    def _preempt(self, seq: Sequence) -> None:
        """Evict a running request: free its KV, keep its outputs, put it back at the FRONT of the queue.
        When readmitted it recomputes the KV of prompt + outputs in one prefill ("recompute" preemption)."""
        self.running.remove(seq)
        self.kv.free(seq)
        seq.num_cached = 0
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _reserve_decode_slots(self) -> None:
        """Every running request needs room for this step's token. If the pool is out of blocks, preempt
        the most recently admitted request (LIFO: it has done the least work, so recomputing it is cheapest)
        until everyone left fits."""
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if self.kv.try_append(seq):
                i += 1
                continue
            self._preempt(self.running[-1])   # may be `seq` itself, in which case the loop just moves on

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _may_admit(self, now: float) -> bool:
        if self.policy == "continuous":
            return True
        # static: only between batches, and only once the batch is full or we've waited long enough
        if self.running or not self.waiting:
            return False
        return len(self.waiting) >= self.max_num_seqs or now - self.waiting[0].arrival_time >= self.batch_wait_s

    def schedule(self, now: float) -> SchedulerOutput:
        """Budget order: (1) running decodes, 1 token each, always; (2) running requests part-way through a
        chunked prefill; (3) new admissions (FCFS). With chunked prefill a prefill takes whatever budget is
        left (min(remaining, budget)) and continues next step; without it, it must fit in one piece."""
        if self.kv is not None:
            self._reserve_decode_slots()
        budget = self.max_num_batched_tokens
        scheduled, chunks = [], []
        partial = []
        for seq in self.running:
            if seq.is_prefill:
                partial.append(seq)
            else:
                scheduled.append(seq)
                chunks.append(1)
                budget -= 1
        for seq in partial:                       # only exists with chunked prefill
            n = min(seq.remaining, budget) if self.chunked_prefill else seq.remaining
            if n > 0:
                scheduled.append(seq)
                chunks.append(n)
                budget -= n
        admitted: list[Sequence] = []
        if self._may_admit(now):
            while (self.waiting and len(self.running) + len(admitted) < self.max_num_seqs
                   and (budget > 0 if self.chunked_prefill else self.waiting[0].step_cost <= budget)
                   and (self.kv is None or self.kv.can_admit(self.waiting[0]))):
                seq = self.waiting.popleft()
                n = min(seq.step_cost, budget) if self.chunked_prefill else seq.step_cost
                budget -= n
                if self.kv is not None:
                    self.kv.admit(seq)             # reserves KV for ALL of its known tokens
                seq.status = SeqStatus.RUNNING
                admitted.append(seq)
                scheduled.append(seq)
                chunks.append(n)
            self.running.extend(admitted)
        return SchedulerOutput(scheduled=scheduled, admitted=admitted, chunks=chunks)
