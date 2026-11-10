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
    def is_prefill(self) -> bool:
        """No KV stored: a new request, or one that was PREEMPTED and lost its KV (M3)."""
        return self.num_cached == 0

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def step_tokens(self) -> tuple[list[int], list[int]]:
        """(token ids, positions) this sequence feeds into the next forward pass.

        Prefill feeds every known token (prompt + any outputs kept across a preemption) and its last logit
        predicts the next token. Decode feeds only the newest output token, whose K/V isn't stored yet.
        """
        if self.is_prefill:
            toks = self.prompt_ids + self.output_ids
            return list(toks), list(range(len(toks)))
        return [self.output_ids[-1]], [self.num_tokens - 1]

    @property
    def step_cost(self) -> int:
        return self.num_tokens if self.is_prefill else 1


@dataclass
class SchedulerOutput:
    scheduled: list[Sequence]   # everyone running this step (decodes + new prefills), in batch order
    admitted: list[Sequence]    # the subset newly admitted this step (need a KV cache)


class Scheduler:
    def __init__(self, policy: str = "continuous", max_num_seqs: int = 8, max_num_batched_tokens: int = 8192,
                 batch_wait_s: float = 0.02, kv=None):
        if policy not in ("static", "continuous"):
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.batch_wait_s = batch_wait_s
        self.kv = kv                      # KV manager (nanoserve/kv_manager.py) or None = unlimited (M2 tests)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        if len(seq.prompt_ids) > self.max_num_batched_tokens:
            raise ValueError(f"prompt of {len(seq.prompt_ids)} tokens exceeds the per-step token budget "
                             f"({self.max_num_batched_tokens}); chunked prefill is not implemented")
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
        if self.kv is not None:
            self._reserve_decode_slots()
        admitted: list[Sequence] = []
        if self._may_admit(now):
            budget = self.max_num_batched_tokens - sum(s.step_cost for s in self.running)
            while (self.waiting and len(self.running) + len(admitted) < self.max_num_seqs
                   and self.waiting[0].step_cost <= budget
                   and (self.kv is None or self.kv.can_admit(self.waiting[0]))):
                seq = self.waiting.popleft()
                budget -= seq.step_cost
                if self.kv is not None:
                    self.kv.admit(seq)
                seq.status = SeqStatus.RUNNING
                admitted.append(seq)
            self.running.extend(admitted)
        return SchedulerOutput(scheduled=list(self.running), admitted=admitted)
