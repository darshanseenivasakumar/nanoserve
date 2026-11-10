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
    cache: object | None = None        # KV cache, allocated by the engine on admission

    @property
    def is_prefill(self) -> bool:
        return not self.output_ids

    def step_tokens(self) -> tuple[list[int], list[int]]:
        """(token ids, positions) this sequence feeds into the next forward pass."""
        if self.is_prefill:
            return list(self.prompt_ids), list(range(len(self.prompt_ids)))
        pos = len(self.prompt_ids) + len(self.output_ids) - 1
        return [self.output_ids[-1]], [pos]

    @property
    def step_cost(self) -> int:
        return len(self.prompt_ids) if self.is_prefill else 1


@dataclass
class SchedulerOutput:
    scheduled: list[Sequence]   # everyone running this step (decodes + new prefills), in batch order
    admitted: list[Sequence]    # the subset newly admitted this step (need a KV cache)


class Scheduler:
    def __init__(self, policy: str = "continuous", max_num_seqs: int = 8, max_num_batched_tokens: int = 8192,
                 batch_wait_s: float = 0.02):
        if policy not in ("static", "continuous"):
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.batch_wait_s = batch_wait_s
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        if len(seq.prompt_ids) > self.max_num_batched_tokens:
            raise ValueError(f"prompt of {len(seq.prompt_ids)} tokens exceeds the per-step token budget "
                             f"({self.max_num_batched_tokens}); chunked prefill is not implemented")
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def finish(self, seq: Sequence) -> None:
        seq.status = SeqStatus.FINISHED
        self.running.remove(seq)

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
        admitted: list[Sequence] = []
        if self._may_admit(now):
            budget = self.max_num_batched_tokens - sum(s.step_cost for s in self.running)
            while (self.waiting and len(self.running) + len(admitted) < self.max_num_seqs
                   and self.waiting[0].step_cost <= budget):
                seq = self.waiting.popleft()
                budget -= seq.step_cost
                seq.status = SeqStatus.RUNNING
                admitted.append(seq)
            self.running.extend(admitted)
        return SchedulerOutput(scheduled=list(self.running), admitted=admitted)
