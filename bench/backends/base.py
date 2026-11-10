"""Core types shared by every backend.

A Backend turns one Request into one RequestRecord. The record holds raw timestamps only; every metric
(TTFT, TPOT, E2E) is *derived* from them, so all engines are measured by exactly the same arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Request:
    id: int
    prompt: str          # already chat-templated text, identical for every engine
    prompt_tokens: int   # counted by us with the HF tokenizer
    max_tokens: int      # engines are forced to produce exactly this many (ignore_eos)


@dataclass
class RequestRecord:
    id: int
    t_submit: float                      # time.perf_counter() when the request was sent
    t_chunks: list[float] = field(default_factory=list)  # arrival time of each streamed chunk
    num_output_tokens: int = 0           # may exceed len(t_chunks) if a server packs tokens per chunk
    prompt_tokens_reported: int | None = None  # what the engine says it saw (sanity check)
    output_text: str = ""
    output_token_ids: list[int] | None = None  # only in-process backends know these
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token: submit -> first chunk."""
        return self.t_chunks[0] - self.t_submit if self.t_chunks else None

    @property
    def e2e(self) -> float | None:
        """End-to-end latency: submit -> last chunk."""
        return self.t_chunks[-1] - self.t_submit if self.t_chunks else None

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first: (last - first) / (tokens - 1).

        Uses the token count, not the chunk count, so servers that send several tokens per chunk are
        measured fairly. Undefined for a single-token output.
        """
        if not self.t_chunks or self.num_output_tokens < 2:
            return None
        return (self.t_chunks[-1] - self.t_chunks[0]) / (self.num_output_tokens - 1)


class Backend(Protocol):
    async def generate(self, req: Request) -> RequestRecord: ...

    def info(self) -> dict: ...

    async def close(self) -> None: ...
