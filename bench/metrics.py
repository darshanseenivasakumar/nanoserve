"""Turn a list of RequestRecords into summary numbers.

Throughput uses *wall time* of the whole run (first submit -> last token), not the sum of per-request
times: with 8 requests in flight, summing would count the same second 8 times.
"""
from __future__ import annotations

import math
import statistics

from bench.backends.base import RequestRecord


def percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0 <= q <= 100) of values. Raises ValueError on empty input."""
    if not values:
        raise ValueError("percentile of empty list")
    xs = sorted(values)
    # Linear interpolation between closest ranks (= numpy.percentile's default). Chosen so our p50/p99
    # are computed exactly like vLLM's own benchmark scripts (which call np.percentile), keeping
    # cross-engine numbers comparable. Caveat: with n=64, p99 sits between the two slowest requests.
    pos = (q / 100) * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize(records: list[RequestRecord], wall_time: float) -> dict:
    ok = [r for r in records if r.error is None and r.t_chunks]
    total_out = sum(r.num_output_tokens for r in ok)
    total_in = sum(r.prompt_tokens_reported or 0 for r in ok)
    return {
        "num_requests": len(records),
        "num_errors": len(records) - len(ok),
        "wall_time_s": wall_time,
        "total_output_tokens": total_out,
        "total_prompt_tokens_reported": total_in,
        "output_tok_per_s": total_out / wall_time if wall_time > 0 else 0.0,
        "req_per_s": len(ok) / wall_time if wall_time > 0 else 0.0,
        "ttft": _dist([r.ttft for r in ok]),
        "tpot": _dist([r.tpot for r in ok if r.tpot is not None]),
        # M9: ITL = every gap between consecutive streamed chunks, pooled over all requests (vLLM's
        # definition). TPOT is one MEAN gap per request, so a single long stall gets averaged away; ITL p99
        # shows it. (With multi-token chunks a gap covers several tokens; our servers send one per chunk.)
        "itl": _dist([b - a for r in ok for a, b in zip(r.t_chunks, r.t_chunks[1:])]),
        "e2e": _dist([r.e2e for r in ok]),
    }
