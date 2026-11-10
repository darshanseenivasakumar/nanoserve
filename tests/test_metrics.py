"""Metric definitions. These pin down exactly what TTFT/TPOT/E2E/throughput mean in this project."""
import pytest
from pytest import approx

from bench.backends.base import RequestRecord
from bench.metrics import percentile, summarize


def rec(i, t_submit, chunks, n, error=None):
    return RequestRecord(id=i, t_submit=t_submit, t_chunks=chunks, num_output_tokens=n, error=error)


def test_record_derived_latencies():
    r = rec(0, 10.0, [10.5, 10.6, 10.7, 10.9], 4)
    assert r.ttft == approx(0.5)              # first token - submit
    assert r.e2e == approx(0.9)               # last token - submit
    assert r.tpot == approx((10.9 - 10.5) / 3)  # gaps between the 4 tokens


def test_tpot_uses_token_count_not_chunk_count():
    # A server may pack several tokens into one streamed chunk: 2 chunks but 5 tokens.
    r = rec(0, 0.0, [1.0, 2.0], 5)
    assert r.tpot == approx((2.0 - 1.0) / 4)


def test_tpot_undefined_for_single_token():
    assert rec(0, 0.0, [1.0], 1).tpot is None


def test_percentile_small_cases():
    assert percentile([5.0, 1.0, 3.0], 50) == approx(3.0)
    assert percentile([5.0, 1.0, 3.0], 0) == approx(1.0)
    assert percentile([5.0, 1.0, 3.0], 100) == approx(5.0)
    assert percentile([7.0], 99) == approx(7.0)


def test_percentile_tail():
    values = [float(v) for v in range(1, 101)]  # 1..100, deliberately unsorted below
    values.reverse()
    assert 99.0 <= percentile(values, 99) <= 100.0
    assert 50.0 <= percentile(values, 50) <= 51.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize():
    records = [
        rec(0, 0.0, [0.2, 0.4, 0.6, 0.8], 4),
        rec(1, 0.0, [0.4, 0.8, 1.2, 1.6], 4),
    ]
    s = summarize(records, wall_time=2.0)
    assert s["num_requests"] == 2
    assert s["num_errors"] == 0
    assert s["total_output_tokens"] == 8
    assert s["output_tok_per_s"] == approx(4.0)   # 8 tokens / 2 s
    assert s["req_per_s"] == approx(1.0)
    assert s["ttft"]["n"] == 2
    assert s["ttft"]["mean"] == approx(0.3)
    assert s["e2e"]["p50"] >= 0.8
    assert set(s["tpot"]) >= {"p50", "p99", "mean", "n"}


def test_summarize_counts_errors_but_excludes_them_from_latency():
    records = [rec(0, 0.0, [0.2, 0.4], 2), rec(1, 0.0, [], 0, error="boom")]
    s = summarize(records, wall_time=1.0)
    assert s["num_errors"] == 1
    assert s["ttft"]["n"] == 1
    assert s["total_output_tokens"] == 2
