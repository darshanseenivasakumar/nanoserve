"""Plain-text results table."""
from __future__ import annotations


def _ms(x):
    return "-" if x is None else f"{x * 1000:.1f}"


def _s(x):
    return "-" if x is None else f"{x:.2f}"


COLS = ["engine", "conc", "n", "out tok/s (min-max)", "req/s", "TTFT p50/p99 ms", "TPOT p50/p99 ms",
        "E2E p50/p99 s", "checks"]
WIDTHS = [12, 4, 4, 22, 6, 17, 16, 13, 7]


def _fmt(cells):
    return "  ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(cells, WIDTHS)))


def format_table(rows: list[dict], header: bool = True) -> str:
    lines = [_fmt(COLS), _fmt(["-" * w for w in WIDTHS])] if header else []
    for row in rows:
        s, sp = row["summary"], row["spread"]
        lines.append(_fmt([
            row["engine"], row["concurrency"], s["ttft"]["n"],
            f"{s['output_tok_per_s']:.1f} ({sp['output_tok_per_s_min']:.0f}-{sp['output_tok_per_s_max']:.0f})",
            f"{s['req_per_s']:.2f}",
            f"{_ms(s['ttft']['p50'])}/{_ms(s['ttft']['p99'])}",
            f"{_ms(s['tpot']['p50'])}/{_ms(s['tpot']['p99'])}",
            f"{_s(s['e2e']['p50'])}/{_s(s['e2e']['p99'])}",
            "OK" if not row["problems"] else f"FAIL {len(row['problems'])}",
        ]))
    return "\n".join(lines)
