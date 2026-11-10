"""Print a one-line summary per saved result file.

    uv run python -m bench.summarize                       # everything
    uv run python -m bench.summarize "*m2session*" "*nanoserve-m2*"
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

RESULTS = Path(__file__).with_name("results")


def line(d: dict) -> str:
    s, sp = d["summary"], d["spread"]
    rng = d["args"].get("output_len_range")
    work = f"var{rng[0]}-{rng[1]}" if rng else f"fixed{d['args']['max_tokens']}"
    ms = lambda x: f"{x * 1e3:7.0f}" if x is not None else "      -"
    return (f"{d['backend']['engine']:24s} {work:10s} c={d['concurrency']:<3d} "
            f"tok/s={s['output_tok_per_s']:7.1f} ({sp['output_tok_per_s_min']:.0f}-{sp['output_tok_per_s_max']:.0f}) "
            f"TTFT p50/p99={ms(s['ttft']['p50'])}/{ms(s['ttft']['p99'])} ms  "
            f"TPOT p50/p99={s['tpot']['p50'] * 1e3:5.1f}/{s['tpot']['p99'] * 1e3:5.1f} ms  "
            f"E2E p50/p99={s['e2e']['p50']:.2f}/{s['e2e']['p99']:.2f} s  problems={len(d['problems'])}")


def main(patterns: list[str]) -> None:
    files = set()
    for pat in patterns or ["*.json"]:
        files.update(glob.glob(str(RESULTS / pat)))
    for f in sorted(files, key=os.path.getmtime):
        print(line(json.load(open(f))))


if __name__ == "__main__":
    main(sys.argv[1:])
