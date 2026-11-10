"""Plot benchmark results: throughput, TTFT and TPOT vs concurrency, one line per engine.

    uv run python -m bench.plot --tag m6 --out bench/plots

Reads bench/results/*_<tag>.json (fixed-length workload) and *_<tag>-var16-256.json (varied lengths).
Each point is the median repeat; error bars on throughput show the min-max across repeats.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).with_name("results")
COLORS = {"vllm": "#1f77b4", "llamacpp": "#2ca02c", "hf-generate": "#7f7f7f"}


def load(tag: str) -> dict:
    data = defaultdict(dict)   # engine -> concurrency -> result
    for f in glob.glob(str(RESULTS / f"*_{tag}.json")):
        d = json.load(open(f))
        data[d["backend"]["engine"]][d["concurrency"]] = d
    return data


def color(engine: str) -> str:
    return COLORS.get(engine, "#d62728" if engine.startswith("nanoserve") else "#9467bd")


def panel(ax, data, metric, title, ylabel, logy=False):
    for engine, by_c in sorted(data.items()):
        cs = sorted(by_c)
        if metric == "tok/s":
            ys = [by_c[c]["summary"]["output_tok_per_s"] for c in cs]
            lo = [y - by_c[c]["spread"]["output_tok_per_s_min"] for c, y in zip(cs, ys)]
            hi = [by_c[c]["spread"]["output_tok_per_s_max"] - y for c, y in zip(cs, ys)]
            ax.errorbar(cs, ys, yerr=[lo, hi], marker="o", capsize=3, label=engine, color=color(engine))
        else:
            key, pct = metric.split(":")
            ys = [by_c[c]["summary"][key][pct] * 1e3 for c in cs]
            ax.plot(cs, ys, marker="o", label=engine, color=color(engine))
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot(tag: str, out: Path, label: str) -> Path | None:
    data = load(tag)
    if not data:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    panel(axes[0], data, "tok/s", f"Output throughput ({label})", "tokens / s", logy=True)
    panel(axes[1], data, "ttft:p50", "TTFT p50", "ms", logy=True)
    panel(axes[2], data, "ttft:p99", "TTFT p99", "ms", logy=True)
    panel(axes[3], data, "tpot:p50", "TPOT p50 (time per output token)", "ms", logy=True)
    fig.suptitle("Qwen2.5-0.5B-Instruct, RTX 3050 6GB laptop, 64 prompts, greedy", fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="m6")
    p.add_argument("--out", default=str(Path(__file__).with_name("plots")))
    args = p.parse_args()
    for tag, label in ((args.tag, "fixed 128 output tokens"), (f"{args.tag}-var16-256", "output 16-256 tokens")):
        path = plot(tag, Path(args.out), label)
        print(path or f"no results for tag {tag}")


if __name__ == "__main__":
    main()
