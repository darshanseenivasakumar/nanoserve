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
COLORS = {"vllm": "#1f77b4", "llamacpp": "#2ca02c", "llamacpp-np8": "#98df8a", "hf-generate": "#7f7f7f"}


def load(tags: list[str]) -> dict:
    """Merge result files of several tags. The first tag that has an engine owns its plain name; the same
    engine from a later tag is labelled "engine [tag]" (e.g. nanoserve before/after a change)."""
    data = defaultdict(dict)   # engine label -> concurrency -> result
    owner = {}
    for tag in tags:
        for f in glob.glob(str(RESULTS / f"*_{tag}.json")):
            d = json.load(open(f))
            eng = d["backend"]["engine"]
            owner.setdefault(eng, tag)
            label = eng if owner[eng] == tag else f"{eng} [{tag}]"
            data[label][d["concurrency"]] = d
    return data


def color(engine: str) -> str:
    if engine.startswith("nanoserve"):
        return "#ff9896" if "[" in engine else "#d62728"
    return COLORS.get(engine.split(" [")[0], "#9467bd")


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


def plot(tags: list[str], out: Path, label: str) -> Path | None:
    data = load(tags)
    tag = tags[0]
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
    p.add_argument("--tag", default="m6", help="comma-separated; first tag wins per engine, e.g. m7,m6")
    p.add_argument("--out", default=str(Path(__file__).with_name("plots")))
    args = p.parse_args()
    tags = args.tag.split(",")
    for tg, label in ((tags, "fixed 128 output tokens"), ([t + "-var16-256" for t in tags], "output 16-256 tokens")):
        path = plot(tg, Path(args.out), label)
        print(path or f"no results for tags {tg}")


if __name__ == "__main__":
    main()
