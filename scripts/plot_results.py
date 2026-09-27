"""Render docs/results-{light,dark}.png from docs/benchmark.json.

Two panels: single-thread cl100k encode throughput and single-thread training time.
Each bar runs to the slowest of the recorded runs; the thin whisker reaches the fastest.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
              "accent": "#2a78d6", "rest": "#a8a69e"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#383835",
             "accent": "#3987e5", "rest": "#76746d"},
}


def ranges(values):
    return min(values), max(values)


def load():
    bench = json.loads((DOCS / "benchmark.json").read_text(encoding="utf-8"))
    single = [r["single"] for r in bench["runs"]]
    enc = lambda k: ranges([s["encode_mb_per_s"]["cl100k_base"][k] for s in single])  # noqa: E731
    train = lambda k: ranges([s["train"][1][k] for s in single])  # noqa: E731
    encode_rows = [
        ("bytepair (Rust)", enc("bytepair_rust_cold"), True),
        ("tiktoken", enc("tiktoken"), False),
        ("bytepair (Python reference)", enc("bytepair_python_cold"), False),
        ("HF tokenizers", enc("hf_tokenizers_batch_of_lines"), False),
    ]
    train_rows = [
        ("bytepair (Rust)", train("bytepair_rust_s"), True),
        ("HF tokenizers", train("hf_tokenizers_s"), False),
        ("bytepair (Python reference)", train("bytepair_python_s"), False),
    ]
    note = f"{bench['cpu'].replace(' 16-Core Processor', '')} · one thread · {len(single)} runs"
    return encode_rows, train_rows, note


def fmt(lo, hi, unit):
    a, b = (f"{lo:.1f}", f"{hi:.1f}")
    return f"{a} {unit}" if a == b else f"{a}–{b} {unit}"


def panel(ax, rows, title, unit, better, theme, lower_is_better):
    t = THEMES[theme]
    ax.set_facecolor(t["surface"])
    names = [r[0] for r in rows]
    ys = range(len(rows))[::-1]
    xmax = max(hi for _, (lo, hi), _ in rows)
    for y, (name, (lo, hi), highlight) in zip(ys, rows):
        # Bar to the conservative end (slowest run), whisker to the other.
        bar_end, whisker_end = (hi, lo) if lower_is_better else (lo, hi)
        color = t["accent"] if highlight else t["rest"]
        ax.barh(y, bar_end, height=0.56, color=color, edgecolor="none", zorder=3)
        ax.plot([bar_end, whisker_end], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.text(max(lo, hi) + xmax * 0.02, y, fmt(lo, hi, unit), va="center", ha="left",
                fontsize=11, color=t["ink"], fontweight="bold" if highlight else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=11, color=t["ink"])
    ax.set_xlim(0, xmax * 1.35)
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.set_title(title, loc="left", fontsize=13, color=t["ink"], fontweight="bold", pad=22)
    ax.text(0, 1.03, better, transform=ax.transAxes, fontsize=10, color=t["ink2"])


def main():
    encode_rows, train_rows, note = load()
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), dpi=150)
        fig.patch.set_facecolor(t["surface"])
        panel(axes[0], encode_rows, "Encode 11.4 MB with cl100k_base", "MB/s",
              "throughput, higher is better", theme, lower_is_better=False)
        panel(axes[1], train_rows, "Train BPE on 11.4 MB to vocab 8192", "s",
              "seconds, lower is better", theme, lower_is_better=True)
        fig.text(0.01, 0.02, f"{note} · bar = slowest run, whisker = fastest · output identical to tiktoken",
                 fontsize=9, color=t["ink2"])
        fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=4)
        out = DOCS / f"results-{theme}.png"
        fig.savefig(out, facecolor=t["surface"])
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
