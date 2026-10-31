"""Draw the F1 and AUPRC figures from results/metrics.csv.

Usage:
    python scripts/08_figures.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CONDITIONS = ["zero-shot", "gold", "self", "random"]
CONDITION_LABELS = ["zero-\nshot", "gold", "self", "random"]
MODELS = {"qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B", "gemma3-4b": "Gemma-3-4B"}
LANGS = {
    "tr": ("Turkish", "#2a78d6"),
    "en": ("English (MT)", "#eb6834"),
    "en-self": ("English (self-translated)", "#199e70"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def draw(metrics: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    # Local models only; the frontier LLMs have no few-shot runs.
    models = [m for m in MODELS if m in set(metrics["system"])]
    lexicon = metrics.loc[metrics["system"] == "lexicon", metric]
    has_ci = f"{metric}_ci_low" in metrics

    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 2.2), sharey=True, squeeze=False)
    x = np.arange(len(CONDITIONS))
    width = 0.8 / len(LANGS)
    offsets = (np.arange(len(LANGS)) - (len(LANGS) - 1) / 2) * width

    for ax, model in zip(axes[0], models):
        for offset, (lang, (_, color)) in zip(offsets, LANGS.items()):
            rows = (metrics[(metrics["system"] == model) & (metrics["lang"] == lang)]
                    .set_index("condition").reindex(CONDITIONS))
            values = rows[metric].to_numpy()
            ax.bar(x + offset, values, width * 0.94, color=color, zorder=2)
            if has_ci:
                err = np.vstack([values - rows[f"{metric}_ci_low"], rows[f"{metric}_ci_high"] - values])
                ax.errorbar(x + offset, values, yerr=err, fmt="none", ecolor=INK,
                            elinewidth=0.8, capsize=1.5, zorder=3)
        if not lexicon.empty:
            ax.axhline(lexicon.iloc[0], color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.set_title(MODELS[model], color=INK, fontsize=8)
        ax.set_xticks(x, CONDITION_LABELS, fontsize=7)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_ylim(0, 0.6)

    axes[0][0].set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in LANGS.values()]
    labels = [name for name, _ in LANGS.values()]
    if not lexicon.empty:
        handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1))
        labels.append("keyword lexicon")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the paper figures.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    metrics = pd.read_csv(args.results / "metrics.csv")
    # Items scored in every condition, so the bars within a panel are comparable.
    metrics = metrics[metrics["subset"] == "common"]
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    draw(metrics, "f1", "F1 (high-impact)", fig_dir / "f1.pdf")
    draw(metrics, "auprc", "AUPRC", fig_dir / "auprc.pdf")


if __name__ == "__main__":
    main()
