"""Generate publication-quality figures for the arXiv preprint.

Run from the repo root:
    python paper/generate_figures.py

Outputs to paper/figures/
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODELS = ["Qwen-2.5-7B", "Llama-3.1-8B", "Mistral-7B"]
MODEL_COLORS = {"Qwen-2.5-7B": "#2176AE", "Llama-3.1-8B": "#E8553A", "Mistral-7B": "#57A773"}
MODEL_MARKERS = {"Qwen-2.5-7B": "o", "Llama-3.1-8B": "s", "Mistral-7B": "^"}


# ── Data ──────────────────────────────────────────────────────────

# Learning curve data from the JSON files
LC_DATA = {
    "Qwen-2.5-7B": {
        "logprob": 0.714,
        "curve": [(25, 0.757), (50, 0.511), (100, 0.609), (250, 0.654),
                  (500, 0.653), (750, 0.616), (881, 0.620)],
    },
    "Llama-3.1-8B": {
        "logprob": 0.650,
        "curve": [(25, 0.624), (50, 0.470), (100, 0.692), (250, 0.681),
                  (500, 0.623), (750, 0.631), (998, 0.646)],
    },
    "Mistral-7B": {
        "logprob": 0.678,
        "curve": [(25, 0.484), (50, 0.624), (100, 0.671), (250, 0.566),
                  (500, 0.650), (750, 0.648), (990, 0.649)],
    },
}

# Cross-dataset data
CROSS_DATASET = {
    "Qwen-2.5-7B":  {"mmlu_logprob": 0.714, "trivia_logprob": 0.828,
                      "mmlu_rllm": 0.665, "trivia_rllm": 0.564},
    "Llama-3.1-8B":  {"mmlu_logprob": 0.650, "trivia_logprob": 0.800,
                      "mmlu_rllm": 0.644, "trivia_rllm": 0.512},
    "Mistral-7B":    {"mmlu_logprob": 0.678, "trivia_logprob": 0.717,
                      "mmlu_rllm": 0.676, "trivia_rllm": 0.562},
}

# Per-signal AUROC for the bar chart
SIGNAL_AUROC = {
    "logprob":  [0.714, 0.650, 0.678],
    "GSA v3":   [0.562, 0.614, 0.638],
    "SC":       [0.504, 0.594, 0.604],
    "KS":       [0.426, 0.413, 0.422],
    "QC":       [0.524, 0.521, 0.522],
    "RouteLLM": [0.665, 0.644, 0.676],
}


# ── Figure 1: Cross-dataset transfer scatter ──────────────────────

def fig1_cross_dataset():
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))

    # Diagonal reference
    ax.plot([0.4, 0.9], [0.4, 0.9], "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.fill_between([0.4, 0.9], [0.4, 0.9], [0.9, 0.9], alpha=0.05, color="green")
    ax.fill_between([0.4, 0.9], [0.4, 0.4], [0.4, 0.9], alpha=0.05, color="red")
    ax.text(0.48, 0.86, "better on\nTriviaQA", fontsize=7, color="green", alpha=0.6, ha="center")
    ax.text(0.84, 0.46, "worse on\nTriviaQA", fontsize=7, color="red", alpha=0.6, ha="center")

    for model in MODELS:
        d = CROSS_DATASET[model]
        c = MODEL_COLORS[model]
        m = MODEL_MARKERS[model]
        # logprob point
        ax.scatter(d["mmlu_logprob"], d["trivia_logprob"], c=c, marker=m,
                   s=80, zorder=3, edgecolors="black", linewidths=0.5,
                   label=f"{model} (logprob)")
        # RouteLLM point
        ax.scatter(d["mmlu_rllm"], d["trivia_rllm"], marker=m,
                   s=80, zorder=3, edgecolors=c, linewidths=1.5,
                   facecolors="none", label=f"{model} (RouteLLM)")

    ax.set_xlabel("AUROC on MMLU-Pro")
    ax.set_ylabel("AUROC on TriviaQA")
    ax.set_xlim(0.45, 0.88)
    ax.set_ylim(0.45, 0.88)
    ax.set_aspect("equal")
    # Place legend below the plot to avoid overlapping data points
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=6.5, ncol=2, framealpha=0.9, columnspacing=1.0)
    ax.set_title("Cross-Dataset Transfer")
    ax.grid(alpha=0.2)

    fig.savefig(f"{OUT}/fig1_cross_dataset.pdf")
    fig.savefig(f"{OUT}/fig1_cross_dataset.png")
    plt.close(fig)
    print(f"  Wrote {OUT}/fig1_cross_dataset.pdf")


# ── Figure 2: RouteLLM learning curve ─────────────────────────────

def fig2_learning_curve():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    for model in MODELS:
        d = LC_DATA[model]
        c = MODEL_COLORS[model]
        m = MODEL_MARKERS[model]
        ns = [p[0] for p in d["curve"]]
        aurocs = [p[1] for p in d["curve"]]

        # RouteLLM curve
        ax.plot(ns, aurocs, color=c, marker=m, markersize=5, linewidth=1.2,
                label=f"RouteLLM ({model})", zorder=2)
        # logprob flat line
        ax.axhline(y=d["logprob"], color=c, linestyle="--", linewidth=1.0,
                   alpha=0.7, zorder=1)
        # Label the flat line
        ax.text(1050, d["logprob"] + 0.008, f'logprob {d["logprob"]:.3f}',
                fontsize=7, color=c, va="bottom")

    ax.set_xlabel("RouteLLM Training Examples ($N$)")
    ax.set_ylabel("AUROC")
    ax.set_xlim(0, 1250)
    ax.set_ylim(0.40, 0.80)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_title("Supervised Learning Curve vs. Zero-Shot Logprob")
    ax.grid(alpha=0.2)

    fig.savefig(f"{OUT}/fig2_learning_curve.pdf")
    fig.savefig(f"{OUT}/fig2_learning_curve.png")
    plt.close(fig)
    print(f"  Wrote {OUT}/fig2_learning_curve.pdf")


# ── Figure 3: Per-signal AUROC grouped bar chart ──────────────────

def fig3_signal_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.2))

    signals = list(SIGNAL_AUROC.keys())
    x = np.arange(len(signals))
    width = 0.22

    for i, model in enumerate(MODELS):
        vals = [SIGNAL_AUROC[s][i] for s in signals]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=MODEL_COLORS[model], edgecolor="black", linewidth=0.3)
        # Value labels on top
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(-0.4, 0.505, "chance", fontsize=7, color="gray", alpha=0.7)

    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=9)
    ax.set_ylim(0.35, 0.80)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Per-Signal AUROC on MMLU-Pro")
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(f"{OUT}/fig3_signal_comparison.pdf")
    fig.savefig(f"{OUT}/fig3_signal_comparison.png")
    plt.close(fig)
    print(f"  Wrote {OUT}/fig3_signal_comparison.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_cross_dataset()
    fig2_learning_curve()
    fig3_signal_comparison()
    print("Done.")
