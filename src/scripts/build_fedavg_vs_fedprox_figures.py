#!/usr/bin/env python3
"""FedAvg against FedProx: the metric/class comparison, and the paired view.

    python src/scripts/build_fedavg_vs_fedprox_figures.py [outdir]

Writes two figures, mirroring the centralised-versus-federated pair:

  fig_avg_vs_prox        three panels -- overall metrics, per-class AUC,
                         per-class recall -- FedAvg mean against FedProx mean over
                         the six matched partitions, with min-max whiskers.
  fig_avg_vs_prox_paired accuracy and macro AUC for each of the six partitions,
                         FedAvg and FedProx side by side.

The six partitions are matched: within each, everything except the aggregation rule
is identical, so a difference is attributable to the proximal term or to noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FED = REPO / "results" / "federated"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

BLUE, ORANGE, GREY, BLACK = "#0072B2", "#E69F00", "#666666", "#000000"
CLASSES = ["HR+/HER2$-$", "TripleNeg", "HER2+"]

PAIRS = [
    ("2 hosp\nbalanced",     "test02_fedavg_2h",          "test03_fedprox_2h"),
    ("3 hosp\nbalanced",     "test04_fedavg_3h",          "test05_fedprox_3h"),
    ("4 hosp\nbalanced",     "test06_fedavg_4h",          "test07_fedprox_4h"),
    ("4 hosp\nskew",         "test08_fedavg_skewed",      "test09_fedprox_skewed"),
    ("3 hosp\ncohort",       "test10_fedavg_cohort",      "test11_fedprox_cohort"),
    ("3 hosp\nsize-matched", "test12_fedavg_sizematched", "test13_fedprox_sizematched"),
]

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    A = [json.loads((FED / a / "test_metrics.json").read_text()) for _, a, _ in PAIRS]
    B = [json.loads((FED / b / "test_metrics.json").read_text()) for _, _, b in PAIRS]
    return A, B


def panel(ax, labels, avg, prox, title, ylim=(0, 0.80)):
    x = np.arange(len(labels))
    w = 0.36
    for off, data, col in [(-w / 2, avg, BLUE), (w / 2, prox, ORANGE)]:
        m = np.array([d.mean() for d in data])
        lo = np.array([d.min() for d in data])
        hi = np.array([d.max() for d in data])
        ax.bar(x + off, m, w, color=col)
        ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black",
                    elinewidth=1.1, capsize=4, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for xi, (a, b) in enumerate(zip(avg, prox)):
        ax.annotate(f"{b.mean() - a.mean():+.3f}", xy=(xi, ylim[1] * 0.945),
                    ha="center", fontsize=9.5)


def fig_metrics(A, B):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5),
                             gridspec_kw={"width_ratios": [5, 3, 3]})
    keys = [("accuracy", "Acc."), ("macro_precision", "Prec."),
            ("macro_recall", "Recall"), ("macro_f1", "F1"), ("auc", "AUC")]
    panel(axes[0], [l for _, l in keys],
          [np.array([a[k] for a in A]) for k, _ in keys],
          [np.array([b[k] for b in B]) for k, _ in keys],
          "Overall (macro-averaged)")
    axes[0].set_ylabel("score")
    panel(axes[1], CLASSES,
          [np.array([a["per_class_auc"][i] for a in A]) for i in range(3)],
          [np.array([b["per_class_auc"][i] for b in B]) for i in range(3)],
          "AUC per class")
    panel(axes[2], CLASSES,
          [np.array([a["per_class_recall"][i] for a in A]) for i in range(3)],
          [np.array([b["per_class_recall"][i] for b in B]) for i in range(3)],
          "Recall per class")

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg (mean of 6)"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx (mean of 6)"),
               plt.Line2D([], [], color="black", lw=1.1, label="min--max")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_avg_vs_prox.{ext}")
    plt.close(fig)


def fig_paired(A, B):
    labels = [p[0] for p in PAIRS]
    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True)
    for ax, key, title, lim in [(axes[0], "accuracy", "Accuracy", (0.40, 0.53)),
                                (axes[1], "auc", "Macro AUC", (0.52, 0.64))]:
        a = np.array([d[key] for d in A])
        b = np.array([d[key] for d in B])
        ax.bar(x - w / 2, a, w, color=BLUE)
        ax.bar(x + w / 2, b, w, color=ORANGE)
        ax.set_ylim(*lim)
        ax.set_ylabel(title)
        ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
        for xi, (u, v) in enumerate(zip(a, b)):
            ax.annotate(f"{v - u:+.3f}", xy=(xi, lim[1] - 0.012), ha="center",
                        fontsize=9.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("partition")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_avg_vs_prox_paired.{ext}")
    plt.close(fig)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = load()
    fig_metrics(A, B)
    fig_paired(A, B)
    print(f"  escrito em {OUT}/fig_avg_vs_prox{{,_paired}}.{{pdf,png}}")
    d = [b["auc"] - a["auc"] for a, b in zip(A, B)]
    print(f"  delta macro AUC por par: {[round(x, 4) for x in d]}")
    print(f"  media {np.mean(d):+.4f} | FedProx maior em {sum(x > 0 for x in d)}/6")


if __name__ == "__main__":
    build()
