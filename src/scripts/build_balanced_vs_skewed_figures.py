#!/usr/bin/env python3
"""Balanced against quantity-skewed partitions (tests 06/07 vs 08/09).

    python src/scripts/build_balanced_vs_skewed_figures.py [outdir]

Writes two figures:

  fig_bal_vs_skew         three panels -- overall metrics, per-class AUC, per-class
                          recall -- balanced against skewed, one bar pair per
                          aggregation rule.
  fig_bal_vs_skew_sites   macro AUC at each hospital, against the number of training
                          patients that hospital holds. This is where quantity skew is
                          visible: the balanced runs give every site 305-306 patients,
                          the skewed runs 135-678.

Both partitions are stratified to within 0.43 percentage points of class share, so the
only thing that differs between them is how much data each site holds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
FED = REPO / "results" / "federated"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

BLUE, ORANGE, GREY, BLACK = "#0072B2", "#E69F00", "#666666", "#000000"
CLASSES = ["HR+/HER2$-$", "TripleNeg", "HER2+"]

BAL = {"FedAvg": "test06_fedavg_4h", "FedProx": "test07_fedprox_4h"}
SKEW = {"FedAvg": "test08_fedavg_skewed", "FedProx": "test09_fedprox_skewed"}
IDS = {"test06": "06", "test07": "07", "test08": "08", "test09": "09"}

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(d):
    return {k: json.loads((FED / v / "test_metrics.json").read_text())
            for k, v in d.items()}


def panel(ax, labels, bal, skew, title, ylim=(0, 0.80)):
    """Four bars per label: balanced/skewed x FedAvg/FedProx."""
    x = np.arange(len(labels))
    w = 0.2
    for off, vals, col, hatch in [
            (-1.5 * w, bal["FedAvg"], BLUE, None),
            (-0.5 * w, skew["FedAvg"], BLUE, "///"),
            (0.5 * w, bal["FedProx"], ORANGE, None),
            (1.5 * w, skew["FedProx"], ORANGE, "///")]:
        ax.bar(x + off, vals, w, color=col, hatch=hatch, edgecolor="white",
               linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)


def fig_metrics():
    B, S = load(BAL), load(SKEW)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5),
                             gridspec_kw={"width_ratios": [5, 3, 3]})
    keys = [("accuracy", "Acc."), ("macro_precision", "Prec."),
            ("macro_recall", "Recall"), ("macro_f1", "F1"), ("auc", "AUC")]
    panel(axes[0], [l for _, l in keys],
          {a: [B[a][k] for k, _ in keys] for a in B},
          {a: [S[a][k] for k, _ in keys] for a in S},
          "Overall (macro-averaged)")
    axes[0].set_ylabel("score")
    panel(axes[1], CLASSES,
          {a: B[a]["per_class_auc"] for a in B},
          {a: S[a]["per_class_auc"] for a in S}, "AUC per class")
    panel(axes[2], CLASSES,
          {a: B[a]["per_class_recall"] for a in B},
          {a: S[a]["per_class_recall"] for a in S}, "Recall per class")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg, balanced"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="///", ec="white",
                      label="FedAvg, skewed"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx, balanced"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="///", ec="white",
                      label="FedProx, skewed"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               ncol=4, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_bal_vs_skew.{ext}")
    plt.close(fig)


def fig_sites():
    """Per-hospital bars, laid out exactly like fig_cohort_sites so the two
    subsections read the same way."""
    pc = pd.read_csv(FED / "final_summary" / "per_client_metrics.csv")

    def series(test, col):
        s = pc[pc.experiment == test].sort_values("site")
        return s[col].to_numpy()

    # site labels carry both patient counts, because the counts ARE the treatment
    bal_n = series("test06", "n_train_patients")
    skew_n = series("test08", "n_train_patients")
    labels = [f"hospital\\_{i + 1}\n{b} / {s}"
              for i, (b, s) in enumerate(zip(bal_n, skew_n))]
    labels = [l.replace("\\_", "_") for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = np.arange(4)
    w = 0.2
    for ax, col, title, ylim in [
            (axes[0], "macro_auc", "Macro AUC", (0, 0.85)),
            (axes[1], "recall_HER2pos", "Recall, HER2+", (0, 0.85))]:
        for off, t, c, hatch in [(-1.5 * w, "test08", BLUE, "///"),
                                 (-0.5 * w, "test06", BLUE, None),
                                 (0.5 * w, "test09", ORANGE, "///"),
                                 (1.5 * w, "test07", ORANGE, None)]:
            vals = series(t, col)
            ax.bar(x + off, vals, w, color=c, hatch=hatch, edgecolor="white",
                   linewidth=0.6)
            for xi, v in zip(x + off, vals):
                if v == 0:
                    ax.annotate("0", xy=(xi, 0.012), ha="center", va="bottom",
                                fontsize=10, color=c, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=8)
        ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("score on the hospital's own validation set")
    for ax in axes:
        ax.set_xlabel("hospital, with its training patients "
                      "(balanced / skewed)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="///", ec="white",
                      label="FedAvg, skewed"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg, balanced"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="///", ec="white",
                      label="FedProx, skewed"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx, balanced"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_bal_vs_skew_sites.{ext}")
    plt.close(fig)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_metrics()
    fig_sites()
    print(f"  escrito em {OUT}/fig_bal_vs_skew{{,_sites}}.{{pdf,png}}")
    B, S = load(BAL), load(SKEW)
    for a in B:
        print(f"  {a}: macro AUC {B[a]['auc']:.4f} -> {S[a]['auc']:.4f}  "
              f"{S[a]['auc'] - B[a]['auc']:+.4f}")
    pc = pd.read_csv(FED / "final_summary" / "per_client_metrics.csv")
    for t in ["test06", "test07", "test08", "test09"]:
        v = pc[pc.experiment == t].macro_auc
        print(f"  {t}: site spread {v.max() - v.min():.4f}")


if __name__ == "__main__":
    build()
