#!/usr/bin/env python3
"""Cohort heterogeneity against its size-matched control (tests 10-13).

    python src/scripts/build_cohort_figures.py [outdir]

Writes two figures:

  fig_cohort        three panels -- overall metrics, per-class AUC, per-class recall --
                    cohort-native against size-matched, one bar pair per aggregation
                    rule.
  fig_cohort_sites  macro AUC and HER2+ recall at each of the three hospitals. The
                    sites hold identical patient counts in both partitions (642, 101,
                    784), so any difference is cohort identity rather than site size.
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

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"
CLASSES = ["HR+/HER2$-$", "TripleNeg", "HER2+"]

COHORT = {"FedAvg": "test10_fedavg_cohort", "FedProx": "test11_fedprox_cohort"}
MIXED = {"FedAvg": "test12_fedavg_sizematched",
         "FedProx": "test13_fedprox_sizematched"}
SITES = ["hospital_1\nDUKE / mixed", "hospital_2\nI-SPY1 / mixed",
         "hospital_3\nI-SPY2 / mixed"]

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(d):
    return {k: json.loads((FED / v / "test_metrics.json").read_text())
            for k, v in d.items()}


def panel(ax, labels, coh, mix, title, ylim=(0, 0.80)):
    x = np.arange(len(labels))
    w = 0.2
    for off, vals, col, hatch in [
            (-1.5 * w, coh["FedAvg"], BLUE, "///"),
            (-0.5 * w, mix["FedAvg"], BLUE, None),
            (0.5 * w, coh["FedProx"], ORANGE, "///"),
            (1.5 * w, mix["FedProx"], ORANGE, None)]:
        ax.bar(x + off, vals, w, color=col, hatch=hatch, edgecolor="white",
               linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)


def fig_metrics():
    C, M = load(COHORT), load(MIXED)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5),
                             gridspec_kw={"width_ratios": [5, 3, 3]})
    keys = [("accuracy", "Acc."), ("macro_precision", "Prec."),
            ("macro_recall", "Recall"), ("macro_f1", "F1"), ("auc", "AUC")]
    panel(axes[0], [l for _, l in keys],
          {a: [C[a][k] for k, _ in keys] for a in C},
          {a: [M[a][k] for k, _ in keys] for a in M},
          "Overall (macro-averaged)")
    axes[0].set_ylabel("score")
    panel(axes[1], CLASSES, {a: C[a]["per_class_auc"] for a in C},
          {a: M[a]["per_class_auc"] for a in M}, "AUC per class")
    panel(axes[2], CLASSES, {a: C[a]["per_class_recall"] for a in C},
          {a: M[a]["per_class_recall"] for a in M}, "Recall per class")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="///", ec="white",
                      label="FedAvg, one cohort per site"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg, size-matched"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="///", ec="white",
                      label="FedProx, one cohort per site"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx, size-matched"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cohort.{ext}")
    plt.close(fig)


def fig_sites():
    pc = pd.read_csv(FED / "final_summary" / "per_client_metrics.csv")

    def series(test, col):
        s = pc[pc.experiment == test].sort_values("site")
        return s[col].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = np.arange(3)
    w = 0.2
    for ax, col, title, ylim in [
            (axes[0], "macro_auc", "Macro AUC", (0, 0.85)),
            (axes[1], "recall_HER2pos", "Recall, HER2+", (0, 0.85))]:
        for off, t, c, hatch in [(-1.5 * w, "test10", BLUE, "///"),
                                 (-0.5 * w, "test12", BLUE, None),
                                 (0.5 * w, "test11", ORANGE, "///"),
                                 (1.5 * w, "test13", ORANGE, None)]:
            vals = series(t, col)
            ax.bar(x + off, vals, w, color=c, hatch=hatch,
                   edgecolor="white", linewidth=0.6)
            # a zero-height bar is indistinguishable from a missing one, and the
            # single most important value in this figure -- HER2+ recall at the
            # I-SPY1 site under FedAvg -- is exactly zero. Label it.
            for xi, v in zip(x + off, vals):
                if v == 0:
                    ax.annotate("0", xy=(xi, 0.012), ha="center", va="bottom",
                                fontsize=10, color=c, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SITES)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=8)
        ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("score on the hospital's own validation set")
    for ax in axes:
        ax.set_xlabel("hospital, with the data it holds "
                      "(cohort-native / size-matched)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="///", ec="white",
                      label="FedAvg, one cohort per site"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg, size-matched"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="///", ec="white",
                      label="FedProx, one cohort per site"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx, size-matched"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_cohort_sites.{ext}")
    plt.close(fig)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_metrics()
    fig_sites()
    print(f"  escrito em {OUT}/fig_cohort{{,_sites}}.{{pdf,png}}")
    C, M = load(COHORT), load(MIXED)
    for a in C:
        print(f"  {a}: cohort {C[a]['auc']:.4f} vs mixed {M[a]['auc']:.4f}  "
              f"{C[a]['auc'] - M[a]['auc']:+.4f}")
        print(f"          HER2+ recall {C[a]['per_class_recall'][2]:.4f} vs "
              f"{M[a]['per_class_recall'][2]:.4f}")


if __name__ == "__main__":
    build()
