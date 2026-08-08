#!/usr/bin/env python3
"""Per-hospital figures for both partition comparisons, in one layout.

    python src/scripts/build_site_figures.py [outdir]

For each of the two comparisons -- balanced against quantity-skewed (tests 06/07 vs
08/09) and cohort-native against size-matched (tests 10/11 vs 12/13) -- it writes:

  fig_<name>_sites        two panels: macro AUC and accuracy, per hospital.
  fig_<name>_sites_class  six panels: AUC per class (top row) and recall per class
                          (bottom row), per hospital.

Both comparisons use identical layout, colours and hatching so the two subsections of
the chapter read the same way. Hatched bars are always the treatment (skewed /
cohort-native); solid bars are always the control (balanced / size-matched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
PC = REPO / "results" / "federated" / "final_summary" / "per_client_metrics.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"
CLS = ["HRposHER2neg", "TripleNeg", "HER2pos"]
CLS_LABEL = ["HR+/HER2$-$", "TripleNeg", "HER2+"]

# name -> (treatment fedavg, control fedavg, treatment fedprox, control fedprox,
#          treatment label, control label, x-axis explanation, site sublabels)
COMPARISONS = {
    "bal_vs_skew": dict(
        t_avg="test08", c_avg="test06", t_prox="test09", c_prox="test07",
        t_lab="skewed", c_lab="balanced",
        xlabel="hospital, with its training patients (balanced / skewed)",
        sub=lambda pc: [f"{b} / {s}" for b, s in zip(
            pc[pc.experiment == "test06"].sort_values("site").n_train_patients,
            pc[pc.experiment == "test08"].sort_values("site").n_train_patients)]),
    "cohort": dict(
        t_avg="test10", c_avg="test12", t_prox="test11", c_prox="test13",
        t_lab="one cohort per site", c_lab="size-matched",
        xlabel="hospital, with the data it holds (cohort-native / size-matched)",
        sub=lambda pc: ["DUKE / mixed", "I-SPY1 / mixed", "I-SPY2 / mixed"]),
}

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10.5,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def bars(ax, pc, spec, col, ylim):
    """Four grouped bars per hospital for one metric column."""
    def series(t):
        return pc[pc.experiment == t].sort_values("site")[col].to_numpy()

    n = len(series(spec["t_avg"]))
    x = np.arange(n)
    w = 0.2
    for off, t, c, hatch in [(-1.5 * w, spec["t_avg"], BLUE, "///"),
                             (-0.5 * w, spec["c_avg"], BLUE, None),
                             (0.5 * w, spec["t_prox"], ORANGE, "///"),
                             (1.5 * w, spec["c_prox"], ORANGE, None)]:
        v = series(t)
        ax.bar(x + off, v, w, color=c, hatch=hatch, edgecolor="white",
               linewidth=0.6)
        for xi, val in zip(x + off, v):
            if val == 0:                       # a zero bar reads as a missing bar
                ax.annotate("0", xy=(xi, ylim[1] * 0.015), ha="center",
                            va="bottom", fontsize=9.5, color=c, fontweight="bold")
    ax.set_xticks(x)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    return x, n


def legend(fig, spec, ncol=2, y=-0.09):
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="///", ec="white",
                       label=f"FedAvg, {spec['t_lab']}"),
         plt.Rectangle((0, 0), 1, 1, color=BLUE, label=f"FedAvg, {spec['c_lab']}"),
         plt.Rectangle((0, 0), 1, 1, color=ORANGE, hatch="///", ec="white",
                       label=f"FedProx, {spec['t_lab']}"),
         plt.Rectangle((0, 0), 1, 1, color=ORANGE, label=f"FedProx, {spec['c_lab']}")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False)


def build_one(name, spec, pc):
    ticks = [f"hospital\\_{i + 1}\n{s}" for i, s in enumerate(spec["sub"](pc))]
    ticks = [t.replace("\\_", "_") for t in ticks]

    # ---- overall: macro AUC and accuracy ---------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, col, title in [(axes[0], "macro_auc", "Macro AUC"),
                           (axes[1], "accuracy", "Accuracy")]:
        bars(ax, pc, spec, col, (0, 0.85))
        ax.set_xticklabels(ticks)
        ax.set_title(title, pad=8)
        ax.set_xlabel(spec["xlabel"])
    axes[0].set_ylabel("score on the hospital's own validation set")
    legend(fig, spec)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_{name}_sites.{ext}")
    plt.close(fig)

    # ---- per class: AUC on top, recall below ------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharey="row")
    for j, (c, lab) in enumerate(zip(CLS, CLS_LABEL)):
        bars(axes[0, j], pc, spec, f"auc_{c}", (0, 0.85))
        axes[0, j].set_title(f"AUC --- {lab}", pad=8)
        axes[0, j].set_xticklabels([])
        bars(axes[1, j], pc, spec, f"recall_{c}", (0, 0.85))
        axes[1, j].set_title(f"Recall --- {lab}", pad=8)
        axes[1, j].set_xticklabels(ticks)
    axes[0, 0].set_ylabel("AUC")
    axes[1, 0].set_ylabel("recall")
    # one shared x-label: three copies of a sentence this long collide
    fig.supxlabel(spec["xlabel"], y=0.035, fontsize=12)
    legend(fig, spec, ncol=4, y=-0.035)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_{name}_sites_class.{ext}")
    plt.close(fig)
    print(f"  fig_{name}_sites.{{pdf,png}} · fig_{name}_sites_class.{{pdf,png}}")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pc = pd.read_csv(PC)
    for name, spec in COMPARISONS.items():
        build_one(name, spec, pc)


if __name__ == "__main__":
    build()
