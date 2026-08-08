#!/usr/bin/env python3
"""Accuracy counterpart of `fig_balanced` (tests 02-07).

    python src/scripts/build_balanced_accuracy_figure.py

Same axes as the macro-AUC figure, but plotting accuracy against the reference that
makes accuracy interpretable: the trivial majority-class rate on the global test set.

NO NOISE-FLOOR BAND IS DRAWN HERE, DELIBERATELY. The 0.067 floor was measured on
macro AUC and does not transfer to accuracy; shading a band of that width on this
axis would assert something never measured. The reference lines are the two rates
that are facts of the split -- always predicting HR+/HER2- (0.5112) and the
centralised run (0.5299).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FED = REPO / "results" / "federated"
OUT = REPO / "latex" / "figuras"

BLUE, ORANGE, GREY, VERM = "#0072B2", "#E69F00", "#999999", "#D55E00"

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def metrics(p: Path) -> dict:
    d = json.loads(p.read_text())
    return d if "accuracy" in d else d["splits"]["test"]


def build() -> None:
    cent = metrics(FED / "test01_centralized" / "seed_42" / "results.json")
    trivial = float(cent["trivial_baseline_accuracy"])

    fedavg = [metrics(FED / n / "test_metrics.json")["accuracy"] for n in
              ("test02_fedavg_2h", "test04_fedavg_3h", "test06_fedavg_4h")]
    fedprox = [metrics(FED / n / "test_metrics.json")["accuracy"] for n in
               ("test03_fedprox_2h", "test05_fedprox_3h", "test07_fedprox_4h")]

    x = np.array([2, 3, 4])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # everything below this line is worse than a constant rule
    ax.axhspan(0.30, trivial, color=VERM, alpha=0.07, zorder=0)
    ax.axhline(trivial, color=VERM, lw=1.8, ls="--", zorder=1)
    ax.axhline(cent["accuracy"], color="black", lw=1.4, zorder=1)

    ax.plot(x, fedavg, "-o", color=BLUE, lw=2, ms=9, zorder=4)
    ax.plot(x, fedprox, "-o", color=ORANGE, lw=2, ms=9, zorder=4)

    ax.annotate("trivial baseline (always HR+/HER2$-$)",
                xy=(1.68, trivial + 0.004), fontsize=11, color=VERM, va="bottom")
    ax.annotate("centralised", xy=(1.68, cent["accuracy"] + 0.004),
                fontsize=11, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(["2\n(612 pat./site)", "3\n(408 pat./site)",
                        "4\n(306 pat./site)"])
    ax.set_xlabel("hospitals in the federation")
    ax.set_ylabel("accuracy")
    ax.set_xlim(1.6, 4.4)
    ax.set_ylim(0.36, 0.57)
    ax.grid(axis="y", color=GREY, alpha=0.28, lw=0.7)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=9, label="FedAvg"),
        plt.Line2D([], [], color=ORANGE, lw=2, marker="o", ms=9, label="FedProx"),
        plt.Line2D([], [], color="black", lw=1.4, label="centralised"),
        plt.Line2D([], [], color=VERM, lw=1.8, ls="--", label="trivial baseline"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, columnspacing=1.3, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_balanced_accuracy.{ext}")
    plt.close(fig)
    print("  fig_balanced_accuracy.pdf / .png")
    print(f"  trivial {trivial:.4f} | centralised {cent['accuracy']:.4f}")
    print(f"  FedAvg  {[round(v, 4) for v in fedavg]}")
    print(f"  FedProx {[round(v, 4) for v in fedprox]}")
    print(f"  runs below the trivial baseline: "
          f"{sum(v < trivial for v in fedavg + fedprox)}/6")


if __name__ == "__main__":
    build()
