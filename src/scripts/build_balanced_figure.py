#!/usr/bin/env python3
"""Figure for the balanced-federation subsection (tests 02-07).

    python src/scripts/build_balanced_figure.py

Hospital count on the x-axis against macro AUC, one line per aggregation rule, with
the centralised baseline and its equivalence margin behind them. The five same-seed
repeats of test06 are drawn at x=4 so the reader can see that the run-to-run spread
of a single configuration exceeds the apparent trend across configurations.
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
POD3 = REPO / "results" / "_pod3" / "guardado"
POD4 = REPO / "results" / "_pod4" / "keep"
OLD = REPO / "old_runs"
OUT = REPO / "latex" / "figuras"

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#999999"
NOISE = 0.067

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def auc(p: Path) -> float:
    d = json.loads(p.read_text())
    return float(d["auc"]) if "auc" in d else float(d["splits"]["test"]["auc"])


def build() -> None:
    cent = auc(FED / "test01_centralized" / "seed_42" / "results.json")
    fedavg = [auc(FED / n / "test_metrics.json") for n in
              ("test02_fedavg_2h", "test04_fedavg_3h", "test06_fedavg_4h")]
    fedprox = [auc(FED / n / "test_metrics.json") for n in
               ("test03_fedprox_2h", "test05_fedprox_3h", "test07_fedprox_4h")]

    t06 = [auc(FED / "test06_fedavg_4h" / "test_metrics.json"),
           auc(OLD / "test06_fedavg_4h__raw_2026-08-04" / "test_metrics.json"),
           auc(POD3 / "test06_run2" / "test_metrics.json"),
           auc(POD3 / "test06_run3" / "test_metrics.json"),
           auc(POD4 / "test06_run4_pod4" / "test_metrics.json")]
    t07 = [auc(FED / "test07_fedprox_4h" / "test_metrics.json"),
           auc(POD4 / "test07_run2_pod4" / "test_metrics.json")]

    x = np.array([2, 3, 4])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.axhspan(cent - NOISE, cent + NOISE, color=GREY, alpha=0.16, zorder=0)
    ax.axhline(cent, color="black", lw=1.4, zorder=1)

    ax.plot(x, fedavg, "-o", color=BLUE, lw=2, ms=9, zorder=4, label="FedAvg")
    ax.plot(x, fedprox, "-o", color=ORANGE, lw=2, ms=9, zorder=4, label="FedProx")

    ax.set_xticks(x)
    ax.set_xticklabels(["2\n(612 pat./site)", "3\n(408 pat./site)",
                        "4\n(306 pat./site)"])
    ax.set_xlabel("hospitals in the federation")
    ax.set_ylabel("macro AUC")
    ax.set_xlim(1.6, 4.4)
    ax.set_ylim(0.52, 0.70)
    ax.grid(axis="y", color=GREY, alpha=0.28, lw=0.7)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=9, label="FedAvg"),
        plt.Line2D([], [], color=ORANGE, lw=2, marker="o", ms=9, label="FedProx"),
        plt.Line2D([], [], color="black", lw=1.4, label="centralised"),
        plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.16,
                      label=f"$\\pm${NOISE:g} noise floor"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, columnspacing=1.3, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_balanced.{ext}")
    plt.close(fig)
    print("  fig_balanced.pdf / .png")
    print(f"  FedAvg range {max(fedavg) - min(fedavg):.4f} | "
          f"FedProx range {max(fedprox) - min(fedprox):.4f} | "
          f"test06 repeat range {max(t06) - min(t06):.4f}")


if __name__ == "__main__":
    build()
