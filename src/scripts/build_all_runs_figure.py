#!/usr/bin/env python3
"""Accuracy and macro AUC of all thirteen runs, centralised included.

    python src/scripts/build_all_runs_figure.py [outdir]

Two panels sharing one x-axis of thirteen experiments. The centralised baseline is the
first bar and is repeated as a dashed reference line across both panels, so the reader
can see which federated runs sit above and below it on each metric independently.

Writes fig_all_runs.{pdf,png} to `outdir` (default: the working directory).
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

NAMES = {2: "test02_fedavg_2h", 3: "test03_fedprox_2h", 4: "test04_fedavg_3h",
         5: "test05_fedprox_3h", 6: "test06_fedavg_4h", 7: "test07_fedprox_4h",
         8: "test08_fedavg_skewed", 9: "test09_fedprox_skewed",
         10: "test10_fedavg_cohort", 11: "test11_fedprox_cohort",
         12: "test12_fedavg_sizematched", 13: "test13_fedprox_sizematched"}
FEDAVG = {2, 4, 6, 8, 10, 12}

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def build() -> None:
    c = json.loads((FED / "test01_centralized" / "seed_42" / "results.json")
                   .read_text())["splits"]["test"]
    ids = [1] + list(NAMES)
    acc = [c["accuracy"]]
    auc = [c["auc"]]
    for i in NAMES:
        d = json.loads((FED / NAMES[i] / "test_metrics.json").read_text())
        acc.append(d["accuracy"])
        auc.append(d["auc"])

    colours = [GREY] + [BLUE if i in FEDAVG else ORANGE for i in NAMES]
    labels = [f"{i:02d}" for i in ids]
    x = np.arange(len(ids))

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)

    for ax, vals, ref, title, lo, hi in [
            (axes[0], acc, c["accuracy"], "Accuracy", 0.38, 0.56),
            (axes[1], auc, c["auc"], "Macro AUC", 0.50, 0.65)]:
        ax.bar(x, vals, 0.66, color=colours)
        ax.axhline(ref, color=BLACK, lw=1.2, ls="--", zorder=4)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(title)
        ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("experiment")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREY, label="Centralised"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="FedAvg"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="FedProx"),
        plt.Line2D([], [], color=BLACK, lw=1.2, ls="--",
                   label="centralised reference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, frameon=False)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_all_runs.{ext}")
    plt.close(fig)
    print(f"  escrito em {OUT}/fig_all_runs.{{pdf,png}}")
    a, u = np.array(acc[1:]), np.array(auc[1:])
    print(f"  accuracy : cent {acc[0]:.4f} | federados {a.min():.4f}-{a.max():.4f} "
          f"| acima do cent: {(a > acc[0]).sum()}/12")
    print(f"  macro AUC: cent {auc[0]:.4f} | federados {u.min():.4f}-{u.max():.4f} "
          f"| acima do cent: {(u > auc[0]).sum()}/12")


if __name__ == "__main__":
    build()
