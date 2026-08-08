#!/usr/bin/env python3
"""Centralised versus federated, across every metric and every class.

    python src/scripts/build_central_vs_fed_figure.py [outdir]

Three panels: overall macro metrics, per-class AUC, per-class recall. The federated
bar is the mean of the twelve runs and the whisker is their full min-max range, so the
reader can see at once whether the centralised bar sits outside that range.

Writes fig_central_vs_fed.{pdf,png} to `outdir` (default: the scratchpad, never the
dissertation's own folder).
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

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#999999"
CLASSES = ["HR+/HER2$-$", "TripleNeg", "HER2+"]
NAMES = {2: "test02_fedavg_2h", 3: "test03_fedprox_2h", 4: "test04_fedavg_3h",
         5: "test05_fedprox_3h", 6: "test06_fedavg_4h", 7: "test07_fedprox_4h",
         8: "test08_fedavg_skewed", 9: "test09_fedprox_skewed",
         10: "test10_fedavg_cohort", 11: "test11_fedprox_cohort",
         12: "test12_fedavg_sizematched", 13: "test13_fedprox_sizematched"}

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlesize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel(ax, labels, cent, fed, title):
    """One grouped-bar panel. `fed` is a list of arrays, one per label."""
    x = np.arange(len(labels))
    w = 0.36
    mean = np.array([f.mean() for f in fed])
    lo = np.array([f.min() for f in fed])
    hi = np.array([f.max() for f in fed])

    ax.bar(x - w / 2, cent, w, color=GREY, label="Centralised")
    ax.bar(x + w / 2, mean, w, color=BLUE, label="Federated (mean of 12)")
    ax.errorbar(x + w / 2, mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor="black", elinewidth=1.1, capsize=4, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.80)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for xi, (c, m) in enumerate(zip(cent, mean)):
        ax.annotate(f"{m - c:+.3f}", xy=(xi, 0.755), ha="center", fontsize=9.5,
                    color="black")


def build() -> None:
    c = json.loads((FED / "test01_centralized" / "seed_42" / "results.json")
                   .read_text())["splits"]["test"]
    F = [json.loads((FED / n / "test_metrics.json").read_text())
         for n in NAMES.values()]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5),
                             gridspec_kw={"width_ratios": [5, 3, 3]})

    keys = [("accuracy", "Acc."), ("macro_precision", "Prec."),
            ("macro_recall", "Recall"), ("macro_f1", "F1"), ("auc", "AUC")]
    panel(axes[0], [l for _, l in keys],
          [c[k] for k, _ in keys],
          [np.array([f[k] for f in F]) for k, _ in keys],
          "Overall (macro-averaged)")
    axes[0].set_ylabel("score")

    panel(axes[1], CLASSES, c["per_class_auc"],
          [np.array([f["per_class_auc"][i] for f in F]) for i in range(3)],
          "AUC per class")
    panel(axes[2], CLASSES, c["per_class_recall"],
          [np.array([f["per_class_recall"][i] for f in F]) for i in range(3)],
          "Recall per class")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREY, label="Centralised"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Federated (mean of 12)"),
        plt.Line2D([], [], color="black", lw=1.1, label="federated min--max"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_central_vs_fed.{ext}")
    plt.close(fig)
    print(f"  escrito em {OUT}/fig_central_vs_fed.{{pdf,png}}")
    for i, cl in enumerate(["HR+/HER2-", "TripleNeg", "HER2+"]):
        a = np.array([f["per_class_auc"][i] for f in F])
        r = np.array([f["per_class_recall"][i] for f in F])
        print(f"  {cl:11s} AUC {a.mean() - c['per_class_auc'][i]:+.4f} | "
              f"recall {r.mean() - c['per_class_recall'][i]:+.4f}  (fed menos cent)")


if __name__ == "__main__":
    build()
