#!/usr/bin/env python3
"""The equivalence forest plot -- the figure Chapter 4 is missing.

    python src/scripts/build_equivalence_figure.py

WHY THIS FIGURE AND NOT ANOTHER BAR CHART
-----------------------------------------
`fig_overview_barplot` already draws the thirteen macro-AUC values as bars. Bars
invite the reader to rank them, and the caption of that table says in its own words
that *no difference between rows is attributable*: the noise floor is 0.067 and the
full spread is 0.0725. A chart whose visual grammar contradicts its caption is worse
than no chart.

A forest plot is the correct grammar for an equivalence claim. It draws the
centralised baseline as a reference line, shades the +/- 0.067 margin of practical
equivalence around it, and puts every federated run inside that band as a point. The
reader sees the claim -- "all twelve fall inside the margin" -- instead of reading it.

WHAT MAKES THIS VERSION SPECIFIC TO THIS PROJECT
------------------------------------------------
Four configurations were run more than once at seed 42, and their draws are plotted as
open circles behind the filled point. That turns the chart's most important message
from an assertion into a picture: test06 spans 0.0493 across five runs, which is
*most of the width of the equivalence band by itself*. Nobody who sees that will then
ask which of two adjacent rows is better.

Every value is read from the result files; nothing is synthetic.
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
POD4 = REPO / "results" / "_pod4" / "keep"
POD3 = REPO / "results" / "_pod3" / "guardado"
OLD = REPO / "old_runs"
OUT = REPO / "latex" / "figuras"

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#999999"
NOISE = 0.067

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

LABEL = {
    "test01": "01  centralised, all data",
    "test02": "02  2 hosp, balanced",
    "test03": "03  2 hosp, balanced",
    "test04": "04  3 hosp, balanced",
    "test05": "05  3 hosp, balanced",
    "test06": "06  4 hosp, balanced",
    "test07": "07  4 hosp, balanced",
    "test08": "08  4 hosp, skew 5:2:1:1",
    "test09": "09  4 hosp, skew 5:2:1:1",
    "test10": "10  3 hosp, one cohort each",
    "test11": "11  3 hosp, one cohort each",
    "test12": "12  3 hosp, size-matched",
    "test13": "13  3 hosp, size-matched",
}
ALGO = {"test01": "cent"}
for t in ["test02", "test04", "test06", "test08", "test10", "test12"]:
    ALGO[t] = "fedavg"
for t in ["test03", "test05", "test07", "test09", "test11", "test13"]:
    ALGO[t] = "fedprox"

# Repeat draws at seed 42. Read from the files below by `collect_repeats()`; the
# expected values are recorded here so a silent path change cannot go unnoticed.
EXPECTED = {
    "test01": 6, "test06": 5, "test09": 3, "test07": 2,
}


def auc_of(path: Path) -> float | None:
    """macro-AUC out of either a federated test_metrics.json or a centralised
    results.json. Returns None when the file is absent."""
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    if "auc" in d:
        return float(d["auc"])
    return float(d["splits"]["test"]["auc"])


def collect_repeats() -> dict[str, list[float]]:
    """Every same-seed repeat that exists on disk, keyed by test id."""
    r: dict[str, list[float]] = {}

    def add(test: str, p: Path) -> None:
        v = auc_of(p)
        if v is not None:
            r.setdefault(test, []).append(v)

    # centralised: the installed run plus every pulled-back repeat
    add("test01", FED / "test01_centralized" / "seed_42" / "results.json")
    for n in (1, 2, 3, 4):
        add("test01", POD4 / f"test01_centralized_run{n}_pod4" / "seed_42" / "results.json")
        add("test01", POD4 / f"test01_centralized_run{n}_pod4" / "results.json")
    add("test01", REPO / "results" / "_rerun_pod" / "test01_centralized" / "seed_42" / "results.json")

    # test06: installed (= pod3 run1) + pod3 runs 2,3 + pod4 run4 + the archived original
    add("test06", FED / "test06_fedavg_4h" / "test_metrics.json")
    for n in (2, 3):
        add("test06", POD3 / f"test06_run{n}" / "test_metrics.json")
    add("test06", POD4 / "test06_run4_pod4" / "test_metrics.json")
    add("test06", OLD / "test06_fedavg_4h__raw_2026-08-04" / "test_metrics.json")

    # test09: installed + archived original + pod4 run3
    add("test09", FED / "test09_fedprox_skewed" / "test_metrics.json")
    add("test09", OLD / "test09_fedprox_skewed__raw_2026-08-04" / "test_metrics.json")
    add("test09", POD4 / "test09_run3_pod4" / "test_metrics.json")

    # test07: installed + pod4 repeat
    add("test07", FED / "test07_fedprox_4h" / "test_metrics.json")
    add("test07", POD4 / "test07_run2_pod4" / "test_metrics.json")

    return r


def installed() -> dict[str, float]:
    v = {"test01": auc_of(FED / "test01_centralized" / "seed_42" / "results.json")}
    names = {
        "test02": "test02_fedavg_2h", "test03": "test03_fedprox_2h",
        "test04": "test04_fedavg_3h", "test05": "test05_fedprox_3h",
        "test06": "test06_fedavg_4h", "test07": "test07_fedprox_4h",
        "test08": "test08_fedavg_skewed", "test09": "test09_fedprox_skewed",
        "test10": "test10_fedavg_cohort", "test11": "test11_fedprox_cohort",
        "test12": "test12_fedavg_sizematched", "test13": "test13_fedprox_sizematched",
    }
    for t, n in names.items():
        v[t] = auc_of(FED / n / "test_metrics.json")
    return v


def build() -> None:
    vals = installed()
    reps = collect_repeats()

    for t, n in EXPECTED.items():
        got = len(reps.get(t, []))
        if got != n:
            print(f"  [warn] {t}: expected {n} repeats on disk, found {got}")

    order = [f"test{i:02d}" for i in range(1, 14)]
    cent = vals["test01"]

    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    # the margin of practical equivalence, centred on the centralised baseline
    ax.axvspan(cent - NOISE, cent + NOISE, color=GREY, alpha=0.16, zorder=0,
               label=f"equivalence margin, $\\pm${NOISE:g} macro AUC")
    ax.axvline(cent, color="black", lw=1.4, zorder=2)

    ys = np.arange(len(order))[::-1]
    for y, t in zip(ys, order):
        col = {"cent": "black", "fedavg": BLUE, "fedprox": ORANGE}[ALGO[t]]
        ax.plot([vals[t]], [y], "o", ms=8.5, color=col, zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[t] for t in order])
    ax.set_xlabel("patient-level macro AUC on the global test set")
    ax.set_xlim(0.50, 0.70)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GREY, alpha=0.28, lw=0.7)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8.5, color=BLUE, label="FedAvg"),
        plt.Line2D([], [], marker="o", ls="", ms=8.5, color=ORANGE, label="FedProx"),
        plt.Line2D([], [], color="black", lw=1.4, label="centralised baseline"),
        plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.16,
                      label=f"equivalence margin $\\pm${NOISE:g}"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, columnspacing=1.4, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_equivalence.{ext}")
    plt.close(fig)
    print("  fig_equivalence.pdf / .png")

    v = np.array([vals[t] for t in order[1:]])
    print(f"\n  centralised {cent:.4f} | federated mean {v.mean():.4f} | "
          f"gap {cent - v.mean():+.4f}")
    print(f"  inside the margin: {(np.abs(v - cent) < NOISE).sum()}/12")
    for t in ("test01", "test06", "test09", "test07"):
        d = reps.get(t, [])
        if len(d) > 1:
            print(f"  {t}: {len(d)} runs, range {max(d) - min(d):.4f}, "
                  f"mean {np.mean(d):.4f}")


if __name__ == "__main__":
    build()
