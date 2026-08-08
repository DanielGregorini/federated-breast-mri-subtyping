#!/usr/bin/env python3
"""Generate every figure used in Chapter 4, from the measured results.

    python src/scripts/build_chapter4_figures.py

Writes PDF + PNG into ``latex/figuras/``. Every value is read from
``results/federated/``; nothing is synthetic. Re-running after a re-collection
regenerates the whole chapter's artwork.

Palette is Okabe-Ito (colour-blind safe). Figures are deliberately compact with
large type, and carry no explanatory prose inside the axes -- that belongs in the
caption, where it can be read at any zoom level.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
FS = REPO / "results" / "federated" / "final_summary"
EXP = FS / "experiments"
FED = REPO / "results" / "federated"
OUT = REPO / "latex" / "figuras"

CLASSES = ["HR+/HER2$-$", "TripleNeg", "HER2+"]
ORDER = [f"test{i:02d}" for i in range(1, 14)]
BASELINE = 0.5112
NOISE = 0.067

# Okabe-Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PURPLE, GREY = "#D55E00", "#CC79A7", "#999999"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def load():
    s = pd.read_csv(FS / "summary.csv")
    s["_o"] = s.experiment.map({t: i for i, t in enumerate(ORDER)})
    s = s.sort_values("_o").reset_index(drop=True)
    M = {r["name"]: json.load(open(EXP / r["name"] / "metrics.json"))
         for _, r in s.iterrows()}
    pc = pd.read_csv(FS / "per_client_metrics.csv")
    return s, M, pc


def label(r) -> str:
    return r.experiment.replace("test", "T")


def colour(r) -> str:
    return {"centralized": GREY, "fedavg": BLUE, "fedprox": ORANGE}[r.algorithm]


# --------------------------------------------------------------- 1. overview
def fig_overview(s, M):
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    x = np.arange(len(s))
    v = [M[n]["global_test"]["auc"] for n in s["name"]]
    ax.bar(x, v, color=[colour(r) for _, r in s.iterrows()], width=0.7,
           edgecolor="white", linewidth=0.6)
    # The noise floor drawn around the centralised reference, so the reader sees
    # at a glance which bars are indistinguishable from it.
    c = M["test01_centralized"]["global_test"]["auc"]
    ax.axhspan(c - NOISE, c + NOISE, color=GREY, alpha=0.18, zorder=0)
    ax.axhline(c, color=GREY, ls="--", lw=1.4, zorder=1)
    ax.axhline(0.5, color=VERM, ls=":", lw=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label(r) for _, r in s.iterrows()])
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.45, 0.72)
    h = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in (GREY, BLUE, ORANGE)]
    h.append(plt.Line2D([0], [0], color=VERM, ls=":", lw=1.4))
    ax.legend(h, ["Centralised", "FedAvg", "FedProx", "Chance"],
              ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "fig_overview_barplot")


# --------------------------------------------------- 2. centralised vs federated
def fig_central_vs_fed(s, M):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    c = M["test01_centralized"]["global_test"]["auc"]
    fed = s[s.algorithm != "centralized"]
    d = [M[n]["global_test"]["auc"] - c for n in fed["name"]]
    x = np.arange(len(fed))
    ax.bar(x, d, color=[colour(r) for _, r in fed.iterrows()], width=0.68,
           edgecolor="white", linewidth=0.6)
    ax.axhspan(-NOISE, NOISE, color=GREY, alpha=0.20, zorder=0)
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([label(r) for _, r in fed.iterrows()])
    ax.set_ylabel("Macro AUC $-$ centralised")
    ax.set_ylim(-0.09, 0.09)
    h = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in (BLUE, ORANGE)]
    h.append(plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.35))
    ax.legend(h, ["FedAvg", "FedProx", "Noise floor"], ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig_central_vs_fed")


# ------------------------------------------------------------ 3. n hospitals
def fig_num_hospitals(s, M):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bal = s[s.partition.isin(["2_clients_balanced", "3_clients_balanced",
                              "4_clients_balanced"])]
    for algo, col, mk in [("fedavg", BLUE, "o"), ("fedprox", ORANGE, "s")]:
        sub = bal[bal.algorithm == algo]
        ax.plot(sub.n_hospitals, [M[n]["global_test"]["auc"] for n in sub["name"]],
                marker=mk, color=col, lw=2.0, ms=9, label=algo.replace("fed", "Fed"))
    c = M["test01_centralized"]["global_test"]["auc"]
    ax.axhspan(c - NOISE, c + NOISE, color=GREY, alpha=0.18, zorder=0)
    ax.axhline(c, color=GREY, ls="--", lw=1.4, label="Centralised")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("Participating hospitals")
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.52, 0.70)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, "fig_num_hospitals")


# ------------------------------------------------------ 4. FedAvg vs FedProx
def fig_fedavg_vs_fedprox(s, M):
    pairs = [("2_clients_balanced", "2C bal."), ("3_clients_balanced", "3C bal."),
             ("4_clients_balanced", "4C bal."), ("4_clients_skewed", "4C skew"),
             ("3_clients_cohort", "3C cohort"),
             ("3_clients_sizematched", "3C size-m.")]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(pairs))
    a = [M[s[(s.partition == p) & (s.algorithm == "fedavg")].iloc[0]["name"]]
         ["global_test"]["auc"] for p, _ in pairs]
    b = [M[s[(s.partition == p) & (s.algorithm == "fedprox")].iloc[0]["name"]]
         ["global_test"]["auc"] for p, _ in pairs]
    ax.bar(x - 0.19, a, 0.36, label="FedAvg", color=BLUE, edgecolor="white")
    ax.bar(x + 0.19, b, 0.36, label="FedProx", color=ORANGE, edgecolor="white")
    c = M["test01_centralized"]["global_test"]["auc"]
    ax.axhline(c, color=GREY, ls="--", lw=1.4, label="Centralised")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in pairs], rotation=12)
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.50, 0.70)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, "fig_fedavg_vs_fedprox")


# ------------------------------------------------------------ 5. convergence
def fig_convergence(s, M):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    shown = [("test06_fedavg_4h", "T06 FedAvg 4C", BLUE),
             ("test09_fedprox_skewed", "T09 FedProx skew", ORANGE),
             ("test10_fedavg_cohort", "T10 FedAvg cohort", VERM),
             ("test12_fedavg_sizematched", "T12 FedAvg size-m.", GREEN)]
    for name, lab, col in shown:
        p = FED / name / "sites" / "rounds.csv"
        if not p.is_file():
            continue
        g = pd.read_csv(p).groupby("round")["agg_val_auc"].mean()
        ax.plot(g.index, g.values, color=col, lw=1.9, label=lab)
        row = s[s.name == name].iloc[0]
        if not pd.isna(row.best_round):
            r = int(row.best_round)
            ax.plot([r], [g.loc[r]], marker="v", color=col, ms=11, mec="white",
                    mew=1.2, zorder=5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Aggregated model, mean local AUC")
    ax.set_xlim(-0.5, 29.5)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    save(fig, "fig_convergence")


# ------------------------------------------------- 6. balanced vs skewed vs cohort
def fig_balanced_vs_skewed(s, M):
    groups = [("4_clients_balanced", "Balanced\n(4 sites)"),
              ("4_clients_skewed", "Quantity skew\n(5:2:1:1)"),
              ("3_clients_sizematched", "Size-matched\ncontrol"),
              ("3_clients_cohort", "Cohort non-IID\n(1 cohort/site)")]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(groups))
    a = [M[s[(s.partition == p) & (s.algorithm == "fedavg")].iloc[0]["name"]]
         ["global_test"]["auc"] for p, _ in groups]
    b = [M[s[(s.partition == p) & (s.algorithm == "fedprox")].iloc[0]["name"]]
         ["global_test"]["auc"] for p, _ in groups]
    ax.bar(x - 0.19, a, 0.36, label="FedAvg", color=BLUE, edgecolor="white")
    ax.bar(x + 0.19, b, 0.36, label="FedProx", color=ORANGE, edgecolor="white")
    c = M["test01_centralized"]["global_test"]["auc"]
    ax.axhspan(c - NOISE, c + NOISE, color=GREY, alpha=0.18, zorder=0)
    ax.axhline(c, color=GREY, ls="--", lw=1.4, label="Centralised")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.50, 0.70)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig_balanced_vs_skewed")


# ------------------------------------------------------------- 7. confusions
def _confusion(ax, cm, title):
    cm = np.asarray(cm, float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=0.8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=13,
                    color="white" if norm[i, j] > 0.45 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=18)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    return im


def fig_confusions(M):
    for name, tag, title in [
            ("test01_centralized", "central", "Centralised (Test 01)"),
            ("test06_fedavg_4h", "federated", "FedAvg, 4 hospitals (Test 06)"),
            ("test10_fedavg_cohort", "cohort", "FedAvg, cohort non-IID (Test 10)")]:
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        _confusion(ax, M[name]["global_test"]["confusion"], title)
        save(fig, f"fig_confusion_{tag}")


# --------------------------------------------- 7b. centralised training curve
def fig_centralised_training():
    """Epoch-by-epoch curve of the centralised baseline.

    One panel with every series: the rate metrics on the left axis and the training
    loss on a right-hand twin axis, since loss is unbounded and does not share the
    units of the rates. The legend sits below the axes so that no curve is covered.
    Epochs are numbered from 1 for reading; the run log indexes from 0, so the
    selected epoch 5 here is `best_epoch = 4` in results.json.
    """
    p = FED / "test01_centralized" / "seed_42" / "rounds.csv"
    if not p.is_file():
        print("  (skipped fig_centralised_training: rounds.csv missing)")
        return
    d = pd.read_csv(p)
    e = d["round"].values + 1                      # 1-indexed for the reader
    sel = int(d["post_val_auc"].idxmax()) + 1

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(e, d.train_acc, color=GREY, lw=2.0, label="Training accuracy")
    ax.plot(e, d.post_val_auc, color=BLUE, lw=2.3, label="Validation macro AUC")
    ax.plot(e, d.post_val_acc, color=ORANGE, lw=1.8, label="Validation accuracy")
    ax.plot(e, d.post_val_bal_acc, color=GREEN, lw=1.8,
            label="Validation balanced accuracy")
    ax.axvline(sel, color=VERM, ls="--", lw=1.6, zorder=0)
    ax.plot([sel], [d.post_val_auc.max()], marker="v", color=VERM, ms=11,
            mec="white", mew=1.2, zorder=5)
    ax.annotate(f"selected epoch {sel}", xy=(sel, d.post_val_auc.max()),
                xytext=(sel + 1.4, d.post_val_auc.max() + 0.07),
                color=VERM, fontsize=12, ha="left")
    ax.axhline(0.5, color="black", ls=":", lw=1.2)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.30, 1.06)
    ax.set_xlim(0.5, 30.5)

    # Loss on its own axis: unbounded, and not in the same units as the rates.
    axl = ax.twinx()
    axl.plot(e, d.train_loss, color=PURPLE, lw=2.3, ls=(0, (5, 2)),
             label="Training loss")
    axl.set_ylabel("Training loss")
    axl.set_ylim(0.0, 1.30)
    axl.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axl.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, fontsize=12,
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    save(fig, "fig_centralised_training")


# -------------------------------------------------------------- 8. per class
def fig_per_class(s, M):
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    x = np.arange(len(s))
    w = 0.26
    for k, (lab, col) in enumerate(zip(CLASSES, (BLUE, ORANGE, VERM))):
        ax.bar(x + (k - 1) * w,
               [M[n]["global_test"]["per_class_auc"][k] for n in s["name"]],
               w, label=lab, color=col, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="black", ls=":", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([label(r) for _, r in s.iterrows()])
    ax.set_ylabel("One-vs-rest AUC")
    ax.set_ylim(0.40, 0.80)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, "fig_per_class")


# ----------------------------------------------------------- 9. per hospital
def fig_per_hospital(s, M, pc):
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    exps = [e for e in ORDER if e in set(pc.experiment)]
    xs, ys, ss, cs = [], [], [], []
    for i, e in enumerate(exps):
        g = pc[pc.experiment == e]
        for _, r in g.iterrows():
            xs.append(i + np.linspace(-0.22, 0.22, len(g))[list(g.site).index(r.site)])
            ys.append(r.macro_auc)
            ss.append(18 + 1.05 * r.n_val_patients)
            cs.append(BLUE if r.algorithm == "fedavg" else ORANGE)
    ax.scatter(xs, ys, s=ss, c=cs, alpha=0.78, edgecolor="white", linewidth=0.8,
               zorder=3)
    gl = [M[s[s.experiment == e].iloc[0]["name"]]["global_test"]["auc"] for e in exps]
    ax.plot(range(len(exps)), gl, marker="_", ms=26, ls="none", color="black",
            mew=2.2, zorder=4, label="Global test set")
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels([e.replace("test", "T") for e in exps])
    ax.set_ylabel("Macro AUC")
    h = [plt.Line2D([0], [0], marker="o", ls="none", color=c_, ms=9)
         for c_ in (BLUE, ORANGE)]
    h.append(plt.Line2D([0], [0], marker="_", ls="none", color="black", ms=16, mew=2.2))
    ax.legend(h, ["FedAvg site", "FedProx site", "Global test set"], ncol=3,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig_per_hospital")


def main() -> None:
    s, M, pc = load()
    print(f"generating Chapter 4 figures from {len(s)} experiments ->  {OUT}")
    fig_overview(s, M)
    fig_central_vs_fed(s, M)
    fig_num_hospitals(s, M)
    fig_fedavg_vs_fedprox(s, M)
    fig_convergence(s, M)
    fig_balanced_vs_skewed(s, M)
    fig_confusions(M)
    fig_centralised_training()
    fig_per_class(s, M)
    fig_per_hospital(s, M, pc)
    print("done")


if __name__ == "__main__":
    main()
