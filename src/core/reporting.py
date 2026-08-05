"""Every figure and file an experiment produces, written automatically.

Nothing here is called by hand. `training.run()` calls `write_all()` once at the
end, and the experiment folder is complete: curves, ROC, PR, confusion matrix,
per-class metrics, classification report and per-patient predictions.

The point is that no manual step stands between a finished run and a
dissertation-ready figure. A run that needs a human afterwards is a run that will
be reported inconsistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display on a headless GPU box
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluation as ev

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})

PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2"]


# --------------------------------------------------------------------------- #
# Curves over epochs                                                           #
# --------------------------------------------------------------------------- #
def plot_curves(history: pd.DataFrame, out_dir: Path, best_epoch: int | None = None
                ) -> None:
    """Loss, accuracy and the monitored metric, train against validation."""
    def _mark(ax):
        if best_epoch:
            ax.axvline(best_epoch, color="k", ls="--", lw=1, alpha=0.6)
            ax.text(best_epoch, ax.get_ylim()[1], f" best ep {best_epoch}",
                    fontsize=7, va="top")

    # ---- loss ---- #
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history.epoch, history.train_loss, label="train", color=PALETTE[0])
    if "val_loss" in history:
        ax.plot(history.epoch, history.val_loss, label="validation", color=PALETTE[1])
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss")
    ax.legend(); _mark(ax)
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight"); plt.close(fig)

    # ---- accuracy ---- #
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history.epoch, history.train_acc, label="train (slice)", color=PALETTE[0])
    if "val_accuracy" in history:
        ax.plot(history.epoch, history.val_accuracy, label="validation (patient)",
                color=PALETTE[1])
    if "val_balanced_accuracy" in history:
        ax.plot(history.epoch, history.val_balanced_accuracy,
                label="validation balanced", color=PALETTE[2], ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_title("Accuracy")
    ax.legend(fontsize=7); _mark(ax)
    fig.savefig(out_dir / "accuracy_curve.png", bbox_inches="tight"); plt.close(fig)

    # ---- AUC + the generalisation gap ---- #
    # The gap is on the same figure on purpose: a change that lowers training
    # accuracy without lowering test AUC has reduced overfitting, and that is
    # invisible if the two are plotted apart.
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    if "val_auc" in history:
        ax[0].plot(history.epoch, history.val_auc, color=PALETTE[1])
        ax[0].axhline(0.5, color="grey", ls=":", lw=1)
        ax[0].set_title("Validation macro-AUC (patient level)")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("macro-AUC"); _mark(ax[0])
    if "val_accuracy" in history:
        gap = history.train_acc - history.val_accuracy
        ax[1].plot(history.epoch, gap, color=PALETTE[3])
        ax[1].axhline(0, color="grey", ls=":", lw=1)
        ax[1].set_title("Generalisation gap  (train acc − val acc)")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("gap"); _mark(ax[1])
    fig.savefig(out_dir / "auc_and_gap_curve.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-split diagnostic figures                                                 #
# --------------------------------------------------------------------------- #
def plot_confusion(cm: list[list[int]], class_names, out_path: Path,
                   title: str = "Confusion matrix") -> None:
    """Counts and row-normalised percentages in one figure."""
    cm = np.asarray(cm, dtype=float)
    pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    fig, axes = plt.subplots(1, 2, figsize=(4.2 * len(class_names), 4.2))
    for ax, data, fmt, lab in ((axes[0], cm, "{:.0f}", "count"),
                               (axes[1], pct, "{:.2f}", "row-normalised")):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title} — {lab}", fontsize=9); ax.grid(False)
        thr = data.max() / 2 if data.max() else 0.5
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > thr else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def plot_roc(y_true, y_prob, class_names, out_path: Path) -> None:
    """One-vs-rest ROC per class."""
    curves = ev.roc_points(y_true, y_prob, len(class_names))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for c, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, color=PALETTE[c % len(PALETTE)],
                label=f"{class_names[c]}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC — one vs rest, patient level"); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def plot_pr(y_true, y_prob, class_names, out_path: Path) -> None:
    """One-vs-rest precision-recall, each with its own prevalence baseline."""
    curves = ev.pr_points(y_true, y_prob, len(class_names))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for c, (recall, precision, base) in curves.items():
        col = PALETTE[c % len(PALETTE)]
        ax.plot(recall, precision, color=col, label=f"{class_names[c]}")
        ax.axhline(base, color=col, ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–recall (dotted = class prevalence)")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def plot_per_class(metrics: dict, class_names, out_path: Path) -> None:
    """AUC, precision, recall and F1 side by side per class."""
    keys = ["per_class_auc", "per_class_precision", "per_class_recall", "per_class_f1"]
    labels = ["AUC", "precision", "recall", "F1"]
    data = np.array([metrics[k] for k in keys], dtype=float)
    x = np.arange(len(class_names)); width = 0.2
    fig, ax = plt.subplots(figsize=(1.9 * len(class_names) + 3, 3.8))
    for i, (row, lab) in enumerate(zip(data, labels)):
        ax.bar(x + (i - 1.5) * width, row, width, label=lab,
               color=PALETTE[i % len(PALETTE)])
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(x, class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Per-class metrics, patient level"); ax.legend(fontsize=8, ncol=4)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# The one entry point                                                          #
# --------------------------------------------------------------------------- #
def write_split(split: str, y_true, y_prob, pids, class_names, out_dir: Path,
                rows: pd.DataFrame | None = None) -> dict:
    """Every artefact for one split. Returns the metrics dict."""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics = ev.compute_metrics(y_true, y_prob, class_names)
    plot_confusion(metrics["confusion"], class_names,
                   figures / f"confusion_matrix_{split}.png",
                   title=f"Confusion — {split}")
    plot_roc(y_true, y_prob, class_names, figures / f"roc_curve_{split}.png")
    plot_pr(y_true, y_prob, class_names, figures / f"pr_curve_{split}.png")
    plot_per_class(metrics, class_names, figures / f"per_class_{split}.png")

    ev.predictions_frame(pids, y_true, y_prob, class_names, rows).to_csv(
        out_dir / f"predictions_{split}.csv", index=False)
    (out_dir / f"classification_report_{split}.txt").write_text(
        ev.report_text(y_true, y_prob, class_names))
    return metrics


def write_all(out_dir: Path, cfg, history: pd.DataFrame, results: dict,
              best_epoch: int, params: dict) -> None:
    """Curves, the metrics table and the run summary. Called once, at the end."""
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    history.to_csv(out_dir / "train_log.csv", index=False)
    plot_curves(history, figures, best_epoch)
    # The three curve figures also live at the top level, where a reader looks
    # first; `figures/` keeps the full set.
    for name in ("loss_curve.png", "accuracy_curve.png", "auc_and_gap_curve.png"):
        src = figures / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    # One flat row per split — this is the file `07_compare_experiments` reads.
    flat = []
    for split, m in results.items():
        row = {"split": split}
        for k, v in m.items():
            if isinstance(v, list) and k != "confusion":
                row.update({f"{k}_{i}": x for i, x in enumerate(v)})
            elif k != "confusion":
                row[k] = v
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / "metrics.csv", index=False)

    (out_dir / "results.json").write_text(json.dumps({
        "config": cfg.as_dict(), "best_epoch": best_epoch,
        "epochs_run": int(len(history)), "parameters": params,
        "splits": results,
    }, indent=2, default=float))
