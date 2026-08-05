"""Metrics, at the patient level and at the slice level.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
Everything reported is per PATIENT. Slice probabilities are aggregated into one
prediction per patient before any metric is computed.

Slices from one patient are near-duplicates, so a slice-level score measures how
well the model recognises the *patient*, not the disease. Slice-level numbers are
still computed and returned — they are useful for spotting overfitting — but they
are labelled as such and never used for model selection.

The headline metric is macro one-vs-rest AUC: it weights classes equally
regardless of prevalence, and it is threshold-free. Everything this project
measured about threshold selection on small validation sets says a tuned
threshold does not transfer, so predictions are plain argmax and no threshold is
ever fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


@torch.no_grad()
def predict(model, loader, device, use_amp: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax probabilities, labels and row positions for a whole loader.

    Returns the full (N, C) probability matrix rather than a single positive
    column: collapsing a 3-class problem to one column throws away two thirds of
    the prediction.
    """
    model.eval()
    probs, labels, idxs = [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
        idxs.append(i.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)


def aggregate_by_patient(probs: np.ndarray, idxs: np.ndarray, rows: pd.DataFrame,
                         how: str = "mean") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collapse slice probabilities into one prediction per patient.

    `how="median"` is what the authors report as best for HER2; `mean` elsewhere.
    """
    # Explicit column list, not a `startswith("p")` filter — that would also
    # match `pid` and try to average patient identifiers.
    prob_cols = [f"class_{c}" for c in range(probs.shape[1])]
    df = pd.DataFrame(probs, columns=prob_cols)
    df["pid"] = rows.iloc[idxs].pid.values
    df["label"] = rows.iloc[idxs].label.values

    agg = df.groupby("pid").agg({**{c: how for c in prob_cols}, "label": "first"})
    p = agg[prob_cols].to_numpy()
    # Renormalise: a median over slices does not sum to 1.
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
    return p, agg["label"].to_numpy(), agg.index.tolist()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: tuple[str, ...] | list[str]) -> dict:
    """Every metric this project reports, from labels and probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_prob.shape[1]
    present = np.unique(y_true)

    # A diverged run produces NaN logits. Report that as NaN metrics rather than
    # crashing halfway through an epoch — the training log then shows exactly
    # which epoch went wrong instead of a stack trace with no context.
    finite = np.isfinite(y_prob).all()
    if not finite:
        y_prob = np.nan_to_num(y_prob, nan=1.0 / n, posinf=1.0, neginf=0.0)
    y_pred = y_prob.argmax(1)

    # A split missing a whole class cannot support a macro-AUC. NaN rather than a
    # number computed over a subset, which would be silently incomparable.
    if len(present) == n and finite:
        macro_auc = float(roc_auc_score(y_true, y_prob[:, 1]) if n == 2 else
                          roc_auc_score(y_true, y_prob, multi_class="ovr",
                                        average="macro"))
        per_class_auc = [float(roc_auc_score((y_true == c).astype(int), y_prob[:, c]))
                         for c in range(n)]
    else:
        macro_auc = float("nan")
        per_class_auc = [float("nan")] * n

    m: dict = {
        "n_patients": int(len(y_true)),
        "auc": macro_auc,                       # the headline metric
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "per_class_auc": [round(a, 4) for a in per_class_auc],
        "per_class_precision": [round(float(v), 4) for v in
                                precision_score(y_true, y_pred, average=None,
                                                labels=list(range(n)), zero_division=0)],
        "per_class_recall": [round(float(v), 4) for v in
                             recall_score(y_true, y_pred, average=None,
                                          labels=list(range(n)), zero_division=0)],
        "per_class_f1": [round(float(v), 4) for v in
                         f1_score(y_true, y_pred, average=None,
                                  labels=list(range(n)), zero_division=0)],
        "class_counts": [int((y_true == c).sum()) for c in range(n)],
        # Rows = truth, columns = prediction. Two runs with the same macro-AUC can
        # predict entirely different classes; without this that stays invisible.
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(n))).tolist(),
        # Accuracy means nothing without this beside it.
        "trivial_baseline_accuracy": float(
            max((y_true == c).mean() for c in range(n))),
    }

    if n == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m.update({
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
            "positive_rate": float((y_true == 1).mean()),
        })
    return m


def report_text(y_true: np.ndarray, y_prob: np.ndarray,
                class_names: tuple[str, ...] | list[str]) -> str:
    """sklearn's classification report, saved verbatim beside the metrics."""
    return classification_report(y_true, y_prob.argmax(1),
                                 labels=list(range(len(class_names))),
                                 target_names=list(class_names), zero_division=0,
                                 digits=4)


def roc_points(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
               ) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """One-vs-rest ROC curve per class, for plotting."""
    out = {}
    for c in range(n_classes):
        binary = (np.asarray(y_true) == c).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, c])
        out[c] = (fpr, tpr, float(roc_auc_score(binary, y_prob[:, c])))
    return out


def pr_points(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
              ) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    """One-vs-rest precision-recall curve per class, with the baseline rate.

    The baseline for a PR curve is the class prevalence, not 0.5 — a PR curve
    without it cannot be read.
    """
    out = {}
    for c in range(n_classes):
        binary = (np.asarray(y_true) == c).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, c])
        out[c] = (recall, precision, float(binary.mean()))
    return out


def predictions_frame(pids: list[str], y_true: np.ndarray, y_prob: np.ndarray,
                      class_names: tuple[str, ...] | list[str],
                      rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per patient: truth, prediction, and every class probability."""
    df = pd.DataFrame({"pid": pids, "label": y_true,
                       "label_name": [class_names[i] for i in y_true],
                       "pred": y_prob.argmax(1),
                       "pred_name": [class_names[i] for i in y_prob.argmax(1)]})
    for c, name in enumerate(class_names):
        df[f"prob_{name}"] = y_prob[:, c]
    df["correct"] = df.label == df.pred
    # Cohort travels with the prediction so per-cohort AUC needs no second join.
    if rows is not None and "cohort" in rows.columns:
        df = df.merge(rows.drop_duplicates("pid")[["pid", "cohort"]], on="pid", how="left")
    return df
