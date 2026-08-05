"""Metrics. Everything here is per PATIENT.

THE RULE THIS MODULE ENFORCES
-----------------------------
Slice probabilities are averaged into one prediction per patient before any metric
is computed. Slices from one patient are near-duplicates, so a slice-level score
measures how well the model recognises the *patient*, not the disease. Slice-level
numbers are still returned, labelled as such, purely as an overfitting signal.

WHY THIS MATTERS MORE IN A FEDERATION
-------------------------------------
The server selects the global model from a metric the clients report. A previous
iteration of this project reported plain training accuracy there, so the server
picked whichever global model had let clients memorise their own shard best — a
number that reaches 99% on this data and carries no information at all.

The metric reported to the server is therefore `val_balanced_accuracy`, computed on
held-out patients the client did not train on. Balanced accuracy rather than plain
accuracy because the classes are imbalanced (2.25:1 pooled) and because it is
defined on the small per-hospital validation sets, where macro-AUC is not: a site
holding 39 patients can easily have a validation split missing a whole class, and
`compute_metrics` correctly returns NaN for AUC in that case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .thesis import thesis_eval


def evaluate(model, loader, rows: pd.DataFrame, device, *, num_classes: int,
             class_names: list[str] | tuple[str, ...], aggregation: str = "mean",
             use_amp: bool = False) -> dict:
    """Patient-level metrics for one loader. Restores train() mode on the way out.

    Returns the full metric dict plus `slice_accuracy` and `slice_loss`. Never
    raises on a degenerate split: a validation set missing a class yields NaN for
    AUC rather than an exception, because losing a whole federated run to a metric
    is a bad trade.
    """
    was_training = model.training
    probs, labels, idxs = thesis_eval.predict(model, loader, device, use_amp)
    p, y, pids = thesis_eval.aggregate_by_patient(probs, idxs, rows, aggregation)
    metrics = thesis_eval.compute_metrics(y, p, class_names)

    metrics["slice_accuracy"] = float((probs.argmax(1) == labels).mean())
    eps = 1e-9
    metrics["slice_loss"] = float(
        -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())
    metrics["n_slices"] = int(len(labels))

    if was_training:
        model.train()
    return metrics


def predictions(model, loader, rows: pd.DataFrame, device, *,
                class_names: list[str] | tuple[str, ...], aggregation: str = "mean",
                use_amp: bool = False) -> pd.DataFrame:
    """One row per patient: truth, prediction and every class probability.

    Written for every experiment so a dissertation figure can be regenerated, and so
    two runs with the same macro-AUC can be shown to disagree about which patients
    they get right.
    """
    probs, _, idxs = thesis_eval.predict(model, loader, device, use_amp)
    p, y, pids = thesis_eval.aggregate_by_patient(probs, idxs, rows, aggregation)
    return thesis_eval.predictions_frame(pids, y, p, class_names, rows)


def report_text(model, loader, rows: pd.DataFrame, device, *,
                class_names: list[str] | tuple[str, ...], aggregation: str = "mean",
                use_amp: bool = False) -> str:
    probs, _, idxs = thesis_eval.predict(model, loader, device, use_amp)
    p, y, _ = thesis_eval.aggregate_by_patient(probs, idxs, rows, aggregation)
    return thesis_eval.report_text(y, p, class_names)


def key_metric(metrics: dict, name: str) -> float:
    """Read the server's selection metric out of a metric dict, by name.

    Indirection on purpose: the metric is named once, in
    `config/experiments.py::FederationConfig.key_metric`, and both the client that
    reports it and the server that selects on it read that one name. A client
    reporting `accuracy` while the server selects on `val_balanced_accuracy` is a
    failure that produces plausible-looking numbers.
    """
    short = name[4:] if name.startswith("val_") else name
    for candidate in (name, short):
        if candidate in metrics:
            value = float(metrics[candidate])
            return value if np.isfinite(value) else 0.0
    raise KeyError(
        f"metric {name!r} not produced by the evaluator. Available: "
        f"{sorted(metrics)}")
