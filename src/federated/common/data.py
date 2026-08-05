"""Loading one site's data. Pure PyTorch — this file does not know NVFLARE exists.

A HOSPITAL FOLDER IS A DATASET FOLDER
-------------------------------------
`scripts/partition_data.py` writes every hospital as the same shape the classifier
phase already uses:

    hospital_1/
    ├── images/<pid>/slice_XXX.png
    ├── train.csv
    └── val.csv

which is exactly what `src` produces, minus `test.csv`. That is not a
coincidence, it is the point: the thesis loader, sampler and augmentation work on a
hospital folder unchanged, so a federated client and the centralised baseline load
their data through literally the same function.

The hospital holds no test split. There is exactly one test set in this project, it
is identical across all nine experiments, and it lives with the server.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .thesis import build_config, thesis_data


def load_site(site_dir: Path, training, *, epochs: int = 1, seed: int | None = None,
              augmentation: str = "default"):
    """train/val loaders and frames for one site.

    Returns `(loaders, frames, cfg)`. The frames come back because every downstream
    step — class weights, patient aggregation, per-cohort reporting — must score
    exactly the rows the loader fed, not a re-read of the CSV.
    """
    site_dir = Path(site_dir)
    if not (site_dir / "train.csv").is_file():
        raise FileNotFoundError(
            f"{site_dir} has no train.csv. Run scripts/partition_data.py first.")

    cfg = build_config(training, site_dir, epochs=epochs, seed=seed,
                       augmentation=augmentation)
    loaders, frames = thesis_data.make_loaders(site_dir, cfg)
    return loaders, frames, cfg


def load_eval_only(data_dir: Path, split: str, training, *, seed: int | None = None):
    """A single split with augmentation off — the global test or global val set.

    Used by `collect_results.py` and by the per-round global convergence curve.
    Augmentation is disabled for every split except `train` inside the thesis
    loader, so an evaluation set can never be augmented by accident.
    """
    data_dir = Path(data_dir)
    csv = data_dir / f"{split}.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found")

    cfg = build_config(training, data_dir, epochs=1, seed=seed, augmentation="none")
    loaders, frames = thesis_data.make_loaders(data_dir, cfg)
    if split not in loaders:
        raise FileNotFoundError(f"{data_dir} has no {split} split")
    return loaders[split], frames[split], cfg


# --------------------------------------------------------------------------- #
# Class weights — and the question of whose frequencies they come from         #
# --------------------------------------------------------------------------- #
def class_weights(rows: pd.DataFrame, num_classes: int, device,
                  override: list[float] | None = None) -> torch.Tensor:
    """Inverse-frequency weights counted per PATIENT.

    `override` is what makes RQ4 testable, and it matters more than it looks.

    LOCAL weights (override=None) are computed from this hospital's own rows. Each
    site then optimises a slightly different objective, and FedAvg averages models
    that were trained on different losses. Under the stratified partitions used by
    tests 02-09 this is harmless — every hospital keeps the global class ratio, so
    the weights agree to three decimals. Under a COHORT partition it stops being
    harmless: DUKE is 64.6% HRposHER2neg against I-SPY2's 38.8%, and the two sites
    would be pulling towards genuinely different decision boundaries.

    GLOBAL weights (override=[...]) give every site the same objective. They are
    consistent, and they leak one vector of class frequencies to the server — a
    small, quantifiable privacy cost.

    That trade-off is RQ4 material, so both are implemented and the choice is a
    field in `config/experiments.py`, not a decision buried in a trainer.
    """
    if override is not None:
        if len(override) != num_classes:
            raise ValueError(
                f"global class weights have {len(override)} entries, "
                f"expected {num_classes}")
        return torch.tensor(override, dtype=torch.float32, device=device)
    return thesis_data.class_weights(rows, num_classes, device)


def patient_class_counts(rows: pd.DataFrame, num_classes: int) -> list[int]:
    """How many PATIENTS carry each class. Slice counts would conflate that with
    how large their tumours are."""
    per_patient = rows.groupby("pid").label.first()
    return np.bincount(per_patient.to_numpy(), minlength=num_classes).tolist()


def global_class_weights(frames: list[pd.DataFrame], num_classes: int) -> list[float]:
    """Inverse-frequency weights over the UNION of every site's patients.

    Computed once by `partition_data.py` from the pooled training split and written
    into each site's manifest, so no site has to see another site's data to use it.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for rows in frames:
        counts += np.asarray(patient_class_counts(rows, num_classes), dtype=np.float64)
    total = counts.sum()
    counts[counts == 0] = 1.0
    return (total / (num_classes * counts)).tolist()


def trivial_baseline(rows: pd.DataFrame) -> float:
    """Majority-class rate among PATIENTS. Accuracy without it is meaningless, and
    it is not a constant — 0.404 on I-SPY2 alone, 0.511 on the pooled cohorts."""
    return thesis_data.trivial_baseline(rows)


def describe_site(name: str, frames: dict[str, pd.DataFrame], num_classes: int) -> str:
    parts = [f"{name}:"]
    for split, rows in frames.items():
        counts = patient_class_counts(rows, num_classes)
        parts.append(f"  {split:<5} {len(rows):>6,} slices / {rows.pid.nunique():>4} "
                     f"patients  per-class {counts}")
    return "\n".join(parts)


def set_seed(seed: int) -> None:
    """Pin what can be pinned — and note what cannot: cuDNN kernel selection, AMP
    and DataLoader worker ordering stay free. Two byte-identical runs differing only
    in seed were measured 0.067 macro-AUC apart on this task."""
    thesis_data.set_seed(seed)
