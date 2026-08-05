#!/usr/bin/env python3
"""Describe how the data is divided, for every one of the nine experiments.

    python scripts/build_distribution_report.py

Writes one figure per experiment to `production/figures/`, one table per experiment
to `production/datasets/`, and a combined summary in CSV and JSON.

WHY THIS RUNS BEFORE ANY TRAINING
---------------------------------
The partition IS an experimental variable — tests 06 and 08 differ in nothing else.
So the split has to be inspectable and reportable before a single epoch is run, not
reconstructed afterwards from a log. Everything here is derived from
`data/partitions/*/partition.json` and `data/global/manifest.json`, both written by
`partition_data.py` / `prepare_data.py` at split time, plus the per-site CSVs for the
cohort breakdown.

Nothing here loads a model or needs a GPU.

READ THE STRATIFICATION PANEL FIRST
-----------------------------------
Every partition in this project is STRATIFIED: each hospital keeps the global class
ratio, so between hospitals only the QUANTITY of data varies. That is a deliberate
limitation and the dissertation states it — tests 08 and 09 are *quantity* skew, not
genuine non-IID label heterogeneity. The normalised class panel is what makes that
visible: if the bars are flat across hospitals, the split is quantity-only. A reader
who misses this will over-claim what tests 08 and 09 measure.

THE COHORT PANEL IS THE ONE TO READ SECOND
------------------------------------------
`SOURCE_DATASET` is `multi_subtype_80mm`: I-SPY2 + I-SPY1 + DUKE. Every hospital
therefore holds a MIX of all three cohorts, because the partitions are stratified by
class, not by cohort.

That matters for interpretation. The cohorts differ systematically — DUKE is 64.6%
HRposHER2neg against I-SPY2's 38.8%, with tumours about five times smaller by volume
— and a source-signature probe reached macro-AUC 0.9978 predicting which cohort a
slice came from, against 0.6078 for the subtype task itself. Mixing the cohorts
evenly across hospitals means that shortcut is available to every site equally, so it
inflates the absolute numbers without creating heterogeneity BETWEEN sites.

If genuine non-IID sites are wanted, `partition_data.py --by-cohort` assigns one real
cohort per hospital (three cohorts, so three hospitals). That is a different
experiment and is not what these figures describe.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from config import experiments as EX                       # noqa: E402

CLASSES = list(EX.CLASS_NAMES)
CLASS_COLOUR = ["#0072B2", "#D55E00", "#009E73"]
SPLIT_COLOUR = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 8.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


# --------------------------------------------------------------------------- #
# Gathering                                                                    #
# --------------------------------------------------------------------------- #
def global_splits() -> pd.DataFrame:
    """The held-out global val and test sets — identical for all nine experiments."""
    manifest = _read_json(EX.GLOBAL_DIR / "manifest.json")
    rows = []
    for split, s in (manifest.get("splits") or {}).items():
        pcp = s.get("per_class_patients") or []
        rows.append({
            "site": f"global_{split}", "split": split,
            "patients": s.get("patients"), "images": s.get("slices"),
            **{f"patients_{c}": (pcp[i] if i < len(pcp) else 0)
               for i, c in enumerate(CLASSES)},
            "trivial_baseline": s.get("trivial_baseline"),
        })
    return pd.DataFrame(rows)


def cohort_counts(partition: str, site: str) -> dict[str, int]:
    """Patients per cohort at one site, read from the site's own CSVs."""
    out: dict[str, int] = {}
    for split in ("train", "val"):
        csv = EX.PARTITIONS_DIR / partition / site / f"{split}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv, usecols=["pid", "cohort"]).drop_duplicates("pid")
        for cohort, n in df.cohort.value_counts().items():
            out[str(cohort)] = out.get(str(cohort), 0) + int(n)
    return out


def partition_frame(partition_name: str) -> pd.DataFrame:
    """One row per (site, split) for a federated partition."""
    pj = _read_json(EX.PARTITIONS_DIR / partition_name / "partition.json")
    rows = []
    for site in pj.get("sites", []):
        name = site.get("site")
        cohorts = cohort_counts(partition_name, name)
        for split in ("train", "val"):
            s = site.get(split) or {}
            pcp = s.get("per_class_patients") or []
            rows.append({
                "site": name, "split": split,
                "patients": s.get("patients", 0), "images": s.get("slices", 0),
                **{f"patients_{c}": (pcp[i] if i < len(pcp) else 0)
                   for i, c in enumerate(CLASSES)},
                "cohorts": json.dumps(cohorts),
            })
    return pd.DataFrame(rows)


def centralized_frame() -> pd.DataFrame:
    """Test 01 pools every hospital's data onto one machine.

    Built from the 4-client balanced partition because that partition is a complete
    cover of the training pool: summing its sites reproduces exactly the patients the
    centralised baseline trains on. Any of the four partitions would give the same
    total — checked by `verify_production.py`, which compares all four.
    """
    df = partition_frame("4_clients_balanced")
    pooled = []
    for split in ("train", "val"):
        sub = df[df.split == split]
        pooled.append({
            "site": "centralized", "split": split,
            "patients": int(sub.patients.sum()), "images": int(sub.images.sum()),
            **{f"patients_{c}": int(sub[f"patients_{c}"].sum()) for c in CLASSES},
            "cohorts": json.dumps({"spy2": int(sub.patients.sum())}),
        })
    return pd.DataFrame(pooled)


def frame_for(experiment) -> pd.DataFrame:
    return (centralized_frame() if experiment.partition is None
            else partition_frame(experiment.partition))


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
def _pct_labels(ax, bars, values, total, fmt="{:,.0f}\n{:.1f}%", tops=None):
    """Count and share above each bar. Both, because a percentage alone hides that
    hospital_3 in the skewed split holds 87 patients, and a count alone hides that
    this is 11% of the federation.

    `tops` is the height to sit above. It matters for stacked bars: `bar.get_height()`
    is only the bottom segment, so without it the total label lands inside the stack
    and reads as if it belonged to the train segment alone.
    """
    heights = tops if tops is not None else [b.get_height() for b in bars]
    headroom = 0.02 * (max(heights) if len(heights) else 1)
    for bar, v, top in zip(bars, values, heights):
        if not np.isfinite(v) or v == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, top + headroom,
                fmt.format(v, 100 * v / total if total else 0),
                ha="center", va="bottom", fontsize=6.5)


def experiment_figure(experiment, df: pd.DataFrame, gl: pd.DataFrame,
                      out: Path) -> None:
    sites = list(dict.fromkeys(df.site))
    train = df[df.split == "train"].set_index("site").reindex(sites)
    val = df[df.split == "val"].set_index("site").reindex(sites)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    algo = (experiment.algorithm or "centralized").upper()
    fig.suptitle(
        f"{experiment.id} — {experiment.name}   |   {algo}   |   "
        f"{experiment.n_clients} "
        f"{'site' if experiment.n_clients == 1 else 'hospitals'}   |   "
        f"{experiment.split_label}",
        fontsize=11, y=1.005)

    # 1. patients per hospital, train + val stacked -------------------------- #
    ax = axes[0][0]
    tp = train.patients.fillna(0).to_numpy(float)
    vp = val.patients.fillna(0).to_numpy(float)
    b1 = ax.bar(sites, tp, color=SPLIT_COLOUR["train"], label="train")
    ax.bar(sites, vp, bottom=tp, color=SPLIT_COLOUR["val"], label="local val")
    _pct_labels(ax, b1, tp + vp, (tp + vp).sum(), tops=tp + vp)
    ax.set_title("Patients per hospital", fontsize=9)
    ax.set_ylabel("patients"); ax.legend(fontsize=7)
    ax.set_ylim(0, (tp + vp).max() * 1.28)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    # 2. images per hospital ------------------------------------------------- #
    ax = axes[0][1]
    ti = train.images.fillna(0).to_numpy(float)
    vi = val.images.fillna(0).to_numpy(float)
    b2 = ax.bar(sites, ti, color=SPLIT_COLOUR["train"], label="train")
    ax.bar(sites, vi, bottom=ti, color=SPLIT_COLOUR["val"], label="local val")
    _pct_labels(ax, b2, ti + vi, (ti + vi).sum(), tops=ti + vi)
    ax.set_title("Images (slices) per hospital", fontsize=9)
    ax.set_ylabel("images"); ax.legend(fontsize=7)
    ax.set_ylim(0, (ti + vi).max() * 1.28)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    # 3. class distribution per hospital, absolute --------------------------- #
    ax = axes[0][2]
    bottom = np.zeros(len(sites))
    for i, c in enumerate(CLASSES):
        v = (train[f"patients_{c}"].fillna(0).to_numpy(float)
             + val[f"patients_{c}"].fillna(0).to_numpy(float))
        ax.bar(sites, v, bottom=bottom, color=CLASS_COLOUR[i], label=c)
        bottom += v
    ax.set_title("Class distribution per hospital (patients)", fontsize=9)
    ax.set_ylabel("patients"); ax.legend(fontsize=6.5)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    # 4. class distribution normalised — the stratification check ------------ #
    ax = axes[1][0]
    totals = bottom.copy()
    bottom = np.zeros(len(sites))
    for i, c in enumerate(CLASSES):
        v = (train[f"patients_{c}"].fillna(0).to_numpy(float)
             + val[f"patients_{c}"].fillna(0).to_numpy(float))
        share = 100 * np.divide(v, totals, out=np.zeros_like(v), where=totals > 0)
        bars = ax.bar(sites, share, bottom=bottom, color=CLASS_COLOUR[i], label=c)
        for bar, s in zip(bars, share):
            if s > 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + s / 2, f"{s:.1f}%", ha="center",
                        va="center", fontsize=6.5, color="white")
        bottom += share
    ax.set_title("Class share per hospital — flat bars = stratified\n"
                 "(only QUANTITY varies, not label distribution)", fontsize=8.5)
    ax.set_ylabel("% of the hospital's patients"); ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    # 5. train / local-val / global-test ------------------------------------- #
    ax = axes[1][1]
    test = gl[gl.split == "test"]
    names = ["train\n(all sites)", "local val\n(all sites)", "global test\n(held out)"]
    counts = [tp.sum(), vp.sum(),
              float(test.patients.iloc[0]) if len(test) else 0.0]
    colours = [SPLIT_COLOUR["train"], SPLIT_COLOUR["val"], SPLIT_COLOUR["test"]]
    bars = ax.bar(names, counts, color=colours)
    _pct_labels(ax, bars, counts, sum(counts))
    ax.set_title("Train / validation / test — patients", fontsize=9)
    ax.set_ylabel("patients"); ax.set_ylim(0, max(counts) * 1.28)
    ax.tick_params(axis="x", labelsize=7)

    # 6. cohort per hospital — single-cohort by design ----------------------- #
    ax = axes[1][2]
    all_cohorts: dict[str, np.ndarray] = {}
    for row_i, site in enumerate(sites):
        merged: dict[str, int] = {}
        for frame in (train, val):
            raw = frame.loc[site, "cohorts"] if site in frame.index else None
            for k, v in (json.loads(raw) if isinstance(raw, str) else {}).items():
                merged[k] = merged.get(k, 0) + v
        for k, v in merged.items():
            all_cohorts.setdefault(k, np.zeros(len(sites)))[row_i] += v
    bottom = np.zeros(len(sites))
    for i, (cohort, vals) in enumerate(sorted(all_cohorts.items())):
        ax.bar(sites, vals, bottom=bottom, color=CLASS_COLOUR[i % len(CLASS_COLOUR)],
               label=cohort)
        bottom += vals
    ax.set_title(f"Cohort per hospital — {len(all_cohorts)} cohort"
                 f"{'s' if len(all_cohorts) != 1 else ''} in this dataset\n"
                 "single-cohort BY DESIGN (see docstring)", fontsize=8.5)
    ax.set_ylabel("patients"); ax.legend(fontsize=6.5)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def overview_figure(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Balanced against skewed, side by side — the RQ2 contrast in one picture."""
    order = ["2_clients_balanced", "3_clients_balanced",
             "4_clients_balanced", "4_clients_skewed"]
    have = [p for p in order if p in frames]
    fig, axes = plt.subplots(1, len(have), figsize=(3.7 * len(have), 3.8),
                             squeeze=False)
    for ax, pname in zip(axes[0], have):
        df = frames[pname]
        sites = list(dict.fromkeys(df.site))
        total = df.groupby("site").patients.sum().reindex(sites).to_numpy(float)
        bars = ax.bar(sites, total,
                      color="#D55E00" if "skewed" in pname else "#0072B2")
        _pct_labels(ax, bars, total, total.sum())
        ax.set_title(EX.PARTITIONS[pname].label, fontsize=8)
        ax.set_ylabel("patients"); ax.set_ylim(0, total.max() * 1.3)
        ax.tick_params(axis="x", rotation=30, labelsize=6.5)
    fig.suptitle("Balanced vs skewed — total patients per hospital "
                 "(class ratios are identical everywhere; only quantity differs)",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def global_figure(gl: pd.DataFrame, out: Path) -> None:
    """The task itself: how many patients and images carry each class."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pooled = centralized_frame()

    ax = axes[0]
    vals = [float(pooled[f"patients_{c}"].sum()) for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=CLASS_COLOUR)
    _pct_labels(ax, bars, vals, sum(vals))
    ax.set_title("Patients per class — training pool", fontsize=9)
    ax.set_ylabel("patients"); ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15, labelsize=7)

    ax = axes[1]
    counts = images_per_class()
    vals = [float(counts.get(c, 0)) for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=CLASS_COLOUR)
    _pct_labels(ax, bars, vals, sum(vals))
    ax.set_title("Images per class — training pool", fontsize=9)
    ax.set_ylabel("images"); ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15, labelsize=7)

    ax = axes[2]
    test = gl[gl.split == "test"]
    vals = [float(test[f"patients_{c}"].iloc[0]) if len(test) else 0.0
            for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=CLASS_COLOUR)
    _pct_labels(ax, bars, vals, sum(vals))
    base = float(test.trivial_baseline.iloc[0]) if len(test) else float("nan")
    ax.axhline(max(vals), ls=":", color="grey", lw=1)
    ax.set_title(f"Patients per class — GLOBAL TEST SET\n"
                 f"trivial baseline {base:.4f} (always predict the majority)",
                 fontsize=8.5)
    ax.set_ylabel("patients"); ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15, labelsize=7)

    fig.suptitle("The classification task — three molecular subtypes", fontsize=10,
                 y=1.06)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def cohort_overview_figure(out: Path) -> None:
    """Patients and images by COHORT, and how the subtypes differ between them.

    This is the panel that justifies reading every result on this dataset with the
    source shortcut in mind: if DUKE is 64.6% HRposHER2neg and I-SPY2 is 38.8%, then
    "which cohort" carries much of "which subtype", and a model can reach a
    respectable macro-AUC without learning any biology.
    """
    src = EX.SOURCE_DATASET
    frames = [pd.read_csv(src / f"{s}.csv", usecols=["pid", "cohort", "label"])
              for s in ("train", "val", "test") if (src / f"{s}.csv").is_file()]
    rows = pd.concat(frames, ignore_index=True)
    pats = rows.drop_duplicates("pid")
    cohorts = sorted(pats.cohort.unique())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    vals = [float((pats.cohort == c).sum()) for c in cohorts]
    bars = ax.bar(cohorts, vals, color=CLASS_COLOUR[:len(cohorts)])
    _pct_labels(ax, bars, vals, sum(vals))
    ax.set_title("Patients per cohort", fontsize=9)
    ax.set_ylabel("patients"); ax.set_ylim(0, max(vals) * 1.3)

    ax = axes[1]
    vals = [float((rows.cohort == c).sum()) for c in cohorts]
    bars = ax.bar(cohorts, vals, color=CLASS_COLOUR[:len(cohorts)])
    _pct_labels(ax, bars, vals, sum(vals))
    ax.set_title("Images per cohort", fontsize=9)
    ax.set_ylabel("images"); ax.set_ylim(0, max(vals) * 1.3)

    ax = axes[2]
    bottom = np.zeros(len(cohorts))
    for i, name in enumerate(CLASSES):
        share = np.array([100 * ((pats.cohort == c) & (pats.label == i)).sum()
                          / max((pats.cohort == c).sum(), 1) for c in cohorts])
        b = ax.bar(cohorts, share, bottom=bottom, color=CLASS_COLOUR[i], label=name)
        for bar, v in zip(b, share):
            if v > 6:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2,
                        f"{v:.1f}%", ha="center", va="center", fontsize=6.5,
                        color="white")
        bottom += share
    ax.set_title("Subtype share WITHIN each cohort\n"
                 "differences here are the source shortcut", fontsize=8.5)
    ax.set_ylabel("% of the cohort's patients"); ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5)

    fig.suptitle("Cohort composition — I-SPY2 + I-SPY1 + DUKE combined",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def images_per_class() -> dict[str, int]:
    """Slice counts per class across the training pool, from the site CSVs.

    Counted from the CSVs rather than from `partition.json`, which records patients
    per class but only a total slice count — patients do not carry equal numbers of
    slices, so images-per-class cannot be derived from patients-per-class.
    """
    counts: dict[str, int] = {c: 0 for c in CLASSES}
    for site in EX.PARTITIONS["4_clients_balanced"].client_names:
        for split in ("train", "val"):
            csv = EX.PARTITIONS_DIR / "4_clients_balanced" / site / f"{split}.csv"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv, usecols=["label"])
            for label, n in df.label.value_counts().items():
                counts[CLASSES[int(label)]] += int(n)
    return counts


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", type=Path, default=EX.FIGURES_DIR)
    p.add_argument("--tables", type=Path, default=EX.DATASETS_DIR)
    args = p.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    gl = global_splits()
    print("=" * 74)
    print("DATASET DISTRIBUTION — nine experiments")
    print("=" * 74)

    combined, frames = [], {}
    for experiment in EX.EXPERIMENTS:
        df = frame_for(experiment)
        if experiment.partition:
            frames[experiment.partition] = df
        stem = f"{experiment.name}_distribution"
        experiment_figure(experiment, df, gl, args.figures / stem)

        table = df.copy()
        table.insert(0, "experiment", experiment.id)
        table.insert(1, "name", experiment.name)
        table.insert(2, "algorithm", experiment.algorithm or "centralized")
        table.insert(3, "partition", experiment.partition or "pooled")
        total_p = table.patients.sum()
        total_i = table.images.sum()
        table["pct_patients"] = (100 * table.patients / total_p).round(2)
        table["pct_images"] = (100 * table.images / total_i).round(2)
        table.to_csv(args.tables / f"{experiment.name}_distribution.csv", index=False)
        combined.append(table)

        sites = table.site.nunique()
        print(f"  {experiment.id}  {experiment.name:<24} {sites} site(s)  "
              f"{int(total_p):>4} patients  {int(total_i):>6,} images  "
              f"-> {stem}.png")

    allrows = pd.concat(combined, ignore_index=True)
    allrows.to_csv(args.tables / "all_distributions.csv", index=False)

    overview_figure(frames, args.figures / "overview_balanced_vs_skewed")
    global_figure(gl, args.figures / "overview_task_and_classes")
    cohort_overview_figure(args.figures / "overview_cohorts")
    gl.to_csv(args.tables / "global_splits.csv", index=False)

    (args.tables / "all_distributions.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(EX.SOURCE_DATASET),
        "classes": CLASSES,
        "split_rule": "patient-level; every slice of a patient goes to one hospital",
        "stratified": True,
        "note": ("All partitions are stratified, so hospitals differ in QUANTITY "
                 "only. Tests 08/09 are quantity skew, not label-distribution "
                 "heterogeneity."),
        "global_splits": gl.to_dict("records"),
        "experiments": {
            e.name: {
                "id": e.id, "algorithm": e.algorithm or "centralized",
                "partition": e.partition or "pooled",
                "n_hospitals": e.n_clients, "split_label": e.split_label,
                "rows": allrows[allrows.name == e.name].to_dict("records"),
            } for e in EX.EXPERIMENTS},
    }, indent=2, default=str))

    print(f"\n  figures -> {args.figures}")
    print(f"  tables  -> {args.tables}")
    print(f"\n  {len(EX.EXPERIMENTS)} experiment figures + 3 overviews")


if __name__ == "__main__":
    main()
