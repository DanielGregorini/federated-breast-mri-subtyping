#!/usr/bin/env python3
"""Describe how the data is divided, for every one of the nine experiments.

    python deployment/code/scripts/build_distribution_report.py

Writes one figure per experiment to `deployment/figures/`, one table per experiment
to `deployment/datasets/`, and a combined summary in CSV and JSON.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

# PNG is the default output. PDF is vector and only needed for print, so it is
# written when --pdf is passed.
SAVE_PDF = False


def save_fig(fig, out: Path) -> None:
    """Write `out.png`, and `out.pdf` as well when --pdf was passed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    if SAVE_PDF:
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)



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


def cohort_counts(partition_dir: str, site: str) -> dict[str, int]:
    """Patients per cohort at one site, read from the site's own CSVs."""
    out: dict[str, int] = {}
    for split in ("train", "val"):
        csv = EX.PARTITIONS_DIR / partition_dir / site / f"{split}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv, usecols=["pid", "cohort"]).drop_duplicates("pid")
        for cohort, n in df.cohort.value_counts().items():
            out[str(cohort)] = out.get(str(cohort), 0) + int(n)
    return out


def partition_frame(partition_dir: str) -> pd.DataFrame:
    """One row per (site, split) for one built partition folder.

    Takes the FOLDER name, not the shape name, so a seed replica such as
    `4_clients_balanced_s19` is read from its own folder.
    """
    pj = _read_json(EX.PARTITIONS_DIR / partition_dir / "partition.json")
    rows = []
    for site in pj.get("sites", []):
        name = site.get("site")
        cohorts = cohort_counts(partition_dir, name)
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


def centralized_frame(suffix: str = "") -> pd.DataFrame:
    """Test 01 pools every hospital's data onto one machine.

    Built from the 4-client balanced partition because that partition is a complete
    cover of the training pool: summing its sites reproduces exactly the patients the
    centralised baseline trains on. Any of the four partitions would give the same
    total.
    """
    df = partition_frame(f"4_clients_balanced{suffix}")
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
    suffix = "" if experiment.seed is None else f"_s{experiment.seed}"
    return (centralized_frame(suffix) if experiment.partition is None
            else partition_frame(experiment.partition_dir))


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
    save_fig(fig, out)


def overview_figure(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Balanced against skewed, side by side."""
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
    save_fig(fig, out)


def global_figure(gl: pd.DataFrame, out: Path, suffix: str = "") -> None:
    """The task itself: how many patients and images carry each class."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pooled = centralized_frame(suffix)

    ax = axes[0]
    vals = [float(pooled[f"patients_{c}"].sum()) for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=CLASS_COLOUR)
    _pct_labels(ax, bars, vals, sum(vals))
    ax.set_title("Patients per class — training pool", fontsize=9)
    ax.set_ylabel("patients"); ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15, labelsize=7)

    ax = axes[1]
    counts = images_per_class(suffix)
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
    save_fig(fig, out)


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
    save_fig(fig, out)


def images_per_class(suffix: str = "") -> dict[str, int]:
    """Slice counts per class across the training pool, from the site CSVs.

    Counted from the CSVs rather than from `partition.json`, which records patients
    per class but only a total slice count — patients do not carry equal numbers of
    slices, so images-per-class cannot be derived from patients-per-class.
    """
    counts: dict[str, int] = {c: 0 for c in CLASSES}
    for site in EX.PARTITIONS["4_clients_balanced"].client_names:
        for split in ("train", "val"):
            csv = (EX.PARTITIONS_DIR / f"4_clients_balanced{suffix}" / site
                   / f"{split}.csv")
            if not csv.is_file():
                continue
            df = pd.read_csv(csv, usecols=["label"])
            for label, n in df.label.value_counts().items():
                counts[CLASSES[int(label)]] += int(n)
    return counts


def patient_assignments(seed: int) -> pd.DataFrame:
    """Every patient, in every partition, with the site and split it landed in.

    This is the listing that makes a split reproducible rather than merely
    described. `partition.json` says a site holds 306 patients; this says WHICH
    306. Read straight from the per-site CSVs, so it cannot disagree with what the
    clients actually load.

    Slice-level listings are the per-site `train.csv` and `val.csv` themselves and
    are not repeated here: they carry one row per image, and the split is decided
    per patient.
    """
    suffix = "" if seed == EX.TRAINING.seed else f"_s{seed}"
    rows = []
    for shape in EX.PARTITIONS:
        folder = f"{shape}{suffix}"
        pdir = EX.PARTITIONS_DIR / folder
        if not (pdir / "partition.json").is_file():
            continue
        for site in sorted(d.name for d in pdir.iterdir() if d.is_dir()):
            for split in ("train", "val"):
                csv = pdir / site / f"{split}.csv"
                if not csv.is_file():
                    continue
                cols = ["pid", "label", "label_name", "cohort", "filename"]
                df = pd.read_csv(csv)
                keep = [c for c in cols if c in df.columns]
                n_slices = df.groupby("pid").size().rename("n_slices")
                per_patient = df[keep].drop_duplicates("pid").join(n_slices, on="pid")
                per_patient.insert(0, "seed", seed)
                per_patient.insert(1, "partition_shape", shape)
                per_patient.insert(2, "partition_dir", folder)
                per_patient.insert(3, "site", site)
                per_patient.insert(4, "split", split)
                rows.append(per_patient.drop(columns=[c for c in ("filename",)
                                                      if c in per_patient.columns]))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["partition_shape", "site", "split", "pid"],
                           ignore_index=True)


def held_out_patients() -> pd.DataFrame:
    """The global validation and test patients.

    Identical for every seed. They come from the `test` column of the BreastDCEDL
    metadata rather than being drawn here, which is what makes results measured at
    different seeds comparable: the scoring set never moves.
    """
    rows = []
    for split in ("val", "test"):
        csv = EX.GLOBAL_DIR / f"{split}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        keep = [c for c in ("pid", "label", "label_name", "cohort") if c in df.columns]
        n_slices = df.groupby("pid").size().rename("n_slices")
        per_patient = df[keep].drop_duplicates("pid").join(n_slices, on="pid")
        per_patient.insert(0, "split", f"global_{split}")
        rows.append(per_patient)
    return (pd.concat(rows, ignore_index=True).sort_values(["split", "pid"],
                                                           ignore_index=True)
            if rows else pd.DataFrame())


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def seed_readme(seed: int, allrows: pd.DataFrame, gl: pd.DataFrame,
                assignments: pd.DataFrame) -> str:
    """The documentation written beside each seed's tables. Numbers come from the
    tables themselves, so the text cannot drift from the data it describes."""
    is_base = seed == EX.TRAINING.seed
    tr = EX.TRAINING

    glob_rows = "\n".join(
        f"| {r['split']} | {int(r['patients'])} | {int(r['images'])} | "
        + " | ".join(str(int(r.get(f'patients_{c}', 0))) for c in CLASSES) + " |"
        for _, r in gl.iterrows())

    fed = allrows[allrows.partition_shape != "pooled"]
    shape_rows = []
    for shape in sorted(fed.partition_shape.unique()):
        sub = fed[fed.partition_shape == shape]
        part = EX.PARTITIONS[shape]
        pj = _read_json(EX.PARTITIONS_DIR / sub.partition_dir.iloc[0] /
                        "partition.json")
        shape_rows.append(
            f"| `{sub.partition_dir.iloc[0]}` | {part.n_clients} | {pj.get('mode')} "
            f"| {int(sub.patients.sum() / sub.name.nunique())} "
            f"| {int(sub.images.sum() / sub.name.nunique()):,} |")

    site_rows = []
    for shape in sorted(fed.partition_shape.unique()):
        sub = fed[fed.partition_shape == shape].drop_duplicates(["site", "split"])
        for _, r in sub.sort_values(["site", "split"]).iterrows():
            site_rows.append(
                f"| `{shape}` | {r['site']} | {r['split']} | {int(r['patients'])} "
                f"| {int(r['images']):,} | "
                + " | ".join(str(int(r[f'patients_{c}'])) for c in CLASSES) + " |")

    n_shapes = fed.partition_shape.nunique()
    return f"""# Dataset organization — seed {seed}

How the data is divided for every experiment run at seed {seed}, and the listings
that make that division reproducible.

{'This is the original split. Every result in `results/thesis/` was measured on it.'
 if is_base else
 'A replica of the seed ' + str(tr.seed) + ' protocol. Same code, same model, same '
 'hyperparameters, same training loop and same evaluation. The only thing that '
 'differs is which training patients each hospital holds.'}

## What the seed changes, and what it does not

The seed controls two things and nothing else:

1. Which of the {int(gl[gl.split == 'train'].patients.sum()) if 'train' in set(gl.split) else 1527} training patients each hospital receives.
2. Which of its own patients each hospital holds back as its local validation split.

It does not touch the global validation and test sets. Those are read from the
`test` column of the BreastDCEDL metadata, not drawn here, so they are identical at
every seed and results measured at different seeds are directly comparable.

It does not touch the images. `partition_data.py` hardlinks the PNGs that
`dataset/multi_subtype_80mm` already holds. Nothing is resampled, re-cropped,
re-normalised or re-encoded.

It does not touch any hyperparameter. Model `{tr.model_name}`, ImageNet-pretrained,
frozen up to `{tr.freeze_until}`, {tr.image_size}x{tr.image_size} input,
{tr.optimizer} at lr {tr.learning_rate}, weight decay {tr.weight_decay}, dropout
{tr.dropout}, label smoothing {tr.label_smoothing}, batch {tr.batch_size} with at
most {tr.max_slices_per_patient_per_batch} slice per patient,
{EX.FEDERATION.num_rounds} rounds x {EX.FEDERATION.local_epochs} local epoch,
selection on `{EX.FEDERATION.key_metric}`, patient-level `{tr.aggregation}`
aggregation. All of it comes from `config/experiments.py` and is shared with every
other seed.

## The held-out sets, identical at every seed

| split | patients | images | {' | '.join(CLASSES)} |
|---|---:|---:|{'---:|' * len(CLASSES)}
{glob_rows}

## The partitions

{n_shapes} shapes, each covering all training patients exactly once.

| folder | hospitals | mode | patients | images |
|---|---:|---|---:|---:|
{chr(10).join(shape_rows)}

`mode` is how patients are dealt out. `stratified` splits within each class, so
every site keeps the global class ratio and the sites differ only in quantity.
`cohort` gives each site one whole source cohort, which is the only genuinely
non-IID partition here. A cohort partition's site membership does not depend on the
seed, because the cohorts are fixed; only its local validation split moves.

## Per hospital and per split

| partition | hospital | split | patients | images | {' | '.join(CLASSES)} |
|---|---|---|---:|---:|{'---:|' * len(CLASSES)}
{chr(10).join(site_rows)}

## The files

| file | what it holds |
|---|---|
| `all_distributions.csv` | one row per experiment, hospital and split, with patient and image counts, per-class counts, cohort counts and percentages |
| `all_distributions.json` | the same, nested per experiment, plus the split rules |
| `global_splits.csv` | the held-out validation and test sets |
| `global_held_out_patients.csv` | every held-out patient, with label, cohort and slice count |
| `patient_assignments.csv` | **the sample listing**: {len(assignments):,} rows, one per patient per partition, saying which hospital and which split it landed in |
| `figures/` | one distribution figure per experiment, plus three overviews |

Slice-level listings are the per-site `train.csv` and `val.csv` under
`deployment/data/partitions/<folder>/<hospital>/`. They carry one row per image and
are what the clients actually read. This folder describes them; it does not replace
them.

## How to rebuild it

```bash
python deployment/code/scripts/partition_data.py --seed {seed}{'' if is_base else f' --suffix _s{seed}'} --hardlink
python deployment/code/scripts/build_distribution_report.py --seed {seed}
```

The allocation is deterministic given the seed, so this reproduces the same split.
"""


def build_for_seed(seed: int, out_root: Path, gl: pd.DataFrame) -> pd.DataFrame:
    """Every figure and table describing how the data is divided at one seed."""
    suffix = "" if seed == EX.TRAINING.seed else f"_s{seed}"
    out = out_root / f"seed_{seed}"
    figures, tables = out / "figures", out
    figures.mkdir(parents=True, exist_ok=True)

    experiments = [e for e in EX.all_runs() if e.train_seed == seed]

    print("=" * 74)
    print(f"DATASET ORGANIZATION — seed {seed}")
    print("=" * 74)

    combined, frames = [], {}
    for experiment in experiments:
        df = frame_for(experiment)
        if experiment.partition:
            frames[experiment.partition] = df
        stem = f"{experiment.name}_distribution"
        experiment_figure(experiment, df, gl, figures / stem)

        table = df.copy()
        table.insert(0, "seed", seed)
        table.insert(1, "experiment", experiment.id)
        table.insert(2, "name", experiment.name)
        table.insert(3, "algorithm", experiment.algorithm or "centralized")
        table.insert(4, "partition_shape", experiment.partition or "pooled")
        table.insert(5, "partition_dir", experiment.partition_dir or "pooled")
        total_p = table.patients.sum()
        total_i = table.images.sum()
        table["pct_patients"] = (100 * table.patients / total_p).round(2)
        table["pct_images"] = (100 * table.images / total_i).round(2)
        combined.append(table)

        sites = table.site.nunique()
        print(f"  {experiment.id:<12} {experiment.name:<32} {sites} site(s)  "
              f"{int(total_p):>4} patients  {int(total_i):>6,} images")

    allrows = pd.concat(combined, ignore_index=True)
    allrows.to_csv(tables / "all_distributions.csv", index=False)

    overview_figure(frames, figures / "overview_balanced_vs_skewed")
    global_figure(gl, figures / "overview_task_and_classes", suffix)
    cohort_overview_figure(figures / "overview_cohorts")
    gl.to_csv(tables / "global_splits.csv", index=False)

    # The listings. These are what let somebody rebuild or audit the split.
    assignments = patient_assignments(seed)
    assignments.to_csv(tables / "patient_assignments.csv", index=False)
    held_out_patients().to_csv(tables / "global_held_out_patients.csv", index=False)
    (tables / "README.md").write_text(seed_readme(seed, allrows, gl, assignments))

    (tables / "all_distributions.json").write_text(json.dumps({
        "seed": seed,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(EX.SOURCE_DATASET),
        "classes": CLASSES,
        "split_rule": "patient-level; every slice of a patient goes to one hospital",
        "global_split_rule": ("read from the `test` column of the BreastDCEDL "
                              "metadata, not drawn here, so it does not depend on "
                              "the seed"),
        "note": ("Stratified partitions vary QUANTITY only. 3_clients_cohort is one "
                 "real cohort per site and its site membership does not depend on "
                 "the seed; only its local validation split does."),
        "global_splits": gl.to_dict("records"),
        "experiments": {
            e.name: {
                "id": e.id, "seed": e.train_seed,
                "algorithm": e.algorithm or "centralized",
                "partition_shape": e.partition or "pooled",
                "partition_dir": e.partition_dir or "pooled",
                "n_hospitals": e.n_clients, "split_label": e.split_label,
                "rows": allrows[allrows.name == e.name].to_dict("records"),
            } for e in experiments},
    }, indent=2, default=str))

    print(f"  -> {out}")
    return allrows


def seed_comparison(per_seed: dict[int, pd.DataFrame], out: Path) -> pd.DataFrame:
    """One row per shape, site and split, with the counts side by side per seed.

    The point of the table is to show what did NOT change. Patients per site and
    per class are fixed by the stratified allocation, so they must match across
    seeds; only WHICH patients, and therefore the image counts, move.
    """
    rows = []
    for seed, df in sorted(per_seed.items()):
        sub = df[df.partition_shape != "pooled"].copy()
        sub["seed"] = seed
        rows.append(sub[["seed", "partition_shape", "site", "split", "patients",
                         "images"] + [f"patients_{c}" for c in CLASSES]])
    long = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["seed", "partition_shape", "site", "split"])
    wide = long.pivot_table(index=["partition_shape", "site", "split"],
                            columns="seed",
                            values=["patients", "images"] +
                                   [f"patients_{c}" for c in CLASSES])
    wide.columns = [f"{a}_seed{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out, index=False)
    return wide


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=EX.FIGURES_DIR,
                   help="root for the per-seed folders")
    p.add_argument("--seed", type=int, action="append", default=None, metavar="N",
                   help="describe this seed only. Repeatable. Default: every seed "
                        "in config/experiments.py::SEEDS that has been built.")
    p.add_argument("--pdf", action="store_true",
                   help="also write a vector .pdf beside every .png")
    args = p.parse_args()

    global SAVE_PDF
    SAVE_PDF = args.pdf

    seeds = args.seed or list(EX.SEEDS)
    gl = global_splits()

    built = {}
    for seed in seeds:
        suffix = "" if seed == EX.TRAINING.seed else f"_s{seed}"
        probe = EX.PARTITIONS_DIR / f"4_clients_balanced{suffix}" / "partition.json"
        if not probe.is_file():
            print(f"seed {seed}: not built, skipped "
                  f"(partition_data.py --seed {seed} --suffix _s{seed})")
            continue
        built[seed] = build_for_seed(seed, args.out, gl)

    if len(built) > 1:
        wide = seed_comparison(built, args.out / "seed_comparison.csv")
        print(f"\n  cross-seed comparison: {len(wide)} rows -> "
              f"{args.out / 'seed_comparison.csv'}")

    print(f"\n  {len(built)} seed(s) described under {args.out}")


if __name__ == "__main__":
    main()
