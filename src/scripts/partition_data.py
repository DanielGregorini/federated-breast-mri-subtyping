#!/usr/bin/env python3
"""Divide the training patients between hospitals. One physical folder per site.

    python scripts/partition_data.py                     # all four partitions
    python scripts/partition_data.py --only 4_clients_skewed

WHAT THIS PRODUCES
------------------
    data/partitions/4_clients_balanced/
    ├── hospital_1/
    │   ├── images/<pid>/slice_NNN.png
    │   ├── train.csv
    │   ├── val.csv          20% of THIS hospital's patients
    │   └── manifest.json
    ├── hospital_2/ ...
    └── partition.json       what was split, how, and the leakage checks

Each hospital folder holds only that hospital's patients, as real files. It costs
disk and buys two things: the layout is exactly what would be `rsync`-ed to a real
hospital machine, and it is impossible for a bug to let one site read another's
data — the files are not there.

THREE RULES, ENFORCED HERE AND RE-CHECKED BY verify_data.py
-----------------------------------------------------------
1. Split by PATIENT, never by slice. Every image of a patient goes to one site.
   Splitting by slice would let a model recognise the patient rather than the
   disease, and it is the exact leak an earlier phase of this project shipped.
2. No patient in two hospitals, and no training patient in the global test set.
3. Each hospital keeps its own validation split, carved from its own patients. It
   is local by construction: a hospital cannot validate on another's patients, and
   that is what makes the metric the server selects on honest.

WHY STRATIFIED, AND WHAT THAT COSTS
-----------------------------------
Every hospital keeps the global class ratio, so the only thing that varies between
sites is QUANTITY. This is a deliberate limitation and the dissertation states it:
tests 08 and 09 are quantity skew, not genuine non-IID heterogeneity, which is why
the previous run of these experiments found no detectable RQ2 effect.

`--stratify none` produces label skew instead, and `--by-cohort` assigns one real
cohort per hospital — DUKE at 64.6% HRposHER2neg against I-SPY2's 38.8%, with
tumours five times smaller by volume. That is real heterogeneity, and it is the
strongest available upgrade to RQ2. It needs a pooled source dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))
# `from scripts.prepare_data import ...` below needs src/ on the path,
# which is the parent of this file's own directory.
SCRIPTS_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_PARENT))

from config import experiments as EX  # noqa: E402


def portable(path) -> str:
    """A path relative to the repository root, for recording in a manifest.

    These manifests are provenance records that travel with the repository, so
    an absolute path in one is wrong the moment anybody clones it somewhere
    else — or the moment the project folder is renamed. Relative to the root it
    stays true, and it is still enough to say which dataset a partition came
    from.
    """
    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(EX.REPO_ROOT))
    except ValueError:
        return str(path)
from scripts.prepare_data import copy_patient_images  # noqa: E402


def stratified_shares(patients: pd.DataFrame, fractions: tuple[float, ...],
                      rng: np.random.Generator) -> list[list]:
    """Split patient ids across sites, keeping each site's class ratio global.

    Done per class rather than globally: shuffling all patients and cutting at the
    fraction boundaries gives each site the right SIZE but lets its class ratio
    wander, which would silently turn a "balanced" test into a mild label-skew test.
    """
    buckets: list[list] = [[] for _ in fractions]
    for label, group in patients.groupby("label"):
        ids = group.pid.to_numpy().copy()
        rng.shuffle(ids)
        # Largest-remainder allocation. Plain rounding loses or invents patients
        # when the shares do not divide evenly, and the 5:2:1:1 split does not.
        exact = np.array(fractions) * len(ids)
        take = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - take))[:len(ids) - take.sum()]:
            take[i] += 1
        start = 0
        for i, n in enumerate(take):
            buckets[i].extend(ids[start:start + n].tolist())
            start += n
    return buckets


def unstratified_shares(patients: pd.DataFrame, fractions: tuple[float, ...],
                        rng: np.random.Generator) -> list[list]:
    """Split by size only, ignoring class — produces label skew as a side effect."""
    ids = patients.pid.to_numpy().copy()
    rng.shuffle(ids)
    exact = np.array(fractions) * len(ids)
    take = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - take))[:len(ids) - take.sum()]:
        take[i] += 1
    out, start = [], 0
    for n in take:
        out.append(ids[start:start + n].tolist())
        start += n
    return out


def cohort_shares(patients: pd.DataFrame, n_clients: int) -> list[list]:
    """One real cohort per hospital. Genuine non-IID: the sites differ in class
    prior, tumour size and scanner, not merely in how many patients they hold."""
    if "cohort" not in patients.columns:
        raise SystemExit("--by-cohort needs a `cohort` column; the source dataset "
                         "is single-cohort. Use a pooled dataset.")
    cohorts = sorted(patients.cohort.unique())
    if len(cohorts) != n_clients:
        raise SystemExit(f"--by-cohort needs one hospital per cohort: "
                         f"{len(cohorts)} cohorts ({', '.join(cohorts)}) but "
                         f"{n_clients} hospitals.")
    return [patients[patients.cohort == c].pid.tolist() for c in cohorts]


def carve_local_val(pids: list, patients: pd.DataFrame, fraction: float,
                    rng: np.random.Generator) -> tuple[list, list]:
    """Hold out `fraction` of a site's patients, stratified, at least one per class.

    Stratified because a site holding 39 patients can easily draw a validation
    split missing a whole class, and `compute_metrics` then returns NaN for AUC —
    the site would report nothing for the metric the server selects on.
    """
    sub = patients[patients.pid.isin(pids)]
    val: list = []
    for _, group in sub.groupby("label"):
        ids = group.pid.to_numpy().copy()
        rng.shuffle(ids)
        n = max(1, int(round(fraction * len(ids)))) if len(ids) > 1 else 0
        val.extend(ids[:n].tolist())
    train = [p for p in pids if p not in set(val)]
    return train, val


def write_site(site_dir: Path, rows: pd.DataFrame, train_pids: list, val_pids: list,
               src_images: Path, hardlink: bool, extra: dict) -> dict:
    site_dir.mkdir(parents=True, exist_ok=True)
    summary = {"site": site_dir.name, **extra}
    for split, pids in (("train", train_pids), ("val", val_pids)):
        sub = rows[rows.pid.isin(pids)].copy()
        sub["split"] = split
        sub.to_csv(site_dir / f"{split}.csv", index=False)
        copy_patient_images(pids, src_images, site_dir / "images", hardlink)
        counts = sub.drop_duplicates("pid").label.value_counts()
        summary[split] = {
            "patients": len(pids),
            "slices": int(len(sub)),
            "per_class_patients": [int(counts.get(c, 0))
                                   for c in range(EX.NUM_CLASSES)],
        }
    (site_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_partition(name: str, partition, rows: pd.DataFrame, patients: pd.DataFrame,
                    src_images: Path, out_root: Path, args) -> dict:
    rng = np.random.default_rng(args.seed)
    print(f"\n{partition.describe()}")

    if args.by_cohort:
        shares = cohort_shares(patients, partition.n_clients)
        mode = "cohort"
    elif partition.stratified and args.stratify != "none":
        shares = stratified_shares(patients, partition.fractions, rng)
        mode = "stratified"
    else:
        shares = unstratified_shares(patients, partition.fractions, rng)
        mode = "unstratified"

    # Global class weights, computed once from the POOLED training split so no site
    # has to see another site's data to use them. Written into every manifest; the
    # client uses them only when class_weight_scope == "global". See RQ4.
    per_patient = patients.label.to_numpy()
    counts = np.bincount(per_patient, minlength=EX.NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    global_weights = (len(per_patient) / (EX.NUM_CLASSES * counts)).tolist()

    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    sites, all_pids = [], []
    for i, pids in enumerate(shares):
        train_pids, val_pids = carve_local_val(
            pids, patients, EX.FEDERATION.local_val_fraction, rng)
        site = write_site(
            out_dir / partition.client_names[i], rows, train_pids, val_pids,
            src_images, args.hardlink,
            {"global_class_weights": [round(w, 6) for w in global_weights],
             "partition": name, "mode": mode})
        sites.append(site)
        all_pids.extend(pids)
        print(f"  {site['site']:<12} train {site['train']['patients']:>4} pat "
              f"{str(site['train']['per_class_patients']):<16} "
              f"val {site['val']['patients']:>3} pat "
              f"{site['train']['slices'] + site['val']['slices']:>6,} slices")

    # A partition that lost or duplicated a patient must fail here, not silently
    # produce a federation training on 3% less data than the baseline it is
    # compared against.
    if len(all_pids) != len(set(all_pids)):
        raise SystemExit(f"{name}: a patient landed in two hospitals")
    if len(all_pids) != len(patients):
        raise SystemExit(f"{name}: {len(all_pids)} patients placed of "
                         f"{len(patients)} available")

    meta = {
        "partition": name, "mode": mode, "n_clients": partition.n_clients,
        "ratio": list(partition.ratio), "fractions": list(partition.fractions),
        "seed": args.seed, "source": portable(src_images.parent),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "local_val_fraction": EX.FEDERATION.local_val_fraction,
        "global_class_weights": global_weights,
        "total_patients": len(all_pids), "sites": sites,
    }
    (out_dir / "partition.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=EX.SOURCE_DATASET)
    p.add_argument("--out", type=Path, default=EX.PARTITIONS_DIR)
    p.add_argument("--only", default=None, help="build one partition by name")
    p.add_argument("--stratify", choices=["class", "none"], default="class")
    p.add_argument("--by-cohort", action="store_true",
                   help="one real cohort per hospital — genuine non-IID for RQ2")
    p.add_argument("--hardlink", action="store_true")
    p.add_argument("--seed", type=int, default=EX.TRAINING.seed)
    args = p.parse_args()

    src = Path(args.source)
    train_csv = src / "train.csv"
    if not train_csv.is_file():
        raise SystemExit(f"{src} is not a prepared dataset (no train.csv)")

    rows = pd.read_csv(train_csv)
    patients = rows.drop_duplicates("pid")[
        ["pid", "label"] + (["cohort"] if "cohort" in rows.columns else [])]
    print(f"source: {src}")
    print(f"training pool: {len(patients)} patients, {len(rows):,} slices")
    counts = patients.label.value_counts().sort_index()
    print("per class: " + ", ".join(
        f"{EX.CLASS_NAMES[c]} {counts.get(c, 0)}" for c in range(EX.NUM_CLASSES)))

    names = [args.only] if args.only else list(EX.PARTITIONS)
    built = {}
    for name in names:
        if name not in EX.PARTITIONS:
            raise SystemExit(f"unknown partition {name!r}. "
                             f"Known: {', '.join(EX.PARTITIONS)}")
        built[name] = build_partition(name, EX.PARTITIONS[name], rows, patients,
                                      src / "images", Path(args.out), args)

    print(f"\nwritten: {args.out}")
    print("next: python scripts/verify_data.py")


if __name__ == "__main__":
    main()
