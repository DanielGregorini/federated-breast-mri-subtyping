#!/usr/bin/env python3
"""Build `data/global/` — the held-out sets that every experiment is scored on.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --source ../dataset/mine_subtype_pooled

WHAT THIS PRODUCES
------------------
    data/global/
    ├── images/<pid>/slice_NNN.png
    ├── test.csv      the official set. Identical for all nine experiments.
    ├── val.csv       the centralised baseline's selection set, and the optional
    │                 per-round convergence curve of the aggregated model.
    └── manifest.json what was built, from what, when.

WHY THE SERVER HOLDS DATA AT ALL
--------------------------------
In a production federation the aggregation server usually holds nothing. Here it
holds a held-out test set, because the nine experiments have to be compared on
identical ground and a test set assembled from per-hospital leftovers would differ
between a 2-client and a 4-client run. That is a benchmarking decision, not a claim
about deployment, and the dissertation states it as such.

The split is NOT re-drawn. It is the split already used by the classifier phase, so
the federated results and the centralised numbers in `all_runs_pod.csv` are measured
on the same 99 patients. Re-drawing it would make every comparison in this project
approximate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
import pathlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

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


def copy_patient_images(pids, src_images: Path, dst_images: Path,
                        hardlink: bool) -> int:
    """Copy one patient's slices at a time. Returns the number of files placed."""
    n = 0
    for pid in pids:
        src = src_images / str(pid)
        dst = dst_images / str(pid)
        if not src.is_dir():
            raise FileNotFoundError(f"no image folder for patient {pid}: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        for png in src.glob("*.png"):
            target = dst / png.name
            if target.exists():
                continue
            if hardlink:
                target.hardlink_to(png)
            else:
                shutil.copy2(png, target)
            n += 1
    return n


def write_split(rows: pd.DataFrame, out_dir: Path, split: str, src_images: Path,
                hardlink: bool) -> dict:
    rows = rows.copy()
    rows["split"] = split
    rows.to_csv(out_dir / f"{split}.csv", index=False)
    pids = rows.pid.unique()
    n_files = copy_patient_images(pids, src_images, out_dir / "images", hardlink)
    counts = rows.drop_duplicates("pid").label.value_counts().sort_index()
    summary = {
        "split": split,
        "patients": int(len(pids)),
        "slices": int(len(rows)),
        "files_placed": n_files,
        "per_class_patients": [int(counts.get(c, 0)) for c in range(EX.NUM_CLASSES)],
        "trivial_baseline": float(
            rows.drop_duplicates("pid").label.value_counts().max() / len(pids)),
    }
    print(f"  {split:<5} {summary['patients']:>4} patients  "
          f"{summary['slices']:>6,} slices  per-class "
          f"{summary['per_class_patients']}  trivial "
          f"{summary['trivial_baseline']:.4f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=EX.SOURCE_DATASET,
                   help="the processed dataset built by notebooks/02_build_dataset.ipynb")
    p.add_argument("--out", type=Path, default=EX.GLOBAL_DIR)
    p.add_argument("--hardlink", action="store_true",
                   help="hardlink instead of copying. Each site still has its own "
                        "path and still cannot read another's folder; this only "
                        "avoids storing the same immutable PNG many times.")
    p.add_argument("--force", action="store_true", help="overwrite an existing build")
    args = p.parse_args()

    src = Path(args.source)
    if not (src / "train.csv").is_file():
        raise SystemExit(f"{src} is not a prepared dataset (no train.csv).\n"
                         "  Build it with notebooks/02_build_dataset.ipynb.")

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit(f"{args.out} already exists. Pass --force to rebuild.")
    if args.out.exists() and args.force:
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    print(f"source: {src}")
    print(f"target: {args.out}\n")

    manifest = {
        "source": portable(src),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardlinked": bool(args.hardlink),
        "classes": EX.CLASS_NAMES,
        "splits": {},
    }
    for split in ("test", "val"):
        csv = src / f"{split}.csv"
        if not csv.is_file():
            print(f"  {split:<5} not present in source — skipped")
            continue
        manifest["splits"][split] = write_split(
            pd.read_csv(csv), args.out, split, src / "images", args.hardlink)

    if "test" not in manifest["splits"]:
        raise SystemExit(f"{src} has no test.csv — there is nothing to score on.")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwritten: {args.out}")
    print("next: python scripts/partition_data.py")


if __name__ == "__main__":
    main()
