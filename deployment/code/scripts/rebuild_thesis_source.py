#!/usr/bin/env python3
"""Rebuild the source dataset exactly as it was on 2026-08-05, when the seed-42
splits were made.

WHY THIS EXISTS
---------------
`dataset/multi_subtype_80mm/` on disk today is NOT the dataset the seed-42
partitions were built from. It was rebuilt on 2026-08-14 and two things changed:

  * the CSV schema lost eight columns
        roi_basis, box3d_row0, box3d_row1, box3d_col0, box3d_col1,
        crop_center_row, crop_center_col, crop_mm
    (35 columns in August, 27 today)
  * 1,352 of the 12,131 training PNGs have different bytes, across 827 of the
    1,527 patients, and `mm_per_px` moved by up to 0.00297

Partitioning the seed-19 and seed-50 splits off today's dataset would therefore
give files in a different format, describing slightly different images, from the
ones every seed-42 result was measured on. The seeds would not be comparable.

The August data is still on disk, in the seed-42 partitions themselves. Every
one of the 1,527 training patients appears exactly once across the four CSVs of
`2_clients_balanced`, and the held-out global val/test rows are in
`deployment/data/global/`. Together they are the whole August dataset, and the
PNGs under those folders are the August bytes (the 2026-08-14 rebuild wrote new
inodes, so the hardlinks placed in August kept the old content).

So this script puts the August dataset back together from its own output.

WHAT IT WRITES
--------------
    dataset/multi_subtype_80mm_thesis/
    ├── config.json
    ├── metadata.csv      16,378 rows   all splits
    ├── train.csv         12,131 rows   the pool the seeds divide up
    ├── val.csv            2,132 rows   global validation, not seeded
    ├── test.csv           2,115 rows   global test, not seeded
    └── images/<pid>/slice_NNN.png      2,063 patients, physical copies

Row order is the source's own order, recovered from today's dataset: every one
of the four August partition CSVs is monotonic in it, which can only happen if
it is the order the August source had.

Physical copies, never hardlinks. A hardlink here would tie the new splits to
the old ones and a rebuild of either would silently change both.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# scripts/ -> code/ -> deployment/ -> repo root
ROOT = Path(__file__).resolve().parents[3]
AUG_PARTITION = ROOT / "deployment/data/partitions/2_clients_balanced"
GLOBAL = ROOT / "deployment/data/global"
TODAY = ROOT / "dataset/multi_subtype_80mm"
# Rebuilt into the dataset folder beside the current build, not over it: the two
# describe the same slices but differ in eight CSV columns and in 1,352 PNGs.
OUT = ROOT / "dataset" / "multi_subtype_80mm_thesis"

AUGUST_COLUMNS = 35


def august_training_table() -> pd.DataFrame:
    """The 12,131 August training rows, in the August source's own order."""
    frames = [pd.read_csv(AUG_PARTITION / site / f"{split}.csv")
              for site in ("hospital_1", "hospital_2")
              for split in ("train", "val")]
    rows = pd.concat(frames, ignore_index=True)

    order = {f: i for i, f in enumerate(pd.read_csv(TODAY / "train.csv").filename)}
    if set(order) != set(rows.filename):
        raise SystemExit("the August partitions and today's dataset do not "
                         "describe the same slices")
    rows["_o"] = rows.filename.map(order)
    rows = rows.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    # The partitions carry the LOCAL split (a fifth of each site is its own
    # validation). In the source these are all training patients, which is what
    # the seed reads and re-divides.
    rows["split"] = "train"
    return rows


def copy_patient_images(pids, src_images: Path, dst_images: Path) -> int:
    """Physical copies, one patient at a time. Returns files placed."""
    def one(pid: str) -> int:
        src, dst = src_images / pid, dst_images / pid
        if not src.is_dir():
            raise FileNotFoundError(f"no image folder for patient {pid}: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dst / png.name)
            n += 1
        return n

    with ThreadPoolExecutor(8) as pool:
        return sum(pool.map(one, [str(p) for p in pids]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)

    train = august_training_table()
    val = pd.read_csv(GLOBAL / "val.csv")
    test = pd.read_csv(GLOBAL / "test.csv")

    for name, df in (("train", train), ("val", val), ("test", test)):
        if len(df.columns) != AUGUST_COLUMNS:
            raise SystemExit(f"{name}.csv has {len(df.columns)} columns, "
                             f"expected the August {AUGUST_COLUMNS}")
    if list(val.columns) != list(train.columns) != list(test.columns):
        raise SystemExit("the three splits disagree on their columns")

    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)

    # metadata.csv is every split in one table, in today's metadata order.
    order = {f: i for i, f in enumerate(pd.read_csv(TODAY / "metadata.csv").filename)}
    meta = pd.concat([train, val, test], ignore_index=True)
    meta["_o"] = meta.filename.map(order)
    meta = meta.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    meta.to_csv(OUT / "metadata.csv", index=False)

    cfg = json.loads((TODAY / "config.json").read_text())
    cfg["_note"] = (
        "Copied from the 2026-08-14 rebuild. The build PARAMETERS did not change "
        "between the two builds; the extraction code did, which is why the CSV "
        "schema and 1,352 PNGs differ. The images and tables beside this file are "
        "the 2026-08-05 ones, recovered from the seed-42 partitions."
    )
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    # Images. Training patients come from the August partition (both sites hold
    # part of the pool), val and test from the August global folder.
    placed = 0
    for site in ("hospital_1", "hospital_2"):
        pids = pd.concat([pd.read_csv(AUG_PARTITION / site / f"{s}.csv")
                          for s in ("train", "val")]).pid.unique()
        placed += copy_patient_images(pids, AUG_PARTITION / site / "images",
                                      OUT / "images")
        print(f"  images from {site:<12} {placed:>6,} placed so far")
    held = pd.concat([val, test]).pid.unique()
    placed += copy_patient_images(held, GLOBAL / "images", OUT / "images")
    print(f"  images from global       {placed:>6,} placed in total")

    print(f"\nsource rebuilt: {OUT.relative_to(ROOT)}")
    print(f"  train {len(train):>6,} rows   {train.pid.nunique():>5,} patients")
    print(f"  val   {len(val):>6,} rows   {val.pid.nunique():>5,} patients")
    print(f"  test  {len(test):>6,} rows   {test.pid.nunique():>5,} patients")
    print(f"  columns {len(train.columns)}")
    print(f"  images  {placed:,} files, {len(list((OUT / 'images').iterdir())):,} patients")


if __name__ == "__main__":
    main()
