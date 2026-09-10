#!/usr/bin/env python3
"""Check every split in deployment/data/partitions against the rules the
dissertation claims for them, and against the August seed-42 format.

Run:  python3 deployment/code/scripts/check_splits.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# scripts/ -> code/ -> deployment/ -> repo root
ROOT = Path(__file__).resolve().parents[3]
PART = ROOT / "deployment/data/partitions"
SRC = ROOT / "dataset" / "multi_subtype_80mm_thesis"
# 2_clients_balanced is the one partition kept in the repository: it holds every
# training patient once, with the August 35-column rows, and its header is what
# every other split is checked against.
AUG = ROOT / "deployment/data/partitions"

AUGUST_HEADER = (AUG / "2_clients_balanced/hospital_1/train.csv").read_text().split("\n")[0]
POOL = pd.read_csv(SRC / "train.csv")
HELD = set(pd.read_csv(SRC / "val.csv").pid) | set(pd.read_csv(SRC / "test.csv").pid)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def image_pairs(part: Path):
    for site in sorted(p for p in part.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            for f in pd.read_csv(site / f"{split}.csv").filename:
                yield site / "images" / f, SRC / "images" / f


def one_partition(part: Path) -> dict:
    meta = json.loads((part / "partition.json").read_text())
    sites = sorted(p for p in part.iterdir() if p.is_dir())
    seen: Counter = Counter()
    rows = 0

    for site in sites:
        for split in ("train", "val"):
            csv = site / f"{split}.csv"
            head = csv.read_text().split("\n")[0]
            check(head == AUGUST_HEADER,
                  f"{csv.relative_to(ROOT)}: header is not the August schema")
            df = pd.read_csv(csv)
            check(set(df.split) == {split},
                  f"{csv.relative_to(ROOT)}: split column is {set(df.split)}")
            seen.update(df.pid.unique())
            rows += len(df)

        # A site's train and val must not share a patient.
        tr = set(pd.read_csv(site / "train.csv").pid)
        va = set(pd.read_csv(site / "val.csv").pid)
        check(not (tr & va), f"{site.name}: {len(tr & va)} patients in both "
                             f"its train and its val")

    dup = [p for p, n in seen.items() if n > 1]
    check(not dup, f"{part.name}: {len(dup)} patients in more than one hospital")
    check(len(seen) == POOL.pid.nunique(),
          f"{part.name}: {len(seen)} patients placed of {POOL.pid.nunique()}")
    check(not (set(seen) & HELD),
          f"{part.name}: {len(set(seen) & HELD)} training patients are also in "
          f"the global val/test set")
    check(rows == len(POOL),
          f"{part.name}: {rows} slices placed of {len(POOL)}")
    check(meta["total_patients"] == len(seen),
          f"{part.name}: partition.json says {meta['total_patients']} patients")

    pairs = list(image_pairs(part))
    with ThreadPoolExecutor(12) as pool:
        def chk(t):
            a, b = t
            if not a.exists():
                return "missing"
            if a.stat().st_ino == b.stat().st_ino:
                return "hardlink"
            return "ok" if sha(a) == sha(b) else "corrupt"
        res = Counter(pool.map(chk, pairs))
    check(res["ok"] == len(pairs),
          f"{part.name}: images {dict(res)}")

    return {"partition": part.name, "seed": meta["seed"], "mode": meta["mode"],
            "sites": len(sites), "patients": len(seen), "slices": rows,
            "images": res["ok"], "image_problems": len(pairs) - res["ok"]}


def main() -> None:
    parts = sorted(p for p in PART.iterdir() if p.is_dir())
    print(f"{'partition':<28}{'seed':>5}{'mode':>14}{'sites':>6}"
          f"{'patients':>10}{'slices':>9}{'images':>9}{'bad':>5}")
    for part in parts:
        r = one_partition(part)
        print(f"{r['partition']:<28}{r['seed']:>5}{r['mode']:>14}{r['sites']:>6}"
              f"{r['patients']:>10,}{r['slices']:>9,}{r['images']:>9,}"
              f"{r['image_problems']:>5}")

    print()
    if failures:
        print(f"FAILED — {len(failures)} problems")
        for f in failures:
            print("  ", f)
        sys.exit(1)
    print(f"OK — {len(parts)} partitions, every rule holds")


if __name__ == "__main__":
    main()
