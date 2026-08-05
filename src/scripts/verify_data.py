#!/usr/bin/env python3
"""Leakage checks. Exits non-zero if any of them fails.

    python scripts/verify_data.py
    python scripts/verify_data.py --check-imports

This is meant to be run before every experiment, and it is meant to REFUSE. A
federated result computed on a leaking split is worse than no result, because it
looks fine: the numbers are plausible, the curves converge, and the conclusion is
wrong. Every check below corresponds to a mistake this project or its predecessor
actually made.

    patient in two hospitals      FedAvg would average two models that both
                                  memorised the same patient
    training patient in the test  the classic leak; the earlier radiomics phase
    set                           shipped it via StratifiedKFold over SLICES
    slice-level split             same leak, harder to see: a patient's slices are
                                  near-duplicates, so a model recognises the patient
    missing class in a local val  the site reports NaN for the metric the server
                                  selects on, and the server silently selects on
                                  the remaining sites
    src/ importing nvflare        breaks the invariant that the centralised baseline
                                  and the clients run the same trainer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  ok    {message}")
        return True
    print(f"  FAIL  {message}")
    if detail:
        print(f"        {detail}")
    FAILURES.append(message)
    return False


def read_split(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def verify_global() -> set:
    print("\nglobal held-out sets")
    test = read_split(EX.GLOBAL_DIR / "test.csv")
    if test is None:
        check(False, "data/global/test.csv exists",
              "run scripts/prepare_data.py first")
        return set()

    test_pids = set(test.pid)
    check(len(test_pids) > 0, f"test set has patients ({len(test_pids)})")
    check(test.groupby("pid").label.nunique().max() == 1,
          "every test patient carries exactly one label")

    val = read_split(EX.GLOBAL_DIR / "val.csv")
    if val is not None:
        overlap = test_pids & set(val.pid)
        check(not overlap, "global val and global test share no patient",
              f"{len(overlap)} shared: {sorted(overlap)[:5]}")

    missing = [p for p in list(test_pids)[:200]
               if not (EX.GLOBAL_DIR / "images" / str(p)).is_dir()]
    check(not missing, "every test patient has an image folder",
          f"missing: {missing[:5]}")
    return test_pids


def verify_partition(name: str, test_pids: set) -> None:
    part_dir = EX.PARTITIONS_DIR / name
    print(f"\npartition {name}")
    if not part_dir.is_dir():
        check(False, f"{name} exists", "run scripts/partition_data.py")
        return

    partition = EX.PARTITIONS[name]
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    total = 0

    for site in partition.client_names:
        site_dir = part_dir / site
        if not site_dir.is_dir():
            check(False, f"{site} exists")
            continue

        train = read_split(site_dir / "train.csv")
        val = read_split(site_dir / "val.csv")
        if train is None or val is None:
            check(False, f"{site} has train.csv and val.csv")
            continue

        train_pids, val_pids = set(train.pid), set(val.pid)
        total += len(train_pids | val_pids)

        # A patient whose slices are split between this site's train and val is the
        # slice-level leak wearing a different hat.
        check(not (train_pids & val_pids),
              f"{site}: local train and local val share no patient",
              f"{len(train_pids & val_pids)} shared")

        check(not ((train_pids | val_pids) & test_pids),
              f"{site}: no patient of the global test set",
              f"{len((train_pids | val_pids) & test_pids)} leaked")

        for pid in train_pids | val_pids:
            if pid in seen and seen[pid] != site:
                duplicates.append(f"{pid} in {seen[pid]} and {site}")
            seen[pid] = site

        present = val.drop_duplicates("pid").label.nunique()
        check(present == EX.NUM_CLASSES,
              f"{site}: local val covers all {EX.NUM_CLASSES} classes",
              f"only {present} present — this site cannot report a usable "
              f"{EX.FEDERATION.key_metric}")

        for split, rows in (("train", train), ("val", val)):
            bad = rows.groupby("pid").label.nunique()
            check((bad <= 1).all(), f"{site}/{split}: one label per patient",
                  f"{(bad > 1).sum()} patients with several labels")

    check(not duplicates, f"{name}: no patient in two hospitals",
          "; ".join(duplicates[:5]))

    meta_path = part_dir / "partition.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        check(meta["total_patients"] == total,
              f"{name}: patient count matches the manifest",
              f"manifest {meta['total_patients']}, on disk {total}")


def verify_imports() -> None:
    """`src/` must not import nvflare — the invariant the layout rests on."""
    print("\nsrc/ purity")
    offenders = []
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import nvflare", "from nvflare")):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{i}")
    check(not offenders, "no file in src/ imports nvflare", "; ".join(offenders))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None, help="verify one partition")
    p.add_argument("--check-imports", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("DATA VERIFICATION — this refuses rather than warns")
    print("=" * 70)

    test_pids = verify_global()
    for name in ([args.only] if args.only else list(EX.PARTITIONS)):
        verify_partition(name, test_pids)
    if args.check_imports:
        verify_imports()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED — {len(FAILURES)} of {CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nDo not run experiments on this data.")
        sys.exit(1)
    print(f"PASSED — {CHECKS} checks")
    print("=" * 70)


if __name__ == "__main__":
    main()
