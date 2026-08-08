#!/usr/bin/env python3
"""Run the centralised baseline exactly as notebooks/03_train_centralized.ipynb does.

    python src/scripts/run_notebook_centralized.py
    python src/scripts/run_notebook_centralized.py --seed 1 --epochs 30

WHY THIS EXISTS ALONGSIDE run_centralized.py
--------------------------------------------
`run_centralized.py` is the federated campaign's baseline: 30 epochs, early
stopping disabled, writing into results/federated/test01_centralized/. The
notebook is the interactive path: it calls `core.training.run`, defaults to 10
epochs, and writes a full auto-report through `core.reporting`, including BOTH
checkpoints --- `checkpoints/best_model.pt` and `checkpoints/last_model.pt` ---
and `history.csv`, the per-epoch training record.

This script reproduces the notebook's configuration field for field so a headless
run and the notebook cannot drift apart. Every value below is copied from the
notebook's CFG cell; the only additions are command-line overrides for seed and
epoch count, so that repeating across seeds does not require editing a file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import dataset_config as config          # noqa: E402
from dataset_config import Config, lr_for_model   # noqa: E402
from core.training import run, get_device, describe_device  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10,
                   help="the notebook's default is 10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--notes", default="centralised baseline (notebook config)")
    args = p.parse_args()

    cfg = Config(
        pipeline="thesis",
        task="subtype",
        cohorts=("spy2", "spy1", "duke"),
        dataset_name="multi_subtype_80mm",
        model=args.model,
        seed=args.seed,
        optimizer="adamw",
        learning_rate=lr_for_model(args.model),
        weight_decay=5e-4,
        scheduler="cosine",
        batch_size=24,
        epochs=args.epochs,
        early_stopping_patience=30,
        dropout=0.5,
        label_smoothing=0.1,
        class_weighted_loss=True,
        mixup_alpha=0.0,
        freeze_until="layer3",
        freeze_bn=False,
        max_slices_per_patient_per_batch=1,
        image_size=224,
        aggregation="mean",
        monitor_metric="auc",
        mixed_precision=True,
        num_workers=8,
        augmentation="default",
        notes=args.notes,
    )

    device = get_device()
    print(cfg.summary())
    print(f"\ndevice : {describe_device(device)}")
    print(f"output : {config.RESULTS_DIR}\n")

    run_dir = run(cfg, progress=True)

    print("\nfiles written:")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            print("  ", f.relative_to(run_dir))

    res_path = run_dir / "results.json"
    if res_path.is_file():
        res = json.loads(res_path.read_text())
        t = res.get("test", res)
        print(f"\nmacro AUC          {t.get('auc', float('nan')):.4f}")
        print(f"balanced accuracy  {t.get('balanced_accuracy', float('nan')):.4f}")
        print(f"accuracy           {t.get('accuracy', float('nan')):.4f}")
        print(f"trivial baseline   {t.get('trivial_baseline_accuracy', 0.5112):.4f}")
    print("\nOne seed is not a result: the measured noise floor is 0.067 macro AUC.")


if __name__ == "__main__":
    main()
