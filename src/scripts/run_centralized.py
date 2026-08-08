#!/usr/bin/env python3
"""Test 01 — the centralised baseline. Not an NVFLARE job.

    python src/scripts/run_centralized.py
    python src/scripts/run_centralized.py --seed 1 --epochs 30

This is the reference every federated run is measured against, so the only thing
that may differ between it and a client is that it sees all the data. It uses the
same `TrainingConfig`, the same model builder, the same augmentation, the same
patient-level evaluator, and — through `src/training.py` — the same training loop.

BUDGET MATCHING, AND WHY IT IS NOT OPTIONAL
-------------------------------------------
30 epochs here against 30 rounds x 1 local epoch there. The model sees the data the
same number of times on both sides. Without that, RQ1 would be reading a difference
in compute as a difference in federation, and the federated arm would lose for a
reason that has nothing to do with federation.

Note this makes the baseline WEAKER than the headline classifier result, and that is
correct rather than a problem: the 0.6159 in `all_runs_pod.csv` came from a
100-epoch sweep with patience 30. The number to compare a federated run against is
the one trained on the same budget, not the best number the project ever produced.
In practice the gap is small — the best epoch on this task lands between 1 and 5.

WHY IT WRITES THE SAME FILES A FEDERATED RUN WRITES
---------------------------------------------------
`rounds.csv`, `predictions_test.csv`, `results.json`. One epoch is one row, exactly
as one round is one row, so `collect_results.py` and the analysis notebooks read all
nine experiments through the same code path and cannot treat the baseline specially
by accident.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# the centralised baseline lives at the repository root; the configuration it
# shares with the federated clients lives under src/federated/
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
PROJECT_ROOT = REPO_ROOT / "src" / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX  # noqa: E402
from common import data as D             # noqa: E402
from common import evaluation as EV      # noqa: E402
from common import models as M           # noqa: E402
from common import training as T         # noqa: E402

ROUND_FIELDS = ["round", "site", "n_train_patients", "n_train_slices", "lr",
                "train_loss", "train_acc",
                "post_val_acc", "post_val_bal_acc", "post_val_auc",
                "post_val_macro_f1", "seconds"]


# The admin-side logger, imported rather than reimplemented: one definition of how a
# run is recorded means the centralised baseline and the federated tests produce logs
# in the same format, which is what makes them comparable in an appendix.
sys.path.insert(0, str(SCRIPTS_DIR))
from run_experiment import _Tee   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=EX.SOURCE_DATASET,
                   help="the pooled dataset — 'all hospitals' data on one machine'")
    p.add_argument("--epochs", type=int, default=EX.FEDERATION.centralized_epochs)
    p.add_argument("--seed", type=int, default=EX.TRAINING.seed)
    p.add_argument("--out", type=Path, default=None)
    # An ABLATION knob, defaulting to the shared config so a normal run is
    # unaffected. It exists because "which freeze setting" is a question the data
    # can answer directly, and answering it by editing config/experiments.py would
    # change what every other experiment trains.
    p.add_argument("--freeze-until", default=EX.TRAINING.freeze_until,
                   help="override TrainingConfig.freeze_until for this run only "
                        f"(default: {EX.TRAINING.freeze_until})")
    # The same knob for batch size. Changing it alters the number of optimiser
    # steps per epoch, so a run using it is an ablation and must carry a --tag.
    p.add_argument("--batch-size", type=int, default=EX.TRAINING.batch_size,
                   help="override TrainingConfig.batch_size for this run only "
                        f"(default: {EX.TRAINING.batch_size})")
    p.add_argument("--tag", default=None,
                   help="label for the output folder, so an ablation cannot "
                        "overwrite a real result")
    args = p.parse_args()

    # Every line also lands in production/logs/test01/train.log, timestamped. The
    # centralised baseline has no server or hospital processes, so this file plus the
    # run's own results folder IS its complete record.
    # A tagged ablation logs beside its own results, never into test01's log —
    # two runs appending to one file interleave mid-line and neither is readable.
    log_path = (EX.LOGS_DIR / "_ablations" / f"{args.tag}_seed_{args.seed}.log"
                if args.tag else EX.LOGS_DIR / "test01" / "train.log")
    sys.stdout = _Tee(sys.stdout, log_path)
    print(f"\n===== run_centralized.py seed={args.seed} "
          f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC) =====")

    experiment = EX.get("test01")
    # A tagged run is an ablation and is kept OUT of the experiment's results
    # folder, so `collect_results.py` and the final summary never mistake it for
    # test01 proper.
    if args.out:
        out = Path(args.out)
    elif args.tag:
        out = EX.RESULTS_DIR / "_ablations" / f"{args.tag}_seed_{args.seed}"
    else:
        out = EX.RESULTS_DIR / experiment.name / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"{experiment.id} — {experiment.objective}")
    print(f"  budget : {args.epochs} epochs (federated arm: "
          f"{EX.FEDERATION.num_rounds} rounds x {EX.FEDERATION.local_epochs})")
    print(f"  seed   : {args.seed}")
    print(f"  out    : {out}")
    print("=" * 74)

    training = EX.TRAINING
    overrides = {}
    if args.freeze_until != EX.TRAINING.freeze_until:
        overrides["freeze_until"] = args.freeze_until
    if args.batch_size != EX.TRAINING.batch_size:
        overrides["batch_size"] = args.batch_size
    if overrides:
        from dataclasses import replace
        training = replace(EX.TRAINING, **overrides)
        for field, value in overrides.items():
            print(f"  ABLATION: {field} {getattr(EX.TRAINING, field)!r} -> "
                  f"{value!r} for this run only")
    D.set_seed(args.seed)
    device = M.get_device()
    use_amp = T.use_amp_on(device, training.mixed_precision)

    # The whole training pool, exactly as the classifier phase used it.
    loaders, frames, cfg = D.load_site(args.source, training, epochs=args.epochs,
                                       seed=args.seed)
    print("\n" + D.describe_site("pooled", frames, EX.NUM_CLASSES))

    # The official test set lives with the server and is identical for all nine.
    test_loader, test_rows, _ = D.load_eval_only(EX.GLOBAL_DIR, "test", training,
                                                 seed=args.seed)
    print(f"  test  {len(test_rows):>6,} slices / {test_rows.pid.nunique():>4} "
          f"patients  trivial {D.trivial_baseline(test_rows):.4f}")

    model = M.build_model(training, EX.NUM_CLASSES).to(device)
    print("\n" + M.describe(model, training.model_name))
    print(f"architecture fingerprint {M.architecture_fingerprint(model)}")

    weights = (D.class_weights(frames["train"], EX.NUM_CLASSES, device)
               if training.class_weighted_loss else None)
    if weights is not None:
        print(f"class weights (patient level): "
              f"{[round(w, 3) for w in weights.tolist()]}")
    criterion = T.build_criterion(weights, training.label_smoothing)
    optimizer = T.build_optimizer(model, training)
    scaler = T.build_scaler(device, use_amp)
    sampler = getattr(loaders["train"], "batch_sampler", None)

    n_patients = int(frames["train"].pid.nunique())
    n_slices = int(len(frames["train"]))
    best_score, best_epoch = -np.inf, 0
    train_acc_at_best = float("nan")
    rounds_csv = out / "rounds.csv"
    history = []

    print("\n" + "-" * 74)
    for epoch in range(args.epochs):
        t0 = time.time()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        # The SAME curve the federated clients follow, evaluated at the same index.
        lr = T.lr_for_round(training.learning_rate, epoch, args.epochs,
                            training.scheduler)
        T.set_lr(optimizer, lr)
        loss, acc = T.train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            use_amp=use_amp, freeze_bn=training.freeze_bn)

        val = EV.evaluate(model, loaders["val"], frames["val"], device,
                          num_classes=EX.NUM_CLASSES, class_names=EX.CLASS_NAMES,
                          aggregation=training.aggregation, use_amp=use_amp)
        seconds = time.time() - t0
        row = {
            "round": epoch, "site": "centralized",
            "n_train_patients": n_patients, "n_train_slices": n_slices,
            "lr": f"{lr:.6e}", "train_loss": f"{loss:.6f}",
            "train_acc": f"{acc:.6f}",
            "post_val_acc": val["accuracy"],
            "post_val_bal_acc": val["balanced_accuracy"],
            "post_val_auc": val["auc"], "post_val_macro_f1": val["macro_f1"],
            "seconds": round(seconds, 1),
        }
        history.append(row)
        new = not rounds_csv.is_file()
        with rounds_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROUND_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

        print(f"epoch {epoch + 1:03d}/{args.epochs}  lr {lr:.2e}  "
              f"loss {loss:.4f}  train_acc {acc:.4f} | val AUC {val['auc']:.4f} "
              f"bal {val['balanced_accuracy']:.4f}  ({seconds:.0f}s)")

        # Selected on validation AUC, the classifier phase's rule. The federated
        # server selects on val_balanced_accuracy because that is what survives a
        # 39-patient local split; here the validation set is 99 patients and AUC is
        # well defined. The difference is stated in docs/EXPERIMENTS.md.
        score = val[training.monitor_metric]
        if np.isfinite(score) and (score > best_score or epoch == 0):
            best_score, best_epoch, train_acc_at_best = score, epoch, acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": cfg.as_dict(), "val_metrics": val},
                       out / "best_model.pt")
    print("-" * 74)
    print(f"best val {training.monitor_metric} {best_score:.4f} at epoch "
          f"{best_epoch + 1}\n")

    # ---- official evaluation, from the SELECTED checkpoint ---------------- #
    ck = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    results = {}
    for split, loader, rows in (("val", loaders["val"], frames["val"]),
                                ("test", test_loader, test_rows)):
        m = EV.evaluate(model, loader, rows, device, num_classes=EX.NUM_CLASSES,
                        class_names=EX.CLASS_NAMES,
                        aggregation=training.aggregation, use_amp=use_amp)
        results[split] = m
        print(f"{split:<5} macro-AUC {m['auc']:.4f} | acc {m['accuracy']:.4f} "
              f"(trivial {m['trivial_baseline_accuracy']:.4f}) | "
              f"balanced {m['balanced_accuracy']:.4f}")

    EV.predictions(model, test_loader, test_rows, device,
                   class_names=EX.CLASS_NAMES, aggregation=training.aggregation,
                   use_amp=use_amp).to_csv(out / "predictions_test.csv", index=False)
    (out / "report_test.txt").write_text(
        EV.report_text(model, test_loader, test_rows, device,
                       class_names=EX.CLASS_NAMES,
                       aggregation=training.aggregation, use_amp=use_amp))

    (out / "results.json").write_text(json.dumps({
        "experiment": experiment.id, "name": experiment.name, "kind": "centralized",
        "seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch,
        "train_acc_at_best": train_acc_at_best,
        "generalisation_gap": {s: train_acc_at_best - results[s]["accuracy"]
                               for s in results},
        "source": str(Path(args.source).resolve()),
        "architecture": M.architecture_fingerprint(model),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.as_dict(), "splits": results,
    }, indent=2, default=float))

    print(f"\nwritten: {out}")
    print("\nreminder: one seed is not a result. The noise floor on this task is "
          "0.067 macro-AUC.\n  Run at least --seed 1 and --seed 42 before "
          "comparing this to anything.")


if __name__ == "__main__":
    main()
