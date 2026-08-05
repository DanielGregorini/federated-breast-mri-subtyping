#!/usr/bin/env python3
"""Evaluate every finished experiment on the ONE global test set, and tabulate.

    python scripts/collect_results.py
    python scripts/collect_results.py --only test06_fedavg_4h

WHY EVALUATION IS CENTRALISED HERE INSTEAD OF DONE BY EACH RUN
--------------------------------------------------------------
Because the comparison is the point. Nine experiments scored by nine pieces of code
is nine chances for them to differ — and this project has already shipped a
`collect_results.py` with `"resnet18"` hard-coded, which silently evaluated
federated runs that had trained a ResNet-50. Here the architecture comes from
`config/experiments.py`, the test set comes from `data/global/`, and both are the
same objects the training used.

Every number is patient-level macro-AUC on the same 99 patients, quoted beside the
trivial baseline. Accuracy without that baseline is meaningless and it is not a
constant — 0.404 on I-SPY2 alone, 0.511 pooled.

READING THE OUTPUT
------------------
The noise floor on this task is 0.067 macro-AUC, measured between two byte-identical
configurations differing only in seed. The summary prints that line every time,
because the previous run of these nine experiments produced a FedAvg-vs-FedProx
difference of 0.004 to 0.021 — four times in the same direction, which is a trend,
and never once outside the noise, which means it is not a fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX  # noqa: E402
from config import federation as FED  # noqa: E402
from common import data as D             # noqa: E402
from common import evaluation as EV      # noqa: E402
from common import models as M           # noqa: E402
from common import training as T         # noqa: E402

# NVFLARE writes the aggregated model under the server's job workspace. The exact
# filename depends on the persistor, so several are tried rather than one guessed.
GLOBAL_MODEL_NAMES = [
    "best_FL_global_model.pt",   # the model selected by key_metric
    "FL_global_model.pt",        # the last round
    # The name a model is given when it is copied out of the server's job
    # workspace and back to this machine. The workspace itself does not survive
    # the GPU host being released, so once a pod is torn down this is the only
    # copy left — and `job.json` still points `job_workspace` at a path that no
    # longer exists. Searched last, so a live workspace always wins.
    "global_model.pt",
]


def find_global_model(exp_dir: Path) -> tuple[Path | None, str]:
    """Locate the aggregated model for one federated experiment.

    Returns `(path, which)` where `which` says whether it is the SELECTED model or
    merely the LAST round — a distinction that matters and is reported in the table,
    because "best" and "last" disagreed twice in this project's history.
    """
    job = exp_dir / "job.json"
    roots: list[Path] = []
    if job.is_file():
        record = json.loads(job.read_text())
        if record.get("job_workspace"):
            roots.append(Path(record["job_workspace"]))
        job_id = record.get("job_id")
        if job_id:
            try:
                roots.append(FED.startup_kit(FED.SERVER_NAME) / job_id)
                roots.append(FED.workspace_dir() / FED.SERVER_NAME / job_id)
            except SystemExit:
                pass
    roots.append(exp_dir)

    for name in GLOBAL_MODEL_NAMES:
        for root in roots:
            if not root.exists():
                continue
            hit = (root / name) if (root / name).is_file() else next(
                iter(sorted(root.rglob(name))), None)
            if hit is not None:
                return hit, model_provenance(hit)
    return None, "missing"


def model_provenance(path: Path) -> str:
    """Whether `path` is the SELECTED global model or merely the last round.

    The two NVFLARE filenames say this outright, but `global_model.pt` — the name
    a model is given when it is copied off a GPU host before the host is released
    — does not. Guessing from the name got it wrong in both directions, so the
    answer is read from the checkpoint itself: the PT persistor records
    `meta_props.current_round`, and a model saved before the final round can only
    be one that beat its predecessors on `key_metric`.

    A model recorded AT the final round is genuinely ambiguous — the best round
    may simply have been the last one — and is reported as `last_round`, which is
    the conservative reading.
    """
    if path.name.startswith("best"):
        return "selected"
    if path.name.startswith("FL_"):
        return "last_round"
    try:
        import torch
        meta = torch.load(path, map_location="cpu",
                          weights_only=False).get("meta_props", {})
        rnd = meta.get("current_round")
    except Exception:
        return "unknown"
    if rnd is None:
        return "unknown"
    return "selected" if rnd < EX.FEDERATION.num_rounds - 1 else "last_round"


def evaluate_checkpoint(path: Path, loader, rows, device, use_amp) -> tuple[dict, object]:
    """Load a checkpoint into the SHARED architecture and score it.

    `strict=True`: a head-shape mismatch loaded leniently gives a randomly
    initialised classifier and no error, which on this task still produces a
    plausible macro-AUC near chance. That would be read as a federated result.
    """
    model = M.build_model(EX.TRAINING, EX.NUM_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if isinstance(state, dict):
        state = {k: torch.as_tensor(v) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    metrics = EV.evaluate(model, loader, rows, device, num_classes=EX.NUM_CLASSES,
                          class_names=EX.CLASS_NAMES,
                          aggregation=EX.TRAINING.aggregation, use_amp=use_amp)
    return metrics, model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None)
    p.add_argument("--out", type=Path, default=EX.RESULTS_DIR / "all_experiments.csv")
    args = p.parse_args()

    device = M.get_device()
    use_amp = T.use_amp_on(device, EX.TRAINING.mixed_precision)
    loader, rows, _ = D.load_eval_only(EX.GLOBAL_DIR, "test", EX.TRAINING)
    baseline = D.trivial_baseline(rows)

    print("=" * 78)
    print(f"GLOBAL TEST SET — {rows.pid.nunique()} patients, {len(rows):,} slices")
    print(f"  trivial baseline (majority class among patients): {baseline:.4f}")
    print(f"  chance macro-AUC: 0.5000")
    print("=" * 78)

    records = []
    for experiment in EX.EXPERIMENTS:
        if args.only and experiment.name != args.only and experiment.id != args.only:
            continue
        exp_dir = EX.RESULTS_DIR / experiment.name
        if not exp_dir.is_dir():
            print(f"\n{experiment.id:<9} not run yet")
            continue

        base = {
            "experiment": experiment.id, "name": experiment.name,
            "kind": experiment.kind, "algorithm": experiment.algorithm or "-",
            "n_clients": experiment.n_clients,
            "partition": experiment.partition or "-",
            "n_test": int(rows.pid.nunique()), "baseline": round(baseline, 4),
        }

        if experiment.kind == "centralized":
            # One folder per seed. Each already evaluated itself on this same set;
            # re-scoring here anyway means every row in the table is produced by
            # one code path.
            for seed_dir in sorted(exp_dir.glob("seed_*")):
                ckpt = seed_dir / "best_model.pt"
                if not ckpt.is_file():
                    continue
                metrics, _ = evaluate_checkpoint(ckpt, loader, rows, device, use_amp)
                info = json.loads((seed_dir / "results.json").read_text()) \
                    if (seed_dir / "results.json").is_file() else {}
                records.append({**base, "seed": info.get("seed"),
                                "model_used": "selected",
                                "best_epoch": info.get("best_epoch"),
                                "test_auc": round(metrics["auc"], 4),
                                "test_acc": round(metrics["accuracy"], 4),
                                "test_bal": round(metrics["balanced_accuracy"], 4),
                                "test_macro_f1": round(metrics["macro_f1"], 4),
                                "per_class_recall": metrics["per_class_recall"]})
                print(f"\n{experiment.id:<9} seed {info.get('seed')}  "
                      f"macro-AUC {metrics['auc']:.4f}  acc {metrics['accuracy']:.4f}"
                      f"  bal {metrics['balanced_accuracy']:.4f}")
            continue

        path, which = find_global_model(exp_dir)
        if path is None:
            print(f"\n{experiment.id:<9} no aggregated model found under {exp_dir}")
            continue
        metrics, model = evaluate_checkpoint(path, loader, rows, device, use_amp)
        records.append({**base, "seed": EX.TRAINING.seed, "model_used": which,
                        "best_epoch": None,
                        "test_auc": round(metrics["auc"], 4),
                        "test_acc": round(metrics["accuracy"], 4),
                        "test_bal": round(metrics["balanced_accuracy"], 4),
                        "test_macro_f1": round(metrics["macro_f1"], 4),
                        "per_class_recall": metrics["per_class_recall"]})
        print(f"\n{experiment.id:<9} {experiment.algorithm:<8} "
              f"{experiment.n_clients}c  macro-AUC {metrics['auc']:.4f}  "
              f"acc {metrics['accuracy']:.4f}  bal "
              f"{metrics['balanced_accuracy']:.4f}  ({which})")

        EV.predictions(model, loader, rows, device, class_names=EX.CLASS_NAMES,
                       aggregation=EX.TRAINING.aggregation, use_amp=use_amp
                       ).to_csv(exp_dir / "predictions_test.csv", index=False)
        (exp_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=float))

    if not records:
        raise SystemExit("\nnothing to collect — no experiment has produced a model.")

    df = pd.DataFrame(records).sort_values("test_auc", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print(df[["experiment", "algorithm", "n_clients", "seed", "test_auc",
              "test_acc", "test_bal", "model_used"]].to_string(index=False))
    print("=" * 78)
    print(f"\nwritten: {args.out}")
    print("\nHOW TO READ THIS")
    print("  The noise floor is 0.067 macro-AUC — measured between two runs of a")
    print("  byte-identical configuration differing only in seed. Treat any gap")
    print("  below that as 'no difference detected', which is a finding and should")
    print("  be reported as one. One seed is not a result.")
    central = df[df.kind == "centralized"].test_auc
    federated = df[df.kind == "federated"].test_auc
    if len(central) and len(federated):
        gap = central.mean() - federated.mean()
        print(f"\n  centralised mean {central.mean():.4f} (n={len(central)})")
        print(f"  federated   mean {federated.mean():.4f} (n={len(federated)})")
        print(f"  gap              {gap:+.4f}  "
              f"{'— above the noise floor' if abs(gap) > 0.067 else '— INSIDE the noise floor'}")


if __name__ == "__main__":
    main()
