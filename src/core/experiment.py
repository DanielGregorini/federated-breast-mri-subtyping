"""Auto-numbered experiment folders.

    from core.experiment import new_experiment
    run_dir = new_experiment(cfg)      # results/test_007_resnet18_subtype/

The number is assigned by scanning `results/` for the highest existing one and
adding one, so two runs never collide and the chronological order is readable
from the folder listing alone. The name carries the model and the task because
that is what a reader needs before opening anything.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

RUN_PATTERN = re.compile(r"^test_(\d+)_")


def next_number(results_dir: Path) -> int:
    """One above the highest number already present. Gaps are never reused —
    a deleted experiment must not have its number silently taken by another."""
    results_dir.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for d in results_dir.iterdir() if d.is_dir()
            for m in [RUN_PATTERN.match(d.name)] if m]
    return max(used, default=0) + 1


def new_experiment(cfg, results_dir: Path | None = None,
                   suffix: str = "") -> Path:
    """Create and return `results/test_NNN_<model>_<task>[_suffix]/`.

    The folder is laid out before training starts, so a crashed run still leaves
    its configuration behind and can be told apart from a run that never began.
    """
    from dataset_config import RESULTS_DIR
    root = Path(results_dir or RESULTS_DIR)
    n = next_number(root)
    parts = [f"test_{n:03d}", cfg.model, cfg.task]
    if suffix:
        parts.append(suffix)
    run_dir = root / "_".join(parts)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Written first, not last: a run that dies at epoch 3 must still say what it
    # was trying to do.
    (run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2,
                                                    default=str))
    (run_dir / "README.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Pipeline **{cfg.pipeline}** · task **{cfg.task}** · model **{cfg.model}**\n\n"
        f"```\n{cfg.summary()}\n```\n\n"
        f"Generated automatically by `core.experiment.new_experiment`. "
        f"Full configuration in `config.json`; metrics in `metrics.csv`; "
        f"figures in `figures/`.\n")
    return run_dir


def load_experiments(results_dir: Path | None = None) -> list[dict]:
    """Every finished run as a flat record, newest number last.

    This is what `07_compare_experiments` reads. Runs without `results.json` —
    crashed or still going — are skipped rather than reported as zeros.
    """
    from dataset_config import RESULTS_DIR
    root = Path(results_dir or RESULTS_DIR)
    if not root.exists():
        return []

    out = []
    for d in sorted(root.iterdir()):
        f = d / "results.json"
        if not (d.is_dir() and f.is_file()):
            continue
        r = json.loads(f.read_text())
        cfg = r.get("config", {})
        rec = {
            "run": d.name,
            "number": int(m.group(1)) if (m := RUN_PATTERN.match(d.name)) else -1,
            "pipeline": cfg.get("pipeline"), "task": cfg.get("task"),
            "model": cfg.get("model"), "seed": cfg.get("seed"),
            "cohorts": ",".join(cfg.get("cohorts", [])),
            "augmentation": cfg.get("augmentation"),
            "freeze_until": cfg.get("freeze_until"),
            "best_epoch": r.get("best_epoch"), "epochs_run": r.get("epochs_run"),
            "trainable_params": r.get("parameters", {}).get("trainable"),
        }
        for split, m_ in r.get("splits", {}).items():
            for k in ("auc", "accuracy", "balanced_accuracy", "macro_f1",
                      "trivial_baseline_accuracy"):
                if k in m_:
                    rec[f"{split}_{k}"] = m_[k]
        # The number that matters for overfitting, computed once here so no
        # notebook has to remember the definition.
        if "train_acc_at_best" in r:
            rec["train_acc"] = r["train_acc_at_best"]
        out.append(rec)
    return out


def delete_experiment(run_dir: Path) -> None:
    """Remove a run. Separate function so deletion is always deliberate."""
    run_dir = Path(run_dir)
    if not RUN_PATTERN.match(run_dir.name):
        raise ValueError(f"{run_dir.name} does not look like an experiment folder")
    shutil.rmtree(run_dir)
