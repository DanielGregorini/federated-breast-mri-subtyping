"""The bridge to the shared trainer. The only file that knows where it lives.

WHY A BRIDGE INSTEAD OF A COPY
------------------------------
The trainer, the model factory, the augmentation policy and the patient-level
evaluator already exist and are already measured. The federated clients need exactly
those, and copying them here would create a second definition that starts drifting on
day one. The previous iteration of this project did copy: `model.py` went into all 28
participant folders and needed a `sync_model.py` to keep them equal. FedAvg only
averages correctly if every site builds an identical network.

So there is one definition, imported from here, and the centralised baseline and every
federated client run the same code.

WHY sys.path AND NOT AN IMPORT
------------------------------
`core/` is not an installed package and has no `__init__.py`. It is a folder whose
modules are imported by putting its parent on `sys.path`, and `dataset_config.py` is
loaded from its file location rather than by name, because `config` is also the name
of the federated configuration package and whichever came first on the path would
silently win.

WHERE IT LOOKS
--------------
1. Beside this file. Inside a submitted job that is `<job>/app/custom/`, which is
   where NVFLARE unpacks the code the job shipped. A hospital machine therefore needs
   nothing from this repository.
2. $BREAST_CORE_ROOT, for a site that keeps the code somewhere of its own.
3. The repository layout, for running from a checkout.

A missing definition raises rather than falls back. A differently shaped network is
something FedAvg would average without complaining.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# common/ -> code/ -> deployment/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent


def _locate_thesis() -> Path:
    """Find ``, or explain precisely what to set."""
    # Inside a submitted job this file is at <job>/app/custom/common/thesis.py and
    # the shared code is at <job>/app/custom/. That is checked first, so a job that
    # shipped its own code never reaches out to the machine it landed on.
    candidates = [Path(__file__).resolve().parent.parent]

    env = os.environ.get("BREAST_CORE_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / "src")

    for path in candidates:
        if (path / "dataset_config.py").is_file() and (path / "core").is_dir():
            return path.resolve()

    raise ImportError(
        "cannot find the shared core (needs dataset_config.py and core/).\n"
        f"  looked in: {', '.join(str(c) for c in candidates)}\n"
        "  fix: export BREAST_CORE_ROOT=/path/to/the repository root\n"
        "  This must be resolved before training starts. A site that cannot build "
        "the shared model must not join the federation with a different one.")


THESIS_ROOT = _locate_thesis()


def _load_thesis_config_module():
    """Load `config.py` from its FILE, not from `sys.path`.

    Both projects expose the name `config` — the thesis as a module (`config.py`),
    this one as a package (`config/`) — so whichever is found first on `sys.path`
    silently wins and the other becomes unimportable. Ordering the path is not a
    fix, it just chooses which project breaks.

    Loading by explicit file location removes the ambiguity entirely: this is the
    only `config` that is ever resolved by name, and it is resolved from a path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "breast_dataset_config", THESIS_ROOT / "dataset_config.py")
    module = importlib.util.module_from_spec(spec)
    # Registered under a unique name so a later `import config` cannot pick it up
    # by accident, and so it is loaded exactly once.
    sys.modules["breast_dataset_config"] = module
    spec.loader.exec_module(module)
    return module


if str(THESIS_ROOT) not in sys.path:
    # Needed for `core`, whose modules import each other relatively. `core` does not
    # import `config` — only `core/dataset_builder.py` does, and this project never
    # builds datasets, so that module is never loaded. Appended rather than
    # prepended so nothing in the thesis folder can shadow this project's own
    # top-level names.
    sys.path.append(str(THESIS_ROOT))

ThesisConfig = _load_thesis_config_module().Config

# ruff: noqa: E402  — the path bootstrap above must run before these imports.
from core import data as thesis_data          # noqa: F401
from core import evaluation as thesis_eval    # noqa: F401
from core import models as thesis_models      # noqa: F401
from core import training as thesis_training  # noqa: F401


def build_config(training, dataset_dir: Path, *, epochs: int, seed: int | None = None,
                 augmentation: str = "default") -> ThesisConfig:
    """Turn this project's `TrainingConfig` into the thesis `Config`.

    Every field is copied explicitly rather than by `**asdict`. A silent mismatch
    between what the federated protocol says it is training and what the trainer
    actually trains is precisely the bug class this project has already paid for
    three times, and an explicit list fails at import when a field is renamed.

    `dataset_dir` is a hospital's own folder in the federated case and the pooled
    dataset in the centralised case. Both have the same shape — `images/` plus
    `train.csv` / `val.csv` — so the same loader serves both.
    """
    cfg = ThesisConfig(
        task="subtype",
        model=training.model_name,
        image_size=training.image_size,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        max_slices_per_patient_per_batch=training.max_slices_per_patient_per_batch,
        optimizer=training.optimizer,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        scheduler=training.scheduler,
        epochs=epochs,
        label_smoothing=training.label_smoothing,
        class_weighted_loss=training.class_weighted_loss,
        dropout=training.dropout,
        freeze_until=training.freeze_until,
        freeze_bn=training.freeze_bn,
        augmentation=augmentation,
        seed=training.seed if seed is None else seed,
        mixed_precision=training.mixed_precision,
        aggregation=training.aggregation,
        monitor_metric=training.monitor_metric,
        early_stopping_patience=0,   # never inside a federated round; see training.py
    )
    # `Config.dataset_dir` is the derived property `DATA_DIR / dataset_name`, and
    # pathlib resolves `anything / "/absolute/path"` to the absolute path. Assigning
    # an absolute `dataset_name` therefore redirects the config at this project's
    # per-hospital folders without a second override mechanism to keep in step.
    cfg.dataset_name = str(Path(dataset_dir).resolve())
    return cfg
