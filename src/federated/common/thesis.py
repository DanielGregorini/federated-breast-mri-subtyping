"""The bridge to `the repository root`. The only file that knows where it lives.

WHY A BRIDGE INSTEAD OF A COPY
------------------------------
The classifier phase produced a trainer, a model factory, an augmentation policy
and a patient-level evaluator, all of them measured and all of them already used to
produce the numbers this dissertation reports. The federated phase needs exactly
those things.

Copying them here would create a second definition that starts drifting on day one.
The previous iteration of this project did copy — `model.py` went into all 28
participant folders and needed a `sync_model.py` to keep them equal — and FedAvg
only averages correctly if every site builds an identical network.

So: one definition, in `core//`, imported from here. The
centralised baseline and every federated client therefore run **literally the same
code** as the classifier phase. That is what makes RQ1 a measurement of federation
rather than a measurement of two codebases.

WHY sys.path AND NOT AN IMPORT
------------------------------
`` is not an installed package and has no `__init__.py` — it is
a project folder whose `config.py` and `core/` are imported by adding the folder to
`sys.path`, which is what its own notebooks do. Reproducing that here keeps the two
projects independent: neither has to be installed for the other to work.

DEPLOYING TO A REAL HOSPITAL MACHINE
------------------------------------
NVFLARE ships a job's `custom/` folder to each site, so `src/` travels with the job.
`core//` does NOT. On a real hospital machine, either
  * place the repository at the same relative path (the default assumption), or
  * set $BREAST_CORE_ROOT to wherever it lives.
Both are checked below, in that order, and a clear error is raised if neither
resolves — a missing model definition must fail loudly at startup, not silently
produce a differently-shaped network that FedAvg would then average.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# training/ -> federated/ -> src/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent


def _locate_thesis() -> Path:
    """Find ``, or explain precisely what to set."""
    env = os.environ.get("BREAST_CORE_ROOT")
    candidates = [Path(env)] if env else []
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
        pipeline="thesis",
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
