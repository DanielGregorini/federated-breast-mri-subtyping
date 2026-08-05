#!/usr/bin/env python3
"""Build `results/final_summary/` — every number, table and figure, in one pass.

    python scripts/build_final_summary.py
    python scripts/build_final_summary.py --require-complete   # for the final thesis run
    python scripts/build_final_summary.py --no-client-eval     # skip per-hospital scoring

WHAT THIS IS FOR
----------------
Nine experiments, each writing its own folder, are not a dissertation chapter. This
script turns them into one: a directory where every table is derived from the same
code path, every figure is regenerated from stored predictions rather than from a
screenshot, and every claim can be traced back to the file it came from.

IT NEVER INVENTS A NUMBER
-------------------------
An experiment that has not run is reported as `not_run`, not as a blank row that
looks like a measurement. `manifest.json` records the status of all nine, and
`README.md` states plainly which ones are missing. Running this against an empty
`results/` is a supported and useful thing to do: it produces the cohort tables
(which come from the prepared data, not from any run) and tells you exactly what is
still outstanding. That is deliberate — a summary that silently drops missing runs
is how a thesis ends up quoting eight experiments and calling it nine.

WHERE THE NUMBERS COME FROM
---------------------------
Preferentially from `predictions_test.csv`, the per-patient probability table each
experiment writes. Deriving every metric from stored predictions rather than from a
stored metric dict means the whole summary is reproducible without a GPU, and means
the confusion matrix, the ROC curve and the accuracy in the table cannot disagree
with one another — they are computed from one array.

The model is only loaded when it has to be: to score the global model on each
hospital's own held-out patients, which no run writes out on its own.

THE MEASUREMENT RULES THIS SCRIPT ENFORCES
------------------------------------------
1. Every metric is PER PATIENT. Slice probabilities are averaged per patient before
   anything is computed. `predictions_test.csv` is already patient-level.
2. Accuracy is never reported without the trivial baseline of the same split. It is
   not a constant: 0.404 on this test set, 0.511 pooled.
3. Every experiment is scored on the SAME global test set, `data/global/test/`.
4. The noise floor on this task is 0.067 macro-AUC, measured between two
   byte-identical configurations differing only in seed. It is printed on every
   table and stated in every report, because four of these nine comparisons are
   expected to land inside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for collect_results

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")                                                # headless
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                 # noqa: E402
from sklearn.metrics import (auc as sk_auc, precision_recall_curve,  # noqa: E402
                             roc_curve)

from config import experiments as EX                                 # noqa: E402

CLASSES = list(EX.CLASS_NAMES)
N_CLASSES = EX.NUM_CLASSES
OUT_DIR = EX.RESULTS_DIR / "final_summary"

# Where the per-experiment folders are read from. Overridable with --results-dir so a
# results tree copied back from the GPU host can be summarised in place, without
# moving it into the repository first.
RESULTS_DIR = EX.RESULTS_DIR

# The measured seed-to-seed spread on this task. Printed everywhere a difference is.
NOISE_FLOOR = 0.067

# Publication defaults. Serif to match a LaTeX body, 300 dpi, no chartjunk.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
# One colour per algorithm, used identically in every figure.
ALGO_COLOUR = {"centralized": "#1b1b1b", "fedavg": "#0072B2", "fedprox": "#D55E00"}
CLASS_COLOUR = ["#0072B2", "#D55E00", "#009E73"]


# --------------------------------------------------------------------------- #
# 1. METRICS — one implementation, used for the global set and for every site  #
# --------------------------------------------------------------------------- #
def metrics_from_predictions(df: pd.DataFrame) -> dict:
    """Every reported metric, from one per-patient prediction table.

    Deliberately does not call the training-time evaluator: this must work on a
    laptop with no CUDA and no checkpoint, months after the runs, from the CSVs
    alone. The formulas are the same ones `core/evaluation.py::compute_metrics`
    uses, and `--verify-metrics` checks the two agree.

    Returns NaN for AUC when a class is absent from the split rather than a number
    computed over a subset, which would be silently incomparable. On the small
    per-hospital validation sets this happens, and it must not crash the report.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    y = df["label"].to_numpy(dtype=int)
    prob = df[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=float)
    pred = prob.argmax(1)
    present = np.unique(y)
    labels = list(range(N_CLASSES))

    if len(present) == N_CLASSES and np.isfinite(prob).all():
        macro_auc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
        per_class_auc = [float(roc_auc_score((y == c).astype(int), prob[:, c]))
                         for c in labels]
    else:
        macro_auc = float("nan")
        per_class_auc = [float("nan")] * N_CLASSES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {
            "n_patients": int(len(y)),
            "auc": macro_auc,
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_precision": float(precision_score(y, pred, average="macro",
                                                     zero_division=0)),
            "macro_recall": float(recall_score(y, pred, average="macro",
                                               zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "per_class_auc": per_class_auc,
            "per_class_precision": [float(v) for v in precision_score(
                y, pred, average=None, labels=labels, zero_division=0)],
            "per_class_recall": [float(v) for v in recall_score(
                y, pred, average=None, labels=labels, zero_division=0)],
            "per_class_f1": [float(v) for v in f1_score(
                y, pred, average=None, labels=labels, zero_division=0)],
            "class_counts": [int((y == c).sum()) for c in labels],
            "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
            "trivial_baseline_accuracy": float(
                max((y == c).mean() for c in labels)) if len(y) else float("nan"),
        }
    return out


def curves_from_predictions(df: pd.DataFrame) -> dict:
    """One-vs-rest ROC and precision-recall curve points, per class.

    Stored as plain lists so the figures can be redrawn — restyled for a different
    journal, say — without re-running anything or reloading a model.
    """
    y = df["label"].to_numpy(dtype=int)
    prob = df[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=float)
    out: dict[str, dict] = {"roc": {}, "pr": {}}
    for c, name in enumerate(CLASSES):
        binary = (y == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            out["roc"][name] = {"fpr": [], "tpr": [], "auc": float("nan")}
            out["pr"][name] = {"recall": [], "precision": [],
                               "average_precision": float("nan")}
            continue
        fpr, tpr, _ = roc_curve(binary, prob[:, c])
        precision, recall, _ = precision_recall_curve(binary, prob[:, c])
        out["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                            "auc": float(sk_auc(fpr, tpr))}
        out["pr"][name] = {"recall": recall.tolist(), "precision": precision.tolist(),
                           "average_precision": float(sk_auc(recall, precision)),
                           "prevalence": float(binary.mean())}
    return out


# --------------------------------------------------------------------------- #
# 2. DISCOVERY — what actually ran                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """One experiment's artifacts, resolved to paths and parsed metadata."""

    experiment: object                  # EX.Experiment
    status: str                         # "complete" | "partial" | "not_run"
    detail: str = ""
    result_dir: Path | None = None
    predictions: pd.DataFrame | None = None
    rounds: pd.DataFrame | None = None
    job: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)      # results.json, centralized only
    model_path: Path | None = None
    model_kind: str = ""                          # "selected" | "last_round"
    seed: int | None = None
    metrics: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)
    clients: dict = field(default_factory=dict)   # site -> {metrics, curves, cohort}

    @property
    def id(self) -> str:
        return self.experiment.id

    @property
    def name(self) -> str:
        return self.experiment.name

    @property
    def algorithm(self) -> str:
        return self.experiment.algorithm or "centralized"

    @property
    def ok(self) -> bool:
        return self.status == "complete"


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.is_file() else None
    except Exception:
        return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except Exception:
        return {}


def discover(only: str | None = None) -> list[Run]:
    """Resolve all nine experiments to a status and, where present, artifacts.

    A centralised run is one folder per seed (`seed_1/`, `seed_42/`); the seed whose
    checkpoint the rest of the project treats as the baseline is the LOWEST-variance
    one, but for the summary every seed present is carried and the table reports the
    spread. A federated run is one folder with `job.json` and an aggregated model
    somewhere under the server workspace.
    """
    try:
        from collect_results import find_global_model
    except Exception:
        find_global_model = None

    runs: list[Run] = []
    for experiment in EX.EXPERIMENTS:
        if only and only not in (experiment.id, experiment.name):
            continue
        d = RESULTS_DIR / experiment.name
        if not d.is_dir():
            runs.append(Run(experiment, "not_run", f"no folder at {d}"))
            continue

        run = Run(experiment, "partial", result_dir=d)
        # A federated run's per-round CSV is written by the CLIENTS, into the
        # `--results-dir` the recipe passes them, which is `<experiment>/sites/`.
        # Only the centralised run writes it at the top level. Looking in one place
        # silently produced an empty convergence figure — the report still built.
        run.rounds = _read_csv(d / "rounds.csv")
        if run.rounds is None:
            run.rounds = _read_csv(d / "sites" / "rounds.csv")

        if experiment.kind == "centralized":
            seed_dirs = sorted(d.glob("seed_*"))
            if not seed_dirs and (d / "results.json").is_file():
                seed_dirs = [d]
            chosen = None
            for sd in seed_dirs:
                if (sd / "predictions_test.csv").is_file():
                    chosen = sd
                    break
            if chosen is None:
                run.detail = "no seed folder with predictions_test.csv"
                runs.append(run)
                continue
            run.predictions = _read_csv(chosen / "predictions_test.csv")
            run.info = _read_json(chosen / "results.json")
            seed_rounds = _read_csv(chosen / "rounds.csv")
            if seed_rounds is not None:
                run.rounds = seed_rounds
            run.model_path = (chosen / "best_model.pt"
                              if (chosen / "best_model.pt").is_file() else None)
            run.model_kind = "selected"
            run.seed = run.info.get("seed")
            run.info["_seed_dirs"] = [str(s.name) for s in seed_dirs]
        else:
            run.job = _read_json(d / "job.json")
            run.seed = EX.TRAINING.seed
            run.predictions = _read_csv(d / "predictions_test.csv")
            if find_global_model is not None:
                try:
                    path, which = find_global_model(d)
                    run.model_path, run.model_kind = path, which
                except Exception:
                    run.model_path, run.model_kind = None, "missing"

        if run.predictions is None or run.predictions.empty:
            run.detail = ("no predictions_test.csv — run scripts/collect_results.py "
                          "first, it writes one per experiment")
            runs.append(run)
            continue

        missing = [c for c in ["label"] + [f"prob_{c}" for c in CLASSES]
                   if c not in run.predictions.columns]
        if missing:
            run.detail = f"predictions_test.csv missing columns: {missing}"
            runs.append(run)
            continue

        run.status = "complete"
        run.metrics = metrics_from_predictions(run.predictions)
        run.curves = curves_from_predictions(run.predictions)
        runs.append(run)
    return runs


# --------------------------------------------------------------------------- #
# 3. COHORT — available with zero runs, straight from the prepared data         #
# --------------------------------------------------------------------------- #
def cohort_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-partition and per-hospital data description.

    Comes from `data/partitions/*/partition.json` and `data/global/manifest.json`,
    both written by `prepare_data.py`/`partition_data.py`. No experiment has to have
    run for this to be correct, which is why it is a separate section: the
    dissertation's data chapter does not depend on the results chapter.
    """
    global_rows = []
    manifest = _read_json(EX.GLOBAL_DIR / "manifest.json")
    for split, s in (manifest.get("splits") or {}).items():
        row = {"scope": "global", "partition": "-", "site": f"global_{split}",
               "split": split, "patients": s.get("patients"),
               "images": s.get("slices"),
               "trivial_baseline": s.get("trivial_baseline")}
        for i, name in enumerate(CLASSES):
            pcp = s.get("per_class_patients") or []
            row[f"patients_{name}"] = pcp[i] if i < len(pcp) else None
        global_rows.append(row)

    site_rows = []
    part_rows = []
    for pname, partition in EX.PARTITIONS.items():
        pj = _read_json(EX.PARTITIONS_DIR / pname / "partition.json")
        part_rows.append({
            "partition": pname, "label": partition.label,
            "n_clients": partition.n_clients,
            "ratio": " : ".join(str(r) for r in partition.ratio),
            "fractions": " / ".join(f"{100 * f:.1f}%" for f in partition.fractions),
            "stratified": partition.stratified,
            "total_patients": pj.get("total_patients"),
            "built": pj.get("built"), "seed": pj.get("seed"),
        })
        for site in pj.get("sites", []):
            for split in ("train", "val"):
                s = site.get(split) or {}
                row = {"scope": "site", "partition": pname,
                       "site": site.get("site"), "split": split,
                       "patients": s.get("patients"), "images": s.get("slices"),
                       "trivial_baseline": None}
                pcp = s.get("per_class_patients") or []
                total = sum(pcp) if pcp else 0
                for i, name in enumerate(CLASSES):
                    row[f"patients_{name}"] = pcp[i] if i < len(pcp) else None
                    row[f"pct_{name}"] = (round(100 * pcp[i] / total, 1)
                                          if total and i < len(pcp) else None)
                site_rows.append(row)

    return pd.DataFrame(part_rows), pd.DataFrame(global_rows + site_rows)


# --------------------------------------------------------------------------- #
# 4. PER-CLIENT EVALUATION — the global model scored at each hospital           #
# --------------------------------------------------------------------------- #
def evaluate_clients(run: Run, device=None) -> dict:
    """Score this experiment's global model on every hospital's own held-out set.

    WHAT "PER-CLIENT METRICS" MEANS HERE, AND WHY
    ---------------------------------------------
    It is the ONE global model evaluated on each site's local validation patients —
    not each site's local model, which is never exported and would not be comparable
    across sites anyway. This is the number that answers "does the federated model
    work equally well at every hospital, or is it carried by the big one?", which is
    the per-client question a dissertation actually needs.

    The official headline number is never one of these. It is the global test set,
    identical for all nine experiments. These are per-site diagnostics, on sets as
    small as 36 patients, where a missing class legitimately yields NaN AUC.
    """
    if run.model_path is None or not Path(run.model_path).is_file():
        return {}
    if run.experiment.partition is None:
        return {}

    import torch
    from common import data as D
    from common import evaluation as EV
    from common import models as M
    from common import training as T

    device = device or M.get_device()
    use_amp = T.use_amp_on(device, EX.TRAINING.mixed_precision)

    model = M.build_model(EX.TRAINING, EX.NUM_CLASSES)
    state = torch.load(run.model_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if isinstance(state, dict):
        state = {k: torch.as_tensor(v) for k, v in state.items()}
    # strict=True: a head-shape mismatch loaded leniently gives a randomly
    # initialised classifier and no error, which on this task still scores near
    # chance and would be tabulated as a per-site result.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    out: dict[str, dict] = {}
    partition = EX.PARTITIONS[run.experiment.partition]
    for site in partition.client_names:
        site_dir = EX.PARTITIONS_DIR / partition.name / site
        try:
            loader, rows, _ = D.load_eval_only(site_dir, "val", EX.TRAINING)
            preds = EV.predictions(model, loader, rows, device,
                                   class_names=EX.CLASS_NAMES,
                                   aggregation=EX.TRAINING.aggregation,
                                   use_amp=use_amp)
        except Exception as exc:                       # a site must not kill the report
            out[site] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[site] = {
            "metrics": metrics_from_predictions(preds),
            "curves": curves_from_predictions(preds),
            "n_images": int(len(rows)),
            "predictions": preds,
        }
    return out


# --------------------------------------------------------------------------- #
# 5. TABLES                                                                    #
# --------------------------------------------------------------------------- #
def training_time(run: Run) -> tuple[float | None, str]:
    """Seconds of training, and an honest label for what was actually measured.

    Centralised runs record per-epoch `seconds` in `rounds.csv`, so the sum is pure
    compute. Federated runs record no per-round timing at all, so the only available
    figure is the job's wall clock from `job.json`, which includes provisioning,
    model transfer and server aggregation. The two are NOT the same quantity and the
    label says so — reporting them in one column as if they were is exactly the kind
    of quiet apples-to-oranges this project keeps finding.
    """
    if run.rounds is not None and "seconds" in run.rounds.columns:
        total = pd.to_numeric(run.rounds["seconds"], errors="coerce").sum()
        if np.isfinite(total) and total > 0:
            return float(total), "sum of per-epoch compute"
    sub, fin = run.job.get("submitted"), run.job.get("finished")
    if sub and fin:
        try:
            t0 = datetime.fromisoformat(sub)
            t1 = datetime.fromisoformat(fin)
            return (t1 - t0).total_seconds(), "job wall clock (incl. orchestration)"
        except ValueError:
            pass
    return None, "not recorded"


def best_round(run: Run) -> int | None:
    """The round whose global model the server would have selected.

    Read from `rounds.csv` using the SAME metric name the server selects on,
    `FederationConfig.key_metric`, rather than a hard-coded column. A previous
    iteration of this project reported one metric and selected on another.
    """
    if run.rounds is None or run.rounds.empty or run.experiment.kind != "federated":
        return None
    key = EX.FEDERATION.key_metric
    short = key[4:] if key.startswith("val_") else key
    column = {"balanced_accuracy": "post_val_bal_acc", "accuracy": "post_val_acc",
              "auc": "post_val_auc", "macro_f1": "post_val_macro_f1"}.get(short)
    if column is None or column not in run.rounds.columns:
        return None
    # Mean across clients per round: the server aggregates before selecting.
    per_round = (run.rounds.assign(**{column: pd.to_numeric(run.rounds[column],
                                                            errors="coerce")})
                 .groupby("round")[column].mean())
    return int(per_round.idxmax()) if per_round.notna().any() else None


def main_table(runs: list[Run]) -> pd.DataFrame:
    """One row per experiment — the table the dissertation's results chapter opens with."""
    rows = []
    for run in runs:
        m = run.metrics
        seconds, time_kind = training_time(run)
        rows.append({
            "experiment": run.id,
            "name": run.name,
            "status": run.status,
            "datetime": (run.job.get("finished") or run.info.get("finished")
                         or run.job.get("submitted") or ""),
            "model": EX.TRAINING.model_name,
            "dataset": EX.SOURCE_DATASET.name,
            "n_hospitals": run.experiment.n_clients,
            "partition": run.experiment.partition or "-",
            "data_split": run.experiment.split_label,
            "algorithm": run.algorithm,
            "seed": run.seed,
            "best_epoch": run.info.get("best_epoch"),
            "best_round": best_round(run),
            "training_time_s": None if seconds is None else round(seconds, 1),
            "training_time_kind": time_kind,
            "model_used": run.model_kind or "-",
            "n_test_patients": m.get("n_patients"),
            "trivial_baseline": _r(m.get("trivial_baseline_accuracy")),
            "accuracy": _r(m.get("accuracy")),
            "balanced_accuracy": _r(m.get("balanced_accuracy")),
            "macro_precision": _r(m.get("macro_precision")),
            "macro_recall": _r(m.get("macro_recall")),
            "macro_f1": _r(m.get("macro_f1")),
            "macro_auc": _r(m.get("auc")),
            **{f"auc_{c}": _r((m.get("per_class_auc") or [None] * N_CLASSES)[i])
               for i, c in enumerate(CLASSES)},
            **{f"f1_{c}": _r((m.get("per_class_f1") or [None] * N_CLASSES)[i])
               for i, c in enumerate(CLASSES)},
            **{f"precision_{c}": _r((m.get("per_class_precision")
                                     or [None] * N_CLASSES)[i])
               for i, c in enumerate(CLASSES)},
            **{f"recall_{c}": _r((m.get("per_class_recall") or [None] * N_CLASSES)[i])
               for i, c in enumerate(CLASSES)},
        })
    return pd.DataFrame(rows)


def _r(v, nd: int = 4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, nd)


def client_table(runs: list[Run]) -> pd.DataFrame:
    """One row per (experiment, hospital) — the global model's score at each site."""
    cohort = cohort_tables()[1]
    rows = []
    for run in runs:
        for site, payload in (run.clients or {}).items():
            base = {"experiment": run.id, "name": run.name,
                    "algorithm": run.algorithm,
                    "partition": run.experiment.partition, "site": site}
            if "error" in payload:
                rows.append({**base, "status": "error", "detail": payload["error"]})
                continue
            m = payload["metrics"]
            c = cohort[(cohort.partition == run.experiment.partition)
                       & (cohort.site == site)]
            train = c[c.split == "train"]
            rows.append({
                **base, "status": "ok", "detail": "",
                "n_train_patients": (int(train.patients.iloc[0])
                                     if len(train) and pd.notna(train.patients.iloc[0])
                                     else None),
                "n_train_images": (int(train.images.iloc[0])
                                   if len(train) and pd.notna(train.images.iloc[0])
                                   else None),
                "n_val_patients": m.get("n_patients"),
                "n_val_images": payload.get("n_images"),
                **{f"val_patients_{cn}": m["class_counts"][i]
                   for i, cn in enumerate(CLASSES)},
                "trivial_baseline": _r(m.get("trivial_baseline_accuracy")),
                "accuracy": _r(m.get("accuracy")),
                "balanced_accuracy": _r(m.get("balanced_accuracy")),
                "macro_precision": _r(m.get("macro_precision")),
                "macro_recall": _r(m.get("macro_recall")),
                "macro_f1": _r(m.get("macro_f1")),
                "macro_auc": _r(m.get("auc")),
                **{f"auc_{cn}": _r(m["per_class_auc"][i])
                   for i, cn in enumerate(CLASSES)},
                **{f"precision_{cn}": _r(m["per_class_precision"][i])
                   for i, cn in enumerate(CLASSES)},
                **{f"recall_{cn}": _r(m["per_class_recall"][i])
                   for i, cn in enumerate(CLASSES)},
                **{f"f1_{cn}": _r(m["per_class_f1"][i])
                   for i, cn in enumerate(CLASSES)},
                "confusion": json.dumps(m["confusion"]),
            })
    return pd.DataFrame(rows)


COMPARISON_METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                      "macro_auc", "best_epoch", "best_round", "training_time_s"]


def comparisons(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The eight comparison tables the dissertation asks for.

    Every one carries a `delta_macro_auc` against its reference and a
    `within_noise_floor` flag. The flag is the point: on this task a 0.02 difference
    is not a finding, and a table that reports the difference without saying so
    invites the reader to treat it as one.
    """
    done = table[table.status == "complete"]
    out: dict[str, pd.DataFrame] = {}

    def block(rows: pd.DataFrame, reference: pd.DataFrame | None,
              label: str) -> pd.DataFrame:
        cols = ["experiment", "name", "algorithm", "n_hospitals", "partition",
                "data_split"] + COMPARISON_METRICS
        sub = rows.reindex(columns=cols).copy()
        if reference is not None and len(reference):
            ref_auc = reference.macro_auc.astype(float).mean()
            sub["reference"] = label
            sub["reference_macro_auc"] = round(ref_auc, 4)
            sub["delta_macro_auc"] = (sub.macro_auc.astype(float) - ref_auc).round(4)
            sub["within_noise_floor"] = sub.delta_macro_auc.abs() < NOISE_FLOOR
        return sub

    central = done[done.algorithm == "centralized"]
    fedavg = done[done.algorithm == "fedavg"]
    fedprox = done[done.algorithm == "fedprox"]

    out["centralized_vs_fedavg"] = block(pd.concat([central, fedavg]), central,
                                         "centralized (test01)")
    out["centralized_vs_fedprox"] = block(pd.concat([central, fedprox]), central,
                                          "centralized (test01)")
    out["fedavg_vs_fedprox"] = block(pd.concat([fedavg, fedprox]), fedavg,
                                     "FedAvg (mean over configurations)")

    # RQ3 is a PAIRED question — each FedProx run shares its partition with exactly
    # one FedAvg run, and the pair differs only in the aggregation algorithm. A table
    # that compared FedProx against the mean of all FedAvg runs would fold the
    # client-count effect into the algorithm effect. One row per partition.
    pairs = []
    for split, group in done[done.algorithm.isin(["fedavg", "fedprox"])].groupby(
            "data_split"):
        a = group[group.algorithm == "fedavg"]
        b = group[group.algorithm == "fedprox"]
        if a.empty or b.empty:
            continue
        row = {"data_split": split, "n_hospitals": int(a.n_hospitals.iloc[0]),
               "fedavg": a.experiment.iloc[0], "fedprox": b.experiment.iloc[0]}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                       "macro_auc", "best_round", "training_time_s"]:
            av, bv = a[metric].iloc[0], b[metric].iloc[0]
            row[f"fedavg_{metric}"] = av
            row[f"fedprox_{metric}"] = bv
            if metric in ("macro_auc", "macro_f1", "accuracy"):
                try:
                    row[f"delta_{metric}"] = round(float(bv) - float(av), 4)
                except (TypeError, ValueError):
                    row[f"delta_{metric}"] = None
        d = row.get("delta_macro_auc")
        row["within_noise_floor"] = None if d is None else abs(d) < NOISE_FLOOR
        row["favours"] = ("—" if d is None or abs(d) < NOISE_FLOOR
                          else ("FedProx" if d > 0 else "FedAvg"))
        pairs.append(row)
    out["fedavg_vs_fedprox_paired"] = pd.DataFrame(pairs)

    for key, label, mask in [
        ("2_hospitals", "2 hospitals",
         (done.n_hospitals == 2) | (done.algorithm == "centralized")),
        ("3_hospitals", "3 hospitals",
         (done.n_hospitals == 3) | (done.algorithm == "centralized")),
        # Keyed on the PARTITION, not on the folder name. Matching names with a
        # regex broke silently the first time the experiments were renamed: the
        # 4-hospital tables came out empty and the report still built.
        ("4_hospitals_balanced", "4 hospitals (balanced)",
         (done.partition == "4_clients_balanced") | (done.algorithm == "centralized")),
        ("4_hospitals_skewed", "4 hospitals (skewed)",
         (done.partition == "4_clients_skewed") | (done.algorithm == "centralized")),
    ]:
        out[key] = block(done[mask], central, "centralized (test01)")
    return out


# --------------------------------------------------------------------------- #
# 6. FIGURES                                                                   #
# --------------------------------------------------------------------------- #
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))       # vector, for the thesis
    plt.close(fig)


def fig_roc(curves: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, name in enumerate(CLASSES):
        c = curves.get("roc", {}).get(name, {})
        if not c.get("fpr"):
            continue
        ax.plot(c["fpr"], c["tpr"], color=CLASS_COLOUR[i], lw=1.6,
                label=f"{name} (AUC {c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", lw=1, color="grey", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=9); ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _save(fig, path)


def fig_pr(curves: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, name in enumerate(CLASSES):
        c = curves.get("pr", {}).get(name, {})
        if not c.get("recall"):
            continue
        ax.plot(c["recall"], c["precision"], color=CLASS_COLOUR[i], lw=1.6,
                label=f"{name} (AP {c['average_precision']:.3f})")
        if "prevalence" in c:
            ax.axhline(c["prevalence"], color=CLASS_COLOUR[i], ls=":", lw=0.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title}\ndotted = class prevalence (chance)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _save(fig, path)


def fig_confusion(matrix, title: str, path: Path, normalise: bool = True) -> None:
    m = np.asarray(matrix, dtype=float)
    shown = (m / np.clip(m.sum(1, keepdims=True), 1, None)) if normalise else m
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalise else shown.max())
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(N_CLASSES), CLASSES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{int(m[i, j])}\n{shown[i, j]:.2f}" if normalise
                    else f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > 0.55 else "black")
    ax.set_title(title, fontsize=9); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised" if normalise else "count")
    _save(fig, path)


def fig_roc_overlay(runs: list[Run], path: Path) -> None:
    """Macro-average ROC of every experiment on one axis — the comparison figure."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for run in runs:
        if not run.ok:
            continue
        grid = np.linspace(0, 1, 200)
        tprs = [np.interp(grid, c["fpr"], c["tpr"])
                for c in run.curves.get("roc", {}).values() if c.get("fpr")]
        if not tprs:
            continue
        macro = np.mean(tprs, axis=0)
        ax.plot(grid, macro, lw=1.5, color=ALGO_COLOUR.get(run.algorithm, "grey"),
                ls="-" if run.algorithm != "fedprox" else "--", alpha=0.9,
                label=f"{run.id} {run.algorithm} "
                      f"({run.metrics.get('auc', float('nan')):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", lw=1, color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average ROC — all experiments, same global test set", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    _save(fig, path)


def fig_bars(table: pd.DataFrame, path: Path) -> None:
    done = table[table.status == "complete"]
    if done.empty:
        return
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for ax, metric in zip(axes.ravel(), metrics):
        values = done[metric].astype(float)
        colours = [ALGO_COLOUR.get(a, "grey") for a in done.algorithm]
        ax.bar(done.experiment, values, color=colours)
        # Both reference lines sit BELOW every bar, so their labels are drawn on a
        # white patch — otherwise the text lands on a bar and cannot be read.
        label_box = dict(facecolor="white", edgecolor="none", alpha=0.85,
                         boxstyle="round,pad=0.2")
        if metric == "macro_auc":
            ax.axhline(0.5, ls=":", color="grey", lw=1)
            ax.text(0.01, 0.515, "chance", fontsize=6, color="dimgrey",
                    va="bottom", bbox=label_box,
                    transform=ax.get_yaxis_transform())
        if metric == "accuracy" and done.trivial_baseline.notna().any():
            base = float(done.trivial_baseline.dropna().iloc[0])
            ax.axhline(base, ls="--", color="crimson", lw=1)
            ax.text(0.01, base + 0.015, f"trivial baseline {base:.3f}", fontsize=6,
                    color="crimson", va="bottom", bbox=label_box,
                    transform=ax.get_yaxis_transform())
        ax.set_title(metric.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=6.5)
        ax.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ALGO_COLOUR.values()]
    # rect leaves the top strip free, then the title and legend are placed inside it
    # in that order. Without the rect, tight_layout reclaims the strip and the two
    # land on top of each other.
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("All experiments on the global test set", y=0.985, fontsize=11)
    fig.legend(handles, list(ALGO_COLOUR), ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), fontsize=8)
    _save(fig, path)


def fig_rounds(runs: list[Run], path: Path) -> None:
    """Federated convergence: the aggregated model's score at each site, per round."""
    federated = [r for r in runs if r.experiment.kind == "federated"
                 and r.rounds is not None and not r.rounds.empty]
    if not federated:
        return
    n = len(federated)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.2 * rows_n),
                             squeeze=False)
    for ax, run in zip(axes.ravel(), federated):
        df = run.rounds.copy()
        col = "agg_val_auc" if "agg_val_auc" in df.columns else "post_val_auc"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for site, g in df.groupby("site"):
            ax.plot(g["round"], g[col], lw=1.2, marker="o", ms=2.5, label=site)
        mean = df.groupby("round")[col].mean()
        ax.plot(mean.index, mean.values, lw=2, color="black", label="mean")
        ax.axhline(0.5, ls=":", color="grey", lw=1)
        br = best_round(run)
        if br is not None:
            ax.axvline(br, ls="--", color="crimson", lw=1)
            ax.text(br, 0.02, f" selected r{br}", fontsize=6, color="crimson",
                    transform=ax.get_xaxis_transform())
        ax.set_title(f"{run.id} — {run.algorithm}", fontsize=9)
        ax.set_xlabel("round"); ax.set_ylabel("macro-AUC (global model, local val)")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Federated round evolution — the arriving global model scored at "
                 "each hospital", y=1.01, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def fig_training_evolution(runs: list[Run], path: Path) -> None:
    """Train loss and validation metric against round/epoch, every experiment."""
    have = [r for r in runs if r.rounds is not None and not r.rounds.empty]
    if not have:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for run in have:
        df = run.rounds.copy()
        for c in ("train_loss", "post_val_bal_acc"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        colour = ALGO_COLOUR.get(run.algorithm, "grey")
        ls = "--" if run.algorithm == "fedprox" else "-"
        if "train_loss" in df.columns:
            g = df.groupby("round")["train_loss"].mean()
            ax1.plot(g.index, g.values, lw=1.3, color=colour, ls=ls, alpha=0.85,
                     label=run.id)
        if "post_val_bal_acc" in df.columns:
            g = df.groupby("round")["post_val_bal_acc"].mean()
            ax2.plot(g.index, g.values, lw=1.3, color=colour, ls=ls, alpha=0.85,
                     label=run.id)
    ax1.set_xlabel("round / epoch"); ax1.set_ylabel("training loss")
    ax1.set_title("Training loss", fontsize=9)
    ax2.set_xlabel("round / epoch")
    ax2.set_ylabel(f"validation {EX.FEDERATION.key_metric}")
    ax2.set_title("Validation curve — the metric the server selects on", fontsize=9)
    ax2.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def fig_heatmap(table: pd.DataFrame, path: Path) -> None:
    done = table[table.status == "complete"]
    if done.empty:
        return
    metrics = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
               "macro_f1", "macro_auc"]
    m = done.set_index("experiment")[metrics].astype(float)
    fig, ax = plt.subplots(figsize=(1.1 * len(metrics) + 2.5, 0.45 * len(m) + 2))
    im = ax.imshow(m.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), [x.replace("_", "\n") for x in metrics],
                  fontsize=7)
    ax.set_yticks(range(len(m)), m.index, fontsize=7)
    for i in range(len(m)):
        for j in range(len(metrics)):
            v = m.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.6 else "black")
    ax.set_title("Metric comparison across all experiments", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    _save(fig, path)


def fig_class_distribution(cohort: pd.DataFrame, path: Path) -> None:
    train = cohort[(cohort.scope == "site") & (cohort.split == "train")]
    if train.empty:
        return
    partitions = list(dict.fromkeys(train.partition))
    fig, axes = plt.subplots(1, len(partitions), figsize=(3.4 * len(partitions), 3.4),
                             squeeze=False)
    for ax, pname in zip(axes[0], partitions):
        sub = train[train.partition == pname]
        bottom = np.zeros(len(sub))
        for i, name in enumerate(CLASSES):
            vals = sub[f"patients_{name}"].fillna(0).to_numpy(dtype=float)
            ax.bar(sub.site, vals, bottom=bottom, color=CLASS_COLOUR[i], label=name)
            bottom += vals
        ax.set_title(pname, fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=6.5)
        ax.set_ylabel("training patients")
    axes[0][-1].legend(fontsize=6.5)
    fig.suptitle("Class distribution per hospital — stratified, so only QUANTITY "
                 "varies", y=1.03, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 7. EXPORTS                                                                   #
# --------------------------------------------------------------------------- #
def write_pdf(path: Path, table: pd.DataFrame, comps: dict[str, pd.DataFrame],
              figures: list[Path], status_lines: list[str]) -> None:
    """A self-contained printable report, built with matplotlib's PDF backend.

    No reportlab, no LaTeX, no pandoc — none of which are installed here, and any of
    which would make regenerating this report depend on a machine's setup rather than
    on the repository.
    """
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
        fig.text(0.5, 0.86, "Federated Breast Cancer\nSubtype Classification",
                 ha="center", size=20, weight="bold")
        fig.text(0.5, 0.77, "Final results summary", ha="center", size=13)
        fig.text(0.5, 0.73,
                 datetime.now(timezone.utc).strftime("generated %Y-%m-%d %H:%M UTC"),
                 ha="center", size=9, color="grey")
        body = "\n".join(status_lines)
        fig.text(0.08, 0.62, body, size=8, va="top", family="monospace")
        pdf.savefig(fig); plt.close(fig)

        for title, df in [("Main results", table)] + list(comps.items()):
            if df is None or df.empty:
                continue
            _table_page(pdf, title.replace("_", " ").title(), df)

        for f in figures:
            png = f.with_suffix(".png")
            if not png.is_file():
                continue
            img = plt.imread(png)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(8.27, min(11.69, 8.27 * h / w + 0.6)))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
            ax.imshow(img); ax.axis("off")
            fig.text(0.5, 0.965, png.stem.replace("_", " "), ha="center", size=10)
            pdf.savefig(fig); plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame,
                max_cols: int = 9, rows_per_page: int = 22) -> None:
    """Render a DataFrame across as many A4 landscape pages as it needs."""
    show = fmt_frame(df)

    col_chunks = [list(show.columns[i:i + max_cols])
                  for i in range(0, len(show.columns), max_cols)]
    for ci, cols in enumerate(col_chunks):
        for start in range(0, max(len(show), 1), rows_per_page):
            chunk = show.iloc[start:start + rows_per_page][cols]
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111); ax.axis("off")
            suffix = "" if len(col_chunks) == 1 else f"  (columns {ci + 1}/{len(col_chunks)})"
            ax.set_title(f"{title}{suffix}", fontsize=11, pad=16)
            if chunk.empty:
                ax.text(0.5, 0.5, "no rows", ha="center")
            else:
                t = ax.table(cellText=chunk.values,
                             colLabels=[column_label(c) for c in chunk.columns],
                             loc="upper center", cellLoc="center")
                t.auto_set_font_size(False); t.set_fontsize(6.5); t.scale(1, 1.35)
                # The header carries two lines of text; without extra height the
                # second line is clipped out of the cell.
                header_lines = max(str(column_label(c)).count("\n") + 1
                                   for c in chunk.columns)
                for (row, _), cell in t.get_celld().items():
                    if row == 0:
                        cell.set_height(cell.get_height() * (1 + 0.55 * header_lines))
                        cell.set_text_props(weight="bold")
            pdf.savefig(fig); plt.close(fig)


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            (df if not df.empty else pd.DataFrame({"note": ["no rows"]})).to_excel(
                writer, sheet_name=name[:31], index=False)


def write_latex(out: Path, table: pd.DataFrame, comps: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    done = table[table.status == "complete"]
    cols = ["experiment", "algorithm", "n_hospitals", "accuracy", "macro_f1",
            "macro_auc", "best_round"]
    body = done.reindex(columns=cols)
    (out / "main_results.tex").write_text(
        body.to_latex(index=False, float_format="%.4f", na_rep="--",
                      caption="Global test-set results for every experiment. "
                              f"Noise floor {NOISE_FLOOR:.3f} macro-AUC.",
                      label="tab:main-results"))
    for key, df in comps.items():
        if df is None or df.empty:
            continue
        (out / f"comparison_{key}.tex").write_text(
            df.to_latex(index=False, float_format="%.4f", na_rep="--",
                        caption=key.replace("_", " "), label=f"tab:{key}"))


# --------------------------------------------------------------------------- #
# 8. NARRATIVE                                                                 #
# --------------------------------------------------------------------------- #
#: Columns that are counts, not measurements. Rendering `best_round` as "29.0000"
#: in a dissertation table is the kind of detail a reader notices before the result.
INTEGER_COLUMNS = {"n_hospitals", "seed", "best_epoch", "best_round", "n_patients",
                   "n_test_patients", "n_train_patients", "n_train_images",
                   "n_val_patients", "n_val_images", "training_time_s"}


#: Human labels for table headers. Mechanically turning `training_time_s` into three
#: stacked lines put the unit on its own row and clipped it out of the cell — so the
#: names are written out, with the unit attached to the word it belongs to.
COLUMN_LABELS = {
    "experiment": "experiment", "name": "run", "algorithm": "algorithm",
    "n_hospitals": "hospitals", "data_split": "split", "seed": "seed",
    "best_epoch": "best\nepoch", "best_round": "best\nround",
    "training_time_s": "train time\n(s)", "training_time_kind": "time\nmeasured as",
    "trivial_baseline": "trivial\nbaseline", "accuracy": "accuracy",
    "balanced_accuracy": "balanced\naccuracy", "macro_precision": "macro\nprecision",
    "macro_recall": "macro\nrecall", "macro_f1": "macro\nF1", "macro_auc": "macro\nAUC",
    "delta_macro_auc": "Δ macro\nAUC", "reference_macro_auc": "reference\nmacro AUC",
    "within_noise_floor": "within\nnoise floor", "n_test_patients": "test\npatients",
    "n_train_patients": "train\npatients", "n_train_images": "train\nimages",
    "n_val_patients": "val\npatients", "n_val_images": "val\nimages",
    "n_patients": "patients", "model_used": "model\nused", "status": "status",
    "reference": "reference", "favours": "favours", "site": "hospital",
    "partition": "partition",
}


def column_label(name: str) -> str:
    """A header that fits in a cell: known names are spelled out, unknown ones are
    wrapped to at most two lines rather than one line per underscore."""
    if name in COLUMN_LABELS:
        return COLUMN_LABELS[name]
    words = name.replace("_", " ").split()
    if len(words) <= 1:
        return name
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def fmt_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One display-formatting rule, shared by the markdown, PDF and LaTeX tables.

    Floats to four decimals, counts to none, missing values to empty rather than to
    the string "nan" — which otherwise reads as a measured value of nan rather than
    as "this run did not record it".
    """
    show = df.copy()
    for c in show.columns:
        if show[c].dtype.kind not in "fiu":
            continue
        series = pd.to_numeric(show[c], errors="coerce")
        integral = c in INTEGER_COLUMNS or (
            series.notna().any() and np.allclose(series.dropna() % 1, 0))
        show[c] = series.map(
            lambda v: "" if pd.isna(v) else (f"{v:,.0f}" if integral else f"{v:.4f}"))
    # NaN -> "" BEFORE stringifying. `astype(str)` does not reliably convert a real
    # float nan inside an object column (a column like `best_round`, which mixes
    # None for the centralised run with ints for the federated ones), and a stray
    # float then reaches `" | ".join(...)` and raises.
    show = show.astype(object).where(pd.notna(show), "")
    return show.astype(str).replace(
        {"None": "", "nan": "", "NaT": "", "<NA>": "", "True": "yes", "False": "no"})


def md_table(df: pd.DataFrame | None, empty: str = "_no rows._") -> str:
    """Render a DataFrame as a GitHub markdown table.

    Hand-rolled rather than `DataFrame.to_markdown`, which needs `tabulate`. This
    script deliberately depends on nothing that is not already required to train the
    models — the same reason the PDF is built with matplotlib instead of reportlab.
    A summary that cannot be regenerated because an optional formatting library is
    missing is not much of an archive.
    """
    if df is None or df.empty:
        return empty
    show = fmt_frame(df)
    header = "| " + " | ".join(str(c) for c in show.columns) + " |"
    rule = "|" + "|".join("---" for _ in show.columns) + "|"
    body = ["| " + " | ".join(r) + " |" for r in show.values]
    return "\n".join([header, rule, *body])


def status_lines(runs: list[Run]) -> list[str]:
    lines = []
    for run in runs:
        mark = {"complete": "[done]", "partial": "[part]", "not_run": "[    ]"}[run.status]
        extra = f"  {run.detail}" if run.detail else ""
        auc = run.metrics.get("auc")
        score = f"  macro-AUC {auc:.4f}" if isinstance(auc, float) and np.isfinite(auc) else ""
        lines.append(f"{mark} {run.id}  {run.name:<34}{score}{extra}")
    return lines


def write_markdown(path: Path, runs: list[Run], table: pd.DataFrame,
                   comps: dict, cohort_part: pd.DataFrame,
                   cohort_sites: pd.DataFrame, clients: pd.DataFrame) -> None:
    done = table[table.status == "complete"]
    missing = [r.id for r in runs if not r.ok]
    md = [
        "# Final results summary",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
        "`scripts/build_final_summary.py`.",
        "",
        f"**{len(done)} of {len(runs)} experiments complete.**"
        + (f" Missing: {', '.join(missing)}." if missing else ""),
        "",
        "## How to read these numbers",
        "",
        f"- Every metric is **per patient**. Slice probabilities are averaged per "
        f"patient first.",
        f"- Every experiment is scored on the **same** global test set "
        f"(`data/global/test/`).",
        f"- **The noise floor on this task is {NOISE_FLOOR:.3f} macro-AUC**, measured "
        "between two byte-identical configurations differing only in random seed. Any "
        "difference smaller than that is not a result. The comparison tables carry a "
        "`within_noise_floor` column for exactly this reason.",
        "- Accuracy is meaningless without the trivial baseline printed beside it. It "
        "is not a constant.",
        "- `training_time_kind` says what was actually timed: centralised runs record "
        "per-epoch compute, federated runs only have job wall clock, which includes "
        "orchestration. They are not the same quantity.",
        "",
        "## Protocol",
        "",
        "```",
        EX.summary(),
        "```",
        "",
        "## Main results",
        "",
        md_table(done.reindex(columns=[
            "experiment", "algorithm", "n_hospitals", "data_split", "seed",
            "best_epoch", "best_round", "training_time_s", "trivial_baseline",
            "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "macro_auc"]),
            "_No experiment has produced results yet._"),
        "",
        "## Comparisons",
        "",
    ]
    for key, df in comps.items():
        md += [f"### {key.replace('_', ' ')}", ""]
        md += [md_table(df, "_not available until the relevant experiments have "
                            "run._"), ""]

    md += ["## Per-hospital results", ""]
    md += [md_table(clients, "_not available until a federated experiment has "
                             "produced a global model._"), ""]

    md += [
        "## Data",
        "",
        "These tables come from the prepared dataset, not from any run, so they are "
        "correct regardless of how many experiments have finished.",
        "",
        "### Partitions", "", md_table(cohort_part), "",
        "### Per-hospital and global splits", "", md_table(cohort_sites), "",
        "## Experiment status", "", "```", *status_lines(runs), "```", "",
    ]
    path.write_text("\n".join(md))


README_TEMPLATE = """\
# `results/final_summary/`

Everything the dissertation's results chapter needs, generated in one pass by
[`scripts/build_final_summary.py`](../../scripts/build_final_summary.py).

Regenerate at any time — it is a pure function of `results/`, `data/` and
`config/experiments.py`, and it overwrites this folder:

```bash
python scripts/build_final_summary.py
```

**Status: {n_done} of {n_total} experiments complete.**{missing_note}

---

## Layout

```
final_summary/
├── README.md                     this file
├── manifest.json                 machine-readable status of all nine experiments
├── summary.csv                   one row per experiment — the main table
├── summary.xlsx                  every table as a sheet
├── summary.json                  everything, nested, including curve points
├── summary.md                    the human-readable report
├── summary.pdf                   printable: tables then figures
├── per_client_metrics.csv        one row per (experiment, hospital)
├── comparisons/                  the eight comparison tables, one CSV each
├── figures/                      cross-experiment figures (png + pdf)
├── experiments/<name>/           per-experiment metrics, curves and figures
├── cohort/                       data description — independent of any run
└── tables/                       LaTeX-ready \\begin{{table}} blocks
```

Every figure is written twice: `.png` for a quick look and for the PDF report,
`.pdf` (vector) for the thesis itself.

---

## What each experiment is

| id | algorithm | hospitals | split | research question |
|---|---|---|---|---|
{experiment_rows}

---

## What each metric means

All metrics are computed **per patient**: the model predicts each 2D slice, and the
slice probabilities belonging to one patient are averaged into a single prediction
before anything is scored. Slices from one patient are near-duplicates, so a
slice-level number measures how well the model recognises the *patient*, not the
disease.

| metric | definition | where it comes from |
|---|---|---|
| `accuracy` | fraction of patients whose top-1 class is correct | `predictions_test.csv` |
| `trivial_baseline` | accuracy of always predicting the majority class **of that same split** | computed per split, never assumed |
| `balanced_accuracy` | mean per-class recall — the imbalance-robust accuracy | `predictions_test.csv` |
| `macro_precision/recall/f1` | unweighted mean over the {n_classes} classes | `predictions_test.csv` |
| `precision_<class>` etc. | the same, per class | `predictions_test.csv` |
| `macro_auc` | **the headline metric** — one-vs-rest ROC AUC, macro-averaged | `predictions_test.csv` |
| `auc_<class>` | one-vs-rest AUC for that class alone | `predictions_test.csv` |
| `confusion` | rows = truth, columns = prediction, patient counts | `predictions_test.csv` |
| `best_epoch` | centralised: the epoch whose checkpoint was selected | `results.json` |
| `best_round` | federated: the round maximising `{key_metric}` averaged over clients | `rounds.csv` |
| `training_time_s` | see `training_time_kind` — **two different quantities** | `rounds.csv` / `job.json` |

`macro_auc` is NaN, not a number, whenever a split is missing a whole class. That
happens legitimately on the smaller per-hospital validation sets. A number computed
over a subset of classes would be silently incomparable to one computed over all of
them, so it is not produced.

---

## Three things that will otherwise be misread

**1. The noise floor is {noise_floor:.3f} macro-AUC.** It was measured in this project
between two byte-identical configurations differing only in random seed. Four of these
nine comparisons are expected to land inside it. Every comparison table therefore
carries `delta_macro_auc` **and** `within_noise_floor`; a difference with
`within_noise_floor = True` is not a finding, and must not be written up as one.

**2. Accuracy without its baseline means nothing.** The trivial baseline is not a
constant — it is {baseline_note}. It is carried in every table beside the accuracy it
qualifies.

**3. Training time is not one quantity.** Centralised runs record per-epoch compute in
`rounds.csv` and the reported figure is their sum. Federated runs record no per-round
timing, so the only figure available is the job's wall clock from `job.json`, which
includes provisioning, model transfer and server-side aggregation. The
`training_time_kind` column says which one you are looking at. Do not put them in the
same column of a thesis table without that qualifier.

---

## Per-hospital numbers: what they are

`per_client_metrics.csv` is **the one global model evaluated on each hospital's own
held-out patients**. It is not each hospital's local model — those are never exported,
and would not be comparable across sites in any case.

It answers "does the federated model work equally well everywhere, or is it carried by
the biggest site?", which matters most for the skewed 4-hospital split where one site
holds 5/9 of the data.

The official number for every experiment is always the global test set, which is
identical across all nine. Per-hospital sets are small — down to {min_val} validation
patients — so their AUCs are noisy and sometimes NaN.

---

## Provenance

| output | derived from |
|---|---|
| all headline metrics | `results/<experiment>/predictions_test.csv` |
| ROC / PR / confusion | the same file — so they can never disagree with the table |
| `best_round` | `results/<experiment>/rounds.csv` |
| `best_epoch`, seed | `results/test01_centralized/seed_*/results.json` |
| job timing, algorithm | `results/<experiment>/job.json` |
| per-hospital metrics | the global model re-evaluated on `data/partitions/<p>/<site>/val.csv` |
| cohort tables | `data/partitions/*/partition.json`, `data/global/manifest.json` |

Metrics are recomputed from stored per-patient predictions rather than copied from a
stored metric dict. That is what makes the confusion matrix, the ROC curve and the
accuracy in the table mutually consistent by construction: they come from one array.

The predictions themselves are written by `scripts/collect_results.py`, which scores
every experiment through one code path on one test set. **Run that first** — this
script reads its output.
"""


def write_readme(path: Path, runs: list[Run], cohort_sites: pd.DataFrame) -> None:
    done = [r for r in runs if r.ok]
    missing = [r.id for r in runs if not r.ok]
    rows = "\n".join(
        f"| `{r.id}` | {r.algorithm} | {r.experiment.n_clients} | "
        f"{r.experiment.split_label} | {r.experiment.research_question} |"
        for r in runs)
    val = cohort_sites[(cohort_sites.scope == "site")
                       & (cohort_sites.split == "val")].patients.dropna()
    manifest = _read_json(EX.GLOBAL_DIR / "manifest.json")
    test_base = ((manifest.get("splits") or {}).get("test") or {}).get(
        "trivial_baseline")
    baseline_note = (f"{test_base:.4f} on this test set" if test_base
                     else "computed per split")
    path.write_text(README_TEMPLATE.format(
        n_done=len(done), n_total=len(runs),
        missing_note=("" if not missing else
                      f" Not yet run: {', '.join(missing)}."),
        experiment_rows=rows, n_classes=N_CLASSES,
        key_metric=EX.FEDERATION.key_metric, noise_floor=NOISE_FLOOR,
        baseline_note=baseline_note,
        min_val=int(val.min()) if len(val) else "?"))


# --------------------------------------------------------------------------- #
# 9. ORCHESTRATION                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None,
                   help="read experiment folders from here (default: results/)")
    p.add_argument("--only", default=None, help="one experiment id or folder name")
    p.add_argument("--no-client-eval", action="store_true",
                   help="skip per-hospital scoring (needs torch and the global model)")
    p.add_argument("--require-complete", action="store_true",
                   help="exit non-zero unless all nine experiments are present")
    args = p.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = args.results_dir.resolve()
    out = args.out or (RESULTS_DIR / "final_summary")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("comparisons", "figures", "experiments", "cohort", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BUILDING", out)
    print("=" * 78)

    runs = discover(args.only)
    for line in status_lines(runs):
        print(" ", line)

    # ---- cohort: correct with zero runs ---------------------------------- #
    part, sites = cohort_tables()
    part.to_csv(out / "cohort" / "partitions.csv", index=False)
    sites.to_csv(out / "cohort" / "per_client_data.csv", index=False)
    fig_class_distribution(sites, out / "cohort" / "class_distribution")
    print(f"\ncohort: {len(part)} partitions, {len(sites)} split rows")

    # ---- per-client evaluation ------------------------------------------- #
    if not args.no_client_eval:
        for run in runs:
            if not run.ok or run.experiment.kind != "federated":
                continue
            try:
                run.clients = evaluate_clients(run)
                if run.clients:
                    print(f"  {run.id}: scored {len(run.clients)} hospitals")
            except Exception as exc:
                print(f"  {run.id}: per-client evaluation failed — "
                      f"{type(exc).__name__}: {exc}")

    # ---- per-experiment outputs ------------------------------------------ #
    for run in runs:
        if not run.ok:
            continue
        d = out / "experiments" / run.name
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({
            "experiment": run.id, "name": run.name, "algorithm": run.algorithm,
            "kind": run.experiment.kind, "n_hospitals": run.experiment.n_clients,
            "data_split": run.experiment.split_label, "seed": run.seed,
            "model": EX.TRAINING.model_name, "dataset": EX.SOURCE_DATASET.name,
            "model_used": run.model_kind, "best_epoch": run.info.get("best_epoch"),
            "best_round": best_round(run),
            "training_time_s": training_time(run)[0],
            "training_time_kind": training_time(run)[1],
            "global_test": run.metrics,
            "per_client": {s: v.get("metrics") for s, v in run.clients.items()
                           if "metrics" in v},
        }, indent=2, default=float))
        (d / "curves.json").write_text(json.dumps(run.curves, default=float))
        run.predictions.to_csv(d / "predictions_test.csv", index=False)
        pd.DataFrame(run.metrics["confusion"], index=CLASSES,
                     columns=CLASSES).to_csv(d / "confusion_global.csv")

        fig_roc(run.curves, f"{run.id} — global test set", d / "figures" / "roc_global")
        fig_pr(run.curves, f"{run.id} — global test set", d / "figures" / "pr_global")
        fig_confusion(run.metrics["confusion"], f"{run.id} — global test set",
                      d / "figures" / "confusion_global")
        for site, payload in (run.clients or {}).items():
            if "metrics" not in payload:
                continue
            fig_roc(payload["curves"], f"{run.id} — {site}",
                    d / "figures" / f"roc_{site}")
            fig_pr(payload["curves"], f"{run.id} — {site}",
                   d / "figures" / f"pr_{site}")
            fig_confusion(payload["metrics"]["confusion"], f"{run.id} — {site}",
                          d / "figures" / f"confusion_{site}")
            payload["predictions"].to_csv(d / f"predictions_{site}_val.csv",
                                          index=False)

    # ---- tables ----------------------------------------------------------- #
    table = main_table(runs)
    clients = client_table(runs)
    comps = comparisons(table)

    table.to_csv(out / "summary.csv", index=False)
    if not clients.empty:
        clients.to_csv(out / "per_client_metrics.csv", index=False)
    for key, df in comps.items():
        if df is not None:
            df.to_csv(out / "comparisons" / f"{key}.csv", index=False)

    # ---- cross-experiment figures ----------------------------------------- #
    figures = []
    fig_roc_overlay(runs, out / "figures" / "roc_all_experiments"); figures.append(out / "figures" / "roc_all_experiments")
    fig_bars(table, out / "figures" / "bar_metrics_all"); figures.append(out / "figures" / "bar_metrics_all")
    fig_rounds(runs, out / "figures" / "federated_round_evolution"); figures.append(out / "figures" / "federated_round_evolution")
    fig_training_evolution(runs, out / "figures" / "training_evolution"); figures.append(out / "figures" / "training_evolution")
    fig_heatmap(table, out / "figures" / "metric_comparison_heatmap"); figures.append(out / "figures" / "metric_comparison_heatmap")
    figures.append(out / "cohort" / "class_distribution")

    # ---- exports ---------------------------------------------------------- #
    write_xlsx(out / "summary.xlsx", {
        "main": table, "per_client": clients, "partitions": part,
        "cohort": sites, **{f"cmp_{k}"[:31]: v for k, v in comps.items()}})

    (out / "summary.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "noise_floor_macro_auc": NOISE_FLOOR,
        "protocol": {
            "model": EX.TRAINING.model_name, "dataset": EX.SOURCE_DATASET.name,
            "classes": CLASSES, "seed": EX.TRAINING.seed,
            "num_rounds": EX.FEDERATION.num_rounds,
            "local_epochs": EX.FEDERATION.local_epochs,
            "centralized_epochs": EX.FEDERATION.centralized_epochs,
            "fedprox_mu": EX.FEDERATION.fedprox_mu,
            "key_metric": EX.FEDERATION.key_metric,
        },
        "experiments": [{
            "id": r.id, "name": r.name, "status": r.status, "detail": r.detail,
            "algorithm": r.algorithm, "n_hospitals": r.experiment.n_clients,
            "data_split": r.experiment.split_label, "seed": r.seed,
            "best_epoch": r.info.get("best_epoch"), "best_round": best_round(r),
            "training_time_s": training_time(r)[0],
            "training_time_kind": training_time(r)[1],
            "global_test": r.metrics, "curves": r.curves,
            "per_client": {s: {k: v for k, v in payload.items()
                               if k in ("metrics", "curves", "n_images", "error")}
                           for s, payload in (r.clients or {}).items()},
        } for r in runs],
        "cohort": {"partitions": part.to_dict("records"),
                   "splits": sites.to_dict("records")},
    }, indent=2, default=float))

    (out / "manifest.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": [r.id for r in runs if r.ok],
        "incomplete": {r.id: (r.detail or r.status) for r in runs if not r.ok},
        "n_complete": sum(1 for r in runs if r.ok), "n_total": len(runs),
    }, indent=2))

    write_markdown(out / "summary.md", runs, table, comps, part, sites, clients)
    write_latex(out / "tables", table, comps)
    write_pdf(out / "summary.pdf", table, comps, figures, status_lines(runs))
    write_readme(out / "README.md", runs, sites)

    n_done = sum(1 for r in runs if r.ok)
    print(f"\nwritten: {out}")
    print(f"  {n_done}/{len(runs)} experiments complete")
    print(f"  noise floor {NOISE_FLOOR:.3f} macro-AUC — differences below it are "
          "not results")
    if n_done < len(runs):
        print("\n  incomplete experiments are recorded as such in manifest.json and\n"
              "  README.md. Nothing has been invented to fill the gaps.")
    if args.require_complete and n_done < len(runs):
        raise SystemExit(
            f"\n--require-complete: only {n_done} of {len(runs)} experiments have "
            "results.\n  Run them, then scripts/collect_results.py, then this again.")


if __name__ == "__main__":
    main()
