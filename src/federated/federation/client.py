#!/usr/bin/env python3
"""The federated client. One hospital, one process, one shard of the data.

This is the script NVFLARE ships to every site and runs there. It is a normal
training script wrapped in three calls:

    flare.init()
    while flare.is_running():
        model = flare.receive()      # global weights arrive
        ... train locally ...
        flare.send(weights)          # only weights leave the hospital

Images never appear in that loop. That is the privacy claim made concrete: the
payload crossing the network is a state dict, and it is the only thing that crosses.

WHAT HAPPENS IN ONE ROUND, AND WHY IN THIS ORDER
------------------------------------------------
    1. receive the global model
    2. evaluate it on this hospital's held-out patients   -> the "agg" curve
    3. train `local_epochs` epochs on this hospital's training patients
    4. evaluate the trained model on the same held-out patients -> the "post" curve
    5. send the weights, reporting the step-4 metric

Steps 2 and 4 answer different questions and both are needed. Step 2 is the
convergence curve of the FEDERATION — how good the aggregated model is, which is what
RQ1 is read from. Step 4 describes the weights actually being sent, which is what the
server must select on: the aggregated model of round r is built from the messages of
round r, so selecting on a metric measured before local training would score the
previous round's model.

WHAT MUST NOT BE REPORTED AS THE KEY METRIC
-------------------------------------------
Training accuracy. A previous iteration of this project sent it, so the server
selected whichever global model let clients memorise their own shard best. On this
data that reaches 99% within a few epochs and carries no information whatsoever.
The metric is `val_balanced_accuracy`, computed on patients this client did not
train on, and its name is pinned in `config/experiments.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Locating the project                                                         #
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    """Find `federated/`, from the environment or by walking up.

    NVFLARE runs this script from inside a job workspace, several directories away
    from the repository, so `Path(__file__).parent` is not enough on its own once
    the file has been copied into a job. $FEDBREAST_ROOT is checked first and is
    what a real hospital machine would set.
    """
    env = os.environ.get("FEDBREAST_ROOT")
    if env and (Path(env) / "config" / "experiments.py").is_file():
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "config" / "experiments.py").is_file():
            return parent
    raise SystemExit(
        "cannot locate federated/ (needs config/experiments.py).\n"
        "  fix: export FEDBREAST_ROOT=/path/to/federated")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402 — the path bootstrap above must run first.
from config import experiments as EX
from common import data as D
from common import evaluation as EV
from common import models as M
from common import training as T

import nvflare.client as flare


ROUND_FIELDS = [
    "round", "site", "n_train_patients", "n_train_slices", "lr",
    "train_loss", "train_acc",
    # the model that ARRIVED this round, scored before any local training
    "agg_val_acc", "agg_val_bal_acc", "agg_val_auc", "agg_val_macro_f1",
    # the model being SENT — this is what the server selects on
    "post_val_acc", "post_val_bal_acc", "post_val_auc", "post_val_macro_f1",
]


def build_logger(log_dir: Path, site: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"client.{site}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f"[{site}] %(message)s"))
    logger.addHandler(sh)
    return logger


def append_round(path: Path, row: dict) -> None:
    """One CSV row per round, so the convergence figures need no log parsing.

    Wrapped so it can never break a round: losing a 30-round federated run because
    a metrics file could not be written would be a bad trade.
    """
    try:
        new = not path.is_file()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROUND_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as exc:
        print(f"[metrics] round {row.get('round')} not written: {exc}", flush=True)


def resolve_site_dir(args, site: str) -> Path:
    """This hospital's own data folder, and nobody else's.

    Order: explicit flag, then $BREAST_SITE_DIR (what a real hospital machine sets),
    then the partition layout this project generates. The site never receives a path
    to the pooled dataset — it cannot read what is not there.
    """
    if args.site_dir:
        return Path(args.site_dir)
    env = os.environ.get("BREAST_SITE_DIR")
    if env:
        return Path(env)
    return EX.PARTITIONS_DIR / args.partition / site


def evaluate(model, loader, rows, device, cfg, use_amp) -> dict:
    """Patient-level metrics, guarded. A metric failure must not kill the round."""
    if loader is None:
        return {}
    try:
        return EV.evaluate(model, loader, rows, device,
                           num_classes=EX.NUM_CLASSES, class_names=EX.CLASS_NAMES,
                           aggregation=cfg.aggregation, use_amp=use_amp)
    except Exception as exc:
        print(f"[eval] skipped: {exc}", flush=True)
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--partition", required=True,
                   help="which split under data/partitions/ this run uses")
    p.add_argument("--site-dir", default=None, help="explicit data folder override")
    p.add_argument("--local-epochs", type=int, default=EX.FEDERATION.local_epochs)
    p.add_argument("--num-rounds", type=int, default=EX.FEDERATION.num_rounds)
    # > 0 turns FedAvg into FedProx. The server sends nothing algorithm-specific:
    # aggregation is identical, and the whole difference is this coefficient.
    p.add_argument("--fedprox-mu", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=EX.TRAINING.seed)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--architecture", default=None,
                   help="expected architecture fingerprint; mismatch is fatal")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    flare.init()
    site = flare.get_site_name()

    site_dir = resolve_site_dir(args, site)
    results_dir = Path(args.results_dir) if args.results_dir else (
        EX.RESULTS_DIR / "sites" / site)
    logger = build_logger(results_dir, site)

    training = EX.TRAINING
    D.set_seed(args.seed)
    device = M.get_device()
    use_amp = T.use_amp_on(device, training.mixed_precision)

    # ---- data: this hospital's shard only ------------------------------- #
    loaders, frames, cfg = D.load_site(site_dir, training, epochs=args.local_epochs,
                                       seed=args.seed)
    train_rows = frames["train"]
    val_loader = loaders.get("val")
    val_rows = frames.get("val")
    n_patients = int(train_rows.pid.nunique())
    n_slices = int(len(train_rows))

    logger.info(f"=== {site} | data={site_dir} ===")
    logger.info(D.describe_site(site, frames, EX.NUM_CLASSES))
    logger.info(f"device={device} amp={use_amp} fedprox_mu={args.fedprox_mu} "
                f"local_epochs={args.local_epochs}")

    if val_loader is None:
        # Not fatal, but it must be loud: without a local validation split this
        # client cannot report the metric the server selects on, and the server
        # would silently fall back to selecting on nothing useful.
        logger.warning("NO LOCAL VALIDATION SPLIT — this client cannot report "
                       f"{EX.FEDERATION.key_metric}. Run scripts/partition_data.py.")

    # ---- model: identical at every site --------------------------------- #
    model = M.build_model(training, EX.NUM_CLASSES).to(device)
    fingerprint = M.verify_architecture(model, args.architecture)
    logger.info(M.describe(model, training.model_name))
    logger.info(f"architecture fingerprint {fingerprint}")

    # ---- loss: local or global class frequencies (RQ4) ------------------- #
    override = None
    if training.class_weight_scope == "global":
        manifest = site_dir / "manifest.json"
        if manifest.is_file():
            override = json.loads(manifest.read_text()).get("global_class_weights")
        if override is None:
            raise SystemExit(
                f"class_weight_scope='global' but {manifest} has no "
                "'global_class_weights'. Re-run scripts/partition_data.py.")
    weights = (D.class_weights(train_rows, EX.NUM_CLASSES, device, override)
               if training.class_weighted_loss else None)
    if weights is not None:
        logger.info(f"class weights ({training.class_weight_scope}, per patient): "
                    f"{[round(w, 3) for w in weights.tolist()]}")
    criterion = T.build_criterion(weights, training.label_smoothing)

    optimizer = T.build_optimizer(model, training)
    scaler = T.build_scaler(device, use_amp)
    sampler = getattr(loaders["train"], "batch_sampler", None)
    rounds_csv = results_dir / "rounds.csv"

    # ------------------------------------------------------------------ #
    # The federated loop                                                  #
    # ------------------------------------------------------------------ #
    while flare.is_running():
        incoming = flare.receive()
        rnd = incoming.current_round if incoming.current_round is not None else 0

        if incoming.params:
            model.load_state_dict(
                {k: torch.as_tensor(v) for k, v in incoming.params.items()})

        # 1. score the model that just ARRIVED, before touching it.
        agg = evaluate(model, val_loader, val_rows, device, cfg, use_amp)
        if agg:
            logger.info(f"round {rnd:03d} | AGG  auc={agg['auc']:.4f} "
                        f"bal={agg['balanced_accuracy']:.4f} acc={agg['accuracy']:.4f}")

        # 2. train locally. The LR follows the same cosine curve the centralised
        #    baseline follows, evaluated from the round index — see src/training.py.
        lr = T.lr_for_round(training.learning_rate, rnd, args.num_rounds,
                            training.scheduler)
        T.set_lr(optimizer, lr)
        anchor = T.snapshot_global(model) if args.fedprox_mu > 0 else None

        loss = acc = float("nan")
        for local_epoch in range(args.local_epochs):
            if hasattr(sampler, "set_epoch"):
                # Vary batch composition across rounds, not just within one.
                sampler.set_epoch(rnd * args.local_epochs + local_epoch)
            loss, acc = T.train_one_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device,
                use_amp=use_amp, freeze_bn=training.freeze_bn,
                prox_mu=args.fedprox_mu, global_params=anchor)
            logger.info(f"round {rnd:03d} | epoch {local_epoch + 1}/"
                        f"{args.local_epochs} lr={lr:.2e} loss={loss:.4f} "
                        f"train_acc={acc:.4f}")

        # 3. score the model being SENT. This is what the server selects on.
        post = evaluate(model, val_loader, val_rows, device, cfg, use_amp)
        if post:
            logger.info(f"round {rnd:03d} | POST auc={post['auc']:.4f} "
                        f"bal={post['balanced_accuracy']:.4f} "
                        f"acc={post['accuracy']:.4f}")

        append_round(rounds_csv, {
            "round": rnd, "site": site, "n_train_patients": n_patients,
            "n_train_slices": n_slices, "lr": f"{lr:.6e}",
            "train_loss": f"{loss:.6f}", "train_acc": f"{acc:.6f}",
            "agg_val_acc": agg.get("accuracy", ""),
            "agg_val_bal_acc": agg.get("balanced_accuracy", ""),
            "agg_val_auc": agg.get("auc", ""),
            "agg_val_macro_f1": agg.get("macro_f1", ""),
            "post_val_acc": post.get("accuracy", ""),
            "post_val_bal_acc": post.get("balanced_accuracy", ""),
            "post_val_auc": post.get("auc", ""),
            "post_val_macro_f1": post.get("macro_f1", ""),
        })

        # 4. send. Metrics carry the server's key metric under its pinned name.
        metrics = {"train_accuracy": acc, "train_loss": loss}
        if post:
            metrics["val_accuracy"] = post["accuracy"]
            metrics["val_balanced_accuracy"] = post["balanced_accuracy"]
            metrics["val_macro_f1"] = post["macro_f1"]
            metrics["val_auc"] = (post["auc"] if post["auc"] == post["auc"] else 0.0)

        payload = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # A DIVERGED SITE MUST NOT BE AVERAGED IN SILENTLY.
        #
        # FedAvg sums tensors position by position. One site sending NaN or Inf
        # poisons every position it touches, and the global model is NaN from that
        # round on — with nothing in the server log saying which site did it, or
        # that anything happened at all. The run completes and every number after
        # it is meaningless.
        #
        # This is not hypothetical here. Training this model on Apple MPS used to
        # corrupt weights to NaN partway through the first epoch, which is why MPS
        # was banned outright. It has been re-measured as fixed on torch 2.12 and
        # MPS is now used by default, so the ban is gone — and this check is what
        # replaces it. It is device-independent on purpose: CUDA runs diverge too.
        bad = [k for k, v in payload.items()
               if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise SystemExit(
                f"round {rnd}: {site} produced non-finite weights in "
                f"{len(bad)} tensor(s) — e.g. {bad[:3]}.\n"
                f"  device={device}, loss={loss}, acc={acc}\n"
                "  REFUSING to send. Averaging this would silently poison the "
                "global model for every remaining round.\n"
                "  On Apple MPS, re-run with BREAST_FORCE_CPU=1 to confirm the "
                "device is the cause.")

        flare.send(flare.FLModel(
            params=payload,
            metrics=metrics,
            # FedAvg weights each site's update by how much data it trained on.
            # Patients, not slices: a site whose patients happen to have larger
            # tumours contributes more slices without holding more evidence.
            meta={"NUM_STEPS_CURRENT_ROUND": n_patients},
        ))
        logger.info(f"round {rnd:03d} | sent ({n_patients} patients, "
                    f"{n_slices} slices, all weights finite)")


if __name__ == "__main__":
    main()
