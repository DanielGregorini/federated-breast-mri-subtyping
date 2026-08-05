"""The training loop. One function trains one experiment, end to end.

    from dataset_config import Config
    from core.training import run
    run_dir = run(Config(model="resnet18", task="subtype"))

`train_one_epoch` returns after a single pass. That shape is deliberate: a
centralised baseline calls it N times in a row, and a federated client calls it
once per round between receiving and sending weights. Neither needs to know about
the other, which is what lets the same code serve both.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from . import evaluation as ev
from . import models as M
from . import reporting as R
from .experiment import new_experiment


def get_device(allow_mps: bool = False) -> torch.device:
    """CUDA (NVIDIA) > CPU. Apple MPS only on explicit request — it is BROKEN here.

    MPS WAS RE-TESTED ON torch 2.12 AND STILL FAILS. THE NUMBERS:
    ------------------------------------------------------------
    A standalone probe — real model, real data, plain AdamW loop — ran a FULL epoch
    on MPS with every loss and every weight finite, at ~80 img/s against ~20 on CPU.
    That looked like the old failure had been fixed by a newer torch.

    It had not. The probe did not exercise this project's actual training loop. Run
    through `run_centralized.py`, MPS produces:

        epoch 001/2  loss nan  train_acc 0.9018 | val AUC nan   (190s)
        epoch 002/2  loss nan  train_acc 0.9161 | val AUC nan   (183s)

    NaN loss, NaN AUC, and a training accuracy of 0.90 in the first epoch where CPU
    gives ~0.43 — the accumulator is as broken as the loss. The identical code on
    CPU is finite throughout, and on CUDA it trains normally.

    So the ban stands. What changed is only the confidence behind it: the failure is
    now known to survive torch 2.12, and to hide from a simple probe. Anything that
    "works on MPS" must be demonstrated through THIS loop, not a reduced one.

    Real training belongs on CUDA. The laptop builds datasets and reads results.
    `get_device(allow_mps=True)` still exists for anyone wanting to retest.
    """
    if os.environ.get("BREAST_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        # An op MPS has not implemented falls back to CPU rather than raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def apply_mps_workaround(model, device: torch.device) -> int:
    """Force every BatchNorm input contiguous on the Apple GPU. Returns how many.

    A no-op on CUDA and CPU, so it can be called unconditionally and never changes
    what an NVIDIA box or a CPU run computes. On MPS it sidesteps a backward bug in
    the Apple backend that a non-contiguous BatchNorm input can trigger.

    Measured on torch 2.12 this run is finite with AND without the hook, so it is
    belt-and-braces rather than load-bearing today. It is cheap, it is scoped to
    MPS, and the failure it guards against was previously silent — which is the
    combination that makes a guard worth keeping.
    """
    if device.type != "mps":
        return 0
    n = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_pre_hook(lambda m, i: (i[0].contiguous(),))
            n += 1
    return n


def workers_for(device: torch.device, requested: int) -> int:
    """DataLoader workers for this device. **0 on MPS**, `requested` elsewhere.

    MPS with worker processes is unstable: DataLoader workers use `spawn` on macOS
    and hang when the parent holds an MPS context.

    On this project that is already enforced upstream — `core/data.py::
    effective_num_workers` returns 0 for the whole of macOS, MPS or not — so this
    function is currently belt-and-braces. It exists because that upstream rule is
    keyed on the OPERATING SYSTEM while the real constraint is the DEVICE: a Linux
    box with an Apple-silicon-like backend, or a macOS run that somehow reached a
    non-zero value, would still need this. Keying the guard to the device is the
    honest form of the rule.

    Decoding PNGs single-threaded is slower in principle, but MPS at 0 workers still
    measured ~80 img/s against ~20 on CPU, so the trade is heavily positive.
    """
    return 0 if device.type == "mps" else requested


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          patience=5, factor=0.5)
    return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp: bool, freeze_bn: bool = False) -> tuple[float, float]:
    """One pass over the training set. Returns (mean loss, slice accuracy).

    The running totals live on the GPU and are read once at the end. Reading
    `.item()` per batch forces the CPU to wait for the GPU every step — measured
    at 60.1 ms per batch against 50.2 ms without, about 20% of the step spent
    purely synchronising.
    """
    model.train()
    if freeze_bn:
        M.set_backbone_eval(model)

    # Apple MPS needs an explicit synchronisation once per step. Without one, the
    # queue runs far enough ahead that weights are corrupted to NaN partway
    # through an epoch — measured here, non-deterministically, with the same seed
    # giving accuracies 0.6312 and 0.6832 and `isfinite(parameters)` False. The
    # same code on CPU and on CUDA is exact. The per-batch `.item()` this loop
    # deliberately removed for speed was hiding it by synchronising as a side
    # effect. CUDA keeps the fast path; MPS pays for correctness.
    sync = torch.mps.synchronize if device.type == "mps" else None

    # ACCUMULATE ON THE HOST OUTSIDE CUDA. Measured, not defensive.
    #
    # Keeping the running totals as 0-dim device tensors is a real speed win on
    # CUDA — reading `.item()` per batch costs about 20% of the step. On Apple MPS
    # it silently produces garbage: a full run through this loop reported
    #
    #     epoch 001/2  loss nan  train_acc 0.9018
    #
    # where CPU gives loss ~1.15 and train_acc ~0.43. An accuracy of 0.90 in the
    # first epoch is not a diverged model, it is a broken accumulator — the loss
    # NaN and the impossible accuracy come from the same place. Reading `.item()`
    # per batch forces a synchronisation and the numbers are correct again, which is
    # exactly what the removed `.item()` used to be hiding.
    #
    # So: device tensors on CUDA, Python floats everywhere else.
    on_device = device.type == "cuda"
    running = torch.zeros((), device=device, dtype=torch.float32) if on_device else 0.0
    correct = torch.zeros((), device=device, dtype=torch.float32) if on_device else 0.0
    seen = 0

    # autocast is entered ONLY when AMP is actually on, which is CUDA only. Entering
    # `torch.autocast("mps", enabled=False)` is not reliably inert across torch
    # versions, and this loop has no reason to enter it at all when disabled.
    from contextlib import nullcontext

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast(device.type, enabled=True) if use_amp else nullcontext()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if on_device:
            running += loss.detach() * y.size(0)
            correct += (logits.detach().argmax(1) == y).sum()
        else:
            running += loss.item() * y.size(0)
            correct += float((logits.detach().argmax(1) == y).sum().item())
        seen += y.size(0)
        if sync is not None:
            sync()

    total_loss = float(running.item()) if on_device else float(running)
    total_ok = float(correct.item()) if on_device else float(correct)
    return total_loss / seen, total_ok / seen


def evaluate_split(model, loader, rows: pd.DataFrame, cfg, device,
                   use_amp: bool) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """Patient-level metrics for one split, plus what is needed to plot them."""
    probs, labels, idxs = ev.predict(model, loader, device, use_amp)
    p, y, pids = ev.aggregate_by_patient(probs, idxs, rows, cfg.aggregation)
    metrics = ev.compute_metrics(y, p, cfg.class_names)
    # Slice-level accuracy alongside, purely as an overfitting signal. Never used
    # to select a model — a slice-level score measures patient recognition.
    metrics["slice_accuracy"] = float((probs.argmax(1) == labels).mean())
    eps = 1e-9
    metrics["slice_loss"] = float(
        -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())
    return metrics, y, p, pids


def run(cfg, results_dir: Path | None = None, suffix: str = "",
        log=print) -> Path:
    """Train one experiment and write everything it produces.

    Returns the experiment folder. Nothing is left for a human to do afterwards.
    """
    run_dir = new_experiment(cfg, results_dir, suffix)
    log(cfg.summary())
    log(f"\nrun folder: {run_dir}\n")

    D.set_seed(cfg.seed)
    device = get_device()
    # AMP on CUDA only. On Apple MPS it was measured 13% SLOWER than fp32 and
    # GradScaler is CUDA-specific — throughput only, the maths is identical.
    use_amp = cfg.mixed_precision and device.type == "cuda"

    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(
            f"dataset not found: {cfg.dataset_dir}\n"
            f"Build it first with notebook 02 (authors) or 03 (mine).")

    loaders, frames = D.make_loaders(cfg.dataset_dir, cfg)
    for split, rows in frames.items():
        log(f"  {split:<5} {len(rows):>7,} slices / {rows.pid.nunique():>5} patients"
            f"   trivial baseline {D.trivial_baseline(rows):.4f}")

    model = M.build_model(cfg.model, cfg.num_classes, pretrained=True,
                          dropout=cfg.dropout).to(device)
    if cfg.freeze_until not in ("none", "", None):
        M.freeze_until(model, cfg.freeze_until)
    if cfg.freeze_bn:
        M.freeze_batchnorm(model)
    log("\n" + M.describe(model, cfg.model))
    params = M._param_counts(model)

    weight = (D.class_weights(frames["train"], cfg.num_classes, device)
              if cfg.class_weighted_loss else None)
    if weight is not None:
        log(f"class weights (patient level): {[round(w, 3) for w in weight.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler = getattr(loaders["train"], "batch_sampler", None)

    history: list[dict] = []
    best_score, best_epoch, stale = -np.inf, 0, 0
    train_acc_at_best = float("nan")

    log("\n" + "-" * 78)
    for epoch in range(1, cfg.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            use_amp, cfg.freeze_bn)

        val_metrics, _, _, _ = evaluate_split(model, loaders["val"], frames["val"],
                                              cfg, device, use_amp)
        # Read the rate BEFORE stepping, so the logged value is the one this epoch
        # actually trained with.
        lr_now = optimizer.param_groups[-1]["lr"]
        if scheduler is not None:
            if cfg.scheduler == "plateau":
                scheduler.step(val_metrics[cfg.monitor_metric])
            else:
                scheduler.step()

        history.append({
            "epoch": epoch, "lr": lr_now, "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_metrics["slice_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_auc": val_metrics["auc"], "val_macro_f1": val_metrics["macro_f1"],
            "seconds": time.time() - t0,
        })
        log(f"epoch {epoch:03d}/{cfg.epochs}  lr {lr_now:.2e}  "
            f"loss {train_loss:.4f}  train_acc {train_acc:.4f} | "
            f"val AUC {val_metrics['auc']:.4f}  bal {val_metrics['balanced_accuracy']:.4f}"
            f"  ({time.time()-t0:.0f}s)")

        score = val_metrics[cfg.monitor_metric]
        improved = np.isfinite(score) and score > best_score + 1e-5
        if improved or epoch == 1:
            best_score, best_epoch, stale = score, epoch, 0
            train_acc_at_best = train_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": cfg.as_dict(), "val_metrics": val_metrics},
                       run_dir / "checkpoints" / "best_model.pt")
        else:
            stale += 1
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "config": cfg.as_dict(), "val_metrics": val_metrics},
                   run_dir / "checkpoints" / "last_model.pt")

        if cfg.early_stopping_patience > 0 and stale >= cfg.early_stopping_patience:
            log(f"early stopping: {stale} epochs without improvement")
            break
    log("-" * 78)
    log(f"best {cfg.monitor_metric} {best_score:.4f} at epoch {best_epoch}\n")

    # ---- final evaluation, from the SELECTED checkpoint ------------------- #
    ck = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    results: dict[str, dict] = {}
    for split in ("val", "test"):
        if split not in loaders:
            continue
        m, y, p, pids = evaluate_split(model, loaders[split], frames[split], cfg,
                                       device, use_amp)
        results[split] = R.write_split(split, y, p, pids, cfg.class_names, run_dir,
                                       frames[split])
        results[split]["slice_accuracy"] = m["slice_accuracy"]
        log(f"{split:<5} macro-AUC {m['auc']:.4f} | acc {m['accuracy']:.4f} "
            f"(trivial {m['trivial_baseline_accuracy']:.4f}) | "
            f"balanced {m['balanced_accuracy']:.4f}")

    hist = pd.DataFrame(history)
    R.write_all(run_dir, cfg, hist, results, best_epoch, params)
    # The generalisation gap needs the training accuracy at the SELECTED epoch,
    # not the last one — the last describes a model that was never used.
    import json
    rj = run_dir / "results.json"
    payload = json.loads(rj.read_text())
    payload["train_acc_at_best"] = train_acc_at_best
    payload["generalisation_gap"] = {
        s: train_acc_at_best - results[s]["accuracy"] for s in results}
    rj.write_text(json.dumps(payload, indent=2, default=float))

    log(f"\nwritten: {run_dir}")
    return run_dir
