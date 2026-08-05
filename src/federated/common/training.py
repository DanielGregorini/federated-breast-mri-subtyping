"""The training loop. One call trains one epoch and returns.

WHY THAT SHAPE
--------------
The centralised baseline calls it thirty times in a row. A federated client calls it
once per round, in between receiving and sending weights. Neither needs to know the
other exists, which is what lets the same code serve both — and what makes RQ1 a
measurement of federation rather than of two different trainers.

WHERE THE CODE ACTUALLY LIVES
-----------------------------
`train_one_epoch` with `prox_mu == 0` **delegates to
`src/core/training.py`**, unchanged. So FedAvg clients and the
centralised baseline run byte-identical training code, and that is a fact about the
call graph rather than a promise in a comment.

FedProx needs a term that depends on the model parameters rather than on the logits,
so it cannot be expressed as a criterion and the loop is forked below. The fork
differs from the shared loop by exactly two things: it keeps a frozen copy of the
received global weights, and it adds `mu/2 * ||w - w_global||^2` to the loss before
the backward pass. Everything else — AMP, gradient clipping, GPU-resident counters —
is mirrored deliberately so the two paths cannot drift on anything that is not
FedProx.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import models as M
from .thesis import thesis_training


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, *,
                    use_amp: bool = False, freeze_bn: bool = False,
                    prox_mu: float = 0.0,
                    global_params: dict[str, torch.Tensor] | None = None
                    ) -> tuple[float, float]:
    """One pass over `loader`. Returns (mean loss, slice accuracy).

    The reported loss is always the TASK loss, with the proximal term excluded.
    Including it would make FedAvg and FedProx losses incomparable across the very
    curves RQ3 is read from, and would make the number drift as `mu` changes rather
    than as the model improves.
    """
    if prox_mu <= 0:
        # The shared path: literally the classifier phase's loop.
        return thesis_training.train_one_epoch(
            model, loader, criterion, optimizer, scaler, device, use_amp, freeze_bn)

    if global_params is None:
        raise ValueError("prox_mu > 0 requires the received global weights. "
                         "A client that ignores them is running FedAvg while "
                         "reporting FedProx, and nothing would warn you.")

    model.train()
    if freeze_bn:
        M.set_backbone_eval(model)

    running = torch.zeros((), device=device, dtype=torch.float32)
    correct = torch.zeros((), device=device, dtype=torch.float32)
    seen = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
            # FedProx: anchor this site's weights to the global model it received.
            prox = sum(((p - global_params[n]) ** 2).sum()
                       for n, p in model.named_parameters()
                       if p.requires_grad and n in global_params)
            total = loss + 0.5 * prox_mu * prox
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += loss.detach() * y.size(0)     # task loss only — see docstring
        correct += (logits.detach().argmax(1) == y).sum()
        seen += y.size(0)
    return float(running.item()) / seen, float(correct.item()) / seen


def snapshot_global(model: nn.Module) -> dict[str, torch.Tensor]:
    """A detached copy of the trainable weights, for the FedProx anchor.

    Only `requires_grad` parameters are kept: frozen layers cannot drift, so
    anchoring them would add a constant zero to the loss and a real cost in memory
    and time. Buffers (BatchNorm running statistics) are excluded because they are
    not parameters and have no gradient to penalise.
    """
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def build_optimizer(model: nn.Module, training) -> torch.optim.Optimizer:
    """AdamW over the TRAINABLE parameters only.

    Filtering matters when `freeze_until` is set: handing frozen parameters to AdamW
    still allocates optimiser state for them, and weight decay would then be applied
    to tensors that never receive a gradient.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if training.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=training.learning_rate,
                                 weight_decay=training.weight_decay)
    if training.optimizer == "sgd":
        return torch.optim.SGD(params, lr=training.learning_rate, momentum=0.9,
                               weight_decay=training.weight_decay)
    raise ValueError(f"unknown optimizer {training.optimizer!r}")


def build_criterion(weights: torch.Tensor | None, label_smoothing: float
                    ) -> nn.Module:
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def build_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """AMP on CUDA only. On Apple MPS it was measured 13% SLOWER than fp32, and
    `GradScaler` is CUDA-specific. Throughput only — the maths is identical."""
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def use_amp_on(device: torch.device, requested: bool) -> bool:
    """Mixed precision on CUDA only — never on MPS, never on CPU.

    Not caution about speed: `torch.amp.GradScaler` is a CUDA construct, and the
    loop here calls `scaler.unscale_()` before clipping gradients. On MPS the scaler
    is a no-op passthrough, so enabling AMP would change what autocast does to the
    forward pass while the loss scaling that makes fp16 numerically safe is absent —
    which is how a silent NaN gets reintroduced on exactly the device that used to
    produce them.

    MPS runs in fp32. It is still ~4x faster than CPU on this machine.
    """
    return bool(requested) and device.type == "cuda"


def lr_for_round(base_lr: float, current_round: int, num_rounds: int,
                 schedule: str = "cosine") -> float:
    """The learning rate this round should train with. Stateless by construction.

    A federated client is re-instantiated by the NVFLARE runtime and holds no memory
    between rounds, so a `CosineAnnealingLR` object cannot survive to be stepped. The
    obvious workarounds are both wrong: dropping the schedule leaves the federated
    arm training at a constant rate while the centralised baseline decays, and
    re-creating the scheduler each round produces a sawtooth that resets to the base
    rate every time. Either turns RQ1 into a comparison of learning-rate schedules.

    Because cosine annealing is a closed-form function of the step index, and the
    server sends `current_round` with every model, the client can simply evaluate it:

        lr(r) = base * (1 + cos(pi * r / T)) / 2

    which is exactly the value `CosineAnnealingLR(T_max=T)` holds at epoch `r` in the
    centralised run. The two arms therefore follow the same curve without the client
    keeping any state.

    In practice the effect is small — the best epoch on this task lands between 1 and
    5, where cosine has decayed by under 3% — but "small" is not a reason to leave a
    known asymmetry in the one comparison the dissertation is built on.
    """
    if schedule in ("none", "", None) or num_rounds <= 1:
        return base_lr
    if schedule == "cosine":
        import math
        r = min(max(current_round, 0), num_rounds)
        return base_lr * (1.0 + math.cos(math.pi * r / num_rounds)) / 2.0
    raise ValueError(
        f"schedule {schedule!r} cannot be evaluated statelessly. Only 'cosine' and "
        f"'none' are supported on the federated path; 'plateau' needs history the "
        f"client does not have.")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
