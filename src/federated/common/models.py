"""The shared model. One definition, built identically at every site.

WHY THIS FILE IS THIN
---------------------
It delegates to `src/core/models.py` and adds nothing of its own.
That is deliberate. FedAvg averages tensors position by position: if two sites build
networks that differ by so much as an inserted Dropout, the averaged weights are
meaningless and nothing warns you. The previous iteration of this project kept 28
copies of `model.py` in step with a `sync_model.py` script, which is a bug waiting
for the one time somebody forgets to run it.

So there is exactly one definition of this network in the repository, and both the
classifier phase and the federated phase import it from the same place.

`verify_architecture` exists so that "every site built the same network" is a
checked fact rather than an assumption — see the docstring.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from .thesis import thesis_models


class FederatedClassifier(nn.Module):
    """The shared network, wrapped so NVFLARE can rebuild it from PRIMITIVE args.

    WHY THIS WRAPPER HAS TO EXIST
    -----------------------------
    `FedAvgRecipe(model=<nn.Module>)` does not ship the object. NVFLARE writes the
    server app to JSON and rebuilds the model there, recording it as a class path
    plus the constructor arguments it can recover from the instance
    (`fed_job_config.py::_get_args`: it walks the `__init__` signature and keeps any
    attribute that differs from the parameter default).

    Handing it a torchvision ResNet breaks that in two ways at once:

    1. **It does not serialise.** `ResNet.__init__` takes `norm_layer=None` and the
       instance stores `self._norm_layer = nn.BatchNorm2d`. `_get_args` sees a value
       differing from the default, `type(...).__name__` is `"type"` which is in
       `dir(builtins)`, so it stores the CLASS itself and `json.dumps` raises
       `Object of type type is not JSON serializable`.
    2. **Worse, if it had serialised it would be the wrong network.** The recorded
       path is `torchvision.models.resnet.ResNet` with almost no arguments, so the
       server would rebuild a default 1000-class ResNet while every client trains a
       3-class one with a `Dropout(0.5)+Linear` head. The run would complete and the
       numbers would be meaningless — the same failure this project already shipped
       once with a ResNet-18 server against ResNet-50 clients.

    THE TWO PROPERTIES THAT MAKE IT SAFE
    ------------------------------------
    * **Every constructor argument is a `str`, `int`, `float` or `bool`**, so the
      whole config is JSON-serialisable and describes the network exactly.
    * **The arguments are stored as attributes.** `_get_args` reads them off the
      instance; a value it cannot recover is silently replaced by the parameter
      default, which is how a server could end up with `dropout=0.0` (a bare
      `Linear`) against clients with `dropout=0.5` (a `Sequential`) — different
      state_dict keys, and FedAvg averaging positions that do not correspond.

    `state_dict` and `load_state_dict` delegate to the inner network, so the keys
    are IDENTICAL to `build_model()` — no `net.` prefix. That is what keeps the
    federated weights interchangeable with the centralised checkpoints, and what
    lets `load_checkpoint` stay `strict=True`.
    """

    def __init__(self, model_name: str = "resnet18", num_classes: int = 3,
                 pretrained: bool = True, dropout: float = 0.5,
                 freeze_until: str = "layer3", freeze_bn: bool = False) -> None:
        super().__init__()
        # Kept as attributes so NVFLARE recovers them — see the docstring.
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout
        self.freeze_until = freeze_until
        self.freeze_bn = freeze_bn

        from types import SimpleNamespace
        self.net = build_model(
            SimpleNamespace(model_name=model_name, pretrained=pretrained,
                            dropout=dropout, freeze_until=freeze_until,
                            freeze_bn=freeze_bn),
            num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.net.load_state_dict(state_dict, *args, **kwargs)


def federated_model(training, num_classes: int) -> "FederatedClassifier":
    """The wrapper, built from the shared `TrainingConfig`.

    One place converts the config into the primitive arguments NVFLARE records, so
    the server's copy cannot drift from what `build_model` gives the clients.
    """
    return FederatedClassifier(
        model_name=training.model_name, num_classes=num_classes,
        pretrained=training.pretrained, dropout=training.dropout,
        freeze_until=training.freeze_until, freeze_bn=training.freeze_bn)


def build_model(training, num_classes: int) -> nn.Module:
    """Build the shared network from a `TrainingConfig`.

    The freezing policy is applied here rather than by the caller, because a site
    that froze different layers would send back an update the server cannot
    distinguish from a genuine one.

    NO DROPOUT HEAD IS ATTACHED HERE
    --------------------------------
    There used to be an `_attach_dropout_head` at this point, because
    `core/models.py::build_model` accepted `dropout` and ignored it for every
    torchvision backbone — so the checkpoints, which store `fc.1.weight`, could not
    be loaded by the network the config described. `core/models.py` now honours
    `dropout` for every backbone, so wrapping the head a second time here would
    apply dropout twice and change the architecture out from under FedAvg.
    """
    model = thesis_models.build_model(
        training.model_name, num_classes,
        pretrained=training.pretrained, dropout=training.dropout)

    # Freezing AFTER the head is attached. `freeze_until` walks named stages
    # (conv1, bn1, layer1, ...) and never touches `fc`, so the order is not load
    # bearing — but doing it last keeps "the head is always trainable" true by
    # construction rather than by inspection.
    if training.freeze_until not in ("none", "", None):
        thesis_models.freeze_until(model, training.freeze_until)
    if training.freeze_bn:
        thesis_models.freeze_batchnorm(model)
    return model


def load_checkpoint(model: nn.Module, path, *, strict: bool = True) -> dict:
    """Load a classifier-phase checkpoint, and say plainly when it does not fit.

    `strict=True` on purpose. A head-shape mismatch loaded leniently gives a model
    with a randomly initialised classifier and no error — which on this task still
    produces a plausible-looking macro-AUC near chance, and would be read as a
    federated result rather than as a loading failure.
    """
    import torch as _torch

    ck = _torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise SystemExit(
            f"checkpoint {path} does not fit the network built from this config.\n"
            f"  {exc}\n"
            "  A 'fc.1.*' key means the checkpoint has Sequential(Dropout, Linear) "
            "and this build made a bare Linear — check TrainingConfig.dropout.") from exc
    return ck


def set_backbone_eval(model: nn.Module) -> None:
    """Re-apply BatchNorm eval() after `model.train()` — needed every epoch when
    `freeze_bn` is on, because `train()` puts everything back."""
    thesis_models.set_backbone_eval(model)


def describe(model: nn.Module, name: str) -> str:
    return thesis_models.describe(model, name)


def param_counts(model: nn.Module) -> dict[str, int]:
    return thesis_models._param_counts(model)


def architecture_fingerprint(model: nn.Module) -> str:
    """A short hash of every parameter NAME and SHAPE — not of the values.

    Two sites running the same code produce the same fingerprint; a site running a
    stale copy, a different backbone or a different class count does not. Values are
    excluded on purpose: weights are supposed to differ between sites, shapes are
    not.

    This exists because of a real failure. A recipe once exported the model as
    `{"path": "model.ClassifierNet"}` with the wrong default arguments, so the
    server built a ResNet-18 while the clients built a ResNet-50. The run completed.
    The numbers were nonsense. A fingerprint logged at round 0 by every participant
    turns that silent failure into a visible one.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
    return h.hexdigest()[:16]


def verify_architecture(model: nn.Module, expected: str | None) -> str:
    """Raise if this site's network does not match `expected`."""
    got = architecture_fingerprint(model)
    if expected and got != expected:
        raise SystemExit(
            f"architecture mismatch: this site built {got}, the job expects {expected}.\n"
            "  FedAvg averages tensors position by position, so continuing would "
            "produce a meaningless global model without raising an error.\n"
            "  Check that every site is running the same src/ and the same "
            "TrainingConfig.")
    return got


def get_device(allow_mps: bool = False) -> torch.device:
    """CUDA (NVIDIA) > CPU. Apple MPS only on explicit request — it is BROKEN here.

    Re-tested on torch 2.12: a standalone probe ran a full epoch on MPS finite and
    4x faster than CPU, but this project's real loop returns `loss nan` and
    `val AUC nan` from the first epoch. See `core/training.py::get_device` for the
    measurements. The ban stands.

    A silently diverged client would be averaged into the global model by a server
    with no way to tell — which is why `federation/client.py` now refuses to send a
    non-finite update, on every device.
    """
    from .thesis import thesis_training
    return thesis_training.get_device(allow_mps=allow_mps)


def apply_mps_workaround(model: nn.Module, device: torch.device) -> int:
    """Contiguous BatchNorm inputs on the Apple GPU. No-op on CUDA and CPU."""
    from .thesis import thesis_training
    return thesis_training.apply_mps_workaround(model, device)


def workers_for(device: torch.device, requested: int) -> int:
    """DataLoader workers for this device — 0 on MPS, `requested` elsewhere."""
    from .thesis import thesis_training
    return thesis_training.workers_for(device, requested)
