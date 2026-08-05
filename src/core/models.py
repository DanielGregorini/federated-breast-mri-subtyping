"""Model zoo. One name in, one ready-to-train network out.

    from core.models import build_model
    net = build_model("resnet18", num_classes=3)

Every architecture listed in `config.MODEL` is wired here, and each was checked to
build and run a forward pass. Adding one means adding a branch below and a
commented line in `config.py` — nothing else in the project changes.

WHY THE HEAD IS BUILT THE WAY IT IS
-----------------------------------
Every backbone ends with exactly one `Dropout(p)` feeding the final `Linear`, where
`p` is `TrainingConfig.dropout`. Uniform on purpose: `dropout` is written into every
`results.json`, so a build that honoured it for some backbones and ignored it for
others would report hyperparameters that do not describe the trained model. That is
precisely what this file used to do — see `_new_head` — and it left seven trained
checkpoints unloadable.

The rule that keeps checkpoints loadable is *never insert into an existing
`Sequential`*. Inserting shifts every index after it, so a checkpoint saved by one
build fails to load into another — a bug this project already paid for once. The
final `Linear` is therefore only ever REPLACED in place, and where a backbone
already ships a `Dropout` feeding it (efficientnet, mobilenet) that `Dropout` is
retuned to `p` rather than stacked with a second one.

The resulting head, per layout:

    resnet, densenet, swin, vit    fc / classifier / head  = Sequential(Dropout, Linear)
    efficientnet, mobilenet        native classifier Dropout retuned to p,
                                   final Linear replaced in place
    convnext                       classifier[2]           = Sequential(Dropout, Linear)

At `dropout=0` no `Sequential` is created and the key layout is exactly
torchvision's own.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

# HuggingFace checkpoints, loaded through `transformers` rather than torchvision.
HF_CHECKPOINTS: dict[str, str] = {
    # The authors' own model. Confirmed against their released checkpoint:
    # 85,800,194 parameters, classifier head (2, 768).
    "vit_mae_base": "facebook/vit-mae-base",
}

# Every model `config.MODEL` may name, with its parameter count at 3 classes.
SUPPORTED: dict[str, str] = {
    "resnet18": "11.2M — the measured winner on the subtype task",
    "resnet34": "21.3M",
    "resnet50": "23.5M — the reference in the comparable literature",
    "efficientnet_b0": "4.0M",
    "efficientnet_b3": "10.7M",
    "convnext_tiny": "27.8M",
    "convnext_small": "49.5M",
    "mobilenet_v3_large": "4.2M",
    "densenet121": "7.0M",
    "vit_b_16": "85.8M — torchvision ViT, ImageNet-pretrained",
    "vit_mae_base": "85.8M — facebook/vit-mae-base, the AUTHORS' model",
    "swin_t": "27.5M",
    "thda_resnet34": "21.3M — dual-attention ResNet, arXiv:2510.13897",
}


class HFClassifier(nn.Module):
    """A HuggingFace image classifier behind a plain `forward -> logits`.

    Exists so the training loop never has to know whether it holds a torchvision
    module or a HuggingFace one.
    """

    def __init__(self, checkpoint: str, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        from transformers import ViTForImageClassification

        self.net = ViTForImageClassification.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.net(x).logits)


class THDAResNet(nn.Module):
    """Dual-attention ResNet-34, after Fridman & Goldstein (arXiv:2510.13897).

    A channel-attention and a spatial-attention block on the final feature map.
    Included because it is the architecture the authors report as their best for
    HER2 (AUC 0.744, against 0.668 for a ViT).
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.5) -> None:
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = tvm.resnet34(weights=weights)
        # Everything except avgpool and fc — we need the spatial map for attention.
        self.features = nn.Sequential(*list(base.children())[:-2])
        c = base.fc.in_features

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // 16), nn.ReLU(inplace=True),
            nn.Linear(c // 16, c), nn.Sigmoid())
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = f * self.channel_att(f).unsqueeze(-1).unsqueeze(-1)
        pooled = torch.cat([f.mean(1, keepdim=True), f.max(1, keepdim=True)[0]], dim=1)
        f = f * self.spatial_att(pooled)
        return self.head(f)


def _new_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    """`Dropout(p) -> Linear`, or a bare `Linear` when `p` is 0.

    Wrapping the new `Linear` in its own two-element `Sequential` is what the
    checkpoints in `results/classifier/checkpoints/` were trained with: every one of them
    stores `fc.1.weight` / `fc.1.bias`. A build that returns a bare `Linear` here
    cannot load them at all, and `strict=False` "loads" them with the classifier
    left at random init — which still scores near chance and reads as a result
    rather than as a failure. Hence `strict=True` everywhere, and hence this.
    """
    head = nn.Linear(in_features, num_classes)
    return nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head


def _retune_dropout_before(seq: nn.Sequential, i: int, dropout: float) -> bool:
    """Set `p` on a `Dropout` that already feeds `seq[i]`. True if there was one.

    efficientnet ships `Sequential(Dropout(0.2), Linear)` and mobilenet_v3
    `Sequential(Linear, Hardswish, Dropout(0.2), Linear)`. Adding a second Dropout
    to those would leave the effective rate at neither 0.2 nor `p`, and would put a
    number in `results.json` that no layer actually uses. Retuning the existing one
    keeps one dropout at exactly `p`, and keeps every index stable.
    """
    if i > 0 and isinstance(seq[i - 1], nn.Dropout):
        seq[i - 1].p = dropout
        return True
    return False


def build_model(name: str, num_classes: int, pretrained: bool = True,
                dropout: float = 0.5) -> nn.Module:
    """Build `name` with a fresh `num_classes` head, regularised by `dropout`."""
    if name not in SUPPORTED:
        raise ValueError(f"model {name!r} is not wired. Supported:\n  "
                         + "\n  ".join(f"{k:<22} {v}" for k, v in SUPPORTED.items()))

    if name in HF_CHECKPOINTS:
        return HFClassifier(HF_CHECKPOINTS[name], num_classes, dropout)
    if name == "thda_resnet34":
        return THDAResNet(num_classes, pretrained, dropout)

    weights = "IMAGENET1K_V1" if pretrained else None
    net = getattr(tvm, name)(weights=weights)

    # Four head layouts across torchvision. The final Linear is replaced in place;
    # nothing is ever inserted into a Sequential that already exists.
    if hasattr(net, "fc"):                                    # resnet
        net.fc = _new_head(net.fc.in_features, num_classes, dropout)
    elif hasattr(net, "classifier"):                          # efficientnet, convnext,
        cls = net.classifier                                  # mobilenet, densenet
        if isinstance(cls, nn.Linear):                        # densenet
            net.classifier = _new_head(cls.in_features, num_classes, dropout)
        else:
            i = max(j for j, m in enumerate(cls) if isinstance(m, nn.Linear))
            if _retune_dropout_before(cls, i, dropout):       # efficientnet, mobilenet
                cls[i] = nn.Linear(cls[i].in_features, num_classes)
            else:                                             # convnext
                cls[i] = _new_head(cls[i].in_features, num_classes, dropout)
    elif hasattr(net, "heads"):                               # torchvision ViT
        net.heads.head = _new_head(net.heads.head.in_features, num_classes, dropout)
    elif hasattr(net, "head"):                                # swin
        net.head = _new_head(net.head.in_features, num_classes, dropout)
    else:
        raise ValueError(f"{name}: head layout not recognised")
    return net


# --------------------------------------------------------------------------- #
# Freezing                                                                     #
# --------------------------------------------------------------------------- #
RESNET_STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def freeze_until(net: nn.Module, stage: str) -> dict[str, int]:
    """Freeze every ResNet stage BEFORE `stage`. Returns parameter counts.

    `freeze_until(net, "layer3")` freezes conv1, bn1, layer1 and layer2, leaving
    layer3, layer4 and the head trainable.

    Measured on ResNet-18: this removes only 683,072 of 11,178,051 parameters
    (6.1%), because the weight is almost all in layer4. If the hypothesis is
    "fewer trainable parameters means less memorisation", `layer4` is the honest
    test — it removes 25%.
    """
    if stage in (None, "none", ""):
        return _param_counts(net)
    if stage not in RESNET_STAGES:
        raise ValueError(f"unknown stage {stage!r}. Known: {RESNET_STAGES}")
    if not all(hasattr(net, s) for s in RESNET_STAGES):
        raise ValueError("freeze_until only applies to ResNet-style backbones")

    for s in RESNET_STAGES[:RESNET_STAGES.index(stage)]:
        for p in getattr(net, s).parameters():
            p.requires_grad_(False)
    return _param_counts(net)


def freeze_batchnorm(net: nn.Module) -> int:
    """Put every BatchNorm2d in eval() and freeze its affine parameters.

    Two separate things, and both matter. Freezing the parameters stops gamma and
    beta adapting to the batch; `eval()` stops the running mean and variance
    moving. With batch 24 and near-duplicate slices of one patient, batch
    statistics are noisy and training-specific — letting them move is a route to
    memorisation that weight decay does not reach.

    Must be reapplied every epoch: `model.train()` puts everything back into
    training mode. `set_backbone_eval` does that inside the loop.
    """
    n = 0
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            n += 1
    return n


def set_backbone_eval(net: nn.Module) -> None:
    """Re-apply BatchNorm eval() after `model.train()`."""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _param_counts(net: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def describe(net: nn.Module, name: str) -> str:
    c = _param_counts(net)
    pct = 100 * c["trainable"] / c["total"]
    return (f"{name}: {c['total']:,} parameters, {c['trainable']:,} trainable "
            f"({pct:.2f}%), {c['frozen']:,} frozen")
