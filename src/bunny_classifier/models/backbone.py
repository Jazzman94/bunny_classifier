"""Backbone factory: an ImageNet-pretrained CNN with a fresh head for our classes.

Transfer learning in one sentence: a network trained on a million photos has
already learned generic edges/textures/shapes in its convolutional stack, so
with ~680 images we only train a new final layer on top of those features
instead of learning vision from scratch.

Requires the `train` extra.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision import models

from bunny_classifier.labels import LABELS

# Anything torchvision's `get_model` can build and whose classifier ends in a
# Linear layer works here; this list is the curated set, not a technical limit.
# ImageNet top-1 and CPU cost are tabulated in docs/training.md.
SUPPORTED_BACKBONES: tuple[str, ...] = (
    # deployable on the 1 GB e2-micro (ROADMAP §6 Phase 6)
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "regnet_y_800mf",
    "resnet18",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    # too large or too slow for the VM — local experiments only
    "swin_t",
    "vit_b_16",
)


def _last_linear_name(model: nn.Module) -> str:
    """Name of the final `nn.Linear`, which for torchvision classifiers is the head.

    Found by traversal instead of a hand-maintained per-architecture map: the
    head lives at `fc`, `classifier.1`, `classifier.2`, `classifier.3`, `head`
    or `heads.head` depending on the family, and every new backbone would
    otherwise need another special case. `named_modules()` yields definition
    order, so the classifier is always last — verified for every entry in
    SUPPORTED_BACKBONES by `tests/test_backbone.py`.
    """
    name: str | None = None
    for candidate, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name = str(candidate)
    if name is None:
        raise ValueError("model has no nn.Linear layer to use as a classification head")
    return name


def _set_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    """Assign `replacement` at a dotted path, e.g. `fc`, `classifier.1`, `heads.head`."""
    *parents, attribute = path.split(".")
    parent: nn.Module = model
    for step in parents:
        parent = cast(nn.Sequential, parent)[int(step)] if step.isdigit() else getattr(parent, step)
    if attribute.isdigit():
        cast(nn.Sequential, parent)[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)


def _replace_head(model: nn.Module, num_classes: int) -> nn.Linear:
    """Swap the ImageNet classifier (1000 outputs) for a randomly initialized one."""
    path = _last_linear_name(model)
    old = cast(nn.Linear, model.get_submodule(path))
    head = nn.Linear(old.in_features, num_classes)
    _set_submodule(model, path, head)
    return head


def build_model(
    backbone: str = "resnet18",
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    num_classes: int = len(LABELS),
) -> nn.Module:
    """Build a classifier for `num_classes`, optionally with the backbone frozen.

    `freeze_backbone=True` (the Phase 2 baseline) turns off gradients for every
    pretrained parameter, so the optimizer only ever updates the new head. That
    is both much faster and much less prone to overfitting on a small dataset:
    a fully trainable ResNet18 has ~11M parameters against our ~680 images.

    `pretrained=True` requests `weights="DEFAULT"`, which resolves to the best
    weights torchvision ships for that architecture — for several models that is
    `IMAGENET1K_V2`, a better training recipe on identical architecture (+4.7
    top-1 for resnet50). Same cost, better features, no code change.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"unsupported backbone {backbone!r}; known: {SUPPORTED_BACKBONES}")
    weights = "DEFAULT" if pretrained else None
    model: nn.Module = models.get_model(backbone, weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    head = _replace_head(model, num_classes)
    for parameter in head.parameters():  # the head is always trainable
        parameter.requires_grad = True
    return model


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts — logged so a run states what it actually trained."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in trainable_parameters(model))
    return trainable, total
