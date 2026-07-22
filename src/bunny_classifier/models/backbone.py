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

SUPPORTED_BACKBONES: tuple[str, ...] = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")

# Where each architecture keeps its final Linear layer. None = a direct attribute
# on the model, an int = the index inside the `classifier` Sequential.
_HEAD_INDEX: dict[str, int | None] = {
    "resnet18": None,
    "mobilenet_v3_small": 3,
    "efficientnet_b0": 1,
}


def _replace_head(model: nn.Module, backbone: str, num_classes: int) -> nn.Linear:
    """Swap the ImageNet classifier (1000 outputs) for a randomly initialized one.

    The casts are unavoidable: `nn.Module.__getattr__` is typed as
    `Tensor | Module`, so the static type of any submodule lookup is a union.
    """
    index = _HEAD_INDEX[backbone]
    if index is None:
        old = cast(nn.Linear, model.fc)
        head = nn.Linear(old.in_features, num_classes)
        model.fc = head
        return head
    classifier = cast(nn.Sequential, model.classifier)
    old = cast(nn.Linear, classifier[index])
    head = nn.Linear(old.in_features, num_classes)
    classifier[index] = head
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
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"unsupported backbone {backbone!r}; known: {SUPPORTED_BACKBONES}")
    weights = "DEFAULT" if pretrained else None
    model: nn.Module = models.get_model(backbone, weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    head = _replace_head(model, backbone, num_classes)
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
