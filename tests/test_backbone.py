"""Every supported backbone must build, freeze and predict correctly.

`pretrained=False` throughout: architecture and head placement are independent
of the weights, so these tests need no downloads.
"""

from __future__ import annotations

import pytest

from bunny_classifier.labels import LABELS

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from bunny_classifier.models.backbone import (  # noqa: E402
    SUPPORTED_BACKBONES,
    build_model,
    parameter_counts,
    trainable_parameters,
)


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_backbone_predicts_our_classes(backbone: str) -> None:
    model = build_model(backbone, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, len(LABELS))


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_freezing_leaves_only_the_head_trainable(backbone: str) -> None:
    model = build_model(backbone, pretrained=False, freeze_backbone=True)
    trainable, total = parameter_counts(model)

    # A Linear(features, 7) head: 7 biases plus 7 weights per input feature.
    assert (trainable - len(LABELS)) % len(LABELS) == 0
    assert trainable < total / 10, "freezing should leave a small fraction trainable"
    assert len(trainable_parameters(model)) == 2  # exactly the head's weight and bias


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_unfrozen_trains_everything(backbone: str) -> None:
    model = build_model(backbone, pretrained=False, freeze_backbone=False)
    trainable, total = parameter_counts(model)
    assert trainable == total


def test_head_replacement_actually_changed_the_output_size() -> None:
    """Guards the head-finding heuristic: a missed head would leave 1000 outputs."""
    for backbone in SUPPORTED_BACKBONES:
        model = build_model(backbone, pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        assert output.shape[1] == len(LABELS), f"{backbone} still has an ImageNet-sized head"


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_model("resnet1000", pretrained=False)


def test_gradients_reach_only_the_head() -> None:
    model = build_model("resnet18", pretrained=False)
    model(torch.zeros(1, 3, 224, 224)).sum().backward()
    with_grad = [name for name, p in model.named_parameters() if p.grad is not None]
    assert with_grad == ["fc.weight", "fc.bias"]
