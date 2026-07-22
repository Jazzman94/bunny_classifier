from __future__ import annotations

from pathlib import Path

import pytest

from bunny_classifier.training.config import TrainConfig


def test_defaults_describe_the_phase_2_baseline() -> None:
    config = TrainConfig()
    assert config.backbone == "resnet18"
    assert config.freeze_backbone is True
    assert config.class_weighting is True
    assert config.eval_test is False  # the test split stays untouched by default


def test_replace_ignores_none_so_unset_cli_flags_do_not_clobber() -> None:
    config = TrainConfig(epochs=30).replace(epochs=None, batch_size=8)
    assert config.epochs == 30
    assert config.batch_size == 8


def test_replace_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig().replace(learning_rte=0.1)


@pytest.mark.parametrize(
    "overrides",
    [{"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0.0}, {"device": "tpu"}],
)
def test_invalid_values_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)  # type: ignore[arg-type]


def test_as_params_is_flat_strings_for_mlflow() -> None:
    params = TrainConfig().as_params()
    assert params["backbone"] == "resnet18"
    assert all(isinstance(value, str) for value in params.values())


def test_from_yaml_overlays_defaults(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="requires the `train` extra")
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 3\nbackbone: efficientnet_b0\n")
    config = TrainConfig.from_yaml(path)
    assert (config.epochs, config.backbone) == (3, "efficientnet_b0")
    assert config.batch_size == TrainConfig().batch_size


def test_from_yaml_rejects_a_non_mapping(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="requires the `train` extra")
    path = tmp_path / "config.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        TrainConfig.from_yaml(path)
