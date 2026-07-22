"""End-to-end smoke test of the training loop on a synthetic dataset.

Runs one epoch of a randomly initialized ResNet18 over noise images: it proves
the wiring (manifest -> dataset -> loop -> metrics -> MLflow artifacts) holds
together, not that the model learns anything.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from bunny_classifier.data.manifest import build_manifest
from bunny_classifier.labels import LABELS
from conftest import make_batch

pytest.importorskip("torch", reason="requires the `train` extra")
pytest.importorskip("mlflow", reason="requires the `train` extra")


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    make_batch(tmp_path, "batch_a", dict.fromkeys(LABELS, 8))
    build_manifest(
        data_dir=tmp_path / "data",
        manifest_path=tmp_path / "data" / "manifest.csv",
        repo_root=tmp_path,
        today=dt.date(2026, 7, 22),
    )
    return tmp_path


@pytest.fixture
def config():  # type: ignore[no-untyped-def]
    from bunny_classifier.training.config import TrainConfig

    return TrainConfig(
        epochs=1,
        batch_size=4,
        num_workers=0,
        device="cpu",
        pretrained=False,  # no network access in CI
        experiment="test",
    )


def test_training_run_produces_metrics_and_artifacts(repo_root: Path, config) -> None:  # type: ignore[no-untyped-def]
    from bunny_classifier.training.train import train

    summary = train(config, repo_root)

    assert summary.best_epoch == 1
    assert 0.0 <= summary.val_macro_f1 <= 1.0
    assert summary.test_macro_f1 is None  # test split untouched unless asked for

    artifacts = next((repo_root / "mlartifacts").rglob("labels.json")).parent
    for name in (
        "labels.json",
        "config.json",
        "model.pt",
        "val_confusion_matrix.csv",
        "val_confusion_matrix.png",
        "val_class_report.csv",
        "val_predictions.png",
    ):
        assert (artifacts / name).exists(), f"missing artifact {name}"
    assert (repo_root / "runs").exists()  # TensorBoard event files


def test_eval_test_scores_the_held_out_split(repo_root: Path, config) -> None:  # type: ignore[no-untyped-def]
    from bunny_classifier.training.train import train

    summary = train(config.replace(eval_test=True), repo_root)
    assert summary.test_macro_f1 is not None


def test_class_weights_are_inverse_frequency(repo_root: Path) -> None:
    import torch

    from bunny_classifier.data.dataset import BunnyDataset
    from bunny_classifier.training.train import class_weights

    dataset = BunnyDataset(repo_root / "data" / "manifest.csv", "train", repo_root)
    weights = class_weights(dataset, torch.device("cpu"))

    assert len(weights) == len(LABELS)
    # Balanced synthetic classes -> all weights equal 1.0; the sum is the class count.
    assert float(weights.sum()) == pytest.approx(len(LABELS), abs=0.5)
