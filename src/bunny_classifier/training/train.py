"""Baseline training loop with MLflow + TensorBoard tracking (ROADMAP §6 Phase 2).

Requires the `train` extra. The loop is deliberately plain — no scheduler, no
early stopping, no fine-tuning of the backbone. Those arrive in Phase 3, and
the point of a baseline is to be the boring number every later run must beat.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from bunny_classifier.data.dataset import BunnyDataset, train_transforms
from bunny_classifier.labels import LABELS
from bunny_classifier.models.backbone import build_model, parameter_counts, trainable_parameters
from bunny_classifier.training.artifacts import (
    PredictionSample,
    render_confusion_matrix,
    render_prediction_grid,
    write_class_report_csv,
    write_confusion_matrix_csv,
    write_labels_json,
)
from bunny_classifier.training.config import TrainConfig
from bunny_classifier.training.metrics import (
    ConfusionMatrix,
    accuracy,
    confusion_matrix,
    macro_f1,
    majority_baseline_macro_f1,
    per_class_metrics,
)

GRID_ERRORS = 12
GRID_TOTAL = 18


@dataclass
class EvalResult:
    loss: float
    targets: list[int]
    predictions: list[int]
    confidences: list[float]

    @property
    def matrix(self) -> ConfusionMatrix:
        return confusion_matrix(self.targets, self.predictions, len(LABELS))


@dataclass
class RunSummary:
    run_id: str
    best_epoch: int
    val_macro_f1: float
    val_accuracy: float
    majority_baseline_macro_f1: float
    test_macro_f1: float | None


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no CUDA device is available")
    return torch.device(spec)


def class_weights(dataset: BunnyDataset, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights: `N / (K * n_c)`, so every class contributes equally.

    Without this, the loss is dominated by the biggest classes and the model can
    minimize it by simply under-predicting `back` (the smallest one).
    """
    counts = Counter(row.label for row in dataset.rows)
    total = sum(counts.values())
    weights = [total / (len(LABELS) * counts[label]) if counts[label] else 0.0 for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * len(targets)
        seen += len(targets)
    return running / seen if seen else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()
    running = 0.0
    seen = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        running += float(criterion(outputs, targets).item()) * len(targets)
        seen += len(targets)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        targets_all.extend(int(value) for value in targets.cpu())
        predictions_all.extend(int(value) for value in predictions.cpu())
        confidences_all.extend(float(value) for value in confidences.cpu())
    return EvalResult(
        loss=running / seen if seen else 0.0,
        targets=targets_all,
        predictions=predictions_all,
        confidences=confidences_all,
    )


def _prediction_samples(dataset: BunnyDataset, result: EvalResult) -> list[PredictionSample]:
    """Errors first, then correct predictions — a grid of successes teaches nothing."""
    errors: list[PredictionSample] = []
    correct: list[PredictionSample] = []
    for row, target, prediction, confidence in zip(
        dataset.rows, result.targets, result.predictions, result.confidences, strict=True
    ):
        sample = PredictionSample(
            path=row.path,
            true_label=LABELS[target],
            predicted_label=LABELS[prediction],
            confidence=confidence,
        )
        (correct if target == prediction else errors).append(sample)
    selected = errors[:GRID_ERRORS]
    return selected + correct[: GRID_TOTAL - len(selected)]


def _write_artifacts(
    out_dir: Path,
    config: TrainConfig,
    result: EvalResult,
    dataset: BunnyDataset,
    repo_root: Path,
    prefix: str,
) -> None:
    matrix = result.matrix
    metrics = per_class_metrics(matrix, LABELS)
    write_confusion_matrix_csv(matrix, LABELS, out_dir / f"{prefix}_confusion_matrix.csv")
    render_confusion_matrix(matrix, LABELS, out_dir / f"{prefix}_confusion_matrix.png")
    write_class_report_csv(metrics, out_dir / f"{prefix}_class_report.csv")
    render_prediction_grid(
        _prediction_samples(dataset, result), repo_root, out_dir / f"{prefix}_predictions.png"
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")


def tracking_uri(repo_root: Path) -> str:
    """Local SQLite store by default; `MLFLOW_TRACKING_URI` wins if set.

    Not the classic `./mlruns` file store: MLflow 3 put that backend into
    maintenance mode and refuses it outright, and the model registry needed in
    Phase 3 has always required a database backend anyway.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{repo_root / 'mlflow.db'}"


def track_uri_hint(repo_root: Path) -> str:
    """The exact command that opens the UI on the store this run wrote to."""
    return (
        f"mlflow ui --backend-store-uri {tracking_uri(repo_root)}\n"
        f"tensorboard --logdir {repo_root / 'runs'}"
    )


def _experiment_id(name: str, artifact_root: Path) -> str:
    """Resolve (creating on first use) the experiment, pinning artifacts inside the repo.

    `set_experiment` would place artifacts relative to the current working
    directory, which silently scatters them when a run is launched elsewhere.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(mlflow.create_experiment(name, artifact_location=artifact_root.resolve().as_uri()))


def train(config: TrainConfig, repo_root: Path, manifest_path: Path | None = None) -> RunSummary:
    manifest = manifest_path or repo_root / "data" / "manifest.csv"
    set_seeds(config.seed)
    device = resolve_device(config.device)

    train_dataset = BunnyDataset(manifest, "train", repo_root, transform=train_transforms())
    val_dataset = BunnyDataset(manifest, "val", repo_root)  # defaults to eval_transforms()
    generator = torch.Generator().manual_seed(config.seed)
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=False,
    )
    # Never shuffled: the prediction grid pairs results with dataset.rows by position.
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    model = build_model(
        config.backbone, pretrained=config.pretrained, freeze_backbone=config.freeze_backbone
    ).to(device)
    weights = class_weights(train_dataset, device) if config.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    run_name = config.run_name or f"{config.backbone}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    trainable, total = parameter_counts(model)
    val_counts = Counter(row.label for row in val_dataset.rows)
    baseline = majority_baseline_macro_f1([val_counts[label] for label in LABELS])

    mlflow.set_tracking_uri(tracking_uri(repo_root))
    experiment_id = _experiment_id(config.experiment, repo_root / "mlartifacts")
    writer = SummaryWriter(log_dir=str(repo_root / "runs" / run_name))

    best_state: dict[str, torch.Tensor] = {}
    best_f1 = -1.0
    best_epoch = 0
    best_result: EvalResult | None = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params(config.as_params())
        # Distinct key from config's `device`: MLflow rejects re-logging a param with
        # a different value, and "auto" resolves to something concrete here.
        mlflow.log_params(
            {
                "resolved_device": str(device),
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "trainable_params": trainable,
                "total_params": total,
            }
        )
        mlflow.log_metric("majority_baseline_macro_f1", baseline)

        for epoch in range(1, config.epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_result = evaluate(model, val_loader, criterion, device)
            val_matrix = val_result.matrix
            val_f1 = macro_f1(val_matrix, LABELS)
            val_accuracy = accuracy(val_matrix)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_result.loss,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            writer.add_scalars("loss", {"train": train_loss, "val": val_result.loss}, epoch)
            writer.add_scalar("val/macro_f1", val_f1, epoch)
            writer.add_scalar("val/accuracy", val_accuracy, epoch)
            print(
                f"epoch {epoch:>3}/{config.epochs}  train_loss {train_loss:.4f}  "
                f"val_loss {val_result.loss:.4f}  val_macro_f1 {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1, best_epoch, best_result = val_f1, epoch, val_result
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        assert best_result is not None  # epochs >= 1 is enforced by TrainConfig
        test_f1: float | None = None
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_labels_json(out_dir / "labels.json")
            _write_artifacts(out_dir, config, best_result, val_dataset, repo_root, "val")
            torch.save(
                {"state_dict": best_state, "backbone": config.backbone, "labels": list(LABELS)},
                out_dir / "model.pt",
            )

            if config.eval_test:
                model.load_state_dict(best_state)
                test_dataset = BunnyDataset(manifest, "test", repo_root)
                test_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
                    test_dataset, batch_size=config.batch_size, shuffle=False
                )
                test_result = evaluate(model, test_loader, criterion, device)
                test_f1 = macro_f1(test_result.matrix, LABELS)
                mlflow.log_metrics(
                    {"test_macro_f1": test_f1, "test_accuracy": accuracy(test_result.matrix)}
                )
                _write_artifacts(out_dir, config, test_result, test_dataset, repo_root, "test")

            mlflow.log_artifacts(str(out_dir))

        mlflow.log_metrics(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "best_val_loss": best_result.loss,
            }
        )
        writer.close()
        return RunSummary(
            run_id=run.info.run_id,
            best_epoch=best_epoch,
            val_macro_f1=best_f1,
            val_accuracy=accuracy(best_result.matrix),
            majority_baseline_macro_f1=baseline,
            test_macro_f1=test_f1,
        )
