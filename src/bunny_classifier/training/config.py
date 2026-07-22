"""Training configuration.

Every run is reproducible from `config + manifest + seed` (ROADMAP §6 Phase 2),
so the whole configuration lives in one dataclass that gets logged verbatim as
MLflow params. Torch-free, so it is importable and testable without the
`train` extra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENT = "bunny-classifier"


@dataclass(frozen=True)
class TrainConfig:
    backbone: str = "resnet18"
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # Phase 2 is the transfer-learning baseline: the pretrained feature extractor
    # is frozen and only the new head is trained. Phase 3 unfreezes.
    freeze_backbone: bool = True
    pretrained: bool = True
    # Weighted cross-entropy compensates the class imbalance instead of discarding
    # images from the larger classes (ROADMAP §3).
    class_weighting: bool = True
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"  # auto | cpu | cuda
    experiment: str = DEFAULT_EXPERIMENT
    run_name: str | None = None
    # The test split stays untouched until a model is actually being shipped;
    # every intermediate decision is made on val (see docs/training.md).
    eval_test: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"device must be auto|cpu|cuda, got {self.device!r}")

    def replace(self, **overrides: Any) -> TrainConfig:
        """Return a copy with `overrides` applied; None values are ignored.

        Lets the CLI pass every flag through unconditionally — an unset flag is
        None and therefore leaves the config (or the YAML file) value in place.
        """
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        applied = {key: value for key, value in overrides.items() if value is not None}
        return TrainConfig(**{**asdict(self), **applied})

    def as_params(self) -> dict[str, str]:
        """Flat string mapping for MLflow params."""
        return {key: str(value) for key, value in asdict(self).items()}

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        import yaml  # lazy: pyyaml ships with the train extra, not the base install

        with path.open() as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls().replace(**loaded)
