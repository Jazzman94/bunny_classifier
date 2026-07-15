"""PyTorch Dataset over the manifest, with the §5.2 preprocessing contract.

Requires the `train` extra (torch, torchvision). Everything torch-free lives
in the sibling modules so the serving image never imports this.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from bunny_classifier.data.images import load_rgb
from bunny_classifier.data.manifest import SPLIT_EXCLUDED, ManifestRow, read_manifest
from bunny_classifier.labels import LABEL_TO_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SHORTER_SIDE = 256
CROP_SIZE = 224


def eval_transforms() -> v2.Compose:
    """Deterministic pipeline — must stay in lockstep with serving preprocessing (§5.2)."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(RESIZE_SHORTER_SIDE),
            v2.CenterCrop(CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transforms() -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(CROP_SIZE, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomRotation(10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BunnyDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images of one split, loaded via the manifest; labels come from LABEL_TO_INDEX."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        repo_root: Path,
        transform: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        if split == SPLIT_EXCLUDED:
            raise ValueError("refusing to build a dataset from excluded images")
        self.rows: list[ManifestRow] = [
            row for row in read_manifest(manifest_path) if row.split == split
        ]
        if not self.rows:
            raise ValueError(f"manifest {manifest_path} has no rows for split {split!r}")
        self.repo_root = repo_root
        self.transform = transform if transform is not None else eval_transforms()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = load_rgb(self.repo_root / row.path)
        return self.transform(image), LABEL_TO_INDEX[row.label]
