from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from PIL import Image

from bunny_classifier.data.manifest import build_manifest
from bunny_classifier.labels import LABEL_TO_INDEX
from conftest import make_batch

torch = pytest.importorskip("torch", reason="requires the `train` extra")


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    batch = make_batch(tmp_path, "batch_a", {"lying": 8, "sitting": 8})
    rgba = Image.new("RGBA", (64, 64), (200, 30, 30, 128))  # semi-transparent
    rgba.save(batch / "lying90.png")
    build_manifest(
        data_dir=tmp_path / "data",
        manifest_path=tmp_path / "data" / "manifest.csv",
        repo_root=tmp_path,
        today=dt.date(2026, 7, 15),
    )
    return tmp_path


def test_dataset_yields_contract_tensors(repo_root: Path) -> None:
    from bunny_classifier.data.dataset import BunnyDataset

    dataset = BunnyDataset(repo_root / "data" / "manifest.csv", "train", repo_root)
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label in set(LABEL_TO_INDEX.values())


def test_dataset_flattens_rgba(repo_root: Path) -> None:
    from bunny_classifier.data.dataset import BunnyDataset, eval_transforms

    dataset = BunnyDataset(repo_root / "data" / "manifest.csv", "train", repo_root)
    for i, row in enumerate(dataset.rows):
        if row.path.endswith("lying90.png"):
            image, label = dataset[i]
            assert image.shape == (3, 224, 224)
            assert label == LABEL_TO_INDEX["lying"]
            break
    else:  # RGBA file may have landed in another split — load it directly instead
        from bunny_classifier.data.images import load_rgb

        tensor = eval_transforms()(load_rgb(repo_root / "data" / "batch_a" / "lying90.png"))
        assert tensor.shape == (3, 224, 224)


def test_eval_transforms_deterministic(repo_root: Path) -> None:
    from bunny_classifier.data.dataset import BunnyDataset

    dataset = BunnyDataset(repo_root / "data" / "manifest.csv", "train", repo_root)
    first, _ = dataset[0]
    second, _ = dataset[0]
    assert torch.equal(first, second)


def test_unknown_split_rejected(repo_root: Path) -> None:
    from bunny_classifier.data.dataset import BunnyDataset

    with pytest.raises(ValueError, match="no rows"):
        BunnyDataset(repo_root / "data" / "manifest.csv", "bogus", repo_root)
