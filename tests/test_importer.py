from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from bunny_classifier.data.importer import apply_batch_import, plan_batch_import
from conftest import noise_image


def _make_subfolder_batch(root: Path, spec: dict[str, int]) -> Path:
    """Create data/batch/<label>/Screenshot ....png subfolders, screenshot-style names."""
    batch_dir = root / "data" / "bunnies_batch_260720"
    seed = 0
    for label, count in spec.items():
        label_dir = batch_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            noise_image(seed).save(label_dir / f"Screenshot from 2026-07-20 15-{i:02d}-00.png")
            seed += 1
    return batch_dir


def test_flattens_and_numbers_per_label(tmp_path: Path) -> None:
    batch_dir = _make_subfolder_batch(tmp_path, {"standing": 3, "sitting": 2})
    plan = plan_batch_import(batch_dir)
    apply_batch_import(plan, batch_dir)

    names = sorted(p.name for p in batch_dir.iterdir() if p.is_file())
    assert names == [
        "sitting01.png",
        "sitting02.png",
        "standing01.png",
        "standing02.png",
        "standing03.png",
    ]
    # Subfolders are removed once emptied.
    assert not (batch_dir / "standing").exists()
    assert not (batch_dir / "sitting").exists()


def test_numbering_follows_sorted_source_name(tmp_path: Path) -> None:
    batch_dir = tmp_path / "data" / "bunnies_batch_260720"
    label_dir = batch_dir / "lying"
    label_dir.mkdir(parents=True)
    noise_image(1).save(label_dir / "Screenshot from 2026-07-20 09-00-00.png")
    noise_image(2).save(label_dir / "Screenshot from 2026-07-20 08-00-00.png")

    apply_batch_import(plan_batch_import(batch_dir), batch_dir)
    # 08:00 sorts before 09:00, so it becomes lying01.
    early = noise_image(2).tobytes()
    assert Image.open(batch_dir / "lying01.png").tobytes() == early


def test_unknown_label_subfolder_is_rejected(tmp_path: Path) -> None:
    batch_dir = _make_subfolder_batch(tmp_path, {"jumping": 1})
    with pytest.raises(ValueError, match="not a known label"):
        plan_batch_import(batch_dir)


def test_existing_target_aborts_before_moving(tmp_path: Path) -> None:
    batch_dir = _make_subfolder_batch(tmp_path, {"standing": 2})
    (batch_dir / "standing01.png").write_bytes(b"pre-existing")
    plan = plan_batch_import(batch_dir)
    with pytest.raises(ValueError, match="target already exists"):
        apply_batch_import(plan, batch_dir)
    # Nothing was moved: the source subfolder is untouched.
    assert len(list((batch_dir / "standing").iterdir())) == 2
