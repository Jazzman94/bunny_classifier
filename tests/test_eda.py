from __future__ import annotations

import datetime as dt
from pathlib import Path

from bunny_classifier.data.eda import run_eda
from bunny_classifier.data.manifest import build_manifest
from conftest import make_batch


def test_eda_writes_artifacts(tmp_path: Path) -> None:
    make_batch(tmp_path, "batch_a", {"lying": 8, "moving": 8})
    manifest = tmp_path / "data" / "manifest.csv"
    build_manifest(tmp_path / "data", manifest, tmp_path, today=dt.date(2026, 7, 15))
    out_dir = tmp_path / "reports" / "eda"
    run_eda(manifest, tmp_path, out_dir)
    assert (out_dir / "class_distribution.csv").is_file()
    assert (out_dir / "image_sizes.csv").is_file()
    assert (out_dir / "samples_lying.png").is_file()
    assert (out_dir / "samples_moving.png").is_file()
