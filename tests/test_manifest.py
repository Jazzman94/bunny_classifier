from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from bunny_classifier.data.hashing import ahash, hamming
from bunny_classifier.data.manifest import (
    SPLIT_EXCLUDED,
    BuildResult,
    build_manifest,
    parse_label,
    read_manifest,
)
from conftest import make_batch, noise_image

DAY1 = dt.date(2026, 7, 15)
DAY2 = dt.date(2026, 8, 1)


def build(repo_root: Path, today: dt.date = DAY1) -> BuildResult:
    return build_manifest(
        data_dir=repo_root / "data",
        manifest_path=repo_root / "data" / "manifest.csv",
        repo_root=repo_root,
        seed=42,
        today=today,
    )


class TestParseLabel:
    def test_valid(self) -> None:
        assert parse_label("standing83.png") == "standing"

    def test_unknown_label(self) -> None:
        with pytest.raises(ValueError, match="unknown label"):
            parse_label("flying01.png")

    def test_malformed_name(self) -> None:
        for bad in ("lying.png", "Lying01.png", "lying01.jpg", "01lying.png"):
            with pytest.raises(ValueError, match=r"convention|unknown"):
                parse_label(bad)


class TestFreshBuild:
    def test_split_sizes_and_minimums(self, tmp_path: Path) -> None:
        make_batch(tmp_path, "batch_a", {"lying": 20, "moving": 8})
        result = build(tmp_path)
        counts = Counter((row.label, row.split) for row in result.rows)
        assert counts[("lying", "test")] == 3  # round(0.15 * 20)
        assert counts[("lying", "val")] == 3
        assert counts[("lying", "train")] == 14
        assert counts[("moving", "test")] == 2  # minimum for a class of 8
        assert counts[("moving", "val")] == 2
        assert counts[("moving", "train")] == 4

    def test_rerun_is_noop(self, tmp_path: Path) -> None:
        make_batch(tmp_path, "batch_a", {"lying": 10, "side": 10})
        build(tmp_path)
        manifest = tmp_path / "data" / "manifest.csv"
        first = manifest.read_bytes()
        build(tmp_path, today=DAY2)  # different day must not touch existing rows
        assert manifest.read_bytes() == first


class TestAppendOnly:
    def test_existing_rows_never_change(self, tmp_path: Path) -> None:
        make_batch(tmp_path, "batch_a", {"lying": 20}, seed_base=0)
        build(tmp_path)
        original = {row.path: row for row in read_manifest(tmp_path / "data" / "manifest.csv")}

        make_batch(tmp_path, "batch_b", {"lying": 10}, seed_base=1000)
        result = build(tmp_path, today=DAY2)
        after = {row.path: row for row in result.rows}
        for path, row in original.items():
            assert after[path].split == row.split
            assert after[path].added_at == row.added_at
        new_rows = [row for row in result.rows if row.path not in original]
        assert len(new_rows) == 10
        assert all(row.added_at == DAY2.isoformat() for row in new_rows)

    def test_new_images_fill_eval_deficits(self, tmp_path: Path) -> None:
        make_batch(tmp_path, "batch_a", {"lying": 20}, seed_base=0)
        build(tmp_path)
        make_batch(tmp_path, "batch_b", {"lying": 20}, seed_base=1000)
        result = build(tmp_path, today=DAY2)
        counts = Counter(row.split for row in result.rows)
        assert counts["test"] == 6  # round(0.15 * 40)
        assert counts["val"] == 6


class TestDedupe:
    def test_exact_duplicate_excluded(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 10})
        noise_image(0).save(batch / "lying99.png")  # same pixels as lying01
        result = build(tmp_path)
        by_path = {row.path: row for row in result.rows}
        assert by_path["data/batch_a/lying01.png"].split != SPLIT_EXCLUDED
        assert by_path["data/batch_a/lying99.png"].split == SPLIT_EXCLUDED

    def test_cross_label_exact_duplicate_excludes_both_and_reports(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 10, "side": 10}, seed_base=0)
        noise_image(0).save(batch / "side99.png")  # same pixels as lying01
        result = build(tmp_path)
        by_path = {row.path: row for row in result.rows}
        assert by_path["data/batch_a/lying01.png"].split == SPLIT_EXCLUDED
        assert by_path["data/batch_a/side99.png"].split == SPLIT_EXCLUDED
        assert any(
            {pair.label_a, pair.label_b} == {"lying", "side"} and pair.hamming == 0
            for pair in result.review_pairs
        )

    def test_cross_label_near_duplicate_reported_not_excluded(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 10, "side": 10}, seed_base=0)
        near = noise_image(0).copy()  # perceptually identical to lying01, one pixel off
        near.putpixel((0, 0), (0, 0, 0))
        near.save(batch / "side99.png")
        result = build(tmp_path)
        by_path = {row.path: row for row in result.rows}
        assert by_path["data/batch_a/lying01.png"].split != SPLIT_EXCLUDED
        assert by_path["data/batch_a/side99.png"].split != SPLIT_EXCLUDED
        assert any(
            {pair.path_a, pair.path_b} == {"data/batch_a/lying01.png", "data/batch_a/side99.png"}
            for pair in result.review_pairs
        )

    def test_same_label_near_duplicates_share_split(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 10}, seed_base=0)
        base = Image.new("RGB", (64, 64), (100, 100, 100))
        variant = base.copy()
        variant.paste(Image.new("RGB", (12, 12), (255, 255, 255)), (0, 0))
        distance = hamming(ahash(base), ahash(variant))
        assert 2 < distance <= 5, f"test setup: expected group-range distance, got {distance}"
        base.save(batch / "lying88.png")
        variant.save(batch / "lying89.png")
        result = build(tmp_path)
        by_path = {row.path: row for row in result.rows}
        assert (
            by_path["data/batch_a/lying88.png"].split == by_path["data/batch_a/lying89.png"].split
        )
        assert by_path["data/batch_a/lying88.png"].split != SPLIT_EXCLUDED


class TestRobustness:
    def test_corrupt_image_errors(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 3})
        (batch / "lying99.png").write_bytes(b"this is not a png")
        with pytest.raises(ValueError, match="corrupt"):
            build(tmp_path)

    def test_deleted_file_warns_but_keeps_row(self, tmp_path: Path) -> None:
        batch = make_batch(tmp_path, "batch_a", {"lying": 10})
        build(tmp_path)
        (batch / "lying01.png").unlink()
        result = build(tmp_path, today=DAY2)
        assert result.missing_paths == ["data/batch_a/lying01.png"]
        assert any(row.path == "data/batch_a/lying01.png" for row in result.rows)
