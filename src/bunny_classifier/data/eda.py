"""EDA artifacts from the manifest: class/split distribution, image sizes, sample grids.

PIL-only on purpose — usable without the train extra.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image

from bunny_classifier.data.images import load_rgb
from bunny_classifier.data.manifest import SPLIT_EXCLUDED, ManifestRow, read_manifest

THUMBNAIL = 128
GRID_SAMPLES = 8


def run_eda(manifest_path: Path, repo_root: Path, out_dir: Path) -> None:
    rows = [row for row in read_manifest(manifest_path) if row.split != SPLIT_EXCLUDED]
    out_dir.mkdir(parents=True, exist_ok=True)
    _class_distribution(rows, out_dir / "class_distribution.csv")
    _size_stats(rows, repo_root, out_dir / "image_sizes.csv")
    by_label: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    for label, label_rows in sorted(by_label.items()):
        _sample_grid(label_rows[:GRID_SAMPLES], repo_root, out_dir / f"samples_{label}.png")


def _class_distribution(rows: list[ManifestRow], out_path: Path) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row.label][row.split] += 1
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "total", "train", "val", "test"])
        for label in sorted(counts):
            splits = counts[label]
            writer.writerow(
                [label, sum(splits.values()), splits["train"], splits["val"], splits["test"]]
            )


def _size_stats(rows: list[ManifestRow], repo_root: Path, out_path: Path) -> None:
    widths: dict[str, list[int]] = defaultdict(list)
    heights: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        with Image.open(repo_root / row.path) as img:
            widths[row.label].append(img.width)
            heights[row.label].append(img.height)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "min_w", "median_w", "max_w", "min_h", "median_h", "max_h"])
        for label in sorted(widths):
            writer.writerow(
                [
                    label,
                    min(widths[label]),
                    int(statistics.median(widths[label])),
                    max(widths[label]),
                    min(heights[label]),
                    int(statistics.median(heights[label])),
                    max(heights[label]),
                ]
            )


def _sample_grid(rows: list[ManifestRow], repo_root: Path, out_path: Path) -> None:
    thumbs = []
    for row in rows:
        img = load_rgb(repo_root / row.path)
        img.thumbnail((THUMBNAIL, THUMBNAIL))
        thumbs.append(img)
    if not thumbs:
        return
    grid = Image.new("RGB", (len(thumbs) * (THUMBNAIL + 4), THUMBNAIL + 4), (255, 255, 255))
    for i, thumb in enumerate(thumbs):
        grid.paste(thumb, (i * (THUMBNAIL + 4) + 2, 2))
    grid.save(out_path)
