"""Human review report for cross-label near-duplicates (ROADMAP §5.3)."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from bunny_classifier.data.images import load_rgb
from bunny_classifier.data.manifest import ReviewPair

THUMBNAIL = 200
CAPTION_HEIGHT = 16


def write_review_report(pairs: list[ReviewPair], repo_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "near_dup_review.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "label_a", "path_b", "label_b", "hamming"])
        for pair in pairs:
            writer.writerow([pair.path_a, pair.label_a, pair.path_b, pair.label_b, pair.hamming])
    if pairs:
        _contact_sheet(pairs, repo_root, out_dir / "near_dup_review.png")


def _contact_sheet(pairs: list[ReviewPair], repo_root: Path, out_path: Path) -> None:
    cell = THUMBNAIL + CAPTION_HEIGHT + 4
    sheet = Image.new("RGB", (2 * (THUMBNAIL + 8), len(pairs) * cell), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, pair in enumerate(pairs):
        for j, (path, label) in enumerate(
            [(pair.path_a, pair.label_a), (pair.path_b, pair.label_b)]
        ):
            img = load_rgb(repo_root / path)
            img.thumbnail((THUMBNAIL, THUMBNAIL))
            x = j * (THUMBNAIL + 8) + 4
            y = i * cell
            sheet.paste(img, (x, y))
            caption = f"{Path(path).name} [{label}] d={pair.hamming}"
            draw.text((x, y + THUMBNAIL + 2), caption, fill=(0, 0, 0))
    sheet.save(out_path)
