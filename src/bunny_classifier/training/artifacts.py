"""Run artifacts: labels.json, confusion matrix, per-class report, prediction grid.

PIL-only rendering, matching `data/eda.py`: the project already depends on
Pillow everywhere, so plots do not justify pulling matplotlib into the
training extra. Torch-free, so these run in CI.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from bunny_classifier.data.images import load_rgb
from bunny_classifier.labels import LABELS
from bunny_classifier.training.metrics import ClassMetrics, ConfusionMatrix

CELL = 52
HEADER_W = 84
HEADER_H = 22
THUMBNAIL = 128
CAPTION_H = 26
LINE_H = 11


@dataclass(frozen=True)
class PredictionSample:
    """One val image with what the model made of it — input to the prediction grid."""

    path: str
    true_label: str
    predicted_label: str
    confidence: float


def write_labels_json(out_path: Path, labels: Sequence[str] = LABELS) -> None:
    """The label contract (§5.1) travels with every model: index in the list = class id."""
    out_path.write_text(json.dumps(list(labels), indent=2) + "\n")


def write_confusion_matrix_csv(
    matrix: ConfusionMatrix, labels: Sequence[str], out_path: Path
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def write_class_report_csv(metrics: Sequence[ClassMetrics], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for m in metrics:
            writer.writerow(
                [m.label, f"{m.precision:.4f}", f"{m.recall:.4f}", f"{m.f1:.4f}", m.support]
            )


def _centered(
    draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], fill: tuple[int, int, int]
) -> None:
    left, top, right, bottom = box
    width = draw.textlength(text)
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - LINE_H) / 2), text, fill=fill
    )


def render_confusion_matrix(matrix: ConfusionMatrix, labels: Sequence[str], out_path: Path) -> None:
    """Row-normalized heatmap; cell text is the raw count.

    Normalizing per row (per true class) is what makes the small classes
    readable — an absolute-count heatmap is dominated by whichever class has
    the most images and hides exactly the failures worth seeing.
    """
    size = len(labels)
    width = HEADER_W + size * CELL
    height = HEADER_H + size * CELL
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for col, label in enumerate(labels):
        x = HEADER_W + col * CELL
        _centered(draw, label, (x, 0, x + CELL, HEADER_H), (0, 0, 0))

    for row_index, (label, row) in enumerate(zip(labels, matrix, strict=True)):
        y = HEADER_H + row_index * CELL
        _centered(draw, label, (0, y, HEADER_W, y + CELL), (0, 0, 0))
        support = sum(row)
        for col_index, count in enumerate(row):
            x = HEADER_W + col_index * CELL
            fraction = count / support if support else 0.0
            # Correct predictions green, confusions red; intensity = share of the true class.
            hue = (34, 139, 34) if row_index == col_index else (200, 30, 30)
            shade = tuple(int(255 - (255 - channel) * fraction) for channel in hue)
            draw.rectangle([x, y, x + CELL, y + CELL], fill=shade, outline=(220, 220, 220))
            text_color = (255, 255, 255) if fraction > 0.55 else (0, 0, 0)
            _centered(draw, str(count), (x, y, x + CELL, y + CELL), text_color)

    image.save(out_path)


def render_prediction_grid(
    samples: Sequence[PredictionSample], repo_root: Path, out_path: Path, columns: int = 6
) -> None:
    """Contact sheet of predictions: green border = correct, red = wrong."""
    if not samples:
        return
    rows = (len(samples) + columns - 1) // columns
    cell_w = THUMBNAIL + 8
    cell_h = THUMBNAIL + CAPTION_H + 8
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for index, sample in enumerate(samples):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        thumb = load_rgb(repo_root / sample.path)
        thumb.thumbnail((THUMBNAIL, THUMBNAIL))
        correct = sample.true_label == sample.predicted_label
        color = (34, 139, 34) if correct else (200, 30, 30)
        draw.rectangle([x + 2, y + 2, x + cell_w - 4, y + THUMBNAIL + 6], outline=color, width=3)
        sheet.paste(thumb, (x + 4 + (THUMBNAIL - thumb.width) // 2, y + 4))
        caption = (
            sample.predicted_label
            if correct
            else f"{sample.predicted_label} != {sample.true_label}"
        )
        draw.text((x + 6, y + THUMBNAIL + 9), caption[:24], fill=color)
        draw.text((x + 6, y + THUMBNAIL + 9 + LINE_H), f"p={sample.confidence:.2f}", fill=(0, 0, 0))

    sheet.save(out_path)
