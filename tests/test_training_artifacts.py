from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from bunny_classifier.labels import LABELS
from bunny_classifier.training.artifacts import (
    PredictionSample,
    render_confusion_matrix,
    render_prediction_grid,
    write_class_report_csv,
    write_confusion_matrix_csv,
    write_labels_json,
)
from bunny_classifier.training.metrics import confusion_matrix, per_class_metrics
from conftest import noise_image


def test_labels_json_is_the_index_ordered_contract(tmp_path: Path) -> None:
    out = tmp_path / "labels.json"
    write_labels_json(out)
    assert json.loads(out.read_text()) == list(LABELS)


def test_confusion_matrix_csv_is_true_by_predicted(tmp_path: Path) -> None:
    matrix = confusion_matrix([0, 1], [0, 0], len(LABELS))
    out = tmp_path / "cm.csv"
    write_confusion_matrix_csv(matrix, LABELS, out)
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["true\\predicted", *LABELS]
    assert rows[1][0] == LABELS[0] and rows[1][1] == "1"
    assert rows[2][0] == LABELS[1] and rows[2][1] == "1"  # a 'cleaning' predicted as 'back'


def test_class_report_csv_has_one_row_per_class(tmp_path: Path) -> None:
    matrix = confusion_matrix([0, 1], [0, 0], len(LABELS))
    out = tmp_path / "report.csv"
    write_class_report_csv(per_class_metrics(matrix, LABELS), out)
    rows = list(csv.DictReader(out.open()))
    assert [row["label"] for row in rows] == list(LABELS)
    assert rows[0]["recall"] == "1.0000"


def test_confusion_matrix_renders(tmp_path: Path) -> None:
    matrix = confusion_matrix([0, 1, 2], [0, 1, 1], len(LABELS))
    out = tmp_path / "cm.png"
    render_confusion_matrix(matrix, LABELS, out)
    with Image.open(out) as image:
        assert image.size[0] > len(LABELS) * 50
        assert image.mode == "RGB"


def test_prediction_grid_renders(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    noise_image(1).save(tmp_path / "data" / "lying01.png")
    samples = [
        PredictionSample("data/lying01.png", "lying", "lying", 0.91),
        PredictionSample("data/lying01.png", "lying", "side", 0.42),
    ]
    out = tmp_path / "grid.png"
    render_prediction_grid(samples, tmp_path, out)
    with Image.open(out) as image:
        assert image.mode == "RGB"


def test_prediction_grid_skips_an_empty_sample_list(tmp_path: Path) -> None:
    out = tmp_path / "grid.png"
    render_prediction_grid([], tmp_path, out)
    assert not out.exists()
