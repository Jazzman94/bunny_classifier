from __future__ import annotations

import pytest

from bunny_classifier.training.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    majority_baseline_macro_f1,
    per_class_metrics,
)

LABELS_3 = ("a", "b", "c")


def test_confusion_matrix_counts_true_by_predicted() -> None:
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], 3)
    assert matrix == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_confusion_matrix_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ"):
        confusion_matrix([0, 1], [0], 3)


def test_confusion_matrix_rejects_out_of_range_class() -> None:
    with pytest.raises(ValueError, match="out of range"):
        confusion_matrix([0], [7], 3)


def test_per_class_metrics_match_hand_computation() -> None:
    # class a: 1 correct of 2 true, 2 predicted -> P 0.5, R 0.5, F1 0.5
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], 3)
    a, b, c = per_class_metrics(matrix, LABELS_3)
    assert (a.precision, a.recall, a.f1, a.support) == (0.5, 0.5, 0.5, 2)
    assert (b.precision, b.recall, b.support) == (0.5, 1.0, 1)
    assert b.f1 == pytest.approx(2 / 3)
    # class c is never predicted: recall 0, and precision is undefined -> 0 by convention
    assert (c.precision, c.recall, c.f1, c.support) == (0.0, 0.0, 0.0, 1)


def test_metrics_of_a_perfect_matrix() -> None:
    matrix = confusion_matrix([0, 1, 2], [0, 1, 2], 3)
    assert macro_f1(matrix, LABELS_3) == 1.0
    assert accuracy(matrix) == 1.0


def test_macro_f1_ignores_class_size() -> None:
    """A model that nails the big class and fails a small one must not score well."""
    targets = [0] * 90 + [1] * 10
    predictions = [0] * 100
    matrix = confusion_matrix(targets, predictions, 2)
    assert accuracy(matrix) == 0.9
    # big: P=0.9, R=1.0 -> F1 0.9474; small: never predicted -> F1 0. Macro halves it.
    assert macro_f1(matrix, ("big", "small")) == pytest.approx(0.473684, abs=1e-5)


def test_label_count_must_match_matrix() -> None:
    with pytest.raises(ValueError, match="labels"):
        per_class_metrics([[1, 0], [0, 1]], LABELS_3)


def test_majority_baseline() -> None:
    # 90/10 split: always predicting the majority gives P=0.9, R=1.0 on it, 0 elsewhere.
    assert majority_baseline_macro_f1([90, 10]) == pytest.approx(0.947368 / 2, abs=1e-5)
    assert majority_baseline_macro_f1([]) == 0.0
    assert majority_baseline_macro_f1([0, 0]) == 0.0


def test_empty_matrix_is_zero_not_a_crash() -> None:
    assert accuracy([[0, 0], [0, 0]]) == 0.0
    assert macro_f1([[0, 0], [0, 0]], ("a", "b")) == 0.0
