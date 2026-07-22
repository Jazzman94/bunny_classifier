"""Classification metrics computed from a confusion matrix.

Pure Python on purpose: no torch, no numpy. The metric definitions are the part
most likely to be wrong in a subtle way, so they stay testable in CI, which
installs neither the `train` nor the `serve` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

ConfusionMatrix = list[list[int]]


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


def confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> ConfusionMatrix:
    """`matrix[true][predicted]` counts. Rows sum to each class's support."""
    if len(targets) != len(predictions):
        raise ValueError(f"targets ({len(targets)}) and predictions ({len(predictions)}) differ")
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < num_classes or not 0 <= prediction < num_classes:
            raise ValueError(f"class index out of range for {num_classes} classes")
        matrix[target][prediction] += 1
    return matrix


def _ratio(numerator: int, denominator: int) -> float:
    """Metric convention: an undefined ratio (no predictions or no support) counts as 0."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def per_class_metrics(matrix: ConfusionMatrix, labels: Sequence[str]) -> list[ClassMetrics]:
    if len(matrix) != len(labels):
        raise ValueError(f"matrix is {len(matrix)}x{len(matrix)} but got {len(labels)} labels")
    metrics = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        metrics.append(
            ClassMetrics(
                label=label,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                support=support,
            )
        )
    return metrics


def macro_f1(matrix: ConfusionMatrix, labels: Sequence[str]) -> float:
    """Unweighted mean of per-class F1 — every class counts equally regardless of size.

    This is the headline metric (ROADMAP §6 Phase 2): with an imbalanced set,
    accuracy can look good while the smallest class is never predicted at all.
    """
    metrics = per_class_metrics(matrix, labels)
    return sum(m.f1 for m in metrics) / len(metrics) if metrics else 0.0


def accuracy(matrix: ConfusionMatrix) -> float:
    total = sum(sum(row) for row in matrix)
    return _ratio(sum(matrix[i][i] for i in range(len(matrix))), total)


def majority_baseline_macro_f1(supports: Sequence[int]) -> float:
    """Macro-F1 of the trivial model that always predicts the most frequent class.

    Logged next to the real score so "0.62 macro-F1" can be read as "better than
    what", instead of being a number without a floor.
    """
    total = sum(supports)
    if not total:
        return 0.0
    majority = max(supports)
    precision = majority / total
    return _f1(precision, 1.0) / len(supports)
