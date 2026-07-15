"""Canonical class labels — the single source of truth (ROADMAP §5.1).

Index in LABELS = class id everywhere: manifest, training, ONNX export,
serving. Alphabetical order, frozen. Do not reorder or extend without
updating ROADMAP §5.1 first.
"""

from typing import Final

LABELS: Final[tuple[str, ...]] = (
    "back",
    "cleaning",
    "lying",
    "moving",
    "side",
    "sitting",
    "standing",
)

LABEL_TO_INDEX: Final[dict[str, int]] = {label: i for i, label in enumerate(LABELS)}
