"""Batch import: flatten per-label subfolders into the '<label><number>.png' layout.

Raw screenshots arrive grouped into one subfolder per label, e.g.

    data/bunnies_batch_260720/standing/Screenshot from 2026-07-20 17-55-03.png

This module renames them into the flat, label-encoded layout the rest of the
pipeline expects and that the manually curated batches already use:

    data/bunnies_batch_260720/standing01.png
    data/bunnies_batch_260720/standing02.png

The label is taken from the subfolder name (validated against `LABELS`);
numbering is per label, starting at 01, ordered by the source filename (which
for screenshots is chronological). This is a pure rename inside one batch, so
each batch keeps its own independent numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bunny_classifier.labels import LABELS


@dataclass
class Rename:
    src: Path
    dst: Path


@dataclass
class ImportPlan:
    renames: list[Rename]
    per_label: dict[str, int]  # label -> number of images


def plan_batch_import(batch_dir: Path) -> ImportPlan:
    """Plan the renames for a batch whose images sit in one subfolder per label.

    Every direct subdirectory must be named after a known label. Raises on an
    unknown subfolder name or a non-PNG file, consistent with the pipeline's
    fail-fast policy on bad input.
    """
    if not batch_dir.is_dir():
        raise ValueError(f"batch directory does not exist: {batch_dir}")

    renames: list[Rename] = []
    per_label: dict[str, int] = {}
    for label_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in LABELS:
            raise ValueError(f"subfolder {label!r} is not a known label; known: {LABELS}")
        images = sorted(p for p in label_dir.iterdir() if p.is_file())
        non_png = [p.name for p in images if p.suffix.lower() != ".png"]
        if non_png:
            raise ValueError(f"non-PNG files in {label_dir}: {non_png}")
        for number, src in enumerate(images, start=1):
            renames.append(Rename(src=src, dst=batch_dir / f"{label}{number:02d}.png"))
        per_label[label] = len(images)
    return ImportPlan(renames=renames, per_label=per_label)


def apply_batch_import(plan: ImportPlan, batch_dir: Path) -> None:
    """Execute a planned import: move each file, then remove the emptied subfolders."""
    for rename in plan.renames:
        if rename.dst.exists():
            raise ValueError(f"target already exists, aborting before any move: {rename.dst}")
    for rename in plan.renames:
        rename.src.rename(rename.dst)
    for label in plan.per_label:
        subfolder = batch_dir / label
        if subfolder.is_dir() and not any(subfolder.iterdir()):
            subfolder.rmdir()
