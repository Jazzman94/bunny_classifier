"""Dataset manifest: scan, validate, dedupe, split (ROADMAP §5.3).

The manifest is append-only: rows already present keep their split forever,
so the test set stays stable as new image batches arrive. Near-duplicate
images of the same class are forced into the same split (or excluded when
practically identical) so they can never leak across the train/test boundary.
Near-duplicates with *different* labels are never auto-resolved — they go to
a human review report.
"""

from __future__ import annotations

import csv
import datetime as dt
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from bunny_classifier.data.hashing import ahash, hamming, pixel_md5
from bunny_classifier.data.images import load_rgb
from bunny_classifier.labels import LABELS

FILENAME_RE = re.compile(r"^([a-z]+?)(\d+)\.png$")
MANIFEST_COLUMNS = ("path", "label", "split", "batch", "phash", "added_at")
SPLIT_EXCLUDED = "excluded"

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Classes with at least this many images get MIN_EVAL_IMAGES in val and in test.
MIN_EVAL_CLASS_SIZE = 8
MIN_EVAL_IMAGES = 2
# Same-label pairs at or below EXCLUDE_HAMMING are near-identical: keep one, exclude the rest.
# Same-label pairs at or below GROUP_HAMMING stay in but must share a split (leakage guard).
# Cross-label pairs at or below GROUP_HAMMING go to the human review report.
EXCLUDE_HAMMING = 2
GROUP_HAMMING = 5


@dataclass
class ManifestRow:
    path: str  # posix path relative to the repo root, e.g. "data/batch_x/lying01.png"
    label: str
    split: str  # train | val | test | excluded
    batch: str
    phash: str  # ahash as 16-char hex
    added_at: str  # ISO date


@dataclass
class ReviewPair:
    path_a: str
    label_a: str
    path_b: str
    label_b: str
    hamming: int


@dataclass
class ScannedImage:
    path: str
    label: str
    batch: str
    ahash_value: int
    pixel_hash: str


@dataclass
class BuildResult:
    rows: list[ManifestRow]
    review_pairs: list[ReviewPair]
    new_count: int
    excluded_count: int
    missing_paths: list[str]  # manifest rows whose file no longer exists on disk


def parse_label(filename: str) -> str:
    match = FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"filename {filename!r} does not match the '<label><number>.png' convention"
        )
    label = match.group(1)
    if label not in LABELS:
        raise ValueError(f"filename {filename!r} has unknown label {label!r}; known: {LABELS}")
    return label


def scan_images(data_dir: Path, repo_root: Path) -> list[ScannedImage]:
    """Scan data_dir batch folders for labeled PNGs; hard-error on bad names or corrupt files."""
    scanned = []
    for path in sorted(data_dir.rglob("*.png")):
        label = parse_label(path.name)
        try:
            with Image.open(path) as img:
                img.load()
        except (UnidentifiedImageError, OSError) as err:
            raise ValueError(f"corrupt or unreadable image: {path}") from err
        rgb = load_rgb(path)
        rel = path.relative_to(repo_root).as_posix()
        batch = path.parent.relative_to(data_dir).as_posix() if path.parent != data_dir else "."
        scanned.append(
            ScannedImage(
                path=rel,
                label=label,
                batch=batch,
                ahash_value=ahash(rgb),
                pixel_hash=pixel_md5(rgb),
            )
        )
    return scanned


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)


@dataclass
class _DedupeResult:
    excluded: set[str]
    groups: dict[str, list[str]]  # group root -> member paths (same-label near-dups)
    review_pairs: list[ReviewPair]


def _dedupe(scanned: list[ScannedImage]) -> _DedupeResult:
    excluded: set[str] = set()
    review_pairs: list[ReviewPair] = []

    # Pass 1 — exact pixel duplicates.
    by_pixel_hash: dict[str, list[ScannedImage]] = defaultdict(list)
    for image in scanned:
        by_pixel_hash[image.pixel_hash].append(image)
    for images in by_pixel_hash.values():
        if len(images) < 2:
            continue
        labels = {image.label for image in images}
        if len(labels) == 1:
            # Same photo, same label: keep the first, drop the rest.
            excluded.update(image.path for image in images[1:])
        else:
            # Same photo labeled differently: contradiction — exclude all, ask the human.
            excluded.update(image.path for image in images)
            for a, b in combinations(images, 2):
                if a.label != b.label:
                    review_pairs.append(ReviewPair(a.path, a.label, b.path, b.label, 0))

    # Pass 2 — perceptual near-duplicates among the survivors.
    survivors = [image for image in scanned if image.path not in excluded]
    uf = _UnionFind([image.path for image in survivors])
    for a, b in combinations(survivors, 2):
        distance = hamming(a.ahash_value, b.ahash_value)
        if distance > GROUP_HAMMING:
            continue
        if a.label != b.label:
            review_pairs.append(ReviewPair(a.path, a.label, b.path, b.label, distance))
        elif distance <= EXCLUDE_HAMMING:
            excluded.add(b.path)  # near-identical; b sorts after a because scan is sorted
        else:
            uf.union(a.path, b.path)

    groups: dict[str, list[str]] = defaultdict(list)
    for image in survivors:
        if image.path not in excluded:
            groups[uf.find(image.path)].append(image.path)
    return _DedupeResult(
        excluded=excluded,
        groups={root: sorted(members) for root, members in groups.items()},
        review_pairs=review_pairs,
    )


def _eval_target(class_size: int) -> int:
    if class_size >= MIN_EVAL_CLASS_SIZE:
        return max(MIN_EVAL_IMAGES, round(TEST_FRACTION * class_size))
    return 1 if class_size >= 3 else 0


def _assign_splits(
    scanned: list[ScannedImage],
    dedupe: _DedupeResult,
    existing: dict[str, ManifestRow],
    seed: int,
) -> dict[str, str]:
    """Assign splits to new, non-excluded images; existing rows are never touched."""
    by_path = {image.path: image for image in scanned}
    new_paths = [
        image.path
        for image in scanned
        if image.path not in existing and image.path not in dedupe.excluded
    ]

    # Group units: same-label near-dup groups move as one; inherit an existing member's split.
    path_to_group = {
        path: members for members in dedupe.groups.values() for path in members if len(members) > 1
    }
    assignments: dict[str, str] = {}
    for path in new_paths:
        for member in path_to_group.get(path, []):
            row = existing.get(member)
            if row is not None and row.split != SPLIT_EXCLUDED:
                assignments[path] = row.split

    # Per-class targets over all included images (existing + new).
    class_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in existing.values():
        if row.split != SPLIT_EXCLUDED:
            class_counts[row.label] += 1
            split_counts[row.label][row.split] += 1
    for path in new_paths:
        class_counts[by_path[path].label] += 1
    for path, split in assignments.items():
        split_counts[by_path[path].label][split] += 1

    unassigned_by_label: dict[str, list[str]] = defaultdict(list)
    for path in new_paths:
        if path not in assignments:
            unassigned_by_label[by_path[path].label].append(path)

    for label in sorted(unassigned_by_label):
        # Units = whole near-dup groups, so group members always land in the same split.
        seen: set[str] = set()
        units: list[list[str]] = []
        for path in unassigned_by_label[label]:
            members = [
                m
                for m in path_to_group.get(path, [path])
                if m not in assignments and m not in seen and m not in existing
            ]
            if members:
                units.append(members)
                seen.update(members)
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(units)

        target_eval = _eval_target(class_counts[label])
        deficits = {
            "test": max(0, target_eval - split_counts[label]["test"]),
            "val": max(0, target_eval - split_counts[label]["val"]),
        }
        for unit in units:
            if deficits["test"] >= len(unit):
                split = "test"
            elif deficits["val"] >= len(unit):
                split = "val"
            else:
                split = "train"
            if split in deficits:
                deficits[split] -= len(unit)
            for member in unit:
                assignments[member] = split
    return assignments


def build_manifest(
    data_dir: Path,
    manifest_path: Path,
    repo_root: Path,
    seed: int = 42,
    today: dt.date | None = None,
) -> BuildResult:
    """Scan data_dir and create or append to the manifest. Re-running is a no-op."""
    scanned = scan_images(data_dir, repo_root)
    existing = (
        {row.path: row for row in read_manifest(manifest_path)} if manifest_path.exists() else {}
    )
    scanned_paths = {image.path for image in scanned}
    missing = sorted(path for path in existing if path not in scanned_paths)

    dedupe = _dedupe(scanned)
    assignments = _assign_splits(scanned, dedupe, existing, seed)

    added_at = (today or dt.date.today()).isoformat()
    rows = list(existing.values())
    new_count = 0
    excluded_count = 0
    for image in scanned:
        if image.path in existing:
            continue
        new_count += 1
        if image.path in dedupe.excluded:
            split = SPLIT_EXCLUDED
            excluded_count += 1
        else:
            split = assignments[image.path]
        rows.append(
            ManifestRow(
                path=image.path,
                label=image.label,
                split=split,
                batch=image.batch,
                phash=f"{image.ahash_value:016x}",
                added_at=added_at,
            )
        )

    rows.sort(key=lambda row: row.path)
    write_manifest(manifest_path, rows)
    return BuildResult(
        rows=rows,
        review_pairs=sorted(dedupe.review_pairs, key=lambda p: (p.path_a, p.path_b)),
        new_count=new_count,
        excluded_count=excluded_count,
        missing_paths=missing,
    )


def read_manifest(manifest_path: Path) -> list[ManifestRow]:
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise ValueError(
                f"unexpected manifest columns {reader.fieldnames}; expected {MANIFEST_COLUMNS}"
            )
        return [ManifestRow(**row) for row in reader]


def write_manifest(manifest_path: Path, rows: list[ManifestRow]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([row.path, row.label, row.split, row.batch, row.phash, row.added_at])
