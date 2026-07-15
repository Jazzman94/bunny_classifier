from __future__ import annotations

import random
from pathlib import Path

from PIL import Image


def noise_image(seed: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Deterministic random-noise image; distinct seeds give perceptually distant images."""
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3))
    return Image.frombytes("RGB", size, data)


def make_batch(repo_root: Path, batch: str, spec: dict[str, int], seed_base: int = 0) -> Path:
    """Create data/<batch>/ with `count` distinct noise images per label."""
    batch_dir = repo_root / "data" / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_base
    for label, count in spec.items():
        for i in range(1, count + 1):
            noise_image(seed).save(batch_dir / f"{label}{i:02d}.png")
            seed += 1
    return batch_dir
