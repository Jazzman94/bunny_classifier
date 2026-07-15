"""Torch-free image loading shared by training and serving (ROADMAP §5.2)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_rgb(path: Path | str) -> Image.Image:
    """Decode an image and flatten any alpha channel over white, per the input contract."""
    with Image.open(path) as img:
        img.load()
        if img.mode == "RGB":
            return img.copy()
        if "A" in img.getbands():
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return img.convert("RGB")
