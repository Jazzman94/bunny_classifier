"""Exact and perceptual image hashing for duplicate detection."""

from __future__ import annotations

import hashlib

from PIL import Image

AHASH_SIZE = 8


def ahash(image: Image.Image, size: int = AHASH_SIZE) -> int:
    """Average hash: grayscale, downscale to size x size, threshold against the mean."""
    small = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    mean = sum(pixels) / len(pixels)
    bits = 0
    for i, value in enumerate(pixels):
        if value > mean:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def pixel_md5(image: Image.Image) -> str:
    """Content hash of decoded RGB pixels — catches re-encoded copies of the same photo."""
    rgb = image.convert("RGB")
    digest = hashlib.md5(f"{rgb.width}x{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()
