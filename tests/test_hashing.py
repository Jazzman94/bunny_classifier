from PIL import Image

from bunny_classifier.data.hashing import ahash, hamming, pixel_md5
from conftest import noise_image


def test_identical_images_have_zero_distance() -> None:
    a, b = noise_image(1), noise_image(1)
    assert hamming(ahash(a), ahash(b)) == 0
    assert pixel_md5(a) == pixel_md5(b)


def test_distinct_noise_images_are_far_apart() -> None:
    a, b = noise_image(1), noise_image(2)
    assert hamming(ahash(a), ahash(b)) > 5
    assert pixel_md5(a) != pixel_md5(b)


def test_pixel_md5_distinguishes_sizes() -> None:
    small = Image.new("RGB", (32, 32), (255, 255, 255))
    large = Image.new("RGB", (64, 64), (255, 255, 255))
    assert pixel_md5(small) != pixel_md5(large)
