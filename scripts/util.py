from PIL import Image
import numpy as np
import os
from read_tiff import read_tiff


def blend(F, B, a):
    if len(a.shape) < len(F.shape):
        a = a[:, :, np.newaxis]
    return a * F + (1 - a) * B


def div_or_zero(a, b):
    result = np.zeros_like(a)
    is_non_zero = b != 0
    result[is_non_zero] = a[is_non_zero] / b[is_non_zero]
    return result


def pad(image, color, r):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)
    h, w, d = image.shape
    result = np.empty((h + 2 * r, w + 2 * r, d), image.dtype)
    for c in range(d):
        result[:, :, c] = color[c]
    result[r : r + h, r : r + w] = image
    return result


def load_image(path):
    if path.lower().rsplit(".", 1)[-1] in {"tif", "tiff"}:
        return read_tiff(path) / 65535.0

    return np.array(Image.open(path)) / 255.0


def save_image(path, image, make_directory=True):
    assert image.dtype in [np.uint8, np.float32, np.float64]

    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if make_directory:
        directory, _ = os.path.split(path)
        if len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

    image = Image.fromarray(image)
    image.save(path)


def lrgb_to_srgb(image, gamma):
    threshold = 0.0031308
    mask = image <= threshold
    linear_term = 12.92 * image
    exponential_term = 1.055 * image ** (1.0 / gamma) - 0.055
    return linear_term * mask + (1 - mask) * exponential_term


def srgb_to_lrgb(image, gamma):
    threshold = 0.04045
    mask = image <= threshold
    linear_term = (1.0 / 12.92) * image
    exponential_term = ((image + 0.055) / 1.055) ** gamma
    return linear_term * mask + (1 - mask) * exponential_term
