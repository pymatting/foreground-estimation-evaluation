from PIL import Image
import numpy as np
import os
from read_tiff import read_tiff
from scipy.ndimage import gaussian_filter, correlate


def find_data_directory():
    if os.path.split(os.getcwd())[-1] == "scripts":
        return "../data"
    return "data"


def calculate_sad_error(image, true_image, mask, weights):
    difference = weights[:, :, np.newaxis] * np.abs(image - true_image)
    return np.sum(difference[mask])


def calculate_mse_error(image, true_image, mask, weights):
    difference = weights[:, :, np.newaxis] * np.square(image - true_image)
    return np.mean(difference[mask])


def calculate_gradient(image, sigma):
    assert len(image.shape) == 2
    r = int(3 * sigma)

    x = np.linspace(-r, r, 2 * r + 1)
    g = np.exp(-0.5 * np.square(x) / (sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
    dg = -1.0 / sigma ** 2 * x * g

    g /= np.linalg.norm(g)
    dg /= np.linalg.norm(dg)

    dx = correlate(correlate(image, dg.reshape(1, -1)), g.reshape(-1, 1))
    dy = correlate(correlate(image, dg.reshape(-1, 1)), g.reshape(1, -1))

    return np.sqrt(dx * dx + dy * dy)


def calculate_gradient_error(image, true_image, mask, weights, sigma=1.4):
    result = 0.0

    for channel in range(image.shape[2]):
        g1 = calculate_gradient(image[:, :, channel], sigma)
        g2 = calculate_gradient(true_image[:, :, channel], sigma)

        difference = weights * np.square(g1 - g2)

        result += np.sum(difference[mask])

    return result


def load_image(path, mode=None):
    if path.lower().rsplit(".", 1)[-1] in {"tif", "tiff"}:
        image = read_tiff(path) / 65535.0
    else:
        image = np.array(Image.open(path))

        if image.dtype == np.uint8:
            image = image / 255.0
        else:
            image = image / 65535.0

    if mode == "gray" and len(image.shape) == 3:
        image = image[:, :, 0]

    return image


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
