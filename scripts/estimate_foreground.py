import util
import numpy as np


def estimate_foreground_naive(image, alpha):
    # Not the smartest of foreground estimation methods,
    # but a good starting point to implement your own.
    return image


def estimate_foreground_multilevel(image, alpha):
    import pymatting

    return pymatting.estimate_foreground_ml(
        image,
        alpha,
        gradient_weight=0.1,
        regularization=5e-3,
        n_small_iterations=10,
        # n_big_iterations=4 decreases error by a few percent, but will increase runtime
        n_big_iterations=2,
    )


def estimate_foregrounds(directory):

    from pymatting import estimate_foreground_ml

    # Add your own method here (method name, estimate_foreground function)
    fg_methods = [
        ("multilevel", estimate_foreground_multilevel),
        ("naive", estimate_foreground_naive),
    ]

    alpha_methods = [
        "gt_training_highres",
    ]

    print("Running foreground estimation")
    for index in range(1, 28):
        name = "GT%02d" % index

        path = f"{directory}/converted/image/{name}.bmp"

        image = util.load_image(path)

        for alpha_method in alpha_methods:
            path = f"{directory}/{alpha_method}/{name}.png"

            alpha = util.load_image(path, "gray")

            for fg_method, estimate_foreground in fg_methods:

                foreground = estimate_foreground(image, alpha)

                print("Image", name, "with", fg_method, "method")

                path = f"{directory}/fg_methods/{fg_method}/{alpha_method}/{name}.bmp"

                util.save_image(path, foreground)


if __name__ == "__main__":
    estimate_foregrounds(util.find_data_directory())
