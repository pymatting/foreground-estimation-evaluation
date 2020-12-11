import os, json, util
import numpy as np


def compute_errors(directory):
    errors_path = f"{directory}/errors.json"

    try:
        with open(errors_path) as f:
            errors = json.load(f)
    except FileNotFoundError:
        errors = {}

    for index in range(1, 28):
        name = "GT%02d" % index

        path = f"{directory}/converted/foreground/{name}.bmp"

        true_foreground = util.load_image(path)

        path = f"{directory}/gt_training_highres/{name}.png"

        alpha = util.load_image(path, "gray")

        is_unknown = np.logical_and(alpha > 0, alpha < 1)

        fg_methods = os.listdir(f"{directory}/fg_methods")

        for fg_method in sorted(fg_methods):
            alpha_methods = os.listdir(f"{directory}/fg_methods/{fg_method}")

            for alpha_method in sorted(alpha_methods):
                path = f"{directory}/fg_methods/{fg_method}/{alpha_method}/{name}.bmp"

                estimated_foreground = util.load_image(path)

                for error_name, calculate_error in [
                    ("SAD", util.calculate_sad_error),
                    ("MSE", util.calculate_mse_error),
                    ("GRAD", util.calculate_gradient_error),
                ]:
                    error = calculate_error(
                        estimated_foreground, true_foreground, is_unknown, alpha
                    )

                    print(name, fg_method, alpha_method, error_name, error)

                    keys = [fg_method, alpha_method, error_name]

                    d = errors
                    for key in keys:
                        d = d.setdefault(key, {})

                    d[name] = error

                # Remove "continue" statement to look at images if you want to
                continue

                import matplotlib.pyplot as plt

                for i, img in enumerate(
                    [estimated_foreground, true_foreground, alpha, is_unknown,]
                ):
                    plt.subplot(2, 2, 1 + i)
                    plt.imshow(img, cmap="gray")
                    plt.axis("off")
                plt.show()

    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=4)


if __name__ == "__main__":
    compute_errors(util.find_data_directory())
