import json, util
import numpy as np
from collections import defaultdict


def print_results(directory):
    with open(f"{directory}/errors.json") as f:
        errors = json.load(f)

    groups = defaultdict(list)

    exponents = {
        "SAD": -3,
        "MSE": 3,
        "GRAD": -3,
    }

    print("")
    print("| Foreground | Alpha method         | Metric | Error         |")
    print("| ---------- | -------------------- | ------ | ------------- |")

    for fg_method, d0 in errors.items():
        for alpha_method, d1 in d0.items():
            for error_name, d2 in d1.items():
                scale = 10 ** exponents[error_name]

                error = scale * np.mean(list(d2.values()))

                result = f"| {fg_method:10} | {alpha_method:20} | {error_name:6} | `{error:5.2f} * 10^{exponents[error_name]:+d}` |"

                print(result)


if __name__ == "__main__":
    print_results(util.find_data_directory())
