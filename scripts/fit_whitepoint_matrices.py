import numpy as np
import json
import os
import util
import numpy as np


def fit_whitepoint_matrices(directory="data", gamma=2.0):
    output_path = os.path.join(directory, "whitepoint_matrices.json")

    matrices = {}

    # fit matrix M to transform from have_lrgb to want_lrgb in least square sense
    for index in range(1, 28):
        print("image", index, "of", 27)

        path = os.path.join(directory, "input_with_gt_fgd/input/GT%02d.tif" % index)
        have_lrgb = util.load_image(path)

        path = os.path.join(directory, "input_training_highres/GT%02d.png" % index)
        want_srgb = util.load_image(path)

        assert have_lrgb.shape[2] == 3
        assert want_srgb.shape[2] == 3

        want_lrgb = util.srgb_to_lrgb(want_srgb, gamma)

        V = have_lrgb.reshape(-1, 3)
        W = want_lrgb.reshape(-1, 3)

        # minimize error function for M
        # i.e. find 3-by-3 matrix M such that
        # want_lrgb and have_lrgb are close
        M = (W.T @ V) @ np.linalg.inv(V.T @ V)

        # convert matrix entries to float so json.dump can handle them
        matrices[index] = [[float(x) for x in row] for row in M]

        # error function
        error = np.mean(np.square(M @ V.T - W.T))

        print("mean squared error: %f" % error)

        # Remove "continue" statement to see differences between images
        continue

        import matplotlib.pyplot as plt

        lrgb = (M @ V.T).T.reshape(want_lrgb.shape)
        lrgb = np.maximum(0, lrgb)
        srgb = util.lrgb_to_srgb(lrgb, gamma)
        srgb = np.clip(srgb, 0, 1)
        have_srgb = srgb

        difference = np.abs(have_srgb - want_srgb)

        nx = 2
        ny = 3
        plt.subplot(ny, nx, 1)
        plt.title("have")
        plt.imshow(have_srgb, vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(ny, nx, 2)
        plt.title("want")
        plt.imshow(want_srgb, vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(ny, nx, 3)
        plt.title("clip(10*|difference|, 0, 1)")
        plt.imshow(np.clip(10 * difference, 0, 1), cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        for channel, name in enumerate(["red", "green", "blue"]):
            plt.subplot(ny, nx, 4 + channel)
            plt.title(name + " channel histogram")
            bins = np.linspace(0, 1, 256)
            values = want_srgb[:, :, channel].flatten()
            plt.hist(values, bins=bins, label="want", alpha=0.5)
            values = have_srgb[:, :, channel].flatten()
            plt.hist(values, bins=bins, label="have", alpha=0.5)
            plt.legend()

        plt.show()

    with open(output_path, "w") as f:
        json.dump(matrices, f, indent=4)


if __name__ == "__main__":
    fit_whitepoint_matrices()
