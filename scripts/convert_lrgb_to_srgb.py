import os, json, util
import numpy as np


def convert_lrgb_to_srgb(directory, gamma=2.0):
    with open(os.path.join(directory, "whitepoint_matrices.json")) as f:
        matrices = json.load(f)

    def transform(image, matrix):
        return (matrix @ image.reshape(-1, 3).T).T.reshape(image.shape)

    for index in range(1, 28):
        print("image", index, "of", 27)

        path = os.path.join(directory, "input_with_gt_fgd/input/GT%02d.tif" % index)
        lrgb = util.load_image(path)
        lrgb = transform(lrgb, np.float64(matrices[str(index)]).reshape(3, 3))
        lrgb = np.maximum(0, lrgb)
        srgb = util.lrgb_to_srgb(lrgb, gamma)
        srgb = np.clip(srgb, 0, 1)

        image = srgb

        path = os.path.join(directory, "input_with_gt_fgd/fgd/GT%02d.tif" % index)
        lrgb = util.load_image(path)
        lrgb = transform(lrgb, np.float64(matrices[str(index)]).reshape(3, 3))
        lrgb = np.maximum(0, lrgb)
        srgb = util.lrgb_to_srgb(lrgb, gamma)
        srgb = np.clip(srgb, 0, 1)

        foreground = srgb

        path = os.path.join(directory, "converted/image/GT%02d.bmp" % index)
        util.save_image(path, image)
        path = os.path.join(directory, "converted/foreground/GT%02d.bmp" % index)
        util.save_image(path, foreground)


if __name__ == "__main__":
    convert_lrgb_to_srgb(util.find_data_directory())
