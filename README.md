# Evaluate quality of foreground estimation methods

The authors of [1] provide an amazing dataset at http://alphamatting.com/datasets.php to evaluate alpha matting and foreground estimation methods.

The foreground images are stored in linear RGB TIFF files without whitepoint correction.

Although it makes more sense to do math in linear color spaces, the standard seems to be to use the sRGB color space instead.

This repository computes a whitepoint transformation matrix to transforms the linear RGB TIFF images into the sRGB color space.

Next, the `estimate_foreground_ml` method from the PyMatting library is used to estimate foreground images and the following three error metrics are computed on the region where 0 < alpha < 1:

* SAD (sum of absolute differences)
* MSE (mean squared error)
* GRAD (gradient error, see [1] for details)

## Installation and Testing

```bash
git clone https://github.com/pymatting/foreground-estimation-evaluation.git
cd foreground-estimation-evaluation
pip3 install -r requirements.txt

python3 scripts/download.py
python3 scripts/fit_whitepoint_matrices.py
python3 scripts/convert_lrgb_to_srgb.py
python3 scripts/estimate_foreground.py
python3 scripts/compute_errors.py
python3 scripts/print_results.py
```

Final output (after roughly 10 minutes):

```
| Foreground | Alpha method         | Metric | Error         |
| ---------- | -------------------- | ------ | ------------- |
| multilevel | gt_training_highres  | SAD    | 20.85 * 10^-3 |
| multilevel | gt_training_highres  | MSE    |  1.44 * 10^+3 |
| multilevel | gt_training_highres  | GRAD   |  8.89 * 10^-3 |
| naive      | gt_training_highres  | SAD    | 41.39 * 10^-3 |
| naive      | gt_training_highres  | MSE    |  5.66 * 10^+3 |
| naive      | gt_training_highres  | GRAD   | 20.44 * 10^-3 |
```

## Notes

* The PNG images from the dataset do not match the TIFF images exactly. For example, `data/input_with_gt_fgd/input/GT12.tif` has a tree in the background, while `data/input_training_highres/GT12.png` does not.
* You can delete the `continue` statement in `scripts/fit_whitepoint_matrices.py` to view the difference between the PNG images and the converted TIFF images.
* Images `data/gt_training_highres/GT25.png` to `GT27.png` load as RGB instead of (indexed) grayscale.
* Add your own foreground/alpha estimation methods in `estimate_foreground.py`.
* The dataset will be downloaded to `data/`.
* sRGB images will appear in `data/converted/image` and `data/converted/foreground`.
* Images are stored as BMP instead of PNG since encoding/decoding of PNGs is a relatively slow process and the space savings are negligible for natural images.

## References

[1] Christoph Rhemann, Carsten Rother, Jue Wang, Margrit Gelautz, Pushmeet Kohli, Pamela Rott. A Perceptually Motivated Online Benchmark for Image Matting.
Conference on Computer Vision and Pattern Recognition (CVPR), June 2009.
