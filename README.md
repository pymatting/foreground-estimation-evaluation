# Evaluate quality of foreground estimation methods

```bash
python3 scripts/download.py
python3 scripts/fit_whitepoint_matrices.py
python3 scripts/convert_lrgb_to_srgb.py
# TODO compute foreground
# TODO measure error
```

Notes:

* The PNG images from the dataset do not match the TIFF images exactly. For example, `data/input_with_gt_fgd/input/GT12.tif` has a tree in the background, while `data/input_training_highres/GT12.png` does not.
* You can delete the `continue` statement in `scripts/fit_whitepoint_matrices.py` to view the difference between the PNG images and the converted TIFF images.
