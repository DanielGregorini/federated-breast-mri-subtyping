# src/pipelines/reference

The BreastDCEDL authors' preprocessing, transcribed from their published code. Its
only job is to be faithful, so do not improve anything here. Changes belong in
[`../thesis/`](../thesis/README.md).

| Rule | Value | Source function |
|---|---|---|
| slices | 4, `range(idx-2, idx+2)` | `predict_patient_images` |
| crop | 224 px fixed, centred on the ROI | `safe_crop_around_roi` |
| normalisation | min-max per slice, joint over channels | `to_rgb` / `minmax` |
| resize | none | verified on all 228 patients with n_xy > 256 |
| augmentation | none exists in their repository | |

## The one deliberate deviation

Their `Image.fromarray(float64, mode="RGB")` reinterprets the buffer as bytes rather
than converting it. On their own sample patient the correlation between what the model
receives and the MRI is 0.0114. It is corrected here and recorded as
`fromarray_fix: true` in the dataset config.

## How to use it

Name it in a `Config` in `src/dataset_config.py`, then build the dataset with
[`notebooks/02_build_dataset.ipynb`](../../../notebooks/02_build_dataset.ipynb).
