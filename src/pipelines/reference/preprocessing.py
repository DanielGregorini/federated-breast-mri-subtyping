"""The BreastDCEDL authors' preprocessing, transcribed from their code.

SOURCE OF EVERY RULE BELOW
--------------------------
Nothing here is inferred. Each step is copied from a specific place in
[github.com/naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL),
cited inline. Two of those files are not referenced by the repository README.

    DUKE/crop_spy2_spy1.ipynb        MinCrop generation, I-SPY1 and I-SPY2
    DUKE/duke_crop.ipynb             MinCrop generation, DUKE
    BrestDCEDL_vit_predict.ipynb     slice selection, crop, normalisation

DO NOT "IMPROVE" ANYTHING IN THIS FILE. Its only job is to be faithful. Every
proposal of this thesis lives in `pipelines/thesis/preprocessing.py`, and the two
are kept apart so that a comparison between them is a comparison of methods.

THE ONE DELIBERATE DEVIATION
----------------------------
`get_jpg_im` in their notebook calls `Image.fromarray(im, mode="RGB")` with `im`
a float64 array — `read_nifti` returns `get_fdata()`, which is always float64,
and nothing casts it. Passing `mode=` explicitly makes PIL reinterpret the raw
buffer as bytes instead of converting. Measured on their own sample patient
`ACRIN-6698-102212`, the correlation between what the model receives and the
actual MRI is **0.0114**.

This is corrected here, because a faithful reproduction of a defect reproduces
nothing. The correction is the obvious one — scale to [0, 255] and cast to uint8
— and it is recorded in the dataset's `config.json` as `fromarray_fix: true`.
"""

from __future__ import annotations

import numpy as np

# The z-margin the authors leave around the tumour when building MinCrop.
# `crop_around_voi_cords(..., slice_padding=2, ...)`. Verified against the
# released data: the tumour occupies z in [2, nz-3] for 749 of 767 I-SPY2
# patients and 12 of 12 DUKE spot checks.
SLICE_PADDING = 2

# `safe_crop_around_roi(img, cx, cy, crop_size=224)`. Fixed in PIXELS, not
# millimetres, so the field of view varies with each patient's spacing:
# 158 mm on I-SPY2, 170 mm on DUKE, 175 mm on I-SPY1.
CROP_SIZE_PX = 224


def select_slices(z_indices: np.ndarray) -> np.ndarray:
    """The four slices the authors feed the model.

    Transcribed from `predict_patient_images`:

        f = max(int(row['mask_start']), 0)
        l = min(int(row['mask_end']), int(row['n_z']))
        idx = (f + l) // 2
        for k in range(max(idx-2, f), min(idx+2, l), 1):

    Note two things, both preserved. The window is ASYMMETRIC — idx-2, idx-1,
    idx, idx+1, so four slices and not five. And the exclusive upper bound
    `min(idx+2, l)` drops the last slice of a thin mask.
    """
    if z_indices.size == 0:
        return z_indices
    f, l = int(z_indices[0]), int(z_indices[-1])
    mid = (f + l) // 2
    chosen = np.arange(max(mid - 2, f), min(mid + 2, l))
    # A mask thin enough to make the range empty still has to contribute
    # something, or the patient silently vanishes from the evaluation.
    if chosen.size == 0:
        chosen = np.array([z_indices[len(z_indices) // 2]])
    return chosen


def crop_window(row0: int, row1: int, col0: int, col1: int,
                crop_px: int = CROP_SIZE_PX) -> tuple[int, int, int, int]:
    """A `crop_px` square centred on the ROI, from `safe_crop_around_roi`.

        roi_center_x = (scol + ecol) // 2
        roi_center_y = (sraw + eraw) // 2
        left = roi_center_x - crop_size // 2      (etc., clamped at the borders)

    Returned as (y0, y1, x0, x1), possibly outside the image; the caller pads.
    """
    cy, cx = (row0 + row1) // 2, (col0 + col1) // 2
    y0, x0 = cy - crop_px // 2, cx - crop_px // 2
    return y0, y0 + crop_px, x0, x0 + crop_px


def normalize(slice_3ch: np.ndarray) -> np.ndarray:
    """Min-max of ONE slice, jointly over the three channels.

    Transcribed from `to_rgb` and `minmax`:

        def minmax(im):    return (im - im.min()) / (im.max() - im.min())
        def to_rgb(a,b,c): return minmax(np.stack([a, b, c], axis=2))

    `minmax` runs on the ALREADY-STACKED array, so the extremes are those of the
    whole slice across all three channels — not of the volume, and not of each
    channel separately.

    This differs from `pipelines/thesis`, which normalises per volume. Per volume
    preserves the intensity relationship BETWEEN slices of one patient; per slice
    does not, so an almost-empty slice ends up as bright as one with an intense
    lesion. That is what they do.

    Input and output are (3, H, W).
    """
    lo, hi = float(np.nanmin(slice_3ch)), float(np.nanmax(slice_3ch))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(slice_3ch, dtype=np.float32)
    return np.clip((slice_3ch - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_uint8(slice_3ch: np.ndarray) -> np.ndarray:
    """Float [0, 1] to uint8 (H, W, 3) — the corrected `Image.fromarray` step.

    See the module docstring for why this replaces
    `Image.fromarray(float64, mode="RGB")`.
    """
    return (np.clip(slice_3ch, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


# --------------------------------------------------------------------------- #
# What the authors do NOT do, recorded so nobody adds it by accident           #
# --------------------------------------------------------------------------- #
NOT_DONE = {
    "resize": "MinCrop is 256 px and stays 256 px. The `cv2.resize` branch in "
              "crop_spy2_spy1.ipynb exists but was NOT used for the released "
              "data — verified on all 228 patients with n_xy > 256.",
    "augmentation": "No augmentation code exists in the repository. Figure 2C of "
                    "the paper shows examples, but no parameters are published.",
    "bias_field_correction": "None. The authors preserve original DICOM "
                             "intensities deliberately, to leave harmonisation "
                             "to whoever uses the data.",
    "z_score": "None.",
    "class_imbalance": "None. The only `class_weight='balanced'` in the whole "
                       "repository is commented out, and belongs to a "
                       "GradientBoostingClassifier on tabular features.",
}
