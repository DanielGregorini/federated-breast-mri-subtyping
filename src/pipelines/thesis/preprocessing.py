"""This thesis's preprocessing. Every deviation from the authors, and its reason.

Kept in a separate module from `pipelines/reference` on purpose: nothing here may
leak into the reproduction, and nothing there constrains what may be proposed
here. A comparison between the two folders is then a comparison of methods.

THE FOUR DEVIATIONS
-------------------
1. Physical crop window (80 mm) instead of a fixed 224 px.
2. Eight slices spread across the lesion instead of four consecutive central ones.
3. Min-max over the whole VOLUME instead of per slice.
4. Optional per-channel percentile clipping (`chanclip`), from the authors' own
   later benchmark.

Each is argued below with the measurement that motivated it — and, where the
measurement went against the proposal, that is stated too.
"""

from __future__ import annotations

import numpy as np

# Default physical window. See `why_physical_window()` for the derivation.
CROP_MM = 80.0

# Slices kept per patient, and how much of each end of the lesion is discarded
# before sampling.
N_SLICES = 8
TRIM_FRACTION = 0.15


def select_slices(z_indices: np.ndarray, n: int = N_SLICES,
                  trim: float = TRIM_FRACTION) -> np.ndarray:
    """Trim the extremes of the lesion, then sample `n` evenly spaced slices.

    Two choices, both deliberate:

    * The trim is PROPORTIONAL, not absolute. A tumour spanning 60 slices loses 9
      at each end; one spanning 8 loses 1. An absolute rule ("drop 3 slices")
      would erase half of a small tumour.
    * The sampling is `linspace` over what remains, not the N central slices.
      Neighbouring slices 1–2 mm apart are practically the same image, so twenty
      of them give twenty copies of one gradient. Spreading covers the lesion end
      to end for the same number of images.

    A patient with fewer than `n` slices keeps all of them: better five images
    than a lost patient. `n_slices_tumor` in the CSV records the original count so
    those cases can be filtered later without reprocessing.
    """
    total = len(z_indices)
    if total == 0:
        return z_indices
    cut = int(np.floor(total * trim))
    core = z_indices[cut:total - cut] if total - 2 * cut >= max(n, 3) else z_indices
    if len(core) <= n:
        return core
    positions = np.linspace(0, len(core) - 1, n)
    return core[np.unique(np.round(positions).astype(int))]


def crop_window(row0: int, row1: int, col0: int, col1: int, xy_spacing: float,
                crop_mm: float = CROP_MM) -> tuple[tuple[int, int, int, int], int]:
    """A square window of `crop_mm` MILLIMETRES centred on the lesion.

    Returns ((y0, y1, x0, x1), side_in_pixels). The window may fall outside the
    image; the caller pads with zeros. Padding beats shrinking: shrinking would
    change the scale only for peripheral tumours, and constant scale is the whole
    point.
    """
    side = max(int(round(crop_mm / xy_spacing)), 8)
    cy, cx = (row0 + row1) // 2, (col0 + col1) // 2
    y0, x0 = cy - side // 2, cx - side // 2
    return (y0, y0 + side, x0, x0 + side), side


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max over the whole 4D volume — all phases, all slices.

    Preserves the intensity relationship BETWEEN slices of one patient and
    BETWEEN phases, which is the enhancement kinetics — the biological signal.
    Per-slice normalisation destroys it: an almost-empty slice ends up as bright
    as one with an intense lesion.

    Input (3, Z, Y, X), output the same.
    """
    lo, hi = float(np.nanmin(volume)), float(np.nanmax(volume))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_channel_clip(volume: np.ndarray, q: float = 0.98) -> np.ndarray:
    """Clip each phase's UPPER tail at its own quantile, then scale.

    The winning strategy in the authors' own later benchmark
    (arXiv:2510.13897), which compared seven normalisations on one model:

        per-channel upper clipping q0.98   AUC 0.744   <- their best
        per-channel upper clipping q0.95   AUC 0.734
        global upper clipping      q0.99   AUC 0.726
        per-slice min-max                  AUC 0.723
        GLOBAL min-max                     AUC 0.700   <- their worst

    MEASURED HERE, AND IT LOST. On the 3-class subtype task with two seeds,
    chanclip scored 0.5830 against 0.6078 for global min-max — the opposite
    direction, though inside the noise floor. Their result was for binary HER2 on
    their preprocessing; it does not transfer. Kept available, not default.
    """
    out = np.empty_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        channel = volume[c]
        hi = float(np.nanquantile(channel, q))
        lo = float(np.nanmin(channel))
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            out[c] = 0.0
            continue
        out[c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# The reasoning, kept with the code                                            #
# --------------------------------------------------------------------------- #
def why_physical_window() -> str:
    return """
A margin of "15-20% of the box" makes the lesion fill the frame identically in
every patient — and by doing so ERASES tumour size, which is worth macro-AUC
0.58-0.68 on its own here and ranks as the most important predictor in the
authors' own feature-importance analysis. An 80 mm frame keeps it: a 60 mm tumour
occupies three times what a 20 mm one does, as in reality.

Fixed in millimetres rather than pixels because xy_spacing ranges from 0.31 to
1.41 mm/px across patients. After resizing to 224 every image sits at the same
0.357 mm per pixel. Without that, the degree of magnification would itself
identify the cohort: DUKE tumours have a median of 29 mm against 61 mm in I-SPY2.

Measured effect: the resampling factor became 1.96x (DUKE) / 1.91x (I-SPY2) /
2.20x (I-SPY1), against roughly 4x versus 2x for a proportional crop.

Why 80 mm specifically: the smallest window containing the whole tumour for the
MEDIAN patient of all three cohorts (I-SPY1 64.8 mm, I-SPY2 61.2 mm, DUKE
29.2 mm) plus about 7 mm of peritumoral tissue per side — the margin reported as
optimal in the peritumoral literature (4-6 mm). It fully contains the lesion in
81% of patients.

HONEST CAVEAT: on the same 99 I-SPY2 test patients, this preprocessing scored
0.5837 +/- 0.011 against 0.6201 +/- 0.024 for the older pipeline. The difference
is inside the 0.067 noise floor, so the verdict is "no difference detected" — but
it is certainly not the improvement that was expected. What it did buy is a
validation-to-test gap of +0.015 against +0.073.
"""


def why_spread_slices() -> str:
    return """
The old behaviour kept EVERY slice containing tumour: a mean of 38 per patient,
a maximum of 150, and 21% of them carrying fewer than 100 tumour pixels. Two
consequences, both harmful: the per-patient mean that produces the final
prediction is diluted by near-empty slices, and a patient with 150 slices
contributes 150 gradient samples against another's 8.

Eight spread slices reduce 63,460 tumour slices to 16,378 images (25.8%) while
covering the lesion end to end.

The trim is geometric rather than area-based on purpose: it works identically
with a segmentation mask or with a bounding box, so it introduces no difference
of treatment between cohorts. In this project a cohort-specific difference is the
most expensive error there is — the source probe reaches macro-AUC 0.9978.
"""
