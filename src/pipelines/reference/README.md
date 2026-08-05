# pipelines/reference/

The BreastDCEDL authors' preprocessing, transcribed from their code with every
rule citing its source file. Two of those files are **not referenced by the
repository README**: `DUKE/crop_spy2_spy1.ipynb` and `DUKE/duke_crop.ipynb`.

**Do not improve anything here.** Its only job is to be faithful. Every proposal
of this thesis lives in `pipelines/thesis/`.

| rule | value | source |
|---|---|---|
| slices | 4, `range(idx-2, idx+2)` — asymmetric | `predict_patient_images` |
| crop | 224 px fixed, centred on the ROI | `safe_crop_around_roi` |
| normalisation | min-max per slice, joint over channels | `to_rgb` / `minmax` |
| resize | none | verified on all 228 patients with n_xy > 256 |
| augmentation | none exists in the repository | — |

## The one deliberate deviation

Their `Image.fromarray(float64, mode="RGB")` reinterprets the buffer as bytes
rather than converting. Measured on their own sample patient, the correlation
between what the model receives and the MRI is **0.0114**. Corrected here and
recorded as `fromarray_fix: true` in the dataset config. A faithful reproduction
of a defect reproduces nothing.
