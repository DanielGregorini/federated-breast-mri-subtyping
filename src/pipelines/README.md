# pipelines — the two preprocessing rule sets

Each pipeline supplies three decisions — which slices to keep, how to crop, and how to
normalise — and `core/dataset_builder.py` does everything else. A comparison between
these two folders is therefore a comparison of methods, with nothing else changing.

| Folder | What it is |
|---|---|
| `thesis/` | What this dissertation proposes: eight evenly spaced slices with 15% of the lesion trimmed from each end, an 80 mm physical crop window, and min-max normalisation over the whole 4-D volume. |
| `reference/` | A faithful reproduction of the rules the BreastDCEDL authors published: four consecutive central slices, a fixed 224-pixel crop, and min-max normalisation per slice. |

## Why they are kept apart

Nothing in `thesis/` may leak into the reproduction, and nothing in `reference/`
constrains what may be proposed. `dataset_config.py::Config` refuses to combine a
pipeline with a task it was not defined for, so a dataset cannot silently be built half
one way and half the other.

The separation is also what makes the normalisation comparison in the preprocessing
document possible: both arms of that figure call the real functions, one from each
folder, rather than a reimplementation of either.

## The three decisions, side by side

| | `thesis/` | `reference/` |
|---|---|---|
| slices per patient | 8, evenly spaced, 15% trimmed each end | 4, `range(idx-2, idx+2)` |
| crop | 80 mm physical window, side = 80 / spacing px | 224 px fixed |
| effective field of view | constant 80 mm | 158–175 mm, varying by cohort |
| final resolution | constant 0.357 mm/px | varies per patient |
| normalisation | min-max over the whole volume | min-max per slice |

Every one of those choices is argued, with the measurement that decided it, in
[the preprocessing document](../../docs/PREPROCESSING_AND_IMAGING.md).
