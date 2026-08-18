# src/pipelines

The two preprocessing rule sets. Each supplies three decisions, and
`core/dataset_builder.py` does everything else, so switching between them changes the
method and nothing else.

| Folder | What it is |
|---|---|
| [`thesis/`](thesis/README.md) | What this dissertation proposes. Eight evenly spaced slices, an 80 mm physical crop window, and min-max normalisation over the whole volume. |
| [`reference/`](reference/README.md) | The rules the BreastDCEDL authors published. Four consecutive central slices, a fixed 224-pixel crop, and min-max normalisation per slice. |

## The three decisions

| | `thesis/` | `reference/` |
|---|---|---|
| slices per patient | 8, evenly spaced, 15% trimmed each end | 4, `range(idx-2, idx+2)` |
| crop | 80 mm physical window, side = 80 / spacing px | 224 px fixed |
| field of view | constant 80 mm | 158 to 175 mm, varying by cohort |
| final resolution | constant 0.357 mm/px | varies per patient |
| normalisation | min-max over the whole volume | min-max per slice |

## How to use it

Pick one in `src/dataset_config.py` when you define a `Config`, then build the
dataset. `Config` refuses to combine a pipeline with a task it was not defined for, so
a dataset cannot be built half one way and half the other.

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Every choice in the table is argued in
[docs/PREPROCESSING_AND_IMAGING.md](../../docs/PREPROCESSING_AND_IMAGING.md).
