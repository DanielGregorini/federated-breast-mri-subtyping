# dataset

The processed 2-D dataset the network trains on. Built from
`raw_dataset_BreastDCEDL/`, and never edited by hand.

One folder per build. The current one is `multi_subtype_80mm/`, named after its task
and its crop window.

```
multi_subtype_80mm/
├── images/<patient>/slice_NNN.png    16,378 RGB PNGs, 224x224
├── metadata.csv                      one row per image, 27 columns
├── train.csv                         12,131 images
├── val.csv                            2,132 images
├── test.csv                           2,115 images
└── config.json                       the settings that produced all of it
```

2,063 patients from three cohorts, at a constant 0.357 mm per pixel. Each image packs
three DCE acquisition time-points into its three colour channels: red is pre-contrast,
green is early post-contrast, blue is late post-contrast.

Splits are at patient level. Every slice of a patient is in exactly one of the three.

## The columns that matter

| Column | What it is |
|---|---|
| `filename` | path under `images/`, relative to the dataset folder |
| `pid` | patient id, the unit every split is made on |
| `cohort` | `spy1`, `spy2` or `duke` |
| `split` | `train`, `val` or `test` |
| `label`, `label_name` | 0, 1, 2 and `HRposHER2neg`, `TripleNeg`, `HER2pos` |
| `slice_index`, `z_rel` | which slice of the volume this is, and where in the lesion |
| `xy_spacing`, `crop_px`, `mm_per_px` | the source spacing, the crop taken, and the resulting resolution |
| `tumor_area_mm2`, `tum_vol` | lesion size, from the mask or the bounding box |
| `roi_source` | `mask` for I-SPY1 and I-SPY2, `bbox` for Duke |

Every column is described in
[docs/DATASET_DOCUMENTATION.md](../docs/DATASET_DOCUMENTATION.md).

## How to build it

Get the raw imaging first, following
[`raw_dataset_BreastDCEDL/README.md`](../raw_dataset_BreastDCEDL/README.md).

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

The builder refuses to finish if a patient appears in two splits, carries two labels,
or has a file missing from disk.

Nothing in this folder is version controlled. It is about 1 GB and is rebuilt from the
raw release.
