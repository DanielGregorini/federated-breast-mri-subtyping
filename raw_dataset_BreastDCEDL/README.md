# raw_dataset_BreastDCEDL — the imaging release

The BreastDCEDL dataset as published. **Never written to by any code in this
repository.** Everything here is input.

Not under version control: it is roughly 35 GB. Obtain it before building the dataset.

## How to download

```bash
python raw_dataset_BreastDCEDL/download_dataset.py
```

About **22 GB** to download, **35 GB** once extracted. The script reads the file list
from the Zenodo API rather than hard-coding it, resumes an interrupted download,
verifies the md5 Zenodo publishes for each file, extracts the archives, and reports
whether the resulting layout is what the builder expects.

| Flag | Effect |
|---|---|
| `--list` | Show what is in the record and exit |
| `--all` | Download every file, not just MinCrop |
| `--no-extract` | Download the archives but leave them packed |

It downloads the **MinCrop** release, not Full. MinCrop ships tumour-centred 256x256
crops and is what this project builds on; Full is 206 GB of whole volumes and is used
nowhere.

**If it fails** — no network, a proxy, a Zenodo outage — it prints manual instructions
rather than a traceback. The short version: open
<https://zenodo.org/records/18114231>, download the four MinCrop files into this folder,
and `tar xzf` each archive in place.

## Version control

This folder is ignored by content: the imaging is far too large to track, but this
README and `download_dataset.py` are kept, so a fresh clone knows how to obtain the
data it is missing.

## What each file is

| Path | What it holds |
|---|---|
| `BreastDCEDL_metadata_min_crop.csv` | One row per patient: cohort, receptor status, the DCE phase indices to use, tumour bounding box, voxel spacing, slice thickness and the official train/validation/test assignment. This is the file `dataset_config.py::METADATA_CSV` points at. |
| `BreastDCEDL_ISPY2_min_crop/dce/` | I-SPY2 DCE volumes, one NIfTI per acquisition time-point, named `<pid>_spy2_..._dce_aqc_<n>.nii.gz` |
| `BreastDCEDL_ISPY2_min_crop/mask/` | I-SPY2 3-D binary tumour masks |
| `BreastDCEDL_ISPY1_min_crop/dce/`, `mask/` | The same for I-SPY1 |
| `BreastDCEDL_DUKE_min_crop/crop_min_dce/` | Duke DCE volumes. **Duke ships no mask** — its annotation is an expert-drawn bounding box carried in the metadata CSV columns `sraw / eraw / scol / ecol` |
| `BreastDCEDL_models.tar.gz` | The authors' released ViT weights. Present for reference; never used in this work |
| `BreastDCEDL_dataset.pdf` | The dataset paper |

## Cohorts

| Cohort | Patients used | Annotation | Character |
|---|---:|---|---|
| I-SPY2 | 982 | 3-D voxel mask | Multi-centre neoadjuvant trial |
| Duke | 914 | Bounding box only | Single-institution clinical series, 14 years |
| I-SPY1 | 167 | 3-D voxel mask | Multi-centre neoadjuvant trial |

The three are **not interchangeable**, and the differences matter enough that the
dissertation devotes an experiment to them: Duke is 64.8% HR+/HER2− against I-SPY2's
38.8%, and its tumours are roughly five times smaller by volume. See
[the dataset report](../docs/DATASET_REPORT.md).

## Citation

Fridman, N. et al. *BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients.* Scientific Data 13, 264 (2026).
<https://doi.org/10.1038/s41597-026-06589-6>
