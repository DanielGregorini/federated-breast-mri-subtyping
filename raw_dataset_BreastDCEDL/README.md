# raw_dataset_BreastDCEDL

The BreastDCEDL imaging release as published. Everything here is input, and no code in
this repository writes to it.

Not under version control. It is about 35 GB on disk, so a fresh clone gets this
README and the download script and has to fetch the imaging itself.

## What each file is

| Path | What it holds |
|---|---|
| `BreastDCEDL_metadata_min_crop.csv` | One row per patient. Cohort, receptor status, the DCE phase indices to use, tumour bounding box, voxel spacing, slice thickness, and the official train/validation/test assignment. This is what `src/dataset_config.py::METADATA_CSV` points at. |
| `BreastDCEDL_ISPY2_min_crop/dce/` | I-SPY2 DCE volumes, one NIfTI per acquisition time-point |
| `BreastDCEDL_ISPY2_min_crop/mask/` | I-SPY2 3-D binary tumour masks |
| `BreastDCEDL_ISPY1_min_crop/dce/`, `mask/` | the same for I-SPY1 |
| `BreastDCEDL_DUKE_min_crop/crop_min_dce/` | Duke DCE volumes. Duke ships no mask, its annotation is a bounding box in the metadata columns `sraw`, `eraw`, `scol`, `ecol` |
| `BreastDCEDL_models.tar.gz` | the authors' released ViT weights, not used in this work |
| `BreastDCEDL_dataset.pdf` | the dataset paper |

## How to download it

Run from the repository root:

```bash
python raw_dataset_BreastDCEDL/download_dataset.py
```

About 22 GB to download and 35 GB once extracted. The script reads the file list from
the Zenodo API, resumes an interrupted download, verifies the md5 Zenodo publishes for
each file, extracts the archives, and reports whether the layout is what the builder
expects.

| Flag | Effect |
|---|---|
| `--list` | show what is in the record and exit |
| `--all` | download every file, not just MinCrop |
| `--no-extract` | download the archives but leave them packed |

It downloads the MinCrop release rather than Full. MinCrop ships tumour-centred
256x256 crops, which is what this project builds on. Full is 206 GB of whole volumes
and is used nowhere here.

If the download fails, the script prints manual instructions instead of a traceback.
The short version: open <https://zenodo.org/records/18114231>, download the four
MinCrop files into this folder, and `tar xzf` each archive in place.

## What to do next

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Turns these volumes into the 2-D dataset under
[`dataset/`](../dataset/README.md).
