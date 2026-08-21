# raw_dataset_BreastDCEDL

The BreastDCEDL imaging release as published. Input only, never written to.

About 35 GB, so it is not under version control. A fresh clone gets this README and
the download script and fetches the imaging itself.

## What is in it

| Path | What it holds |
|---|---|
| `BreastDCEDL_metadata_min_crop.csv` | one row per patient: cohort, receptor status, DCE phase indices, tumour bounding box, voxel spacing, and the authors' official split |
| `BreastDCEDL_ISPY2_min_crop/`, `BreastDCEDL_ISPY1_min_crop/` | `dce/` volumes and `mask/` tumour masks |
| `BreastDCEDL_DUKE_min_crop/crop_min_dce/` | Duke volumes. Duke ships no mask, its annotation is the bounding box in the metadata |
| `BreastDCEDL_models.tar.gz` | the authors' ViT weights, unused here |
| `BreastDCEDL_dataset.pdf` | the dataset paper |

## How to get it

```bash
python raw_dataset_BreastDCEDL/download_dataset.py
```

22 GB to download, 35 GB extracted. Reads the file list from the Zenodo API, resumes
an interrupted transfer, checks each md5, and extracts in place. `--list` shows the
record without downloading, `--no-extract` leaves the archives packed.

It fetches the MinCrop release, which ships tumour-centred 256x256 crops. Full is
206 GB of whole volumes and is used nowhere here.

If it fails it prints manual instructions: open
<https://zenodo.org/records/18114231>, download the four MinCrop files into this
folder, and `tar xzf` each one.

## Next

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Turns these volumes into [`dataset/`](../dataset/README.md).
