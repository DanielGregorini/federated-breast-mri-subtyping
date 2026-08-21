# Federated Learning for Breast Cancer Molecular Subtype Classification

Master's dissertation, Daniel Mahl Gregorini.

A federated learning system that trains a breast cancer molecular subtype classifier
across several hospitals without any medical image leaving the institution that
produced it.

It runs on NVIDIA FLARE 2.8.0 in production mode. Real PKI, one operating-system
process per hospital, mutual TLS, and jobs submitted through the admin API. Not the
simulator.

| | |
|---|---|
| **Task** | 3-class molecular subtype from DCE-MRI: HR+/HER2-, Triple Negative, HER2+ |
| **Dataset** | BreastDCEDL (Fridman et al., 2026), Duke + I-SPY1 + I-SPY2, 2,063 patients, 16,378 images |
| **Model** | ResNet-18, ImageNet-pretrained, 11,178,051 parameters |
| **Experiments** | 1 centralised baseline and 12 federated runs, FedAvg and FedProx, 2 to 4 hospitals |

BreastDCEDL: [paper](https://doi.org/10.1038/s41597-026-06589-6) ·
[download](https://zenodo.org/records/18114231) ·
[code](https://github.com/naomifridman/BreastDCEDL)

---

## The pipeline

```
raw_dataset_BreastDCEDL/        3-D NIfTI volumes, three DCE phases per patient
        |
        |   notebooks/02_build_dataset.ipynb
        v
dataset/                        16,378 RGB PNGs, 224x224, constant 0.357 mm/px
        |                       R = pre-contrast, G = early post, B = late post
        |
        +---> notebooks/04_train_centralized.ipynb -> results/classifier/
        |     one machine, all patients pooled
        |
        +---> notebooks/07_federated_setup.ipynb
              deployment/data/   per-hospital splits, by patient, never by slice
                    |
                    v
              NVIDIA FLARE  ->  results/federated/
              server and 2 to 4 hospital clients, 30 rounds x 1 local epoch
```

Every slice of a patient stays in one split and one hospital.

<p align="center">
  <img src="docs/images/preprocessing_figures/fig_p5_flowchart.png" width="360">
</p>

The same slice at every preprocessing step, produced by the functions the dataset
builder calls:

![Preprocessing walkthrough](docs/images/preprocessing_figures/fig_p1_walkthrough.png)

Final training images, one per cohort and class, each a real file from `dataset/`:

![Example training images](docs/images/report_figures/fig3_examples_cohort_class.png)

---

## Repository layout

| Folder | What it holds |
|---|---|
| [`raw_dataset_BreastDCEDL/`](raw_dataset_BreastDCEDL/README.md) | The BreastDCEDL imaging release. Input only. Its README explains how to obtain it. |
| [`dataset/`](dataset/README.md) | The processed 2-D dataset the network trains on. PNG slices, split manifests and the build configuration. |
| [`src/`](src/README.md) | The dataset builder, the preprocessing and the shared trainer. Everything about turning volumes into a trained model on one machine. |
| [`deployment/`](deployment/README.md) | The running system. The federated code in `code/`, the PKI startup kits, the generated jobs, the per-hospital data and the per-participant logs. |
| [`results/`](results/README.md) | The dissertation record in `thesis/`, plus `classifier/` and `federated/` where your own runs land. |
| [`docs/`](docs/README.md) | Every document and every figure. |
| [`notebooks/`](notebooks/README.md) | The whole pipeline as notebooks, numbered in the order they run. Each carries its own logic. |

The repository root holds only `README.md` and `requirements.txt`.

---

## How to run it

Six steps, in order.

### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Get the raw imaging

Run from the repository root. About 22 GB to download, 35 GB once extracted.

```bash
python raw_dataset_BreastDCEDL/download_dataset.py
```

Writes into [`raw_dataset_BreastDCEDL/`](raw_dataset_BreastDCEDL/README.md). Add
`--list` to see what is in the record without downloading anything.

### 2. Build the processed dataset

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Turns the NIfTI volumes into 16,378 RGB PNG slices under
[`dataset/`](dataset/README.md). The builder refuses to finish if a patient appears in
two splits, carries two labels, or has a file missing from disk.

### 3. Train the centralised baseline

```bash
jupyter notebook notebooks/04_train_centralized.ipynb
```

One machine, all training patients pooled. Writes a numbered run folder into
[`results/classifier/`](results/README.md). The scripted equivalent is
`python deployment/code/scripts/run_centralized.py --seed 42`.

### 4. Split the patients between hospitals

```bash
jupyter notebook notebooks/07_federated_setup.ipynb
```

Writes the global test set and the six partitions into
[`deployment/data/`](deployment/README.md).

### 5. Set up the federation

```bash
bash deployment/code/scripts/provision.sh
python deployment/code/scripts/generate_jobs.py
```

The first writes one PKI startup kit per participant. The second writes the twelve
job folders and copies them where the admin console looks for them.

### 6. Run a federated experiment

Start the server and the hospitals from their startup kits in a terminal, then submit
the job:

```bash
python deployment/code/scripts/run_experiment.py test10
```

Collect and summarise what finished:

```bash
python deployment/code/scripts/collect_results.py
python deployment/code/scripts/build_final_summary.py
```

Results land in [`results/federated/`](results/README.md). Starting the participants,
moving a hospital to its own machine and troubleshooting are all in
[`deployment/README.md`](deployment/README.md) and
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Documentation

Everything is in [`docs/`](docs/README.md). Three to start with.

**[DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md)** describes how the
dataset is organised, column by column, with example images.

**[PREPROCESSING_AND_IMAGING.md](docs/PREPROCESSING_AND_IMAGING.md)** describes what a
DCE-MRI study is, why the channel assignment follows from it, and every preprocessing
step with the measurement that decided it.

**[RESULTS.md](docs/RESULTS.md)** reports centralised against federated and compares
both with the published literature.

---

## Licence and attribution

The code is released under the [MIT licence](LICENSE).

The imaging is not ours. It comes from the BreastDCEDL MinCrop release, which is
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and is
redistributed here under that licence with attribution. The images are not the
released volumes. They are 80 mm physical crops, normalised over the whole volume,
resized to a constant 0.357 mm per pixel, with eight slices per patient and the three
contrast phases fused as RGB. [`ATTRIBUTION.md`](ATTRIBUTION.md) records every change
in full, as CC BY requires.

## Citation

BreastDCEDL aggregates three public collections held by
[The Cancer Imaging Archive](https://www.cancerimagingarchive.net/). Cite the release
and the collections it pools.

**The release:**

> Fridman, N., Solway, B., Fridman, T., et al. *BreastDCEDL: A standardized deep
> learning-ready breast DCE-MRI dataset of 2,070 patients.* Scientific Data **13**, 264
> (2026). <https://doi.org/10.1038/s41597-026-06589-6>

**The three collections it aggregates:**

| Collection | Patients used here |
|---|---:|
| I-SPY1 (ACRIN 6657), TCIA | 167 |
| I-SPY2, TCIA | 982 |
| Duke-Breast-Cancer-MRI, TCIA | 914 |

**I-SPY1:**

> Newitt, D., Hylton, N., on behalf of the I-SPY 1 Network and ACRIN 6657 Trial Team.
> *Multi-center breast DCE-MRI data and segmentations from patients in the I-SPY
> 1/ACRIN 6657 trials* [Data set]. The Cancer Imaging Archive (2016).
> <https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK>

**I-SPY2:**

> Li, W., Newitt, D.C., Gibbs, J., et al. *I-SPY 2 Breast Dynamic Contrast Enhanced
> MRI Trial (ISPY2)* [Data set]. The Cancer Imaging Archive (2022).
> <https://doi.org/10.7937/TCIA.D8Z0-9T85>

**Duke:** the collection is additionally described in:

> Saha, A., Harowicz, M.R., Grimm, L.J., et al. *A machine learning approach to
> radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features.*
> British Journal of Cancer **119**(4), 508-516 (2018).

Each collection carries its own dataset DOI on its TCIA landing page.
