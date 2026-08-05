# `multi_subtype_80mm` — dataset specification

Audited 2026-08-05 14:00 UTC by `scripts/audit_dataset.py`. Every number below is read from the data, not from a previous document.

**Source:** `/Users/daniel/Developer/tese/federated-breast-mri-subtyping/dataset/multi_subtype_80mm`

## Totals

| | |
|---|---|
| total patients | **2,063** |
| total images | **16,378** |
| image dimensions | **224x224**, RGB |
| image format | **PNG** |
| classes | **3** |
| class names | `HRposHER2neg`, `TripleNeg`, `HER2pos` |
| cohorts | `spy2`, `duke`, `spy1` |

## Classes

| index | dissertation name | dataset name | patients | images |
|---|---|---|---:|---:|
| 0 | HR+/HER2- | `HRposHER2neg` | 1,042 | 8,230 |
| 1 | Triple Negative | `TripleNeg` | 564 | 4,495 |
| 2 | HER2+ | `HER2pos` | 457 | 3,653 |

## Cohorts

| cohort | patients | images |
|---|---:|---:|
| `spy2` | 982 | 7,835 |
| `duke` | 914 | 7,212 |
| `spy1` | 167 | 1,331 |

## Train / validation / test

The validation and test splits are **global** — held out before any partitioning, identical for all thirteen experiments, and never trained on. The training pool is what the hospitals divide.

| split | patients | images | per class (patients) | trivial baseline |
|---|---:|---:|---|---:|
| **train pool** | 1,527 | 12,131 | — | — |
| global val | 268 | 2,132 | [132, 76, 60] | 0.4925 |
| global test | 268 | 2,115 | [137, 78, 53] | 0.5112 |

Accuracy is meaningless without the trivial baseline beside it — it is the accuracy of always predicting the majority class of that same split, and it is not a constant.

## Preprocessing

**Unchanged by this project.** The federated pipeline consumes this dataset; it does not build it. Produced by `src/core/dataset_builder.py` with `pipelines/thesis/preprocessing.py`, driven by `notebooks/03_build_dataset_mine.ipynb`.

| parameter | value |
|---|---|
| cohorts | ['spy2', 'spy1', 'duke'] |
| ROI / cropping strategy | `area_max` — the slice with the largest tumour area defines the crop centre |
| physical window | **80.0 mm** — a fixed physical window, not a proportional one, so tumour SIZE is preserved as signal |
| resampling | to a constant **80.0 mm / 224 px = 0.35714 mm/px** |
| saved size | 224x224 px |
| slices per patient | **8**, spread through the tumour volume |
| minimum tumour | 10 px, else the slice is dropped |
| normalization | `minmax` |
| intensity clip | `chanclip_q` 0.98 |
| volume trim | `trim_frac` 0.15 |

### Channel definition

Each PNG is **RGB, and the three channels are three DCE-MRI phases**, not a colour image:

| channel | phase |
|---|---|
| R | pre-contrast |
| G | early post-contrast |
| B | late post-contrast |

Verified: the three channels differ from one another in the audited sample, as three distinct phases must.

### Patients excluded

`config.json` records **no build errors** — `errors: {}`.

The build kept **2,063** patients. Exclusions happen at build time for a missing DCE phase, a missing or empty tumour mask, or a tumour smaller than 10 px on every slice; the surviving set is what this dataset contains.

## Federated hospital partitions

Patients are divided **by patient, never by slice** — every image of a patient goes to exactly one hospital.

Most partitions are **stratified**: every hospital carries the global class ratio, so the sites differ in QUANTITY and nothing else. That is quantity skew, the weakest form of non-IID data. The `3_clients_cohort` partition is deliberately **not** stratified — each hospital receives one complete source cohort — and `3_clients_sizematched` is its control, holding the same three site sizes with the cohorts mixed back together. The `class spread` reported under each partition below is the largest difference between any two hospitals in the share of a single class, in percentage points.

### `2_clients_balanced` — 2 hospitals, balanced (50/50)

stratified · class spread **0.1 pp**

Ratio 1:1 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 764 | 50.0% | 6,084 | 387 | 205 | 172 | 50.6/26.8/22.5 | duke 321 · spy1 48 · spy2 395 |
| `hospital_2` | 763 | 50.0% | 6,047 | 386 | 205 | 172 | 50.6/26.9/22.5 | duke 321 · spy1 53 · spy2 389 |

### `3_clients_balanced` — 3 hospitals, balanced (33.3 each)

stratified · class spread **0.1 pp**

Ratio 1:1:1 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 510 | 33.4% | 4,061 | 258 | 137 | 115 | 50.6/26.9/22.6 | duke 222 · spy1 34 · spy2 254 |
| `hospital_2` | 510 | 33.4% | 4,063 | 258 | 137 | 115 | 50.6/26.9/22.6 | duke 210 · spy1 37 · spy2 263 |
| `hospital_3` | 507 | 33.2% | 4,007 | 257 | 136 | 114 | 50.7/26.8/22.5 | duke 210 · spy1 30 · spy2 267 |

### `4_clients_balanced` — 4 hospitals, balanced (25 each)

stratified · class spread **0.2 pp**

Ratio 1:1:1:1 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 383 | 25.1% | 3,053 | 194 | 103 | 86 | 50.6/26.9/22.4 | duke 160 · spy1 29 · spy2 194 |
| `hospital_2` | 382 | 25.0% | 3,039 | 193 | 103 | 86 | 50.5/27.0/22.5 | duke 161 · spy1 19 · spy2 202 |
| `hospital_3` | 381 | 25.0% | 3,034 | 193 | 102 | 86 | 50.7/26.8/22.6 | duke 169 · spy1 28 · spy2 184 |
| `hospital_4` | 381 | 25.0% | 3,005 | 193 | 102 | 86 | 50.7/26.8/22.6 | duke 152 · spy1 25 · spy2 204 |

### `4_clients_skewed` — 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)

stratified · class spread **0.4 pp**

Ratio 5:2:1:1 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 848 | 55.5% | 6,753 | 429 | 228 | 191 | 50.6/26.9/22.5 | duke 357 · spy1 53 · spy2 438 |
| `hospital_2` | 340 | 22.3% | 2,705 | 172 | 91 | 77 | 50.6/26.8/22.6 | duke 150 · spy1 25 · spy2 165 |
| `hospital_3` | 170 | 11.1% | 1,339 | 86 | 46 | 38 | 50.6/27.1/22.4 | duke 77 · spy1 9 · spy2 84 |
| `hospital_4` | 169 | 11.1% | 1,334 | 86 | 45 | 38 | 50.9/26.6/22.5 | duke 58 · spy1 14 · spy2 97 |

### `3_clients_cohort` — 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)

ONE COHORT PER HOSPITAL · class spread **27.5 pp**

Ratio 642:101:784 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 642 | 42.0% | 5,074 | 426 | 102 | 114 | 66.4/15.9/17.8 | duke 642 |
| `hospital_2` | 101 | 6.6% | 804 | 42 | 27 | 32 | 41.6/26.7/31.7 | spy1 101 |
| `hospital_3` | 784 | 51.3% | 6,253 | 305 | 281 | 198 | 38.9/35.8/25.3 | spy2 784 |

### `3_clients_sizematched` — 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort

stratified · class spread **0.3 pp**

Ratio 642:101:784 · 1,527 patients · 12,131 images

| hospital | patients | % | images | HRposHER2neg | TripleNeg | HER2pos | class % | cohorts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `hospital_1` | 642 | 42.0% | 5,116 | 325 | 172 | 145 | 50.6/26.8/22.6 | duke 270 · spy1 43 · spy2 329 |
| `hospital_2` | 101 | 6.6% | 800 | 51 | 27 | 23 | 50.5/26.7/22.8 | duke 43 · spy1 4 · spy2 54 |
| `hospital_3` | 784 | 51.3% | 6,215 | 397 | 211 | 176 | 50.6/26.9/22.4 | duke 329 · spy1 54 · spy2 401 |

## Patient-level isolation — confirmed

All **127** checks passed.

- No patient appears at two hospitals, in any partition.
- No hospital holds a patient from the global validation or test split.
- Local train and local validation are disjoint at every hospital.
- Every hospital holds all three classes.
- Hospitals + global val + global test reconcile exactly to the source totals (2,063 patients, 16,378 images) in all four partitions.

## Image provenance

200/200 sampled images are the **same inode** as their source file. The federated layout is hardlinked to `multi_subtype_80mm`, so the bytes are the same on disk — the images cannot have been regenerated by a different pipeline.
