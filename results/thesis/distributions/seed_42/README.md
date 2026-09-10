# Dataset organization — seed 42

How the data is divided for every experiment run at seed 42, and the listings
that make that division reproducible.

This is the original split. Every result in `results/thesis/` was measured on it.

## What the seed changes, and what it does not

The seed controls two things and nothing else:

1. Which of the 1527 training patients each hospital receives.
2. Which of its own patients each hospital holds back as its local validation split.

It does not touch the global validation and test sets. Those are read from the
`test` column of the BreastDCEDL metadata, not drawn here, so they are identical at
every seed and results measured at different seeds are directly comparable.

It does not touch the images. Every PNG is a physical copy of the image that
`seed_runs/data/source/images` holds, with no hardlinks anywhere. That source is
the dataset as it stood on 2026-08-05, when the seed 42 splits were made.
Nothing is resampled, re-cropped, re-normalised or re-encoded. Each seed sees
byte-identical pixels.

It does not touch any hyperparameter. Model `resnet18`, ImageNet-pretrained,
frozen up to `layer3`, 224x224 input,
adamw at lr 0.0001, weight decay 0.0005, dropout
0.5, label smoothing 0.1, batch 24 with at
most 1 slice per patient,
30 rounds x 1 local epoch,
selection on `val_balanced_accuracy`, patient-level `mean`
aggregation. All of it comes from `config/experiments.py` and is shared with every
other seed.

## The held-out sets, identical at every seed

| split | patients | images | HRposHER2neg | TripleNeg | HER2pos |
|---|---:|---:|---:|---:|---:|
| test | 268 | 2115 | 137 | 78 | 53 |
| val | 268 | 2132 | 132 | 76 | 60 |

## The partitions

6 shapes, each covering all training patients exactly once.

| folder | hospitals | mode | patients | images |
|---|---:|---|---:|---:|
| `2_clients_balanced` | 2 | stratified | 1527 | 12,131 |
| `3_clients_balanced` | 3 | stratified | 1527 | 12,131 |
| `3_clients_cohort` | 3 | cohort | 1527 | 12,131 |
| `3_clients_sizematched` | 3 | stratified | 1527 | 12,131 |
| `4_clients_balanced` | 4 | stratified | 1527 | 12,131 |
| `4_clients_skewed` | 4 | stratified | 1527 | 12,131 |

`mode` is how patients are dealt out. `stratified` splits within each class, so
every site keeps the global class ratio and the sites differ only in quantity.
`cohort` gives each site one whole source cohort, which is the only genuinely
non-IID partition here. A cohort partition's site membership does not depend on the
seed, because the cohorts are fixed; only its local validation split moves.

## Per hospital and per split

| partition | hospital | split | patients | images | HRposHER2neg | TripleNeg | HER2pos |
|---|---|---|---:|---:|---:|---:|---:|
| `2_clients_balanced` | hospital_1 | train | 612 | 4,870 | 310 | 164 | 138 |
| `2_clients_balanced` | hospital_1 | val | 152 | 1,214 | 77 | 41 | 34 |
| `2_clients_balanced` | hospital_2 | train | 611 | 4,849 | 309 | 164 | 138 |
| `2_clients_balanced` | hospital_2 | val | 152 | 1,198 | 77 | 41 | 34 |
| `3_clients_balanced` | hospital_1 | train | 408 | 3,245 | 206 | 110 | 92 |
| `3_clients_balanced` | hospital_1 | val | 102 | 816 | 52 | 27 | 23 |
| `3_clients_balanced` | hospital_2 | train | 408 | 3,249 | 206 | 110 | 92 |
| `3_clients_balanced` | hospital_2 | val | 102 | 814 | 52 | 27 | 23 |
| `3_clients_balanced` | hospital_3 | train | 406 | 3,206 | 206 | 109 | 91 |
| `3_clients_balanced` | hospital_3 | val | 101 | 801 | 51 | 27 | 23 |
| `3_clients_cohort` | hospital_1 | train | 514 | 4,067 | 341 | 82 | 91 |
| `3_clients_cohort` | hospital_1 | val | 128 | 1,007 | 85 | 20 | 23 |
| `3_clients_cohort` | hospital_2 | train | 82 | 652 | 34 | 22 | 26 |
| `3_clients_cohort` | hospital_2 | val | 19 | 152 | 8 | 5 | 6 |
| `3_clients_cohort` | hospital_3 | train | 627 | 5,005 | 244 | 225 | 158 |
| `3_clients_cohort` | hospital_3 | val | 157 | 1,248 | 61 | 56 | 40 |
| `3_clients_sizematched` | hospital_1 | train | 514 | 4,096 | 260 | 138 | 116 |
| `3_clients_sizematched` | hospital_1 | val | 128 | 1,020 | 65 | 34 | 29 |
| `3_clients_sizematched` | hospital_2 | train | 81 | 640 | 41 | 22 | 18 |
| `3_clients_sizematched` | hospital_2 | val | 20 | 160 | 10 | 5 | 5 |
| `3_clients_sizematched` | hospital_3 | train | 628 | 4,976 | 318 | 169 | 141 |
| `3_clients_sizematched` | hospital_3 | val | 156 | 1,239 | 79 | 42 | 35 |
| `4_clients_balanced` | hospital_1 | train | 306 | 2,439 | 155 | 82 | 69 |
| `4_clients_balanced` | hospital_1 | val | 77 | 614 | 39 | 21 | 17 |
| `4_clients_balanced` | hospital_2 | train | 305 | 2,423 | 154 | 82 | 69 |
| `4_clients_balanced` | hospital_2 | val | 77 | 616 | 39 | 21 | 17 |
| `4_clients_balanced` | hospital_3 | train | 305 | 2,426 | 154 | 82 | 69 |
| `4_clients_balanced` | hospital_3 | val | 76 | 608 | 39 | 20 | 17 |
| `4_clients_balanced` | hospital_4 | train | 305 | 2,399 | 154 | 82 | 69 |
| `4_clients_balanced` | hospital_4 | val | 76 | 606 | 39 | 20 | 17 |
| `4_clients_skewed` | hospital_1 | train | 678 | 5,395 | 343 | 182 | 153 |
| `4_clients_skewed` | hospital_1 | val | 170 | 1,358 | 86 | 46 | 38 |
| `4_clients_skewed` | hospital_2 | train | 273 | 2,171 | 138 | 73 | 62 |
| `4_clients_skewed` | hospital_2 | val | 67 | 534 | 34 | 18 | 15 |
| `4_clients_skewed` | hospital_3 | train | 136 | 1,073 | 69 | 37 | 30 |
| `4_clients_skewed` | hospital_3 | val | 34 | 266 | 17 | 9 | 8 |
| `4_clients_skewed` | hospital_4 | train | 135 | 1,066 | 69 | 36 | 30 |
| `4_clients_skewed` | hospital_4 | val | 34 | 268 | 17 | 9 | 8 |

## The files

| file | what it holds |
|---|---|
| `all_distributions.csv` | one row per experiment, hospital and split, with patient and image counts, per-class counts, cohort counts and percentages |
| `all_distributions.json` | the same, nested per experiment, plus the split rules |
| `global_splits.csv` | the held-out validation and test sets |
| `global_held_out_patients.csv` | every held-out patient, with label, cohort and slice count |
| `patient_assignments.csv` | **the sample listing**: 9,162 rows, one per patient per partition, saying which hospital and which split it landed in |
| `figures/` | one distribution figure per experiment, plus three overviews |

Slice-level listings are the per-site `train.csv` and `val.csv` under
`deployment/data/partitions/<folder>/<hospital>/`. They carry one row per image and
are what the clients actually read. This folder describes them; it does not replace
them.

## How to rebuild it

```bash
python deployment/code/scripts/partition_data.py --seed 42 --hardlink
python deployment/code/scripts/build_distribution_report.py --seed 42
```

The allocation is deterministic given the seed, so this reproduces the same split.
