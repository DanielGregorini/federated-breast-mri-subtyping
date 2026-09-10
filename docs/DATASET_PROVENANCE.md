# Dataset provenance: why `multi_subtype_80mm/` is not what the results used

Every reported result was measured on the dataset as it stood on 2026-08-05. The
folder `dataset/multi_subtype_80mm/` holds a later build and is not that dataset.
This document says how the two differ, where the August data survives, and how to
rebuild it.

```
deployment/code/scripts/rebuild_thesis_source.py   rebuilds the August dataset
deployment/code/scripts/check_splits.py            verifies the rebuilt splits
deployment/data/partitions/2_clients_balanced/     the seed of the rebuild, 722 MB
deployment/data/global/                            held-out val and test, 254 MB
results/thesis/distributions/                      the split record, 19 MB
```

The derived data is not kept in the repository. It is 15 GB of PNGs and it
regenerates byte for byte from the two folders above, which is verified below.

## The problem this folder exists to solve

`dataset/multi_subtype_80mm/` on disk today is **not** the dataset the seed 42
results were measured on. It was rebuilt on 2026-08-14 and two things changed:

| | 2026-08-05, used by seed 42 | 2026-08-14, on disk today |
|---|---|---|
| CSV columns | 35 | 27 |
| dropped | | `roi_basis`, `box3d_row0`, `box3d_row1`, `box3d_col0`, `box3d_col1`, `crop_center_row`, `crop_center_col`, `crop_mm` |
| `mm_per_px` | 0.35714 | 0.35834 (max shift 0.00297) |
| training PNGs | reference | 1,352 of 12,131 differ, across 827 of 1,527 patients |

Splitting seeds 19 and 50 off today's dataset would produce files in a different
format describing slightly different images from the ones seed 42 was measured on.
The seeds would not be comparable, which is the whole point of running them.

## Where the August data was recovered from

It was never lost. Every one of the 1,527 training patients appears exactly once
across the four CSVs of `deployment/data/partitions/2_clients_balanced`, and the
held-out rows are in `deployment/data/global`. The PNGs under those folders are the
August bytes: the 2026-08-14 rebuild wrote new inodes, so the hardlinks placed in
August kept pointing at the old content.

`01_rebuild_august_source.py` puts the dataset back together from its own output.
Row order was recovered from today's dataset, which is safe because all four August
partition CSVs are monotonic in it.

## The three seeds

| | seed 42 | seed 19 | seed 50 |
|---|---|---|---|
| status | reported in the dissertation | new | new |
| partition folders | `<shape>` | `<shape>_s19` | `<shape>_s50` |
| built from | `dataset/multi_subtype_80mm_thesis/` | same | same |

Six shapes at each seed: `2_clients_balanced`, `3_clients_balanced`,
`3_clients_cohort`, `3_clients_sizematched`, `4_clients_balanced`,
`4_clients_skewed`.

The seed changes exactly two things. Which training patient goes to which hospital,
and which of a hospital's own patients it holds back as local validation. It does
not touch the global val and test sets, which come from the BreastDCEDL `test`
column and are the same 268 + 268 patients at every seed. It does not touch a single
pixel. It does not touch a hyperparameter.

How far the seeds actually move the patients, as the share of patients that keep
both the same hospital and the same local split:

| shape | seed 19 vs 42 | seed 50 vs 42 |
|---|---:|---:|
| 2_clients_balanced | 35.1% | 32.7% |
| 3_clients_balanced | 23.3% | 22.2% |
| 3_clients_cohort | 66.5% | 68.2% |
| 3_clients_sizematched | 27.8% | 29.2% |
| 4_clients_balanced | 16.6% | 17.5% |
| 4_clients_skewed | 26.2% | 26.0% |

`3_clients_cohort` stays high on purpose. A site there holds one whole source
cohort, so its membership cannot depend on the seed. Only its local validation
carve moves, and 1 - 2(0.2)(0.8) = 68% is exactly the expected agreement.

## Evidence that the format matches August

Rebuilding **seed 42** from the reconstructed source and diffing it against the
August folders:

| | result |
|---|---|
| per-site `train.csv` and `val.csv` | 38 of 38 byte-identical |
| per-site `manifest.json` | 19 of 19 byte-identical |
| `partition.json` | differs only in `built`, `source` and `shape` |
| per-site patient and class counts | identical in all six shapes |

`built` is a timestamp, `source` is a path, and `shape` is a key added to the
generator after August. No field the training reads differs.

The same August schema then holds for the new seeds. All 114 CSVs across the 18
partitions carry the identical 35-column header.

## Checks that passed

`02_check_splits.py`, over all 18 partitions:

| check | result |
|---|---|
| images present and byte-identical to the source | 218,358 of 218,358 |
| hardlinks | 0, everything is a physical copy |
| patients placed | 1,527 in every partition |
| a patient in two hospitals | none |
| a patient in both a site's train and its val | none |
| a training patient also in global val or test | none |
| slices placed | 12,131 in every partition |
| CSV header equals the August 35-column header | 114 of 114 |

Determinism: rebuilding seeds 19 and 50 a second time reproduced all 126 CSV,
manifest and partition files byte for byte.

## The dataset organization record

`reports/seed_<n>/` holds, for each seed:

- `README.md` — what the seed changes, the held-out sets, per hospital and per
  split counts, and the class balance of every site
- `patient_assignments.csv` — 9,162 rows, one per patient per partition, naming
  the hospital, the split, the label, the cohort and the slice count
- `all_distributions.csv` and `.json` — the same counts in tabular form
- `global_held_out_patients.csv` — the 536 patients no seed ever trains on
- `global_splits.csv` — the val and test class counts
- `figures/*.png` — one distribution figure per experiment plus three overviews

`reports/seed_comparison.csv` puts the three seeds side by side, 38 rows, one per
shape, site and split.

## Rebuilding this folder

```bash
python3 deployment/code/scripts/rebuild_thesis_source.py

python3 deployment/code/scripts/partition_data.py \
  --source dataset/multi_subtype_80mm_thesis \
  --out deployment/data/partitions --seed 42
for S in 19 50; do
  python3 deployment/code/scripts/partition_data.py \
    --source dataset/multi_subtype_80mm_thesis \
    --out deployment/data/partitions \
    --seed $S --suffix _s$S
done

python3 deployment/code/scripts/check_splits.py

```

The `global/` folder is a plain physical copy of `deployment/data/global`.

The splitter is the repository's own `deployment/code/scripts/partition_data.py`,
unmodified, and the report generator is
`deployment/code/scripts/build_distribution_report.py`, also unmodified. Only their
input and output paths point here.

## The runs

`runs/` holds everything the campaign produced on the rented pod between
2026-09-08 18:57 and 2026-09-09 08:50 UTC. 39 runs: 36 federated and 3 centralized,
all scored on the same 268 held-out patients. See `runs/README.md`.

| | |
|---|---|
| `runs/all_experiments.csv` | the 39 runs in one table |
| `runs/analysis/` | cross-seed tables and the statistical summary |
| `runs/centralized/seed_{42,19,50}/` | weights, config, per-epoch curves, predictions |
| `runs/federated/<36 folders>/` | aggregated models, metrics, predictions, per-round CSVs |
| `runs/logs/` | server, hospitals, and the queue drivers |
| `runs/aborted/` | two runs the server killed, kept as a record of the incident |

Brought back with `tar` over ssh and verified by SHA-256: 286 of 286 files identical
to what was on the pod. The manifest is `runs/.pod.sha`.

The headline result: a site holding one whole source cohort loses 0.0574 macro-AUC
against a site of the same size holding a mix of cohorts, in 6 of 6 paired
comparisons, Wilcoxon p = 0.0312. Nothing else measured moved outside the 0.067
noise floor.


## The rebuild is verified, not assumed

Rebuilding the source from `deployment/data/partitions/2_clients_balanced/` and
`deployment/data/global/`, then regenerating the splits from it:

| step | result |
|---|---|
| source rebuilt | 16,383 of 16,383 files byte-identical to the one the runs used |
| seed 19 splits regenerated | 72,849 of 72,849 files byte-identical |
| seed 42 splits regenerated | 38 of 38 CSVs and 19 of 19 manifests byte-identical to the August originals |

`2_clients_balanced` is the one split kept in the repository because it is the only
one needed: it holds all 1,527 training patients exactly once, carries the August
35-column rows, and its PNGs are the August pixels. The 2026-08-14 rebuild wrote new
inodes, so the hardlinks placed in August kept pointing at the old content.
