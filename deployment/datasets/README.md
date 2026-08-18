# deployment/datasets

Tables describing how the patients were divided between the hospitals. No images and
no model weights.

They are a record, not an input. Nothing in the pipeline reads them back. They exist
so a finished campaign can be described without re-deriving the split.

## The files

### `all_distributions.csv`

One row per experiment, site and split. 78 rows covering all thirteen experiments.

| Column | What it is |
|---|---|
| `experiment` | `test01` to `test13` |
| `name` | the experiment's folder name |
| `algorithm` | `centralized`, `fedavg` or `fedprox` |
| `partition` | which split under `data/partitions/` this row belongs to |
| `site` | `hospital_1` to `hospital_4`, or `centralized` |
| `split` | `train` or `val` |
| `patients` | patients at that site in that split |
| `images` | slices at that site in that split |
| `patients_HRposHER2neg`, `patients_TripleNeg`, `patients_HER2pos` | patients per class |
| `cohorts` | JSON, patients per source cohort at that site |
| `pct_patients`, `pct_images` | that site's share of the experiment |

`all_distributions.json` holds the same rows plus the generation timestamp and the
splitting rule.

### `global_splits.csv`

Two rows, the held-out sets no hospital ever receives.

| Column | What it is |
|---|---|
| `site` | `global_test` or `global_val` |
| `patients`, `images` | size of the split |
| `patients_<class>` | patients per class |
| `trivial_baseline` | accuracy of always predicting the majority class of that split |

### `dataset_audit.json`

The output of the integrity audit. Source dataset totals, the global splits, every
partition, and the list of checks that ran and whether any failed.

## How to rebuild them

```bash
python deployment/code/scripts/build_distribution_report.py
```

Writes `all_distributions.csv`, `all_distributions.json` and `global_splits.csv` here,
and the matching figures into `../../results/federated/distributions/`. Add `--pdf` to also write vector copies of
the figures.

```bash
python deployment/code/scripts/audit_dataset.py
```

Writes `dataset_audit.json`. Exits non-zero if any check fails.

Both read `data/partitions/*/partition.json` and `data/global/manifest.json`, so
rebuild a partition first if you changed one.
