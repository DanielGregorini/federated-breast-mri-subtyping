# results/federated — the thirteen reported results

Everything the dissertation reports comes from this folder and from nowhere else. One
subfolder per experiment, plus `final_summary/`. Nothing else lives here.

| Folder | What it is |
|---|---|
| `test01_centralized/seed_42/` | the centralised baseline, macro AUC 0.6068 |
| `test02_fedavg_2h/` … `test13_fedprox_sizematched/` | the twelve federated runs |
| `final_summary/` | the cross-experiment tables, comparisons and figures |
| `all_experiments.csv` | one row per experiment, written by `src/scripts/collect_results.py` |
| `per_hospital_02_07.csv` | the global model scored on each hospital's own validation patients |

## Where the data comes from

Every experiment is scored on the **same** global test set,
`deployment/data/global/test.csv`: 268 patients, 2,115 slices, trivial baseline 0.5112.
No hospital ever receives it. It is built by
[`notebooks/06_federated_setup.ipynb`](../../notebooks/06_federated_setup.ipynb) from
`dataset/multi_subtype_80mm/`.

The per-hospital training data comes from `deployment/data/partitions/<name>/`, built by
the same notebook. Patients are split by patient and never by slice.

## What a federated experiment folder holds

| File | What it is |
|---|---|
| `test_metrics.json` | macro AUC, balanced accuracy, per-class AUC / precision / recall / F1, the confusion matrix, the trivial baseline |
| `predictions_test.csv` | one row per test patient, 268 rows, with the cohort |
| `sites/rounds.csv` | per-round, per-site metrics across all 30 rounds |
| `sites/train.log` | each hospital's training log |
| `job.json` | job id, submission and completion times, algorithm, mu, client count, status |
| `global_model.pt` | the selected global model, about 43 MB. Not in version control |

`test01_centralized/seed_42/` differs: it holds `results.json`, `rounds.csv` (per epoch,
not per round), `best_model.pt`, `predictions_test.csv` and `report_test.txt`.

## Where the outputs go

`src/scripts/collect_results.py` re-scores every finished experiment on the global test
set and writes `all_experiments.csv`. `src/scripts/build_final_summary.py` then builds
`final_summary/`. Both are driven from
[`notebooks/07_federated_run.ipynb`](../../notebooks/07_federated_run.ipynb).

## Before quoting any number

The noise floor is **0.067 macro AUC**, measured between two runs of a byte-identical
configuration differing only in seed. The trivial accuracy baseline is **0.5112**. Every
experiment is a single run at seed 42. Differences below the noise floor are reported as
*no difference detected*. See [the results document](../../docs/RESULTS.md).
