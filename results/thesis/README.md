# results/thesis

The thirteen experiments the dissertation reports, as they were run. One folder per
experiment, plus the summary built from all of them.

Nothing in here is regenerated.

| Path | What it is |
|---|---|
| `test01_centralized/seed_42/` | the centralised baseline |
| `test02_fedavg_2h/` to `test13_fedprox_sizematched/` | the twelve federated runs |
| `final_summary/` | the cross-experiment tables, comparisons and figures |
| `all_experiments.csv` | one row per experiment |
| `per_hospital_02_07.csv` | the global model scored on each hospital's own validation patients |

## What a federated experiment folder holds

| File | What it is |
|---|---|
| `test_metrics.json` | macro AUC, balanced accuracy, per-class AUC, precision, recall and F1, the confusion matrix, and the trivial baseline |
| `predictions_test.csv` | one row per test patient, 268 rows, with the cohort |
| `sites/rounds.csv` | per-round, per-site metrics across all 30 rounds |
| `sites/train.log` | each hospital's training log |
| `job.json` | job id, submission and completion times, algorithm, mu, client count, status |
| `global_model.pt` | the selected global model, about 43 MB. Not in version control |

`test01_centralized/seed_42/` is a centralised run and holds `results.json`,
`rounds.csv` (one row per epoch), `best_model.pt`, `predictions_test.csv` and
`report_test.txt`.

## Where the numbers come from

Every experiment is scored on the same global test set,
`deployment/data/global/test.csv`: 268 patients and 2,115 slices, which no hospital
sees during training. The per-hospital training data comes from
`deployment/data/partitions/<name>/`. Both are built by
[`notebooks/06_federated_setup.ipynb`](../../notebooks/06_federated_setup.ipynb).

## How to read it

```bash
jupyter notebook notebooks/05_compare_experiments.ipynb
```

Puts every run in one table. `final_summary/summary.md` is the same material as a
readable report, and `final_summary/README.md` explains each column.
