# results/federated

Where a federated campaign you run yourself lands. One folder per experiment.

| File | What it is |
|---|---|
| `job.json` | job id, submission and completion times, algorithm, mu, client count, status |
| `global_model.pt` | the selected global model |
| `test_metrics.json` | every metric on the global test set |
| `predictions_test.csv` | one row per test patient |
| `sites/rounds.csv` | per-round, per-site metrics |
| `sites/train.log` | each hospital's training log |

`final_summary/` appears once you build it, holding the tables and figures that
compare the experiments in this folder against each other.

## distributions/

How the patients were divided between the hospitals. One figure per experiment, plus
three overviews, and the tables behind them.

### The tables

#### `all_distributions.csv`

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

#### `global_splits.csv`

Two rows, the held-out sets no hospital ever receives.

| Column | What it is |
|---|---|
| `site` | `global_test` or `global_val` |
| `patients`, `images` | size of the split |
| `patients_<class>` | patients per class |
| `trivial_baseline` | accuracy of always predicting the majority class of that split |

#### `dataset_audit.json`

The output of the integrity audit. Source dataset totals, the global splits, every
partition, and the list of checks that ran and whether any failed.

Rebuild them with:

```bash
python deployment/code/scripts/build_distribution_report.py
```

## How to fill it

Start the federation and submit an experiment. See
[`deployment/README.md`](../../deployment/README.md).

Then score the finished models on the global test set:

```bash
python deployment/code/scripts/collect_results.py
```

And build the summary:

```bash
python deployment/code/scripts/build_final_summary.py
```

Notebook 08 drives both:

```bash
jupyter notebook notebooks/08_federated_run.ipynb
```

Run folders here are not version controlled.
