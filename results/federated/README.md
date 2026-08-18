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
compare the experiments in this folder against each other. `distributions/` holds one
figure per experiment showing how the patients were divided between the hospitals.

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

Notebook 07 drives both:

```bash
jupyter notebook notebooks/07_federated_run.ipynb
```

Run folders here are not version controlled.
