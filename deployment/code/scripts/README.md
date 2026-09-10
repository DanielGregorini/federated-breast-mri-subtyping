# deployment/code/scripts

The scripts that drive a federated campaign, from building the splits to summarising
the results. Run them from the repository root. Every one reads
[`../config/experiments.py`](../config/experiments.py) and none defines a
hyperparameter of its own.

## Building the splits

| Script | What it does |
|---|---|
| `prepare_data.py` | Carves the global validation and test sets out of the processed dataset into `deployment/data/global/`. `--hardlink` avoids storing the same PNG many times. |
| `partition_data.py` | Divides the training patients between hospitals into `deployment/data/partitions/`. `--only NAME` builds one partition, `--seed N --suffix _sN` builds another draw beside the seed-42 one, `--by-cohort` forces one complete cohort per site, `--stratify none` lets the class ratio differ between sites. |

```bash
python deployment/code/scripts/prepare_data.py --hardlink
python deployment/code/scripts/partition_data.py --hardlink
python deployment/code/scripts/partition_data.py --seed 19 --suffix _s19 --hardlink
python deployment/code/scripts/partition_data.py --seed 50 --suffix _s50 --hardlink
```

`prepare_data.py` is run once. The global validation and test sets come from the
BreastDCEDL metadata rather than being drawn here, so they do not depend on a seed
and every seed is scored on the same 268 test patients.

Notebook 07 does the same work interactively for seed 42.

## Setting up and running

| Script | What it does |
|---|---|
| `provision.sh` | Writes one PKI startup kit per participant into `deployment/workspace/`. |
| `generate_jobs.py` | Writes the complete NVFLARE job folders under `deployment/jobs/`, each carrying the code the hospitals run, and copies them into the admin's `transfer/` directory. Twelve by default; `--seed N`, repeatable, adds a replica of each reading `partitions/<shape>_sN/`. |
| `run_experiment.py` | Submits one job through the admin API and follows it to completion. The federation has to be running already. `--dry-run` builds and describes without submitting. |
| `run_centralized.py` | The centralised baseline, `test01`. One machine, all training patients, no server and no clients, using the same trainer the federated clients use. |

```bash
bash deployment/code/scripts/provision.sh
python deployment/code/scripts/generate_jobs.py --seed 19 --seed 50
python deployment/code/scripts/run_experiment.py test06
python deployment/code/scripts/run_experiment.py test06_s19
```

Starting and stopping the participants is done from their startup kits. See
[`../../README.md`](../../README.md).

## Collecting results

| Script | What it does |
|---|---|
| `collect_results.py` | Scores every finished global model on the one global test set and writes `results/federated/<experiment>/`. |
| `build_final_summary.py` | Builds `results/federated/final_summary/` with the cross-experiment tables and figures. `--no-client-eval` skips the slow per-hospital evaluation. |
| `build_distribution_report.py` | Writes the distribution figures and the split tables into `results/thesis/distributions/`, one folder per seed. |

```bash
python deployment/code/scripts/collect_results.py
python deployment/code/scripts/build_final_summary.py
```

`build_final_summary.py` and `build_distribution_report.py` write PNG. Pass `--pdf` to
also write a vector copy of each figure.
