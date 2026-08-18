# deployment/code/scripts

The scripts that drive a federated campaign, from building the splits to summarising
the results. Run them from the repository root. Every one reads
[`../config/experiments.py`](../config/experiments.py) and none defines a
hyperparameter of its own.

## Building the splits

| Script | What it does |
|---|---|
| `prepare_data.py` | Carves the global validation and test sets out of the processed dataset into `deployment/data/global/`. `--hardlink` avoids storing the same PNG many times. |
| `partition_data.py` | Divides the training patients between hospitals into `deployment/data/partitions/`. `--only NAME` builds one partition, `--by-cohort` gives each site one complete cohort, `--stratify none` lets the class ratio differ between sites. |
| `verify_data.py` | Leakage checks. No patient in two sites, no training patient in the test set, every local validation split covering all three classes. `--check-imports` also checks that the shared trainer has not picked up an `nvflare` import. Exits non-zero on failure. |
| `audit_dataset.py` | Verifies the hardlinks by inode and writes `deployment/datasets/dataset_audit.json`. |

```bash
python deployment/code/scripts/prepare_data.py --hardlink
python deployment/code/scripts/partition_data.py --hardlink
python deployment/code/scripts/verify_data.py
```

Notebook 06 does the same work interactively.

## Setting up and running

| Script | What it does |
|---|---|
| `provision.sh` | Writes one PKI startup kit per participant into `deployment/workspace/`. |
| `generate_jobs.py` | Writes the twelve complete NVFLARE job folders under `deployment/jobs/`, each carrying the code the hospitals run, and copies them into the admin's `transfer/` directory. `--check` fails instead if one has drifted. |
| `verify_production.py` | 218 pre-flight checks. Writes nothing and has to pass before any federation starts. |
| `run_experiment.py` | Submits one job through the admin API and follows it to completion. The federation has to be running already. `--dry-run` builds and describes without submitting. |
| `run_centralized.py` | The centralised baseline, `test01`. One machine, all training patients, no server and no clients, using the same trainer the federated clients use. |

```bash
bash deployment/code/scripts/provision.sh
python deployment/code/scripts/generate_jobs.py
python deployment/code/scripts/verify_production.py
python deployment/code/scripts/run_experiment.py test06
```

Starting and stopping the participants is done from their startup kits. See
[`../../README.md`](../../README.md).

## Collecting results

| Script | What it does |
|---|---|
| `collect_results.py` | Scores every finished global model on the one global test set and writes `results/federated/<experiment>/`. |
| `build_final_summary.py` | Builds `results/federated/final_summary/` with the cross-experiment tables and figures. `--no-client-eval` skips the slow per-hospital evaluation. |
| `build_distribution_report.py` | Writes the distribution figures into `results/federated/distributions/` and the tables into `deployment/datasets/`. |
| `snapshot_config.py` | Writes the resolved configuration into `deployment/config/` as a record. |

```bash
python deployment/code/scripts/collect_results.py
python deployment/code/scripts/build_final_summary.py
```

`build_final_summary.py` and `build_distribution_report.py` write PNG. Pass `--pdf` to
also write a vector copy of each figure.
