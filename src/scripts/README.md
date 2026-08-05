# src/scripts — the operational scripts

Run them from the repository root. Every one reads
`src/src/federated/config/experiments.py`; none defines a hyperparameter of its own.

## Building the data

| Script | What it does |
|---|---|
| `prepare_data.py` | Carves the global validation and test sets, hardlinking the PNGs so nothing is duplicated |
| `partition_data.py` | Divides the training patients between hospitals. `--by-cohort` gives one real cohort per site; `--stratify none` gives label skew |
| `verify_data.py` | 111 checks: no patient in two sites, no training patient in the test set, every local validation split covers all three classes |
| `audit_dataset.py` | Verifies the hardlinks by inode and writes the dataset specification |

## Running experiments

| Script | What it does |
|---|---|
| `generate_jobs.py` | Writes `deployment/jobs/` from the experiment table. `--check` fails if a job was edited by hand |
| `verify_production.py` | 219 pre-flight checks. Writes nothing. Must pass before any federation starts |
| `start_federation.sh` | Starts the server, waits for the admin port, then starts each hospital as its own process |
| `run_experiment.py` | Submits one job through the admin API and monitors it to completion |
| `run_all_experiments.py` | The whole matrix |
| `run_centralized.py` | The centralised baseline, `test01` |
| `stop_federation.sh` | Stops everything by PID |

`run_centralized.py` trains the centralised baseline. It is **not** an NVFLARE job — one
machine, all the training data, no server and no clients — but it runs the same trainer
the federated clients run, which is what makes the comparison a measurement of
federation rather than of two different trainers.

## Collecting results

| Script | What it does |
|---|---|
| `collect_results.py` | Scores every global model on the one global test set |
| `build_final_summary.py` | Builds `results/federated/final_summary/`: tables, LaTeX, figures. `--no-client-eval` skips the expensive per-hospital evaluation |
| `build_distribution_report.py` | Per-experiment data distribution figures |
| `snapshot_config.py` | Writes the resolved configuration into `deployment/config/` as a record |

## Building figures

| Script | What it does |
|---|---|
| `build_dataset_report_figures.py` | Every figure in the dataset report |
| `build_preprocessing_walkthrough.py` | The step-by-step preprocessing figures and the pipeline flowchart |

Both call the same functions the dataset builder calls, so a figure and the report's
tables cannot disagree.
