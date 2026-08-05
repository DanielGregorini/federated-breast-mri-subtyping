# `production/scripts/`

Thin wrappers. **Every one of them execs the real script in `../../scripts/`** and
adds nothing — no arguments, no defaults, no logic.

They exist so that the deployment can be driven from inside `production/` without
having to know where the implementation lives, and so that a future move to a
hospital's own machine has one obvious entry point per action.

They deliberately hold no configuration. This project has shipped three bugs whose
single cause was two copies of one setting drifting apart, so there is exactly one
definition of every hyperparameter (`config/experiments.py`) and one implementation
of every action (`scripts/`). A wrapper that added a flag would be a second place to
look, and eventually a second answer.

| wrapper | runs | does |
|---|---|---|
| `provision.sh` | `scripts/provision.sh` | PKI startup kits into `workspace/` |
| `verify.sh` | `scripts/verify_production.py` | the full pre-flight check |
| `distributions.sh` | `scripts/build_distribution_report.py` | figures + tables into `figures/`, `datasets/` |
| `start.sh N TEST` | `scripts/start_federation.sh` | server + N hospitals, logs into `logs/TEST/` |
| `run.sh TEST` | `scripts/run_experiment.py` | submit one experiment through the admin API |
| `stop.sh` | `scripts/stop_federation.sh` | stop every participant |
| `collect.sh` | `scripts/collect_results.py` | score every finished model on the one test set |
| `summary.sh` | `scripts/build_final_summary.py` | build `results/final_summary/` |

Run them from anywhere; each resolves its own location.
