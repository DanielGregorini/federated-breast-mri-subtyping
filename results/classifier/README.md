# results/classifier — local centralised runs

Where a run from [`notebooks/03_train_centralized.ipynb`](../../notebooks/03_train_centralized.ipynb)
lands. One auto-numbered folder per run, `test_NNN_<model>_<task>/`.

The number is one above the highest already present, and gaps are never reused, so a
deleted run cannot have its number silently taken by a later one. **The counter reads
this folder**, so if you archive old runs elsewhere the numbering restarts at 001.

This folder is empty until you train something.

## What a run folder holds

| File | What it is |
|---|---|
| `config.json` | the exact configuration, written before training starts |
| `checkpoints/best_model.pt` | the selected model, chosen on validation macro-AUC |
| `checkpoints/last_model.pt` | the final epoch, whatever it was |
| `train_log.csv` | one row per epoch |
| `metrics.csv` | one flat row per split |
| `results.json` | config, best epoch, parameter counts, every metric, the generalisation gap |
| `predictions_{val,test}.csv` | one row per patient with every class probability and the cohort |
| `classification_report_{val,test}.txt` | sklearn's report, saved verbatim |
| `figures/` | confusion matrix, ROC, PR and per-class, per split, plus the three curves |

A run without `results.json` either crashed or is still going.

## Where the data comes from

`dataset/multi_subtype_80mm/`, built by
[`notebooks/02_build_dataset.ipynb`](../../notebooks/02_build_dataset.ipynb).

## What is not here

**The classifier phase.** 21 runs across 13 architectures and 5 data configurations,
which chose the configuration the federated campaign then held fixed. It was archived to
`unused/old_runs/classifier/`, with its run table `all_runs_pod.csv`, the raw per-run
output in `_from_pod/` and the 7 retained checkpoints including the `FREEZE_R18_*.pt`
pair that `TrainingConfig` was read out of.

**The reported centralised baseline.** That is `results/thesis/test01_centralized/`,
macro AUC 0.6068, produced by `src/scripts/run_centralized.py` on the same pipeline as
the twelve federated runs. Nothing in this folder is reported in the dissertation.
