# results/classifier

Centralised runs from
[`notebooks/04_train_centralized.ipynb`](../../notebooks/04_train_centralized.ipynb).
One folder per run, named `test_NNN_<model>_<task>/`.

The number is one above the highest already present, and gaps are never reused, so a
deleted run cannot have its number taken by a later one. The counter reads this
folder, so archiving old runs elsewhere restarts the numbering at 001.

## What a run folder holds

| Path | What it is |
|---|---|
| `config.json` | the exact configuration, written before training starts |
| `checkpoints/best_model.pt` | the selected model, chosen on validation macro AUC |
| `checkpoints/last_model.pt` | the final epoch, whatever it was |
| `train_log.csv` | one row per epoch |
| `metrics.csv` | one flat row per split |
| `results.json` | the configuration, the best epoch, parameter counts, every metric, and the generalisation gap |
| `predictions_{val,test}.csv` | one row per patient, with every class probability and the cohort |
| `classification_report_{val,test}.txt` | sklearn's report, saved verbatim |
| `figures/` | confusion matrix, ROC, PR and per-class figures per split, plus the three curves |

A run without `results.json` either crashed or is still going.

`all_experiments.csv` is one row per run, written by
[`notebooks/06_compare_experiments.ipynb`](../../notebooks/06_compare_experiments.ipynb).

## How to produce one

The dataset has to exist first, built by
[`notebooks/02_build_dataset.ipynb`](../../notebooks/02_build_dataset.ipynb).

```bash
jupyter notebook notebooks/04_train_centralized.ipynb
```

Then read one run, or compare them all:

```bash
jupyter notebook notebooks/05_evaluate_run.ipynb
jupyter notebook notebooks/06_compare_experiments.ipynb
```
