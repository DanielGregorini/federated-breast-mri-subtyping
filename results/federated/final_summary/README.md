# `results/final_summary/`

Everything the dissertation's results chapter needs, generated in one pass by
[`scripts/build_final_summary.py`](../../scripts/build_final_summary.py).

Regenerate at any time — it is a pure function of `results/`, `data/` and
`config/experiments.py`, and it overwrites this folder:

```bash
python scripts/build_final_summary.py
```

**Status: 13 of 13 experiments complete.**

---

## Layout

```
final_summary/
├── README.md                     this file
├── manifest.json                 machine-readable status of all nine experiments
├── summary.csv                   one row per experiment — the main table
├── summary.xlsx                  every table as a sheet
├── summary.json                  everything, nested, including curve points
├── summary.md                    the human-readable report
├── summary.pdf                   printable: tables then figures
├── per_client_metrics.csv        one row per (experiment, hospital)
├── comparisons/                  the eight comparison tables, one CSV each
├── figures/                      cross-experiment figures (png + pdf)
├── experiments/<name>/           per-experiment metrics, curves and figures
├── cohort/                       data description — independent of any run
└── tables/                       LaTeX-ready \begin{table} blocks
```

Every figure is written twice: `.png` for a quick look and for the PDF report,
`.pdf` (vector) for the thesis itself.

---

## What each experiment is

| id | algorithm | hospitals | split | research question |
|---|---|---|---|---|
| `test01` | centralized | 1 | all data pooled | RQ1 — the reference every federated run is measured against. |
| `test02` | fedavg | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | RQ1 — federated versus centralised. |
| `test03` | fedprox | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | RQ3 — FedAvg versus FedProx at two clients. |
| `test04` | fedavg | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | RQ1 — the shape of degradation as sites increase. |
| `test05` | fedprox | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | RQ3 — FedAvg versus FedProx at three clients. |
| `test06` | fedavg | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | RQ1 — the headline comparison against Test 01. Also the IID control for RQ2. |
| `test07` | fedprox | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | RQ3 — FedAvg versus FedProx at four clients. |
| `test08` | fedavg | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | RQ2 — the impact of heterogeneity, against Test 06. |
| `test09` | fedprox | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | RQ4 — does FedProx mitigate the skew that Test 08 exposes? |
| `test10` | fedavg | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | RQ2 — the primary test, against test12. |
| `test12` | fedavg | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | RQ2 — the control that isolates cohort identity. |
| `test11` | fedprox | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | RQ3 — against test10, under real heterogeneity. |
| `test13` | fedprox | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | RQ3 — against test12. |

---

## What each metric means

All metrics are computed **per patient**: the model predicts each 2D slice, and the
slice probabilities belonging to one patient are averaged into a single prediction
before anything is scored. Slices from one patient are near-duplicates, so a
slice-level number measures how well the model recognises the *patient*, not the
disease.

| metric | definition | where it comes from |
|---|---|---|
| `accuracy` | fraction of patients whose top-1 class is correct | `predictions_test.csv` |
| `trivial_baseline` | accuracy of always predicting the majority class **of that same split** | computed per split, never assumed |
| `balanced_accuracy` | mean per-class recall — the imbalance-robust accuracy | `predictions_test.csv` |
| `macro_precision/recall/f1` | unweighted mean over the 3 classes | `predictions_test.csv` |
| `precision_<class>` etc. | the same, per class | `predictions_test.csv` |
| `macro_auc` | **the headline metric** — one-vs-rest ROC AUC, macro-averaged | `predictions_test.csv` |
| `auc_<class>` | one-vs-rest AUC for that class alone | `predictions_test.csv` |
| `confusion` | rows = truth, columns = prediction, patient counts | `predictions_test.csv` |
| `best_epoch` | centralised: the epoch whose checkpoint was selected | `results.json` |
| `best_round` | federated: the round maximising `val_balanced_accuracy` averaged over clients | `rounds.csv` |
| `training_time_s` | see `training_time_kind` — **two different quantities** | `rounds.csv` / `job.json` |

`macro_auc` is NaN, not a number, whenever a split is missing a whole class. That
happens legitimately on the smaller per-hospital validation sets. A number computed
over a subset of classes would be silently incomparable to one computed over all of
them, so it is not produced.

---

## Three things that will otherwise be misread

**1. The noise floor is 0.067 macro-AUC.** It was measured in this project
between two byte-identical configurations differing only in random seed. Four of these
nine comparisons are expected to land inside it. Every comparison table therefore
carries `delta_macro_auc` **and** `within_noise_floor`; a difference with
`within_noise_floor = True` is not a finding, and must not be written up as one.

**2. Accuracy without its baseline means nothing.** The trivial baseline is not a
constant — it is 0.5112 on this test set. It is carried in every table beside the accuracy it
qualifies.

**3. Training time is not one quantity.** Centralised runs record per-epoch compute in
`rounds.csv` and the reported figure is their sum. Federated runs record no per-round
timing, so the only figure available is the job's wall clock from `job.json`, which
includes provisioning, model transfer and server-side aggregation. The
`training_time_kind` column says which one you are looking at. Do not put them in the
same column of a thesis table without that qualifier.

---

## Per-hospital numbers: what they are

`per_client_metrics.csv` is **the one global model evaluated on each hospital's own
held-out patients**. It is not each hospital's local model — those are never exported,
and would not be comparable across sites in any case.

It answers "does the federated model work equally well everywhere, or is it carried by
the biggest site?", which matters most for the skewed 4-hospital split where one site
holds 5/9 of the data.

The official number for every experiment is always the global test set, which is
identical across all nine. Per-hospital sets are small — down to 19 validation
patients — so their AUCs are noisy and sometimes NaN.

---

## Provenance

| output | derived from |
|---|---|
| all headline metrics | `results/<experiment>/predictions_test.csv` |
| ROC / PR / confusion | the same file — so they can never disagree with the table |
| `best_round` | `results/<experiment>/rounds.csv` |
| `best_epoch`, seed | `results/test01_centralized/seed_*/results.json` |
| job timing, algorithm | `results/<experiment>/job.json` |
| per-hospital metrics | the global model re-evaluated on `data/partitions/<p>/<site>/val.csv` |
| cohort tables | `data/partitions/*/partition.json`, `data/global/manifest.json` |

Metrics are recomputed from stored per-patient predictions rather than copied from a
stored metric dict. That is what makes the confusion matrix, the ROC curve and the
accuracy in the table mutually consistent by construction: they come from one array.

The predictions themselves are written by `scripts/collect_results.py`, which scores
every experiment through one code path on one test set. **Run that first** — this
script reads its output.
