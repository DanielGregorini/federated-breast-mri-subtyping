# Final results summary

Generated 2026-08-07 13:45 UTC by `scripts/build_final_summary.py`.

**13 of 13 experiments complete.**

## How to read these numbers

- Every metric is **per patient**. Slice probabilities are averaged per patient first.
- Every experiment is scored on the **same** global test set (`data/global/test/`).
- **The noise floor on this task is 0.067 macro-AUC**, measured between two byte-identical configurations differing only in random seed. Any difference smaller than that is not a result. The comparison tables carry a `within_noise_floor` column for exactly this reason.
- Accuracy is meaningless without the trivial baseline printed beside it. It is not a constant.
- `training_time_kind` says what was actually timed: centralised runs record per-epoch compute, federated runs only have job wall clock, which includes orchestration. They are not the same quantity.

## Protocol

```
FEDERATED PROTOCOL — breast-cancer molecular subtype
  dataset  : multi_subtype_80mm
  classes  : 3 — HRposHER2neg, TripleNeg, HER2pos
  model    : resnet18 (ImageNet pretrained=True)
  federated: 30 rounds x 1 local epoch
  central  : 30 epochs (same data budget)
  lr 0.0001 · batch 24 · dropout 0.5 · seed 42
  FedProx mu: 0.01
  model selection: val_balanced_accuracy (held-out client data)

  13 experiments, 6 partitions
    test01  test01_centralized                   1c  —        all data pooled
    test02  test02_fedavg_2h                     2c  fedavg   2 hospitals, balanced (50/50)  ->  50.0% / 50.0%
    test03  test03_fedprox_2h                    2c  fedprox  2 hospitals, balanced (50/50)  ->  50.0% / 50.0%
    test04  test04_fedavg_3h                     3c  fedavg   3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3%
    test05  test05_fedprox_3h                    3c  fedprox  3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3%
    test06  test06_fedavg_4h                     4c  fedavg   4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0%
    test07  test07_fedprox_4h                    4c  fedprox  4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0%
    test08  test08_fedavg_skewed                 4c  fedavg   4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1%
    test09  test09_fedprox_skewed                4c  fedprox  4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1%
    test10  test10_fedavg_cohort                 3c  fedavg   3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3%
    test12  test12_fedavg_sizematched            3c  fedavg   3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3%
    test11  test11_fedprox_cohort                3c  fedprox  3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3%
    test13  test13_fedprox_sizematched           3c  fedprox  3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3%
```

## Main results

| experiment | algorithm | n_hospitals | data_split | seed | best_epoch | best_round | training_time_s | trivial_baseline | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | macro_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | centralized | 1 | all data pooled | 42 | 4 |  | 268 | 0.5112 | 0.5299 | 0.4503 | 0.4606 | 0.4503 | 0.4523 | 0.6068 |
| test02 | fedavg | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 42 |  | 29 | 622 | 0.5112 | 0.4328 | 0.3949 | 0.3939 | 0.3949 | 0.3931 | 0.5816 |
| test03 | fedprox | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 42 |  | 29 | 351 | 0.5112 | 0.4328 | 0.4025 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |
| test04 | fedavg | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 42 |  | 27 | 313 | 0.5112 | 0.4851 | 0.4198 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |
| test05 | fedprox | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 42 |  | 28 | 333 | 0.5112 | 0.4590 | 0.4127 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |
| test06 | fedavg | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 42 |  | 0 | 779 | 0.5112 | 0.4925 | 0.4327 | 0.4574 | 0.4327 | 0.4338 | 0.6077 |
| test07 | fedprox | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 42 |  | 0 | 335 | 0.5112 | 0.4739 | 0.4393 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |
| test08 | fedavg | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 42 |  | 21 | 323 | 0.5112 | 0.4888 | 0.4259 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |
| test09 | fedprox | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 42 |  | 21 | 849 | 0.5112 | 0.4776 | 0.4285 | 0.4449 | 0.4285 | 0.4301 | 0.6152 |
| test10 | fedavg | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 42 |  | 7 | 469 | 0.5112 | 0.4291 | 0.3582 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |
| test12 | fedavg | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 42 |  | 17 | 471 | 0.5112 | 0.4478 | 0.4183 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |
| test11 | fedprox | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 42 |  | 16 | 482 | 0.5112 | 0.4590 | 0.4105 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |
| test13 | fedprox | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 42 |  | 27 | 485 | 0.5112 | 0.4664 | 0.3885 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |

## Comparisons

### centralized vs fedavg

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.3939 | 0.3949 | 0.3931 | 0.5816 |  | 29 | 622 | centralized (test01) | 0.6068 | -0.0252 | yes |
| test04 | test04_fedavg_3h | fedavg | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4851 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |  | 27 | 313 | centralized (test01) | 0.6068 | -0.0078 | yes |
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4925 | 0.4574 | 0.4327 | 0.4338 | 0.6077 |  | 0 | 779 | centralized (test01) | 0.6068 | 0.0009 | yes |
| test08 | test08_fedavg_skewed | fedavg | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4888 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |  | 21 | 323 | centralized (test01) | 0.6068 | -0.0086 | yes |
| test10 | test10_fedavg_cohort | fedavg | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4291 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |  | 7 | 469 | centralized (test01) | 0.6068 | -0.0642 | yes |
| test12 | test12_fedavg_sizematched | fedavg | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4478 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |  | 17 | 471 | centralized (test01) | 0.6068 | -0.0232 | yes |

### centralized vs fedprox

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test03 | test03_fedprox_2h | fedprox | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |  | 29 | 351 | centralized (test01) | 0.6068 | -0.0151 | yes |
| test05 | test05_fedprox_3h | fedprox | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4590 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |  | 28 | 333 | centralized (test01) | 0.6068 | -0.0110 | yes |
| test07 | test07_fedprox_4h | fedprox | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4739 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |  | 0 | 335 | centralized (test01) | 0.6068 | 0.0007 | yes |
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4776 | 0.4449 | 0.4285 | 0.4301 | 0.6152 |  | 21 | 849 | centralized (test01) | 0.6068 | 0.0084 | yes |
| test11 | test11_fedprox_cohort | fedprox | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4590 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |  | 16 | 482 | centralized (test01) | 0.6068 | -0.0390 | yes |
| test13 | test13_fedprox_sizematched | fedprox | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4664 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |  | 27 | 485 | centralized (test01) | 0.6068 | -0.0186 | yes |

### fedavg vs fedprox

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.3939 | 0.3949 | 0.3931 | 0.5816 |  | 29 | 622 | FedAvg (mean over configurations) | 0.5854 | -0.0039 | yes |
| test04 | test04_fedavg_3h | fedavg | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4851 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |  | 27 | 313 | FedAvg (mean over configurations) | 0.5854 | 0.0135 | yes |
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4925 | 0.4574 | 0.4327 | 0.4338 | 0.6077 |  | 0 | 779 | FedAvg (mean over configurations) | 0.5854 | 0.0222 | yes |
| test08 | test08_fedavg_skewed | fedavg | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4888 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |  | 21 | 323 | FedAvg (mean over configurations) | 0.5854 | 0.0127 | yes |
| test10 | test10_fedavg_cohort | fedavg | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4291 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |  | 7 | 469 | FedAvg (mean over configurations) | 0.5854 | -0.0429 | yes |
| test12 | test12_fedavg_sizematched | fedavg | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4478 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |  | 17 | 471 | FedAvg (mean over configurations) | 0.5854 | -0.0019 | yes |
| test03 | test03_fedprox_2h | fedprox | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |  | 29 | 351 | FedAvg (mean over configurations) | 0.5854 | 0.0062 | yes |
| test05 | test05_fedprox_3h | fedprox | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4590 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |  | 28 | 333 | FedAvg (mean over configurations) | 0.5854 | 0.0103 | yes |
| test07 | test07_fedprox_4h | fedprox | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4739 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |  | 0 | 335 | FedAvg (mean over configurations) | 0.5854 | 0.0221 | yes |
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4776 | 0.4449 | 0.4285 | 0.4301 | 0.6152 |  | 21 | 849 | FedAvg (mean over configurations) | 0.5854 | 0.0297 | yes |
| test11 | test11_fedprox_cohort | fedprox | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4590 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |  | 16 | 482 | FedAvg (mean over configurations) | 0.5854 | -0.0177 | yes |
| test13 | test13_fedprox_sizematched | fedprox | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4664 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |  | 27 | 485 | FedAvg (mean over configurations) | 0.5854 | 0.0027 | yes |

### fedavg vs fedprox paired

| data_split | n_hospitals | fedavg | fedprox | fedavg_accuracy | fedprox_accuracy | delta_accuracy | fedavg_macro_precision | fedprox_macro_precision | fedavg_macro_recall | fedprox_macro_recall | fedavg_macro_f1 | fedprox_macro_f1 | delta_macro_f1 | fedavg_macro_auc | fedprox_macro_auc | delta_macro_auc | fedavg_best_round | fedprox_best_round | fedavg_training_time_s | fedprox_training_time_s | within_noise_floor | favours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 2 | test02 | test03 | 0.4328 | 0.4328 | 0.0000 | 0.3939 | 0.4028 | 0.3949 | 0.4025 | 0.3931 | 0.4001 | 0.0070 | 0.5816 | 0.5917 | 0.0101 | 29 | 29 | 622 | 351 | yes | — |
| 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 3 | test04 | test05 | 0.4851 | 0.4590 | -0.0261 | 0.4333 | 0.4208 | 0.4198 | 0.4127 | 0.4231 | 0.4116 | -0.0115 | 0.5990 | 0.5958 | -0.0032 | 27 | 28 | 313 | 333 | yes | — |
| 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 3 | test12 | test13 | 0.4478 | 0.4664 | 0.0186 | 0.4187 | 0.3925 | 0.4183 | 0.3885 | 0.4153 | 0.3884 | -0.0269 | 0.5836 | 0.5882 | 0.0046 | 17 | 27 | 471 | 485 | yes | — |
| 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 3 | test10 | test11 | 0.4291 | 0.4590 | 0.0299 | 0.3523 | 0.4302 | 0.3582 | 0.4105 | 0.3536 | 0.4144 | 0.0608 | 0.5426 | 0.5678 | 0.0252 | 7 | 16 | 469 | 482 | yes | — |
| 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 4 | test06 | test07 | 0.4925 | 0.4739 | -0.0186 | 0.4574 | 0.4389 | 0.4327 | 0.4393 | 0.4338 | 0.4362 | 0.0024 | 0.6077 | 0.6075 | -0.0002 | 0 | 0 | 779 | 335 | yes | — |
| 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 4 | test08 | test09 | 0.4888 | 0.4776 | -0.0112 | 0.4374 | 0.4449 | 0.4259 | 0.4285 | 0.4292 | 0.4301 | 0.0009 | 0.5982 | 0.6152 | 0.0170 | 21 | 21 | 323 | 849 | yes | — |

### 2 hospitals

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.3939 | 0.3949 | 0.3931 | 0.5816 |  | 29 | 622 | centralized (test01) | 0.6068 | -0.0252 | yes |
| test03 | test03_fedprox_2h | fedprox | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |  | 29 | 351 | centralized (test01) | 0.6068 | -0.0151 | yes |

### 3 hospitals

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test04 | test04_fedavg_3h | fedavg | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4851 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |  | 27 | 313 | centralized (test01) | 0.6068 | -0.0078 | yes |
| test05 | test05_fedprox_3h | fedprox | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4590 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |  | 28 | 333 | centralized (test01) | 0.6068 | -0.0110 | yes |
| test10 | test10_fedavg_cohort | fedavg | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4291 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |  | 7 | 469 | centralized (test01) | 0.6068 | -0.0642 | yes |
| test12 | test12_fedavg_sizematched | fedavg | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4478 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |  | 17 | 471 | centralized (test01) | 0.6068 | -0.0232 | yes |
| test11 | test11_fedprox_cohort | fedprox | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4590 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |  | 16 | 482 | centralized (test01) | 0.6068 | -0.0390 | yes |
| test13 | test13_fedprox_sizematched | fedprox | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4664 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |  | 27 | 485 | centralized (test01) | 0.6068 | -0.0186 | yes |

### 4 hospitals balanced

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4925 | 0.4574 | 0.4327 | 0.4338 | 0.6077 |  | 0 | 779 | centralized (test01) | 0.6068 | 0.0009 | yes |
| test07 | test07_fedprox_4h | fedprox | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4739 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |  | 0 | 335 | centralized (test01) | 0.6068 | 0.0007 | yes |

### 4 hospitals skewed

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test08 | test08_fedavg_skewed | fedavg | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4888 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |  | 21 | 323 | centralized (test01) | 0.6068 | -0.0086 | yes |
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4776 | 0.4449 | 0.4285 | 0.4301 | 0.6152 |  | 21 | 849 | centralized (test01) | 0.6068 | 0.0084 | yes |

## Per-hospital results

| experiment | name | algorithm | partition | site | status | detail | n_train_patients | n_train_images | n_val_patients | n_val_images | val_patients_HRposHER2neg | val_patients_TripleNeg | val_patients_HER2pos | trivial_baseline | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | auc_HRposHER2neg | auc_TripleNeg | auc_HER2pos | precision_HRposHER2neg | precision_TripleNeg | precision_HER2pos | recall_HRposHER2neg | recall_TripleNeg | recall_HER2pos | f1_HRposHER2neg | f1_TripleNeg | f1_HER2pos | confusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test02 | test02_fedavg_2h | fedavg | 2_clients_balanced | hospital_1 | ok |  | 612 | 4,870 | 152 | 1,214 | 77 | 41 | 34 | 0.5066 | 0.5526 | 0.5294 | 0.5306 | 0.5294 | 0.5259 | 0.6572 | 0.6779 | 0.6955 | 0.5982 | 0.7015 | 0.5641 | 0.3261 | 0.6104 | 0.5366 | 0.4412 | 0.6528 | 0.5500 | 0.3750 | [[47, 8, 22], [10, 22, 9], [10, 9, 15]] |
| test02 | test02_fedavg_2h | fedavg | 2_clients_balanced | hospital_2 | ok |  | 611 | 4,849 | 152 | 1,198 | 77 | 41 | 34 | 0.5066 | 0.4342 | 0.4122 | 0.4103 | 0.4122 | 0.4078 | 0.5785 | 0.6149 | 0.5559 | 0.5648 | 0.6032 | 0.3200 | 0.3077 | 0.4935 | 0.3902 | 0.3529 | 0.5429 | 0.3516 | 0.3288 | [[38, 22, 17], [15, 16, 10], [10, 12, 12]] |
| test03 | test03_fedprox_2h | fedprox | 2_clients_balanced | hospital_1 | ok |  | 612 | 4,870 | 152 | 1,214 | 77 | 41 | 34 | 0.5066 | 0.4868 | 0.4583 | 0.4577 | 0.4583 | 0.4545 | 0.6498 | 0.6644 | 0.6950 | 0.5900 | 0.6719 | 0.4222 | 0.2791 | 0.5584 | 0.4634 | 0.3529 | 0.6099 | 0.4419 | 0.3117 | [[43, 15, 19], [10, 19, 12], [11, 11, 12]] |
| test03 | test03_fedprox_2h | fedprox | 2_clients_balanced | hospital_2 | ok |  | 611 | 4,849 | 152 | 1,198 | 77 | 41 | 34 | 0.5066 | 0.4539 | 0.4126 | 0.4125 | 0.4126 | 0.4117 | 0.5746 | 0.6386 | 0.5489 | 0.5364 | 0.6143 | 0.3864 | 0.2368 | 0.5584 | 0.4146 | 0.2647 | 0.5850 | 0.4000 | 0.2500 | [[43, 15, 19], [14, 17, 10], [13, 12, 9]] |
| test04 | test04_fedavg_3h | fedavg | 3_clients_balanced | hospital_1 | ok |  | 408 | 3,245 | 102 | 816 | 52 | 27 | 23 | 0.5098 | 0.4510 | 0.4108 | 0.4096 | 0.4108 | 0.4100 | 0.6307 | 0.6204 | 0.6311 | 0.6406 | 0.5800 | 0.3571 | 0.2917 | 0.5577 | 0.3704 | 0.3043 | 0.5686 | 0.3636 | 0.2979 | [[29, 12, 11], [11, 10, 6], [10, 6, 7]] |
| test04 | test04_fedavg_3h | fedavg | 3_clients_balanced | hospital_2 | ok |  | 408 | 3,249 | 102 | 814 | 52 | 27 | 23 | 0.5098 | 0.4510 | 0.4286 | 0.4210 | 0.4286 | 0.4231 | 0.6381 | 0.6158 | 0.6558 | 0.6428 | 0.5652 | 0.4062 | 0.2917 | 0.5000 | 0.4815 | 0.3043 | 0.5306 | 0.4407 | 0.2979 | [[26, 13, 13], [10, 13, 4], [10, 6, 7]] |
| test04 | test04_fedavg_3h | fedavg | 3_clients_balanced | hospital_3 | ok |  | 406 | 3,206 | 101 | 801 | 51 | 27 | 23 | 0.5050 | 0.5347 | 0.4762 | 0.4874 | 0.4762 | 0.4761 | 0.5986 | 0.6231 | 0.5886 | 0.5842 | 0.6140 | 0.4483 | 0.4000 | 0.6863 | 0.4815 | 0.2609 | 0.6481 | 0.4643 | 0.3158 | [[35, 9, 7], [12, 13, 2], [10, 7, 6]] |
| test05 | test05_fedprox_3h | fedprox | 3_clients_balanced | hospital_1 | ok |  | 408 | 3,245 | 102 | 816 | 52 | 27 | 23 | 0.5098 | 0.4608 | 0.4377 | 0.4275 | 0.4377 | 0.4295 | 0.6529 | 0.6362 | 0.6430 | 0.6797 | 0.5957 | 0.3200 | 0.3667 | 0.5385 | 0.2963 | 0.4783 | 0.5657 | 0.3077 | 0.4151 | [[28, 12, 12], [12, 8, 7], [7, 5, 11]] |
| test05 | test05_fedprox_3h | fedprox | 3_clients_balanced | hospital_2 | ok |  | 408 | 3,249 | 102 | 814 | 52 | 27 | 23 | 0.5098 | 0.4510 | 0.4308 | 0.4378 | 0.4308 | 0.4296 | 0.6189 | 0.6400 | 0.6459 | 0.5707 | 0.5909 | 0.4800 | 0.2424 | 0.5000 | 0.4444 | 0.3478 | 0.5417 | 0.4615 | 0.2857 | [[26, 9, 17], [7, 12, 8], [11, 4, 8]] |
| test05 | test05_fedprox_3h | fedprox | 3_clients_balanced | hospital_3 | ok |  | 406 | 3,206 | 101 | 801 | 51 | 27 | 23 | 0.5050 | 0.5149 | 0.4573 | 0.4616 | 0.4573 | 0.4575 | 0.6307 | 0.6471 | 0.6446 | 0.6003 | 0.6071 | 0.4444 | 0.3333 | 0.6667 | 0.4444 | 0.2609 | 0.6355 | 0.4444 | 0.2927 | [[34, 9, 8], [11, 12, 4], [11, 6, 6]] |
| test06 | test06_fedavg_4h | fedavg | 4_clients_balanced | hospital_1 | ok |  | 306 | 2,439 | 77 | 614 | 39 | 21 | 17 | 0.5065 | 0.4805 | 0.4191 | 0.4253 | 0.4191 | 0.4208 | 0.6101 | 0.5972 | 0.6556 | 0.5775 | 0.5814 | 0.4444 | 0.2500 | 0.6410 | 0.3810 | 0.2353 | 0.6098 | 0.4103 | 0.2424 | [[25, 5, 9], [10, 8, 3], [8, 5, 4]] |
| test06 | test06_fedavg_4h | fedavg | 4_clients_balanced | hospital_2 | ok |  | 305 | 2,423 | 77 | 616 | 39 | 21 | 17 | 0.5065 | 0.4156 | 0.3433 | 0.3411 | 0.3433 | 0.3412 | 0.5560 | 0.5493 | 0.5578 | 0.5608 | 0.5581 | 0.2778 | 0.1875 | 0.6154 | 0.2381 | 0.1765 | 0.5854 | 0.2564 | 0.1818 | [[24, 7, 8], [11, 5, 5], [8, 6, 3]] |
| test06 | test06_fedavg_4h | fedavg | 4_clients_balanced | hospital_3 | ok |  | 305 | 2,426 | 76 | 608 | 39 | 20 | 17 | 0.5132 | 0.5395 | 0.4515 | 0.4711 | 0.4515 | 0.4521 | 0.7026 | 0.7166 | 0.6705 | 0.7208 | 0.6122 | 0.4375 | 0.3636 | 0.7692 | 0.3500 | 0.2353 | 0.6818 | 0.3889 | 0.2857 | [[30, 4, 5], [11, 7, 2], [8, 5, 4]] |
| test06 | test06_fedavg_4h | fedavg | 4_clients_balanced | hospital_4 | ok |  | 305 | 2,399 | 76 | 606 | 39 | 20 | 17 | 0.5132 | 0.4868 | 0.4254 | 0.4297 | 0.4254 | 0.4255 | 0.5992 | 0.5606 | 0.6518 | 0.5852 | 0.5814 | 0.4000 | 0.3077 | 0.6410 | 0.4000 | 0.2353 | 0.6098 | 0.4000 | 0.2667 | [[25, 7, 7], [10, 8, 2], [8, 5, 4]] |
| test07 | test07_fedprox_4h | fedprox | 4_clients_balanced | hospital_1 | ok |  | 306 | 2,439 | 77 | 614 | 39 | 21 | 17 | 0.5065 | 0.4805 | 0.4412 | 0.4415 | 0.4412 | 0.4398 | 0.6433 | 0.6242 | 0.7185 | 0.5873 | 0.6389 | 0.4000 | 0.2857 | 0.5897 | 0.3810 | 0.3529 | 0.6133 | 0.3902 | 0.3158 | [[23, 5, 11], [9, 8, 4], [4, 7, 6]] |
| test07 | test07_fedprox_4h | fedprox | 4_clients_balanced | hospital_2 | ok |  | 305 | 2,423 | 77 | 616 | 39 | 21 | 17 | 0.5065 | 0.4416 | 0.3972 | 0.3977 | 0.3972 | 0.3971 | 0.5838 | 0.6181 | 0.5587 | 0.5745 | 0.5946 | 0.3043 | 0.2941 | 0.5641 | 0.3333 | 0.2941 | 0.5789 | 0.3182 | 0.2941 | [[22, 11, 6], [8, 7, 6], [7, 5, 5]] |
| test07 | test07_fedprox_4h | fedprox | 4_clients_balanced | hospital_3 | ok |  | 305 | 2,426 | 76 | 608 | 39 | 20 | 17 | 0.5132 | 0.4737 | 0.3837 | 0.3641 | 0.3837 | 0.3726 | 0.6338 | 0.6625 | 0.6348 | 0.6042 | 0.6279 | 0.3810 | 0.0833 | 0.6923 | 0.4000 | 0.0588 | 0.6585 | 0.3902 | 0.0690 | [[27, 5, 7], [8, 8, 4], [8, 8, 1]] |
| test07 | test07_fedprox_4h | fedprox | 4_clients_balanced | hospital_4 | ok |  | 305 | 2,399 | 76 | 606 | 39 | 20 | 17 | 0.5132 | 0.4211 | 0.3967 | 0.3965 | 0.3967 | 0.3943 | 0.5785 | 0.5184 | 0.6437 | 0.5733 | 0.5278 | 0.3889 | 0.2727 | 0.4872 | 0.3500 | 0.3529 | 0.5067 | 0.3684 | 0.3077 | [[19, 7, 13], [10, 7, 3], [7, 4, 6]] |
| test08 | test08_fedavg_skewed | fedavg | 4_clients_skewed | hospital_1 | ok |  | 678 | 5,395 | 170 | 1,358 | 86 | 46 | 38 | 0.5059 | 0.5118 | 0.4720 | 0.4674 | 0.4720 | 0.4682 | 0.6383 | 0.6525 | 0.6602 | 0.6021 | 0.6341 | 0.4444 | 0.3235 | 0.6047 | 0.5217 | 0.2895 | 0.6190 | 0.4800 | 0.3056 | [[52, 20, 14], [13, 24, 9], [17, 10, 11]] |
| test08 | test08_fedavg_skewed | fedavg | 4_clients_skewed | hospital_2 | ok |  | 273 | 2,171 | 67 | 534 | 34 | 18 | 15 | 0.5075 | 0.5373 | 0.5196 | 0.5066 | 0.5196 | 0.5062 | 0.7527 | 0.7077 | 0.7619 | 0.7885 | 0.6552 | 0.4800 | 0.3846 | 0.5588 | 0.6667 | 0.3333 | 0.6032 | 0.5581 | 0.3571 | [[19, 10, 5], [3, 12, 3], [7, 3, 5]] |
| test08 | test08_fedavg_skewed | fedavg | 4_clients_skewed | hospital_3 | ok |  | 136 | 1,073 | 34 | 266 | 17 | 9 | 8 | 0.5000 | 0.4412 | 0.4126 | 0.4327 | 0.4126 | 0.4159 | 0.5225 | 0.5640 | 0.4267 | 0.5769 | 0.6923 | 0.2308 | 0.3750 | 0.5294 | 0.3333 | 0.3750 | 0.6000 | 0.2727 | 0.3750 | [[9, 6, 2], [3, 3, 3], [1, 4, 3]] |
| test08 | test08_fedavg_skewed | fedavg | 4_clients_skewed | hospital_4 | ok |  | 135 | 1,066 | 34 | 268 | 17 | 9 | 8 | 0.5000 | 0.5882 | 0.5583 | 0.5443 | 0.5583 | 0.5502 | 0.6698 | 0.6713 | 0.8044 | 0.5337 | 0.6471 | 0.7000 | 0.2857 | 0.6471 | 0.7778 | 0.2500 | 0.6471 | 0.7368 | 0.2667 | [[11, 1, 5], [2, 7, 0], [4, 2, 2]] |
| test09 | test09_fedprox_skewed | fedprox | 4_clients_skewed | hospital_1 | ok |  | 678 | 5,395 | 170 | 1,358 | 86 | 46 | 38 | 0.5059 | 0.5118 | 0.4569 | 0.4561 | 0.4569 | 0.4564 | 0.6333 | 0.6521 | 0.6662 | 0.5815 | 0.6437 | 0.4468 | 0.2778 | 0.6512 | 0.4565 | 0.2632 | 0.6474 | 0.4516 | 0.2703 | [[56, 15, 15], [14, 21, 11], [17, 11, 10]] |
| test09 | test09_fedprox_skewed | fedprox | 4_clients_skewed | hospital_2 | ok |  | 273 | 2,171 | 67 | 534 | 34 | 18 | 15 | 0.5075 | 0.5970 | 0.5588 | 0.5496 | 0.5588 | 0.5514 | 0.7496 | 0.7380 | 0.7789 | 0.7321 | 0.7188 | 0.5455 | 0.3846 | 0.6765 | 0.6667 | 0.3333 | 0.6970 | 0.6000 | 0.3571 | [[23, 6, 5], [3, 12, 3], [6, 4, 5]] |
| test09 | test09_fedprox_skewed | fedprox | 4_clients_skewed | hospital_3 | ok |  | 136 | 1,073 | 34 | 266 | 17 | 9 | 8 | 0.5000 | 0.4706 | 0.4276 | 0.4306 | 0.4276 | 0.4242 | 0.5643 | 0.5813 | 0.5156 | 0.5962 | 0.6250 | 0.3333 | 0.3333 | 0.5882 | 0.4444 | 0.2500 | 0.6061 | 0.3810 | 0.2857 | [[10, 5, 2], [3, 4, 2], [3, 3, 2]] |
| test09 | test09_fedprox_skewed | fedprox | 4_clients_skewed | hospital_4 | ok |  | 135 | 1,066 | 34 | 268 | 17 | 9 | 8 | 0.5000 | 0.5000 | 0.4600 | 0.4500 | 0.4600 | 0.4542 | 0.6252 | 0.6055 | 0.8133 | 0.4567 | 0.6250 | 0.6000 | 0.1250 | 0.5882 | 0.6667 | 0.1250 | 0.6061 | 0.6316 | 0.1250 | [[10, 1, 6], [2, 6, 1], [4, 3, 1]] |
| test10 | test10_fedavg_cohort | fedavg | 3_clients_cohort | hospital_1 | ok |  | 514 | 4,067 | 128 | 1,007 | 85 | 20 | 23 | 0.6641 | 0.4531 | 0.4183 | 0.4283 | 0.4183 | 0.3875 | 0.6168 | 0.6120 | 0.5593 | 0.6791 | 0.7500 | 0.1818 | 0.3529 | 0.4941 | 0.5000 | 0.2609 | 0.5957 | 0.2667 | 0.3000 | [[42, 35, 8], [7, 10, 3], [7, 10, 6]] |
| test10 | test10_fedavg_cohort | fedavg | 3_clients_cohort | hospital_2 | ok |  | 82 | 652 | 19 | 152 | 8 | 5 | 6 | 0.4211 | 0.3684 | 0.3167 | 0.2205 | 0.3167 | 0.2571 | 0.4837 | 0.6250 | 0.4286 | 0.3974 | 0.4615 | 0.2000 | 0.0000 | 0.7500 | 0.2000 | 0.0000 | 0.5714 | 0.2000 | 0.0000 | [[6, 2, 0], [3, 1, 1], [4, 2, 0]] |
| test10 | test10_fedavg_cohort | fedavg | 3_clients_cohort | hospital_3 | ok |  | 627 | 5,005 | 157 | 1,248 | 61 | 56 | 40 | 0.3885 | 0.4522 | 0.4255 | 0.4444 | 0.4255 | 0.4214 | 0.5802 | 0.5458 | 0.6054 | 0.5895 | 0.4419 | 0.5000 | 0.3913 | 0.6230 | 0.4286 | 0.2250 | 0.5170 | 0.4615 | 0.2857 | [[38, 14, 9], [27, 24, 5], [21, 10, 9]] |
| test12 | test12_fedavg_sizematched | fedavg | 3_clients_sizematched | hospital_1 | ok |  | 514 | 4,096 | 128 | 1,020 | 65 | 34 | 29 | 0.5078 | 0.4297 | 0.3955 | 0.3982 | 0.3955 | 0.3938 | 0.6343 | 0.6911 | 0.6367 | 0.5751 | 0.6296 | 0.3077 | 0.2571 | 0.5231 | 0.3529 | 0.3103 | 0.5714 | 0.3288 | 0.2812 | [[34, 16, 15], [11, 12, 11], [9, 11, 9]] |
| test12 | test12_fedavg_sizematched | fedavg | 3_clients_sizematched | hospital_2 | ok |  | 81 | 640 | 20 | 160 | 10 | 5 | 5 | 0.5000 | 0.6500 | 0.6667 | 0.6444 | 0.6667 | 0.6530 | 0.7422 | 0.6800 | 0.8000 | 0.7467 | 0.6667 | 0.6667 | 0.6000 | 0.6000 | 0.8000 | 0.6000 | 0.6316 | 0.7273 | 0.6000 | [[6, 2, 2], [1, 4, 0], [2, 0, 3]] |
| test12 | test12_fedavg_sizematched | fedavg | 3_clients_sizematched | hospital_3 | ok |  | 628 | 4,976 | 156 | 1,239 | 79 | 42 | 35 | 0.5064 | 0.4359 | 0.4047 | 0.4032 | 0.4047 | 0.4023 | 0.5908 | 0.5981 | 0.6122 | 0.5622 | 0.5942 | 0.3404 | 0.2750 | 0.5190 | 0.3810 | 0.3143 | 0.5541 | 0.3596 | 0.2933 | [[41, 21, 17], [14, 16, 12], [14, 10, 11]] |
| test11 | test11_fedprox_cohort | fedprox | 3_clients_cohort | hospital_1 | ok |  | 514 | 4,067 | 128 | 1,007 | 85 | 20 | 23 | 0.6641 | 0.4375 | 0.3890 | 0.3769 | 0.3890 | 0.3637 | 0.6051 | 0.6140 | 0.5574 | 0.6439 | 0.7193 | 0.1613 | 0.2500 | 0.4824 | 0.2500 | 0.4348 | 0.5775 | 0.1961 | 0.3175 | [[41, 21, 23], [8, 5, 7], [8, 5, 10]] |
| test11 | test11_fedprox_cohort | fedprox | 3_clients_cohort | hospital_2 | ok |  | 82 | 652 | 19 | 152 | 8 | 5 | 6 | 0.4211 | 0.5263 | 0.4972 | 0.4868 | 0.4972 | 0.4845 | 0.6274 | 0.7386 | 0.4000 | 0.7436 | 0.5556 | 0.3333 | 0.5714 | 0.6250 | 0.2000 | 0.6667 | 0.5882 | 0.2500 | 0.6154 | [[5, 2, 1], [2, 1, 2], [2, 0, 4]] |
| test11 | test11_fedprox_cohort | fedprox | 3_clients_cohort | hospital_3 | ok |  | 627 | 5,005 | 157 | 1,248 | 61 | 56 | 40 | 0.3885 | 0.4076 | 0.3834 | 0.3904 | 0.3834 | 0.3728 | 0.5841 | 0.5540 | 0.6160 | 0.5823 | 0.4286 | 0.4211 | 0.3214 | 0.6393 | 0.2857 | 0.2250 | 0.5132 | 0.3404 | 0.2647 | [[39, 12, 10], [31, 16, 9], [21, 10, 9]] |
| test13 | test13_fedprox_sizematched | fedprox | 3_clients_sizematched | hospital_1 | ok |  | 514 | 4,096 | 128 | 1,020 | 65 | 34 | 29 | 0.5078 | 0.4922 | 0.4412 | 0.4412 | 0.4412 | 0.4411 | 0.6143 | 0.6642 | 0.6180 | 0.5608 | 0.6308 | 0.3714 | 0.3214 | 0.6308 | 0.3824 | 0.3103 | 0.6308 | 0.3768 | 0.3158 | [[41, 13, 11], [13, 13, 8], [11, 9, 9]] |
| test13 | test13_fedprox_sizematched | fedprox | 3_clients_sizematched | hospital_2 | ok |  | 81 | 640 | 20 | 160 | 10 | 5 | 5 | 0.5000 | 0.7500 | 0.7333 | 0.7389 | 0.7333 | 0.7313 | 0.7778 | 0.8000 | 0.8267 | 0.7067 | 0.8000 | 0.6667 | 0.7500 | 0.8000 | 0.8000 | 0.6000 | 0.8000 | 0.7273 | 0.6667 | [[8, 1, 1], [1, 4, 0], [1, 1, 3]] |
| test13 | test13_fedprox_sizematched | fedprox | 3_clients_sizematched | hospital_3 | ok |  | 628 | 4,976 | 156 | 1,239 | 79 | 42 | 35 | 0.5064 | 0.4167 | 0.3417 | 0.3368 | 0.3417 | 0.3377 | 0.5995 | 0.6332 | 0.5950 | 0.5702 | 0.5568 | 0.2683 | 0.1852 | 0.6203 | 0.2619 | 0.1429 | 0.5868 | 0.2651 | 0.1613 | [[49, 18, 12], [21, 11, 10], [18, 12, 5]] |

## Data

These tables come from the prepared dataset, not from any run, so they are correct regardless of how many experiments have finished.

### Partitions

| partition | label | n_clients | ratio | fractions | stratified | total_patients | built | seed |
|---|---|---|---|---|---|---|---|---|
| 2_clients_balanced | 2 hospitals, balanced (50/50) | 2 | 1 : 1 | 50.0% / 50.0% | yes | 1,527 | 2026-08-05T14:00:05+00:00 | 42 |
| 3_clients_balanced | 3 hospitals, balanced (33.3 each) | 3 | 1 : 1 : 1 | 33.3% / 33.3% / 33.3% | yes | 1,527 | 2026-08-05T14:00:11+00:00 | 42 |
| 4_clients_balanced | 4 hospitals, balanced (25 each) | 4 | 1 : 1 : 1 : 1 | 25.0% / 25.0% / 25.0% / 25.0% | yes | 1,527 | 2026-08-05T14:00:17+00:00 | 42 |
| 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10) | 4 | 5 : 2 : 1 : 1 | 55.6% / 22.2% / 11.1% / 11.1% | yes | 1,527 | 2026-08-05T14:00:23+00:00 | 42 |
| 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2) | 3 | 642 : 101 : 784 | 42.0% / 6.6% / 51.3% | no | 1,527 | 2026-08-05T14:00:35+00:00 | 42 |
| 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort | 3 | 642 : 101 : 784 | 42.0% / 6.6% / 51.3% | yes | 1,527 | 2026-08-05T14:00:29+00:00 | 42 |

### Per-hospital and global splits

| scope | partition | site | split | patients | images | trivial_baseline | patients_HRposHER2neg | patients_TripleNeg | patients_HER2pos | pct_HRposHER2neg | pct_TripleNeg | pct_HER2pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global | - | global_test | test | 268 | 2,115 | 0.5112 | 137 | 78 | 53 |  |  |  |
| global | - | global_val | val | 268 | 2,132 | 0.4925 | 132 | 76 | 60 |  |  |  |
| site | 2_clients_balanced | hospital_1 | train | 612 | 4,870 |  | 310 | 164 | 138 | 50.7000 | 26.8000 | 22.5000 |
| site | 2_clients_balanced | hospital_1 | val | 152 | 1,214 |  | 77 | 41 | 34 | 50.7000 | 27.0000 | 22.4000 |
| site | 2_clients_balanced | hospital_2 | train | 611 | 4,849 |  | 309 | 164 | 138 | 50.6000 | 26.8000 | 22.6000 |
| site | 2_clients_balanced | hospital_2 | val | 152 | 1,198 |  | 77 | 41 | 34 | 50.7000 | 27.0000 | 22.4000 |
| site | 3_clients_balanced | hospital_1 | train | 408 | 3,245 |  | 206 | 110 | 92 | 50.5000 | 27.0000 | 22.5000 |
| site | 3_clients_balanced | hospital_1 | val | 102 | 816 |  | 52 | 27 | 23 | 51.0000 | 26.5000 | 22.5000 |
| site | 3_clients_balanced | hospital_2 | train | 408 | 3,249 |  | 206 | 110 | 92 | 50.5000 | 27.0000 | 22.5000 |
| site | 3_clients_balanced | hospital_2 | val | 102 | 814 |  | 52 | 27 | 23 | 51.0000 | 26.5000 | 22.5000 |
| site | 3_clients_balanced | hospital_3 | train | 406 | 3,206 |  | 206 | 109 | 91 | 50.7000 | 26.8000 | 22.4000 |
| site | 3_clients_balanced | hospital_3 | val | 101 | 801 |  | 51 | 27 | 23 | 50.5000 | 26.7000 | 22.8000 |
| site | 4_clients_balanced | hospital_1 | train | 306 | 2,439 |  | 155 | 82 | 69 | 50.7000 | 26.8000 | 22.5000 |
| site | 4_clients_balanced | hospital_1 | val | 77 | 614 |  | 39 | 21 | 17 | 50.6000 | 27.3000 | 22.1000 |
| site | 4_clients_balanced | hospital_2 | train | 305 | 2,423 |  | 154 | 82 | 69 | 50.5000 | 26.9000 | 22.6000 |
| site | 4_clients_balanced | hospital_2 | val | 77 | 616 |  | 39 | 21 | 17 | 50.6000 | 27.3000 | 22.1000 |
| site | 4_clients_balanced | hospital_3 | train | 305 | 2,426 |  | 154 | 82 | 69 | 50.5000 | 26.9000 | 22.6000 |
| site | 4_clients_balanced | hospital_3 | val | 76 | 608 |  | 39 | 20 | 17 | 51.3000 | 26.3000 | 22.4000 |
| site | 4_clients_balanced | hospital_4 | train | 305 | 2,399 |  | 154 | 82 | 69 | 50.5000 | 26.9000 | 22.6000 |
| site | 4_clients_balanced | hospital_4 | val | 76 | 606 |  | 39 | 20 | 17 | 51.3000 | 26.3000 | 22.4000 |
| site | 4_clients_skewed | hospital_1 | train | 678 | 5,395 |  | 343 | 182 | 153 | 50.6000 | 26.8000 | 22.6000 |
| site | 4_clients_skewed | hospital_1 | val | 170 | 1,358 |  | 86 | 46 | 38 | 50.6000 | 27.1000 | 22.4000 |
| site | 4_clients_skewed | hospital_2 | train | 273 | 2,171 |  | 138 | 73 | 62 | 50.5000 | 26.7000 | 22.7000 |
| site | 4_clients_skewed | hospital_2 | val | 67 | 534 |  | 34 | 18 | 15 | 50.7000 | 26.9000 | 22.4000 |
| site | 4_clients_skewed | hospital_3 | train | 136 | 1,073 |  | 69 | 37 | 30 | 50.7000 | 27.2000 | 22.1000 |
| site | 4_clients_skewed | hospital_3 | val | 34 | 266 |  | 17 | 9 | 8 | 50.0000 | 26.5000 | 23.5000 |
| site | 4_clients_skewed | hospital_4 | train | 135 | 1,066 |  | 69 | 36 | 30 | 51.1000 | 26.7000 | 22.2000 |
| site | 4_clients_skewed | hospital_4 | val | 34 | 268 |  | 17 | 9 | 8 | 50.0000 | 26.5000 | 23.5000 |
| site | 3_clients_cohort | hospital_1 | train | 514 | 4,067 |  | 341 | 82 | 91 | 66.3000 | 16.0000 | 17.7000 |
| site | 3_clients_cohort | hospital_1 | val | 128 | 1,007 |  | 85 | 20 | 23 | 66.4000 | 15.6000 | 18.0000 |
| site | 3_clients_cohort | hospital_2 | train | 82 | 652 |  | 34 | 22 | 26 | 41.5000 | 26.8000 | 31.7000 |
| site | 3_clients_cohort | hospital_2 | val | 19 | 152 |  | 8 | 5 | 6 | 42.1000 | 26.3000 | 31.6000 |
| site | 3_clients_cohort | hospital_3 | train | 627 | 5,005 |  | 244 | 225 | 158 | 38.9000 | 35.9000 | 25.2000 |
| site | 3_clients_cohort | hospital_3 | val | 157 | 1,248 |  | 61 | 56 | 40 | 38.9000 | 35.7000 | 25.5000 |
| site | 3_clients_sizematched | hospital_1 | train | 514 | 4,096 |  | 260 | 138 | 116 | 50.6000 | 26.8000 | 22.6000 |
| site | 3_clients_sizematched | hospital_1 | val | 128 | 1,020 |  | 65 | 34 | 29 | 50.8000 | 26.6000 | 22.7000 |
| site | 3_clients_sizematched | hospital_2 | train | 81 | 640 |  | 41 | 22 | 18 | 50.6000 | 27.2000 | 22.2000 |
| site | 3_clients_sizematched | hospital_2 | val | 20 | 160 |  | 10 | 5 | 5 | 50.0000 | 25.0000 | 25.0000 |
| site | 3_clients_sizematched | hospital_3 | train | 628 | 4,976 |  | 318 | 169 | 141 | 50.6000 | 26.9000 | 22.5000 |
| site | 3_clients_sizematched | hospital_3 | val | 156 | 1,239 |  | 79 | 42 | 35 | 50.6000 | 26.9000 | 22.4000 |

## Experiment status

```
[done] test01  test01_centralized                  macro-AUC 0.6068
[done] test02  test02_fedavg_2h                    macro-AUC 0.5816
[done] test03  test03_fedprox_2h                   macro-AUC 0.5917
[done] test04  test04_fedavg_3h                    macro-AUC 0.5990
[done] test05  test05_fedprox_3h                   macro-AUC 0.5958
[done] test06  test06_fedavg_4h                    macro-AUC 0.6077
[done] test07  test07_fedprox_4h                   macro-AUC 0.6075
[done] test08  test08_fedavg_skewed                macro-AUC 0.5982
[done] test09  test09_fedprox_skewed               macro-AUC 0.6152
[done] test10  test10_fedavg_cohort                macro-AUC 0.5426
[done] test12  test12_fedavg_sizematched           macro-AUC 0.5836
[done] test11  test11_fedprox_cohort               macro-AUC 0.5678
[done] test13  test13_fedprox_sizematched          macro-AUC 0.5882
```
