# Final results summary

Generated 2026-08-05 10:04 UTC by `scripts/build_final_summary.py`.

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
| test02 | fedavg | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 42 |  | 25 | 313 | 0.5112 | 0.4030 | 0.3742 | 0.3915 | 0.3742 | 0.3744 | 0.5594 |
| test03 | fedprox | 2 | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 42 |  | 29 | 351 | 0.5112 | 0.4328 | 0.4025 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |
| test04 | fedavg | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 42 |  | 27 | 313 | 0.5112 | 0.4851 | 0.4198 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |
| test05 | fedprox | 3 | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 42 |  | 28 | 333 | 0.5112 | 0.4590 | 0.4127 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |
| test06 | fedavg | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 42 |  | 0 | 317 | 0.5112 | 0.4776 | 0.4522 | 0.4646 | 0.4522 | 0.4378 | 0.6531 |
| test07 | fedprox | 4 | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 42 |  | 0 | 335 | 0.5112 | 0.4739 | 0.4393 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |
| test08 | fedavg | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 42 |  | 21 | 323 | 0.5112 | 0.4888 | 0.4259 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |
| test09 | fedprox | 4 | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 42 |  | 2 | 387 | 0.5112 | 0.4515 | 0.4210 | 0.4365 | 0.4210 | 0.4197 | 0.6250 |
| test10 | fedavg | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 42 |  | 7 | 469 | 0.5112 | 0.4291 | 0.3582 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |
| test12 | fedavg | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 42 |  | 17 | 471 | 0.5112 | 0.4478 | 0.4183 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |
| test11 | fedprox | 3 | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 42 |  | 16 | 482 | 0.5112 | 0.4590 | 0.4105 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |
| test13 | fedprox | 3 | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 42 |  | 27 | 485 | 0.5112 | 0.4664 | 0.3885 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |

## Comparisons

### centralized vs fedavg

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4030 | 0.3915 | 0.3742 | 0.3744 | 0.5594 |  | 25 | 313 | centralized (test01) | 0.6068 | -0.0474 | yes |
| test04 | test04_fedavg_3h | fedavg | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4851 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |  | 27 | 313 | centralized (test01) | 0.6068 | -0.0078 | yes |
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4776 | 0.4646 | 0.4522 | 0.4378 | 0.6531 |  | 0 | 317 | centralized (test01) | 0.6068 | 0.0463 | yes |
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
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4515 | 0.4365 | 0.4210 | 0.4197 | 0.6250 |  | 2 | 387 | centralized (test01) | 0.6068 | 0.0182 | yes |
| test11 | test11_fedprox_cohort | fedprox | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4590 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |  | 16 | 482 | centralized (test01) | 0.6068 | -0.0390 | yes |
| test13 | test13_fedprox_sizematched | fedprox | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4664 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |  | 27 | 485 | centralized (test01) | 0.6068 | -0.0186 | yes |

### fedavg vs fedprox

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4030 | 0.3915 | 0.3742 | 0.3744 | 0.5594 |  | 25 | 313 | FedAvg (mean over configurations) | 0.5893 | -0.0299 | yes |
| test04 | test04_fedavg_3h | fedavg | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4851 | 0.4333 | 0.4198 | 0.4231 | 0.5990 |  | 27 | 313 | FedAvg (mean over configurations) | 0.5893 | 0.0097 | yes |
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4776 | 0.4646 | 0.4522 | 0.4378 | 0.6531 |  | 0 | 317 | FedAvg (mean over configurations) | 0.5893 | 0.0638 | yes |
| test08 | test08_fedavg_skewed | fedavg | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4888 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |  | 21 | 323 | FedAvg (mean over configurations) | 0.5893 | 0.0089 | yes |
| test10 | test10_fedavg_cohort | fedavg | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4291 | 0.3523 | 0.3582 | 0.3536 | 0.5426 |  | 7 | 469 | FedAvg (mean over configurations) | 0.5893 | -0.0467 | yes |
| test12 | test12_fedavg_sizematched | fedavg | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4478 | 0.4187 | 0.4183 | 0.4153 | 0.5836 |  | 17 | 471 | FedAvg (mean over configurations) | 0.5893 | -0.0057 | yes |
| test03 | test03_fedprox_2h | fedprox | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4328 | 0.4028 | 0.4025 | 0.4001 | 0.5917 |  | 29 | 351 | FedAvg (mean over configurations) | 0.5893 | 0.0024 | yes |
| test05 | test05_fedprox_3h | fedprox | 3 | 3_clients_balanced | 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 0.4590 | 0.4208 | 0.4127 | 0.4116 | 0.5958 |  | 28 | 333 | FedAvg (mean over configurations) | 0.5893 | 0.0065 | yes |
| test07 | test07_fedprox_4h | fedprox | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4739 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |  | 0 | 335 | FedAvg (mean over configurations) | 0.5893 | 0.0182 | yes |
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4515 | 0.4365 | 0.4210 | 0.4197 | 0.6250 |  | 2 | 387 | FedAvg (mean over configurations) | 0.5893 | 0.0357 | yes |
| test11 | test11_fedprox_cohort | fedprox | 3 | 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 0.4590 | 0.4302 | 0.4105 | 0.4144 | 0.5678 |  | 16 | 482 | FedAvg (mean over configurations) | 0.5893 | -0.0215 | yes |
| test13 | test13_fedprox_sizematched | fedprox | 3 | 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 0.4664 | 0.3925 | 0.3885 | 0.3884 | 0.5882 |  | 27 | 485 | FedAvg (mean over configurations) | 0.5893 | -0.0011 | yes |

### fedavg vs fedprox paired

| data_split | n_hospitals | fedavg | fedprox | fedavg_accuracy | fedprox_accuracy | delta_accuracy | fedavg_macro_precision | fedprox_macro_precision | fedavg_macro_recall | fedprox_macro_recall | fedavg_macro_f1 | fedprox_macro_f1 | delta_macro_f1 | fedavg_macro_auc | fedprox_macro_auc | delta_macro_auc | fedavg_best_round | fedprox_best_round | fedavg_training_time_s | fedprox_training_time_s | within_noise_floor | favours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 2 | test02 | test03 | 0.4030 | 0.4328 | 0.0298 | 0.3915 | 0.4028 | 0.3742 | 0.4025 | 0.3744 | 0.4001 | 0.0257 | 0.5594 | 0.5917 | 0.0323 | 25 | 29 | 313 | 351 | yes | — |
| 3 hospitals, balanced (33.3 each)  ->  33.3% / 33.3% / 33.3% | 3 | test04 | test05 | 0.4851 | 0.4590 | -0.0261 | 0.4333 | 0.4208 | 0.4198 | 0.4127 | 0.4231 | 0.4116 | -0.0115 | 0.5990 | 0.5958 | -0.0032 | 27 | 28 | 313 | 333 | yes | — |
| 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort  ->  42.0% / 6.6% / 51.3% | 3 | test12 | test13 | 0.4478 | 0.4664 | 0.0186 | 0.4187 | 0.3925 | 0.4183 | 0.3885 | 0.4153 | 0.3884 | -0.0269 | 0.5836 | 0.5882 | 0.0046 | 17 | 27 | 471 | 485 | yes | — |
| 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)  ->  42.0% / 6.6% / 51.3% | 3 | test10 | test11 | 0.4291 | 0.4590 | 0.0299 | 0.3523 | 0.4302 | 0.3582 | 0.4105 | 0.3536 | 0.4144 | 0.0608 | 0.5426 | 0.5678 | 0.0252 | 7 | 16 | 469 | 482 | yes | — |
| 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 4 | test06 | test07 | 0.4776 | 0.4739 | -0.0037 | 0.4646 | 0.4389 | 0.4522 | 0.4393 | 0.4378 | 0.4362 | -0.0016 | 0.6531 | 0.6075 | -0.0456 | 0 | 0 | 317 | 335 | yes | — |
| 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 4 | test08 | test09 | 0.4888 | 0.4515 | -0.0373 | 0.4374 | 0.4365 | 0.4259 | 0.4210 | 0.4292 | 0.4197 | -0.0095 | 0.5982 | 0.6250 | 0.0268 | 21 | 2 | 323 | 387 | yes | — |

### 2 hospitals

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test02 | test02_fedavg_2h | fedavg | 2 | 2_clients_balanced | 2 hospitals, balanced (50/50)  ->  50.0% / 50.0% | 0.4030 | 0.3915 | 0.3742 | 0.3744 | 0.5594 |  | 25 | 313 | centralized (test01) | 0.6068 | -0.0474 | yes |
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
| test06 | test06_fedavg_4h | fedavg | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4776 | 0.4646 | 0.4522 | 0.4378 | 0.6531 |  | 0 | 317 | centralized (test01) | 0.6068 | 0.0463 | yes |
| test07 | test07_fedprox_4h | fedprox | 4 | 4_clients_balanced | 4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0% | 0.4739 | 0.4389 | 0.4393 | 0.4362 | 0.6075 |  | 0 | 335 | centralized (test01) | 0.6068 | 0.0007 | yes |

### 4 hospitals skewed

| experiment | name | algorithm | n_hospitals | partition | data_split | accuracy | macro_precision | macro_recall | macro_f1 | macro_auc | best_epoch | best_round | training_time_s | reference | reference_macro_auc | delta_macro_auc | within_noise_floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test01 | test01_centralized | centralized | 1 | - | all data pooled | 0.5299 | 0.4606 | 0.4503 | 0.4523 | 0.6068 | 4 |  | 268 | centralized (test01) | 0.6068 | 0.0000 | yes |
| test08 | test08_fedavg_skewed | fedavg | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4888 | 0.4374 | 0.4259 | 0.4292 | 0.5982 |  | 21 | 323 | centralized (test01) | 0.6068 | -0.0086 | yes |
| test09 | test09_fedprox_skewed | fedprox | 4 | 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)  ->  55.6% / 22.2% / 11.1% / 11.1% | 0.4515 | 0.4365 | 0.4210 | 0.4197 | 0.6250 |  | 2 | 387 | centralized (test01) | 0.6068 | 0.0182 | yes |

## Per-hospital results

_not available until a federated experiment has produced a global model._

## Data

These tables come from the prepared dataset, not from any run, so they are correct regardless of how many experiments have finished.

### Partitions

| partition | label | n_clients | ratio | fractions | stratified | total_patients | built | seed |
|---|---|---|---|---|---|---|---|---|
| 2_clients_balanced | 2 hospitals, balanced (50/50) | 2 | 1 : 1 | 50.0% / 50.0% | yes | 1,527 | 2026-08-03T23:03:03+00:00 | 42 |
| 3_clients_balanced | 3 hospitals, balanced (33.3 each) | 3 | 1 : 1 : 1 | 33.3% / 33.3% / 33.3% | yes | 1,527 | 2026-08-03T23:03:11+00:00 | 42 |
| 4_clients_balanced | 4 hospitals, balanced (25 each) | 4 | 1 : 1 : 1 : 1 | 25.0% / 25.0% / 25.0% / 25.0% | yes | 1,527 | 2026-08-03T23:03:22+00:00 | 42 |
| 4_clients_skewed | 4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10) | 4 | 5 : 2 : 1 : 1 | 55.6% / 22.2% / 11.1% / 11.1% | yes | 1,527 | 2026-08-03T23:03:33+00:00 | 42 |
| 3_clients_cohort | 3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2) | 3 | 642 : 101 : 784 | 42.0% / 6.6% / 51.3% | no | 1,527 | 2026-08-05T00:52:03+00:00 | 42 |
| 3_clients_sizematched | 3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort | 3 | 642 : 101 : 784 | 42.0% / 6.6% / 51.3% | yes | 1,527 | 2026-08-05T00:52:19+00:00 | 42 |

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
[done] test02  test02_fedavg_2h                    macro-AUC 0.5594
[done] test03  test03_fedprox_2h                   macro-AUC 0.5917
[done] test04  test04_fedavg_3h                    macro-AUC 0.5990
[done] test05  test05_fedprox_3h                   macro-AUC 0.5958
[done] test06  test06_fedavg_4h                    macro-AUC 0.6531
[done] test07  test07_fedprox_4h                   macro-AUC 0.6075
[done] test08  test08_fedavg_skewed                macro-AUC 0.5982
[done] test09  test09_fedprox_skewed               macro-AUC 0.6250
[done] test10  test10_fedavg_cohort                macro-AUC 0.5426
[done] test12  test12_fedavg_sizematched           macro-AUC 0.5836
[done] test11  test11_fedprox_cohort               macro-AUC 0.5678
[done] test13  test13_fedprox_sizematched          macro-AUC 0.5882
```
