# Training and preprocessing parameters — complete reference

**Every value in this document was read from source code or from a run artefact.**
Nothing is inferred from library defaults or carried over from prose. Where the code and
the project's own prose disagree, the code wins and the disagreement is stated.

Verified 2026-08-04 against `src/federated/config/experiments.py`, `src/federated/common/training.py`,
`src/federated/common/data.py`, `src/core/training.py`,
`src/core/data.py`, `src/scripts/run_centralized.py`, the run logs in
`deployment/logs/`, and the recorded configs in
`results/classifier/_from_pod/`.

---

## 0. How to read this document — three distinct phases

The project ran two separate experimental campaigns with **different** hyperparameters.
Conflating them is the main way to misreport this work.

| phase | what it is | where the config lives | status |
|---|---|---|---|
| **PRELIMINARY** | The classifier phase — 21 runs, 13 architectures, 5 data configurations. Used to *choose* the configuration. | `src/dataset_config.py`; per-run `results/_from_pod/multi/<run>/config.json` | superseded; not the dissertation's numbers |
| **FINAL** | The dissertation campaign — test01 (centralised) + test02–09 (federated), 2026-08-03/04 | `src/federated/config/experiments.py` — **single source of truth** | **these are the reported results** |
| **PLANNED / NOT RUN** | Implemented and available, never executed | same file / `scripts/partition_data.py` | must not be reported as results |

**Three corrections to earlier summaries in this project, found while verifying:**

1. **Gradient clipping is used and had been omitted.** `clip_grad_norm_(…, max_norm=1.0)`
   is applied in both the shared trainer and the FedProx fork.
2. **The final campaign has no early-stopping mechanism at all.** `early_stopping` does
   not exist in `src/federated/config/experiments.py`, and `scripts/run_centralized.py`
   contains no such code — the loop runs all 30 epochs and tracks the best. The phrase
   "early stopping disabled (`early_stopping_patience = 0`)" in `docs/PROJECT_CONTEXT.md`
   describes the *classifier-phase* field, which is `100 epochs / patience 30`, not the
   federated code.
3. **`backbone_lr_scale = 0.1` was used in the preliminary phase and is absent from the
   final campaign.** The final arms apply one learning rate to all trainable parameters.

---

## 1. Main training parameters

`Used in` states the arm each value applies to. "Both" means the identical value, from the
same `TrainingConfig` object — the centralised baseline and the federated clients run the
same trainer, which is what makes RQ1 a measurement of federation rather than of two
different trainers.

| Parameter | Value | Used in | Source |
|---|---|---|---|
| **Model / backbone** | ResNet-18 (torchvision) | both | `experiments.py::TrainingConfig.model_name` |
| **Pretrained weights** | ImageNet (`pretrained = True`) | both | `TrainingConfig.pretrained` |
| Total parameters | 11,178,051 | both | `sites/train.log` |
| Trainable parameters | 10,494,979 (93.89%) | both | `sites/train.log` |
| Frozen parameters | 683,072 (6.11%) | both | `sites/train.log` |
| `freeze_until` | `"layer3"` → conv1 + bn1 + layer1 + layer2 frozen | both | `TrainingConfig.freeze_until` |
| `freeze_bn` | `False` | both | `TrainingConfig.freeze_bn` |
| Classifier head | `Sequential(Dropout(0.5), Linear(512, 3))` | both | `core/models.py::_new_head` |
| Architecture fingerprint | `2d3031acc2075813` | both | `train.log`, `results.json` |
| **Input image size** | 224 × 224 RGB | both | `TrainingConfig.image_size` |
| Number of classes | 3 | both | `TrainingConfig.num_classes` |
| **Optimizer** | AdamW, **over trainable parameters only** | both | `src/training.py::build_optimizer` |
| **Learning rate** | 1e-4 (base) | both | `TrainingConfig.learning_rate` |
| **Weight decay** | 5e-4 | both | `TrainingConfig.weight_decay` |
| **Batch size** | 24 | both | `TrainingConfig.batch_size` |
| **Number of epochs** | **30** (centralised) | centralised | `FederationConfig.centralized_epochs`; `results.json → epochs: 30` |
| **Local epochs per round** | **1** | federated | `FederationConfig.local_epochs` |
| **Communication rounds** | **30** | federated | `FederationConfig.num_rounds`; `rounds.csv` holds rounds 0–29 |
| **Loss function** | `nn.CrossEntropyLoss(weight=…, label_smoothing=0.1)` | both | `src/training.py::build_criterion` |
| Label smoothing | 0.1 | both | `TrainingConfig.label_smoothing` |
| **Class weights** | inverse frequency, counted **per patient** | both | `src/data.py::class_weights` |
| — centralised values | `[0.658, 1.241, 1.48]` | centralised | `logs/test01/*.log` |
| — federated example (test06, hospital_3) | `[0.66, 1.24, 1.473]` | federated | `sites/train.log` |
| `class_weight_scope` | `"local"` — each site uses its own rows | federated | `TrainingConfig.class_weight_scope` |
| **Dropout** | 0.5 | both | `TrainingConfig.dropout` |
| **LR scheduler** | cosine, **evaluated in closed form**: `lr(r) = base·(1 + cos(π·r/T))/2`, T = 30 | both | `src/training.py::lr_for_round`; `run_centralized.py:168` calls the same function |
| **Gradient clipping** | **`clip_grad_norm_`, max_norm = 1.0** | both | `core/training.py:186`; `src/training.py:78` (FedProx fork) |
| **Mixed precision** | requested `True`; **effective on CUDA only** | both | `TrainingConfig.mixed_precision`; `src/training.py::use_amp_on` |
| — AMP actually active in the campaign | yes (`device=cuda amp=True`) | both | `sites/train.log` |
| — GradScaler | `torch.amp.GradScaler("cuda")`, `unscale_()` before clipping | both | `src/training.py::build_scaler` |
| **FedProx μ** | **0.01** | tests 03, 05, 07, 09 | `FederationConfig.fedprox_mu`; `job.json` |
| **FedAvg μ** | **0.0** (no proximal term) | tests 02, 04, 06, 08 | `job.json → fedprox_mu: 0.0` |
| FedOpt server optimiser | SGD, lr 1.0, momentum 0.6, `device="cpu"`, client μ = 0 | tests 10–13 (**cancelled**) | `federation/recipes.py` |
| Aggregation weighting | `NUM_STEPS_CURRENT_ROUND = n_patients` (patients, not slices) | federated | `federation/client.py` |
| **Random seed** | **42** — one run per job | both | `TrainingConfig.seed`; `results.json → seed: 42` |
| **Early stopping** | **none — no such mechanism in the final campaign** | both | absent from `experiments.py`; absent from `run_centralized.py` |
| Checkpoint selection, centralised | best validation **macro-AUC** (`monitor_metric = "auc"`) | centralised | `TrainingConfig.monitor_metric`; `run_centralized.py:205` |
| Checkpoint selection, federated | best **`val_balanced_accuracy`** reported by clients from held-out patients | federated | `FederationConfig.key_metric` |
| `max_slices_per_patient_per_batch` | 1 | both | `TrainingConfig` + `core/data.py::PatientBatchSampler` |
| Slice → patient aggregation | `mean` of slice probabilities | both | `TrainingConfig.aggregation` |
| `num_workers` | 8 | both | `TrainingConfig.num_workers` |
| MixUp | **not present in the final config** (was `mixup_alpha = 0.0`, i.e. off, in the preliminary phase) | neither | absent from `experiments.py` |
| `backbone_lr_scale` | **not present in the final config** (was 0.1 in the preliminary phase) | neither | absent from `experiments.py` |

### 1.1 Validation strategy

| arm | validation set | size | metric used to select |
|---|---|---|---|
| **Centralised (test01)** | the global validation split | 268 patients / 2,132 images | macro-AUC |
| **Federated (test02–09)** | each hospital's **own** 20% local hold-out (`local_val_fraction = 0.2`) | 34–170 patients per site | `val_balanced_accuracy` |

Both are computed on **held-out patients**; neither is training accuracy. A previous
iteration of this project reported training accuracy to the server, which then selected
whichever global model let clients memorise their own shard best (99%+).

Balanced accuracy rather than macro-AUC on the federated side because a site holding few
patients can draw a validation split missing a class, which makes macro-AUC NaN.

**The official evaluation is separate from both.** All nine experiments are scored on the
same global test set — 268 patients, 2,115 images, trivial baseline **0.5112** — with
slice probabilities averaged per patient first.

### 1.2 Train / validation / test split

Taken from the **`split` column of the BreastDCEDL MinCrop release** (`0 = train,
1 = test, 2 = val`); not re-derived. **Patient-level: every slice of a patient is in
exactly one split.**

| split | patients | images | per-class patients (HR+/HER2−, TripleNeg, HER2+) | trivial baseline |
|---|---:|---:|---|---:|
| train | 1,527 | 12,131 | 773 / 410 / 344 | — |
| validation | 268 | 2,132 | 132 / 76 / 60 | 0.4925 |
| test | 268 | 2,115 | 137 / 78 / 53 | **0.5112** |

Confirmed verbatim in `production/logs/test01/*.log`.

---

## 2. Data augmentation

Profile **`default`**, from `src/core/data.py::AugmentConfig`. **Identical
in the preliminary and the final phases**, and identical for the centralised and the
federated arms.

**Applied at load time, every epoch, to the training split only.** The PNGs on disk are
unaugmented; validation and test receive no augmentation.

| Augmentation | Parameter | Value | Applied to |
|---|---|---|---|
| Horizontal flip | probability | **0.50** | training only |
| Vertical flip | probability | **0.00** — disabled | — (never applied) |
| Rotation | degrees | **±15°**, uniform | training only |
| Rotation | probability | 1.00 | training only |
| Scale (zoom) | range | **0.9 – 1.1**, uniform | training only |
| Scale (zoom) | probability | 1.00 | training only |
| Translation (shift) | fraction of image | **±0.08 (±8%)**, uniform, both axes | training only |
| Translation (shift) | probability | 1.00 | training only |
| Affine composition | rotation + scale + translation as **one** `affine_grid` / `grid_sample` | `mode="bilinear"`, `padding_mode="zeros"`, `align_corners=False` | training only |
| Brightness | multiplicative range | **×U(0.8, 1.2)**, then `clamp(0, 1)` | training only |
| Brightness | probability | 1.00 | training only |
| Gaussian noise | σ | **U(0.005, 0.03)**, added then `clamp(0, 1)` | training only |
| Gaussian noise | probability | **0.25** | training only |
| Cutout | probability | **0.00** — disabled | — (never applied) |
| Cutout | side (if enabled) | `min(h,w) × U(0.1, 0.3)` | — (never applied) |
| MixUp / CutMix | — | **not used** | — |
| Elastic deformation | — | **not used** | — |
| Random crop | — | **not used** — framing is fixed at build time by the 80 mm window | — |
| Contrast | — | **not used** (only brightness) | — |
| **ImageNet normalisation** | mean | `[0.485, 0.456, 0.406]` | **train, validation and test** |
| **ImageNet normalisation** | std | `[0.229, 0.224, 0.225]` | **train, validation and test** |

**Order of operations** (`core/data.py::apply_augment`, then `SliceDataset.__getitem__`):

```
PNG → /255 → [0,1]
    → horizontal flip → vertical flip (off)
    → ONE affine transform (rotation + scale + translation)
    → brightness → gaussian noise → cutout (off)
    → ImageNet (x − mean)/std          ← ALWAYS LAST, and applied to every split
```

Rotation, scale and translation are drawn **independently** but composed into a single
affine transform, so the image is interpolated once rather than three times.

Normalisation is last so that brightness and noise operate in `[0, 1]` space, where
clamping to `[0, 1]` is meaningful.

**Vertical flip is deliberately 0.0:** a cranio-caudal flip produces anatomy that does not
exist. Left/right is acceptable — it reads as the contralateral breast.

### 2.1 The one augmentation ablation that was run

Profile `half` — same amplitudes, half the probability on rotation / scale / translation /
brightness. **Preliminary phase only, two seeds. It was rejected.**

| | `default` | `half` |
|---|---:|---:|
| training accuracy at best epoch | 0.57 | **0.99** |
| train − test gap | 0.135 | **0.512** |
| test macro-AUC | — | **−0.040** |

The current augmentation is the only regulariser in this project with a measured effect.

---

## 3. Preprocessing parameters (build time)

From `data/multi_subtype_80mm/config.json`, verbatim, and confirmed against
`metadata.csv`. **One dataset, used by both arms.**

| Parameter | Value | Source |
|---|---|---|
| Source | BreastDCEDL MinCrop (Zenodo 18114231) | — |
| Cohorts | `["spy2", "spy1", "duke"]` | `config.json` |
| Slices kept per patient | **8**, evenly spaced | `n_slices` |
| Trim per end of lesion | **15%**, proportional | `trim_frac` |
| Minimum tumour pixels per slice | 10 | `min_tumor_px` |
| ROI basis | `area_max` — largest-area slice, for mask **and** box | `roi_basis` |
| Crop window | **80.0 mm physical**, `side_px = max(round(80 / xy_spacing), 8)`, zero-padded | `crop_mm` |
| Output size | **224 × 224** | `save_size` |
| Resize filter | PIL **LANCZOS** | `dataset_builder.py` |
| Intensity normalisation | **min–max over the whole 4-D volume** (all 3 phases, all slices), before cropping | `normalization: "minmax"` |
| Bit depth | uint8 (8-bit), RGB PNG, `optimize=True` | `dataset_builder.py` |
| Channels | R = pre-contrast, G = early post-contrast, B = late post-contrast | `dataset_builder.py` |
| Non-finite handling | NaN, ±inf → 0 on read | `dataset_builder.py` |
| Resulting pixel spacing | **0.35714 mm/px, constant** for all 16,378 images | `metadata.csv` (verified: one unique value) |
| Source pixel spacing range | 0.3125 – 1.4062 mm/px | `metadata.csv` |
| Dataset size | **2,063 patients · 16,378 images** | `config.json`, verified |
| Slices per patient, realised | mean 7.94, median 8, range 1–8; 51 patients have fewer than 8 | verified from `metadata.csv` |
| `chanclip_q` | 0.98 — **recorded but not applied** (`normalization = "minmax"`) | `config.json` |

Full description with reasoning: `docs/PREPROCESSING.md`.

---

## 4. Preliminary phase — what differed, and it matters

The classifier phase chose the configuration; it is **not** the dissertation's result.
Read from `results/_from_pod/multi/FREEZE_R18_s42/config.json`.

| Parameter | PRELIMINARY (classifier phase) | FINAL (test01–09) |
|---|---|---|
| epochs | **100** | **30** |
| early stopping | **patience 30** | **none** |
| `backbone_lr_scale` | **0.1** | **absent** — one LR for all trainable params |
| `num_workers` | 16 | 8 |
| `monitor_metric` | `patient_auc` | `auc` (centralised) / `val_balanced_accuracy` (federated server) |
| `mixup_alpha` | 0.0 (off) | field absent |
| `slice_selection` / `n_central` / `train_mode` | `"all"` / 5 / `"full"` | fields absent |
| seeds | 1 **and** 42 (two per configuration) | **42 only**, one run per job |
| model / lr / wd / batch / dropout / label smoothing / freeze / augmentation | **identical to final** | identical |

The final campaign's configuration was read out of the winning preliminary checkpoint
(`FREEZE_R18`), which is why everything except the schedule-length group matches.

**Why `FREEZE_R18` and not the highest number:** `R18_s42` scored 0.6263 but its sibling
seed scored 0.5894 — 0.037 apart on the same configuration. `FREEZE_R18` scored 0.6140 and
0.6178 — 0.003 apart. Against a measured noise floor of 0.067 the two are indistinguishable
in AUC, but one is reproducible.

---

## 5. Per-test values — where the experiments differ

**Everything not in this table is identical across all nine tests.** Only the client
count, the partition and the aggregation algorithm vary.

| Test | Kind | Clients | Partition | Algorithm | μ | Rounds × local epochs | Seed |
|---|---|---:|---|---|---:|---|---:|
| test01 | centralised | — | all pooled (1,527 patients) | — | — | 30 epochs | 42 |
| test02 | federated | 2 | `2_clients_balanced` (1:1) | FedAvg | 0.0 | 30 × 1 | 42 |
| test03 | federated | 2 | `2_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42 |
| test04 | federated | 3 | `3_clients_balanced` (1:1:1) | FedAvg | 0.0 | 30 × 1 | 42 |
| test05 | federated | 3 | `3_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42 |
| test06 | federated | 4 | `4_clients_balanced` (1:1:1:1) | FedAvg | 0.0 | 30 × 1 | 42 |
| test07 | federated | 4 | `4_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42 |
| test08 | federated | 4 | `4_clients_skewed` (**5:2:1:1**) | FedAvg | 0.0 | 30 × 1 | 42 |
| test09 | federated | 4 | `4_clients_skewed` | FedProx | **0.01** | 30 × 1 | 42 |
| test10–13 | federated | 2/3/4/4 | as 02/04/06/08 | **FedOpt** (server SGD lr 1.0, mom. 0.6) | 0 | 30 × 1 | 42 |

**test10–13 were cancelled.** test12/13 failed at launch (`FedOptRecipe` rejects
`key_metric`); test10 reached round 19 of 30 on CPU and was cancelled by the user; test11
never ran. **No FedOpt result may be reported.**

**Patients per site**, from `production/datasets/all_distributions.csv`:

| Partition | Patients per hospital |
|---|---|
| `2_clients_balanced` | 393 / 391 |
| `3_clients_balanced` | 262 / 262 / 260 |
| `4_clients_balanced` | 198 / 196 / 195 / 195 |
| `4_clients_skewed` | 435 / 175 / 87 / 87 |

**⚠ All four partitions are stratified** — every hospital keeps the global class ratio,
maximum spread 0.4 percentage points. Tests 08/09 therefore measure **quantity skew, not
label or feature non-IID heterogeneity.** This must be stated in the dissertation.

---

## 6. Implemented but never run — do not report as results

| item | status |
|---|---|
| `--by-cohort` partition (one real cohort per hospital) | implemented in `scripts/partition_data.py`, **never run** |
| `--stratify none` (label skew) | implemented, **never run** |
| `class_weight_scope = "global"` | implemented, **never run** — this is the real RQ4 experiment |
| FedOpt (tests 10–13) | implemented, **cancelled** |
| `freeze_until = "layer4"` | supported, **never run** |
| `chanclip` normalisation | implemented; run in the preliminary phase (lost by 0.025), **not used in the final dataset** |
| Multiple seeds on the federated campaign | **never run** — every federated number is one seed |

---

## 7. Concise final hyperparameter table

The single table for the dissertation. Values are those of the **final campaign**
(test01–test09). "Both" = identical in the centralised and federated arms.

| Hyperparameter | Value | Arm |
|---|---|---|
| Backbone | ResNet-18, ImageNet-pretrained | both |
| Trainable / total parameters | 10,494,979 / 11,178,051 | both |
| Frozen layers | conv1, bn1, layer1, layer2 (`freeze_until = layer3`) | both |
| Input | 224 × 224 RGB | both |
| Output classes | 3 | both |
| Head | Dropout(0.5) → Linear(512, 3) | both |
| Optimizer | AdamW (trainable params only) | both |
| Learning rate | 1e-4, cosine `lr(r) = base·(1+cos(πr/30))/2` | both |
| Weight decay | 5e-4 | both |
| Batch size | 24, ≤ 1 slice per patient per batch | both |
| Loss | Cross-entropy, class-weighted per patient, label smoothing 0.1 | both |
| Class weights | `[0.658, 1.241, 1.480]` pooled; computed locally per site | both |
| Dropout | 0.5 | both |
| Gradient clipping | max-norm 1.0 | both |
| Mixed precision | on (CUDA only) | both |
| Epochs | 30 | centralised |
| Rounds × local epochs | 30 × 1 (= 30 epochs, budget-matched) | federated |
| Aggregation | FedAvg, weighted by patient count | federated |
| FedProx μ | 0.01 | tests 03, 05, 07, 09 |
| Model selection | val macro-AUC | centralised |
| Model selection | val balanced accuracy (client hold-out, 20%) | federated |
| Early stopping | none | both |
| Seed | 42, single run per experiment | both |
| Train / val / test | 1,527 / 268 / 268 patients, patient-level, from the release | both |
| Test-set trivial baseline | 0.5112 | both |
| Reported metric | patient-level macro-AUC (slice probabilities averaged first) | both |
| Augmentation | hflip 0.5 · rot ±15° · zoom 0.9–1.1 · shift ±8% · brightness ×0.8–1.2 · noise p 0.25 | training split only |
| Preprocessing | 8 slices, 15% trim, 80 mm window, volume min–max, 224 px, 0.357 mm/px | both |

---

## 8. Statistical caveat that must accompany any of these numbers

**Measured noise floor: 0.067 macro-AUC.** Two runs of a byte-identical configuration
differing only in seed scored 0.7023 and 0.6351. `seed` fixes initialisation and the split
but not cuDNN kernel selection, AMP behaviour, or DataLoader worker ordering.

Every federated experiment in this campaign is **a single run at seed 42**, and the full
spread across the nine tests is 0.093. No comparison within that table is attributable to
the factor that distinguishes two runs. The defensible claim is that federated training
produces models in the **same range** as centralised training on this task — and nothing
sharper.
