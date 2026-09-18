# Training and preprocessing parameters — complete reference

**Every value in this document was read from source code or from a run artefact.**
Nothing is inferred from library defaults or carried over from prose. Where the code and
the project's own prose disagree, the code wins and the disagreement is stated.

Verified 2026-08-04 against `deployment/code/config/experiments.py`, `deployment/code/common/training.py`,
`deployment/code/common/data.py`, `src/core/training.py`,
`src/core/data.py`, `deployment/code/scripts/run_centralized.py`, the run logs in
`deployment/logs/`, and the recorded configs in
`unused/old_runs/classifier/_from_pod/`, where the classifier phase was archived.

**Re-verified 2026-09-15** for the experiment table, the seed set and every FedOpt entry,
against `deployment/code/config/experiments.py` and `results/thesis/all_experiments.csv`.
The rest of this document still carries the 2026-08-04 verification date.

---

## 0. How to read this document — three distinct phases

The project ran two separate experimental campaigns with **different** hyperparameters.
Conflating them is the main way to misreport this work.

| phase | what it is | where the config lives | status |
|---|---|---|---|
| **PRELIMINARY** | The classifier phase — 21 runs, 13 architectures, 5 data configurations. Used to *choose* the configuration. | `src/dataset_config.py`; per-run `results/_from_pod/multi/<run>/config.json` | superseded; not the dissertation's numbers |
| **FINAL** | The dissertation campaign — test01 (centralised) + test02–09 (federated), 2026-08-03/04 | `deployment/code/config/experiments.py` — **single source of truth** | **these are the reported results** |
| **PLANNED / NOT RUN** | Implemented and available, never executed | same file / `deployment/code/scripts/partition_data.py` | must not be reported as results |

**Three corrections to earlier summaries in this project, found while verifying:**

1. **Gradient clipping is used and had been omitted.** `clip_grad_norm_(…, max_norm=1.0)`
   is applied in both the shared trainer and the FedProx fork.
2. **The final campaign has no early-stopping mechanism at all.** `early_stopping` does
   not exist in `deployment/code/config/experiments.py`, and `deployment/code/scripts/run_centralized.py`
   contains no such code — the loop runs all 30 epochs and tracks the best. The phrase
   "early stopping disabled (`early_stopping_patience = 0`)" seen in the classifier-phase
   record describes that phase's field, which is `100 epochs / patience 30`, not the
   federated code.
3. **`backbone_lr_scale = 0.1` was used in the preliminary phase and is absent from the
   final campaign.** The final arms apply one learning rate to all trainable parameters.

---

## 1. Main training parameters

`Used in` states the arm each value applies to. "Both" means the identical value, from the
same `TrainingConfig` object — the centralised baseline and the federated clients run the
same trainer, which is what makes the comparison a measurement of federation rather than of two
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
| **Optimiser** | AdamW, **over trainable parameters only** | both | `src/training.py::build_optimizer` |
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
| **FedProx μ** | **0.01** | tests 03, 05, 07, 09, 11, 13 | `FederationConfig.fedprox_mu`; `job.json` |
| **FedAvg μ** | **0.0** (no proximal term) | tests 02, 04, 06, 08, 10, 12 | `job.json → fedprox_mu: 0.0` |
| FedOpt server optimiser | SGD, lr 1.0, momentum 0.6, `device="cpu"`, client μ = 0 | **cancelled; no test id** | `config/experiments.py::FederationConfig.fedopt_lr`, `.fedopt_momentum` |
| Aggregation weighting | `NUM_STEPS_CURRENT_ROUND = n_patients` (patients, not slices) | federated | `federation/client.py` |
| **Random seed** | **42, 19 and 50** — every configuration run once per seed | both | `config/experiments.py::SEEDS`; `results/thesis/all_experiments.csv` |
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

**The official evaluation is separate from both.** All thirteen experiments are scored on the
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

Confirmed verbatim in `deployment/logs/test01/*.log`.

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

| Parameter | PRELIMINARY (classifier phase) | FINAL (test01–13) |
|---|---|---|
| epochs | **100** | **30** |
| early stopping | **patience 30** | **none** |
| `backbone_lr_scale` | **0.1** | **absent** — one LR for all trainable params |
| `num_workers` | 16 | 8 |
| `monitor_metric` | `patient_auc` | `auc` (centralised) / `val_balanced_accuracy` (federated server) |
| `mixup_alpha` | 0.0 (off) | field absent |
| `slice_selection` / `n_central` / `train_mode` | `"all"` / 5 / `"full"` | fields absent |
| seeds | 1 **and** 42 (two per configuration) | **42, 19 and 50**, three runs per configuration |
| model / lr / wd / batch / dropout / label smoothing / freeze / augmentation | **identical to final** | identical |

The final campaign's configuration was read out of the winning preliminary checkpoint
(`FREEZE_R18`), which is why everything except the schedule-length group matches.

**Why `FREEZE_R18` and not the highest number:** `R18_s42` scored 0.6263 but its sibling
seed scored 0.5894 — 0.037 apart on the same configuration. `FREEZE_R18` scored 0.6140 and
0.6178 — 0.003 apart. Against a measured noise floor of 0.067 the two are indistinguishable
in AUC, but one is reproducible.

---

## 5. Per-test values — where the experiments differ

**Everything not in this table is identical across all thirteen tests.** Only the client
count, the partition and the aggregation algorithm vary.

| Test | Kind | Clients | Partition | Algorithm | μ | Rounds × local epochs | Seed |
|---|---|---:|---|---|---:|---|---:|
| test01 | centralised | — | all pooled (1,527 patients) | — | — | 30 epochs | 42, 19, 50 |
| test02 | federated | 2 | `2_clients_balanced` (1:1) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test03 | federated | 2 | `2_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |
| test04 | federated | 3 | `3_clients_balanced` (1:1:1) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test05 | federated | 3 | `3_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |
| test06 | federated | 4 | `4_clients_balanced` (1:1:1:1) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test07 | federated | 4 | `4_clients_balanced` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |
| test08 | federated | 4 | `4_clients_skewed` (**5:2:1:1**) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test09 | federated | 4 | `4_clients_skewed` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |
| test10 | federated | 3 | `3_clients_cohort` (**642 / 101 / 784, one cohort per site**) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test11 | federated | 3 | `3_clients_cohort` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |
| test12 | federated | 3 | `3_clients_sizematched` (same three sizes, cohorts mixed) | FedAvg | 0.0 | 30 × 1 | 42, 19, 50 |
| test13 | federated | 3 | `3_clients_sizematched` | FedProx | **0.01** | 30 × 1 | 42, 19, 50 |

**FedOpt once occupied test10–13 and was removed from the experiment table on
2026-08-05.** Those four FedOpt runs were cancelled: test12/13 failed at launch
(`FedOptRecipe` rejects `key_metric`), test10 reached round 19 of 30 on CPU and was
cancelled by the user, test11 never ran. **No FedOpt result may be reported.** The four ids
now belong to the heterogeneity pair listed above, which completed at all three seeds. See
`docs/PROJECT_HISTORY.md` §11.4.

**test10 and test12 are a matched pair and are meaningless apart**, as are test11 and
test13. Both partitions hold the same three site sizes. The only difference is whether a
site's patients come from one cohort or from a stratified mix of all three, so what is
measured between them is cohort identity rather than how much data a site holds.

**Patients per site**, training split at seed 42, from
`results/thesis/distributions/seed_42/all_distributions.csv`. The same file exists for
seeds 19 and 50.

| Partition | Train patients per hospital | Max class-share spread |
|---|---|---:|
| `2_clients_balanced` | 612 / 611 | 0.1 pp |
| `3_clients_balanced` | 408 / 408 / 406 | 0.2 pp |
| `4_clients_balanced` | 306 / 305 / 305 / 305 | 0.2 pp |
| `4_clients_skewed` | 678 / 273 / 136 / 135 | 0.7 pp |
| `3_clients_cohort` | 514 / 82 / 627 | **27.4 pp** |
| `3_clients_sizematched` | 514 / 81 / 628 | 0.3 pp |

These are the counts after the 20% local validation carve. The site allocations they come
from are 642 / 101 / 784 for both three-site partitions.

**Four of the six partitions are stratified.** Every hospital keeps the global class ratio
to within 0.7 percentage points, so the three balanced partitions and `4_clients_skewed`
vary quantity and nothing else. Tests 08/09 therefore measure **quantity skew, not label or
feature non-IID heterogeneity**, and this must be stated in the dissertation.
`3_clients_cohort` is the one partition that is not stratified: one real cohort per site,
27.4 pp of class-share spread. `3_clients_sizematched` is its control, holding the same
three site sizes with the cohorts mixed back together, so tests 10 to 13 isolate cohort
identity from site size.

---

## 6. Implemented but never run — do not report as results

| item | status |
|---|---|
| `--stratify none` (label skew) | implemented, **never run** |
| `class_weight_scope = "global"` | implemented, **never run** |
| FedOpt | implemented, **cancelled**; the test ids were reused |
| `freeze_until = "layer4"` | supported, **never run** |
| `chanclip` normalisation | implemented; run in the preliminary phase (lost by 0.025), **not used in the final dataset** |

**Two entries left this table.** The `--by-cohort` partition was run, as tests 10 and 11,
with `3_clients_sizematched` as its control in tests 12 and 13. The campaign was then
repeated at seeds 42, 19 and 50, giving 39 runs. Both are recorded in `results/thesis/`.

---

## 7. Concise final hyperparameter table

The single table for the dissertation. Values are those of the **final campaign**
(test01–test13). "Both" = identical in the centralised and federated arms.

| Hyperparameter | Value | Arm |
|---|---|---|
| Backbone | ResNet-18, ImageNet-pretrained | both |
| Trainable / total parameters | 10,494,979 / 11,178,051 | both |
| Frozen layers | conv1, bn1, layer1, layer2 (`freeze_until = layer3`) | both |
| Input | 224 × 224 RGB | both |
| Output classes | 3 | both |
| Head | Dropout(0.5) → Linear(512, 3) | both |
| Optimiser | AdamW (trainable params only) | both |
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
| Seed | 42, 19 and 50, one run per experiment per seed | both |
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

**The campaign now carries three seeds.** Every configuration was run once at seed 42,
once at 19 and once at 50, which is 13 configurations by 3 seeds and 39 runs in total. The
seed effect is therefore measured rather than assumed: across the three seeds of one
configuration the macro-AUC range averages **0.0308** and reaches **0.0537** at its widest
(`results/thesis/analysis/resumo.json`). Read any two configurations against that range
before treating a difference between them as real.
