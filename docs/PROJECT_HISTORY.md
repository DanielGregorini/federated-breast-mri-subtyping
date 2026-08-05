# PROJECT HISTORY — chronological technical record

Companion to `docs/PROJECT_CONTEXT.md`. That document describes **what the project is
now**; this one describes **how it got there, in order**, including everything that was
tried and abandoned.

**Written 2026-08-04, extended 2026-08-05** from the actual repository. Dates come from
file timestamps, `results.json` `finished` fields, `job.json` submission times and log
headers. Where a date could not be established it is marked **DATE NOT VERIFIED**.

**The 2026-08-05 additions**, in the order they happened: the RQ2 campaign (tests 10–13,
§11.5), the removal of FedOpt from the experiment table (§11.4), the repository
reorganisation (§13, item 5), and the Apple-MPS root cause (§12, bug 16, which this
document previously recorded as never established).

**Why this document exists.** Several of this project's most valuable results are
failures, and a failure is only useful if you can say what preceded it. A dissertation
has to be able to explain how it reached its design.

---

## 1. INITIAL DATASET — the radiomics phase

**Phase 1 — `radiomic_ai/`** (now `unused/legacy_projects/radiomic_ai/`). **DATE NOT
VERIFIED — predates all surviving logs.**

PyRadiomics feature extraction plus a Random Forest classifier. Superseded entirely.

**It contains one bug worth remembering:** `classifier_manager.py:387` used
`StratifiedKFold` on **slices**, not patients. Neighbouring slices of one tumour are
near-duplicates, so the same patient appeared on both sides of every fold boundary. This
is the same class of leak that later became the project's first hard rule:

> **Split by patient. Always. Enforced by a verification step that refuses to pass.**

---

## 2. DATASET INVESTIGATION — the four-source catalogue and its collapse

**Phase 2 — the 4-source, 4-class catalogue.** **DATE NOT VERIFIED.**

A catalogue of **1,488 patients** pooled from **DUKE + I-SPY1 + I-SPY2 + NACT**,
**206,888 slices**, classified into four molecular subtypes (Luminal A, Luminal B, HER2+,
TNBC).

**The results looked good — and that was the problem.** Luminal B reached **F1 ≈ 0.98** on
what the literature calls the hardest class to distinguish.

### The audit (`unused/old_runs/results_01_to_08/05_pipeline_audit/`)

Motivated by suspicion of a code bug. **Seven integrity checks were run and all passed:**
TIFF dimensions against `original_shape`; subtype-ID mapping; prepared labels against the
catalogue (**0 divergences in 20,028 rows**); label integer against label name; one label
per patient; patient ID against filename; **zero train/val patient overlap**.

The most serious suspicion — swapped bounding-box coordinates — was ruled out by
measurement over 400 slices:

| interpretation | mean intensity | fraction in air |
|---|---:|---:|
| x = column, y = row (what the code does) | **92.5** | 1.9% |
| swapped | 50.5 | 21.6% |
| whole image (reference) | 26.3 | — |

**There was no bug.**

### The positive control that changed the project

The identical pipeline — same images, same patients, same architecture — trained to
predict the **dataset source** instead of the subtype:

| target | patient accuracy | macro-AUC |
|---|---:|---:|
| **dataset source** | **0.8864** | **0.9667** |
| molecular subtype | 0.3864 | 0.5895 |

Per source: DUKE 0.989 · I-SPY1 0.979 · I-SPY2 0.994.

**The model was answering "which dataset is this?" instead of "which subtype is this?"**
Luminal B was 90% I-SPY2 because DUKE contains only **3** Luminal B patients in total.

Two design flaws were quantified at the same time, neither a bug: the tumour occupied a
median of **9.7%** of the 128 px window, and the fixed crop covered 0.25 of a DUKE slice
against 0.50 of every other source.

### The lab-balanced controls

`03_lab_balanced_4class/` — 288 patients, 72 per class. macro-AUC **0.5895**, accuracy
0.3864 against a trivial baseline of 0.2500. **All 13 patients predicted Luminal B were
I-SPY2.** None from DUKE, I-SPY1 or NACT.

`04_lab_balanced_3class/` — same patients, same split, **only the label granularity
changed**. The cleanest controlled comparison in the project:

| | 4 classes | 3 classes |
|---|---:|---:|
| accuracy | 0.3864 | 0.4545 |
| trivial baseline | 0.2500 | 0.5000 |
| **accuracy − baseline** | **+13.6 pp** | **−4.5 pp** |
| TripleNeg AUC | 0.614 | 0.614 |
| HER2+ AUC | 0.565 | 0.590 |

**Accuracy rose from 38.6% to 45.5% and the model got worse.** Collapsing the Luminal A/B
boundary helped nothing — the two untouched classes are unchanged. One genuine gain:
calibration inverted in the right direction for the first time (confidence 0.651 on
correct predictions vs 0.578 on wrong ones).

**Outcome: the whole 4-source catalogue was abandoned.**

### Other datasets investigated and discarded

| dataset | why discarded |
|---|---|
| Full I-SPY2 (54 GB NIfTI) | superseded by BreastDCEDL MinCrop; deleted by the user |
| MAMA-MIA | downloaded (expert segmentations + 22 real site IDs), then deleted. **Worth reacquiring** |
| `breast-mri-molecular-cancer-subtype` (Duke .mha, 34 GB) | ships no masks; `dataset.json` filenames do not resolve |
| NACT | only 50 patients; dropped with the catalogue |

---

## 3. PREPROCESSING DEVELOPMENT

**Phase 3 — I-SPY2 only, three classes.** Single-source design. Luminal A and B merged
into `HRposHER2neg`, giving three classes. Performance dropped to a **truthful 0.62
macro-AUC**.

### `02_preprocessing_ablation/` — seven centralised runs, all on one 224-patient split

| run | what it tested | macro-AUC |
|---|---|---:|
| **A** | baseline (replicated channels, normal sampling) | **0.6537** |
| B | 3 DCE phases fused as real RGB channels | 0.6300 |
| C | balanced sampling | 0.6270 |
| D | fused + balanced | 0.6418 |
| E | old pipeline, no corrections | 0.6432 |
| F | label-noise filter only | 0.6108 |
| G | joint 3-phase normalisation only | 0.6396 |

**Everything tested failed.** The label-noise filter is the interesting one: **its effect
depends on the normalisation** — isolated with per-image normalisation it costs −0.032;
combined with joint normalisation it gains +0.014. Not a reliable improvement.

**Also documented here:** an earlier claimed gain from 0.616 to 0.654 was **wrong** — the
validation split had changed (only 34 of 224 patients in common), so roughly 70% of that
"gain" was an easier validation set. This is the origin of the rule *compare only on the
same test set*.

### `06_ispy2_final/` — the first result on a held-out test set

I-SPY2 only, DCE phases 0/2/5 (pre / early / **delayed**, never used before), tight tumour
crop, 679/97/195 patient split, **three seeds**:

| seed | accuracy | bal-acc | macro-F1 | macro-AUC |
|---:|---:|---:|---:|---:|
| 42 | 0.3949 | 0.3548 | 0.3337 | 0.6228 |
| 43 | 0.4051 | 0.3681 | 0.3495 | 0.6070 |
| 44 | 0.3744 | 0.3156 | 0.3090 | 0.6357 |
| **mean ± sd** | 0.3915 ± 0.016 | 0.3462 ± 0.027 | 0.3307 ± 0.020 | **0.6218 ± 0.014** |

**The class ordering matched the literature for the first time** — the strongest evidence
the signal had become biological:

| class | n | mean AUC |
|---|---:|---:|
| **TripleNegative** | 71 | **0.678** |
| Luminal B | 31 | 0.642 |
| Luminal A | 75 | 0.602 |
| HER2+ | 18 | 0.566 |

TripleNegative is the most imaging-detectable subtype and became the best class. In the
contaminated runs Luminal B was artificially the best.

**The headline of that campaign:**

| configuration | confounder AUC | subtype AUC | gap |
|---|---:|---:|---:|
| 4 sources, 128 px, phases 0/1/2 | 0.967 (source) | 0.589 | **0.378** |
| I-SPY2, tight crop, phases 0/2/5 | 0.699 (resolution) | 0.622 | **0.077** |

The distance between what the model could detect that is *not* biology and what it could
detect that *is* fell from 0.378 to 0.077. The shortcut did not vanish — the 384×384
subgroup remained almost perfectly identifiable (AUC 0.972), real intra-cohort acquisition
heterogeneity relevant to RQ2 — but it stopped dominating.

**Controls from that campaign:** the tight crop gained +0.015 against a ±0.014 seed spread
— no measurable effect, despite being the audit's top recommendation. **Averaging all
slices beat using only the largest-tumour slice on all three seeds** (0.622 vs 0.593),
confirming the project's aggregation choice against common practice. `stacking/` holds the
2.5D ablation: 3 adjacent slices 0.6066 ± 0.008 and 9 channels of 3 phases × 3 slices
0.6083 ± 0.028, **both below** the 0.6218 base.

### The slice-selection ladder

| strategy | why tested | outcome | status |
|---|---|---|---|
| all tumour slices | maximum data | 38 per patient, near-duplicates; dilutes the patient mean | abandoned |
| top-20 consecutive | more slices around the lesion | consecutive slices are the same image | abandoned |
| top-5 by area | the authors use ~4–5 central | beat "all slices" by **+0.058** | kept as a variant |
| central 5 (geometric midpoint) | the authors' formula | the midpoint of an asymmetric tumour lands on near-empty slices | replaced by area rank |
| **8 spread, 15% trim** | trade redundancy for diversity | 63,460 → 16,378 images | **CURRENT (mine)** |
| **4, `range(idx−2, idx+2)`** | the authors' exact rule | exactly 4.00 slices/patient | **CURRENT (authors)** |

### The crop ladder

| strategy | why tested | outcome | status |
|---|---|---|---|
| no crop (whole 256 slice) | control | tumour mask covers 1.5% of the frame | abandoned |
| proportional crop (bbox + %) | tighten onto the lesion | **lost to no crop at all**; erases tumour size | abandoned |
| tight 4 mm / 6 mm margin | peritumoral literature | 6 mm measured **−0.021** vs 4 mm | abandoned |
| fixed 100 mm / 120 mm | crop ladder | settled by the 80 mm derivation | abandoned |
| **80 mm physical** | keep size, kill the resolution signature | resampling equalised to 1.91–2.20× | **CURRENT (mine)** |
| **224 px fixed** | the authors' rule | 158–175 mm effective, varies by cohort | **CURRENT (authors)** |

### The normalisation ladder

| strategy | outcome | status |
|---|---|---|
| **min-max per volume** | baseline; preserves kinetics between slices and phases | **CURRENT (mine)** |
| **min-max per slice** | the authors' rule | **CURRENT (authors)** |
| `pclip` (p0.5/p99.5 global) | **−0.034**, lost on all 3 seeds | abandoned |
| `chanclip` (per-channel q0.98) | **−0.025** here, both seeds same direction — despite the authors measuring it **best** (0.744 vs 0.700) | abandoned |
| subtraction + per-patient z-score | −0.021 | abandoned |

### Other interventions measured in this period

| intervention | Δ macro-AUC |
|---|---:|
| fusing DCE phases as RGB | −0.024 (kept anyway — it makes the channels meaningful) |
| balanced sampling | −0.027 |
| label-noise filter (bbox ≥ 100 px²) | −0.032 |
| joint 3-phase normalisation | −0.004 |
| tight tumour crop (4 mm) | +0.015 (noise) |
| multi-task decomposition (3 binary heads + CBAM) | **hurt TripleNeg by 0.040** |
| 2.5D frame stacking (3 / 9 channels) | −0.015 / −0.014 |
| 3D volumes (96³) | 0.53–0.58 |
| background masked out | never run — literature says worst config (−4.9%), and impossible for DUKE |
| batch size 8 / 24 / 64 | 24 already best |
| learning rate 3e-5 / 1e-4 / 3e-4 | 1e-4 best; 3e-4 worst (0.5728) |
| training length 10 / 20 / 60 / 200 / 300 epochs | no effect |

**The conclusion of the whole preprocessing programme:** seven literature-grounded
interventions, all null or negative, everything landing between 0.59 and 0.65. **That is
not a run of bad luck — it is the result: the ceiling is in the data.**

---

## 4. DATASET VERSIONS — the migration to BreastDCEDL

**Phase 5 — migration to the published [BreastDCEDL](https://zenodo.org/records/18114231)
release.** Standardised, deep-learning-ready, and with a paper to compare against.

**First target: pCR.** It gave validation AUC **0.501** — chance. The target moved to
3-class subtype, which is biology already present in the image.

### The MinCrop geometry, reverse-engineered

The authors publish MinCrop but the *rule* that produced it is only in two notebooks the
README does not reference. It was reverse-engineered and then **verified against the real
files**:

| check | result |
|---|---|
| metadata box == true mask box when `n_xy == 256` | **767 / 767 (100%)** |
| tumour occupies `z ∈ [2, nz−3]` | 749 / 767 (97.7%) |
| DUKE: `volume_depth == z_span + 4` | 12 / 12 |
| patients with `n_xy > 256` were **cropped, not resized** | **228 / 228** |

`n_xy` is the size of the **original** scan, not the MinCrop volume. When it equals 256 no
in-plane crop was applied and the metadata box lands exactly on MinCrop pixels. **914 of
916 DUKE patients are in that case**; six re-centring rules were tested for the others and
the best recovered only 62%, so they were dropped. The z convention was later confirmed
against the authors' own `crop_around_voi_cords(..., slice_padding=2, output_size=256)`.

**This is what made DUKE usable for the first time.**

### The multi-cohort build

DUKE included for the first time: **16,378 images, 2,063 patients, 0 errors.**

Since DUKE has no mask to overlay, that the crops actually contain tumour was verified by
**physics** — enhancement (G−R) at the centre vs the periphery:

| cohort | ROI source | centre | periphery | ratio | centre > periphery |
|---|---|---:|---:|---:|---:|
| spy2 | mask | 30.14 | 7.32 | 4.12× | 95% |
| spy1 | mask | 34.97 | 8.84 | 3.96× | 90% |
| **duke** | **box** | 23.54 | 5.02 | **4.69×** | **96%** |

### The honest test of the new preprocessing

`SPY2_R18` ran the new preprocessing on the **same 99 I-SPY2 test patients** that
previously gave 0.6201:

| | macro-AUC | seeds | val−test gap |
|---|---:|---:|---:|
| old preprocessing (all slices, no physical window) | 0.6201 ± 0.024 | 3 | — |
| **new (8 spread, 80 mm)** | **0.5837 ± 0.011** | 2 | **+0.015** |

**Verdict: no difference detected** (0.036, inside the 0.067 floor) — but certainly **not**
the improvement expected. What it bought was a validation-to-test gap of +0.015 against
+0.073 for the pooled runs, and much more even per-class recall `[0.60, 0.50, 0.37]`.

### The dataset versions that existed at various times

| dataset | pipeline | images | patients | test | fate |
|---|---|---:|---:|---:|---|
| `mine_subtype` (I-SPY2 only) | mine | 7,835 | 982 | 99 | superseded |
| `mine_subtype_pooled` → **renamed `multi_subtype_80mm`** | mine | **16,378** | **2,063** | **268** | **ACTIVE** |
| `authors_pcr` | authors | 5,800 | 1,451 | 176 | reproduction only |
| `authors_her2` | authors | 8,229 | 2,063 | 268 | reproduction only |
| `authors_subtype` | cross | 8,229 | 2,063 | 268 | trained, but `Config` refuses the combination |
| `multi_subtype_80mm_chanclip` | mine | — | 2,063 | 268 | the 2×2 ablation arm; retired |
| `multi_subtype_80mm_SOURCEPROBE` | mine | — | 2,063 | 268 | the probe; retired (regenerable in seconds) |
| `spy2only_80mm` | mine | — | 982 | 99 | the honest-test arm |
| 5 older I-SPY2/I-SPY1 variants | mine | — | — | — | `unused/old_datasets/` |

**Phase 7 — refactor and cleanup.** The code was reorganised into two deliberately
isolated pipelines (`src/pipelines/reference/` and `src/pipelines/thesis/`, with `Config` refusing to
mix them) and everything superseded was **moved, never deleted**, into `unused/` (34 GB).
`multi_subtype_80mm` became the single active dataset.

---

## 5. MODEL EXPERIMENTS

Thirteen architectures were wired, built and forward-checked. Full table in
`PROJECT_CONTEXT.md` §7.2. The short version:

* **ResNet-18 (11.2M)** — 0.6078 ± 0.026 pooled, 0.6159 ± 0.003 frozen. **The measured
  winner**, and the only model whose validation did not lie (val−test gap −0.009 against
  +0.062 and +0.059 for the others). 6× faster than ResNet-50 (12 min vs 74.5).
* **ResNet-50 (23.5M)** — 0.6044 ± 0.018. Indistinguishable and six times slower.
* **ViT-MAE-base (85.8M)** — 0.6298 on the authors' pipeline, one seed. The authors' own
  model. Best single 3-class number, but its best epoch was **2 of 32**.
* **THDA-ResNet-34 (21.3M)** — 0.5710 ± 0.007. The authors' best for HER2 (0.744 in their
  paper) and it loses here.
* **Binary-task benchmark** (TripleNeg vs rest): ResNet-101 0.5711 · ResNet-152 0.6430 ·
  ConvNeXt T/S/B 0.6181/0.5815/0.5881 · Swin-T 0.6609 · Swin-V2-T 0.5645 · EfficientNet-B0
  0.6011 · DenseNet-121 0.6709 · MobileNetV3-S 0.6011.
* **Sequential and 3D:** Zhang CNN 0.5460 · ConvLSTM 0.5691 · R(2+1)D-18 0.5817 · R3D-18
  0.5631 · MC3-18 0.5393.
* **The linear probe — 4,098 trainable parameters — scored 0.6813**, matching full
  fine-tuning of 23.5M, with an overfitting gap of 0.020 against 0.37.

**Conclusion: architecture does not matter.** From 1.5M to 87.6M parameters, everything
lands in the same 0.55–0.63 band. With a noise floor of 0.067 the ordering distinguishes
nothing. **The linear probe is the single strongest piece of evidence in the project.**

The hyperparameter ladder on the binary task
(`unused/old_runs/results_01_to_08/08_federated_final/SUMMARY.txt`, 2026-08-01):
ResNet-50 at 10/20/200 epochs → 0.6639 / 0.6652 / **0.7023**; DenseNet-121 → 0.6744 /
0.6465 / 0.6765; augmentation 0.0 → 0.6260 and 1.8 → 0.6364; lr 3e-4 → 0.5728, 3e-5 →
0.6473; batch 8 → 0.6264, batch 64 → 0.6238; 6 mm margin → 0.6443; multi-task 3 binary
heads + attention → mean AUC 0.5907 and 3-class composed 0.5490; linear probe `last.pt`
→ **0.6813**.

---

## 6. OVERFITTING INVESTIGATION

### What was observed

Best epochs of **1–5** are routine; `train_acc` reaches 0.99 within tens of epochs on any
configuration allowed to run long enough. The clearest single record is the centralised
baseline of 2026-08-03 (30 epochs, **early stopping deliberately disabled**): training
accuracy climbs 0.44 → **0.9955** and training loss falls 1.15 → 0.36, while **validation
AUC peaks at epoch 4 (0.6661) and never improves again**.

Per-run train/test gaps are tabulated in `PROJECT_CONTEXT.md` §8.2 — they range from
**−0.039** (ViT, best epoch 2) to **+0.525** (chanclip with halved augmentation, best
epoch 26).

### What was ruled out

* **Capacity.** The 4,098-parameter linear probe beat 23.5M fully fine-tuned. If capacity
  were the bottleneck this could not happen.
* **A pipeline bug.** Seven integrity checks passed with 0 divergences in 20,028 rows, and
  the source probe reached 0.9978 — a broken pipeline could not.
* **Dataset size as the sole cause.** The effective sample size is the **patient** count
  (1,527 training patients), not the 12,131 slices; but doubling patients from 982 to
  2,063 did not raise the ceiling.

### What was measured

| intervention | effect |
|---|---|
| **freezing conv1–layer2** | +0.008 AUC, **seed spread 0.026 → 0.003** |
| **halving augmentation** | **−0.040 AUC, train acc 0.57 → 0.99, gap 0.135 → 0.512** |
| dropout 0.5 | the default — and it was **silently disabled by a bug** for a period (§12.1) |
| chanclip | −0.025 |
| pclip | −0.034 |
| balanced sampling | −0.027 |
| label-noise filter | −0.032 |
| patient-aware batch sampler | kept; effect never isolated |
| MixUp | never conclusive; `mixup_alpha = 0.0` |
| Focal / CB-Loss / samplers | **never run** — ratio 2.25:1 is below the band where they help |
| logit adjustment | **never run** — worth doing, post-hoc, no retraining |

### Validation overestimation

Validation-to-test gaps of **+0.049 to +0.097**. **Everything selected on validation
failed to transfer:** decision threshold (moved results both ways, up to −0.13 in recall),
slice aggregation (7 variants, mean ≈ 0, spread ±0.05), ensemble composition
(validation-picked 0.7005 vs all-runs 0.7066), best vs last checkpoint (2 wins each).

**Rule adopted: report threshold-free AUC at a fixed 0.5 threshold. Do not tune on ~100
validation patients.**

### The noise floor — the discovery that reframed everything before it

Two runs of a **byte-identical** configuration differing only in seed scored **0.7023 and
0.6351 — a gap of 0.067**. `seed` fixes initialisation and the split but not cuDNN kernel
selection, AMP, or DataLoader worker ordering. **An earlier reading of ±0.001 was a lucky
pair and is wrong.**

**This invalidated most single-run comparisons made earlier in the project**, including
the entire architecture benchmark, and is now enforced in the reporting code with a
`within_noise_floor` column on every comparison table.

### The conclusion

**The ceiling is signal, not capacity, and 0.55–0.63 is the correct answer for this task
on this data.** A 106-study, 12,989-patient systematic review concludes conventional
quantitative MRI features "might play a limited role" in subtype prediction, and Zhang et
al. report 0.79/0.91 within-centre collapsing to 0.52/0.44 **cross-centre** — which is
exactly the line our numbers sit on.

---

## 7. COHORT / SOURCE PROBE

Created in `05_pipeline_audit/` (4-source catalogue) and re-run on the current pooled
dataset as `SONDA_r18`.

| dataset | probe macro-AUC | subtype macro-AUC | gap |
|---|---:|---:|---:|
| old 4-source catalogue | 0.967 | 0.589 | 0.378 |
| I-SPY2 only, tight crop | 0.699 (resolution) | 0.622 | 0.077 |
| **current 3-cohort pooled** | **0.9978** | **0.6078** | **0.390** |

Current probe: **accuracy 0.9813** (263 of 268 patients), balanced accuracy 0.9801,
per-class AUC `[0.9963, 1.000, 0.997]`, trivial baseline 0.5075, best epoch 42 of 72.
Per-source accuracy in the earlier audit: DUKE 0.989 · I-SPY1 0.979 · I-SPY2 0.994.

**How to read a probe score:** ≥0.90 the result is contaminated · ~0.70 report it beside
the result · ~0.50 pooling is safe.

**The mechanism, demonstrated:** the same 72 I-SPY2 Luminal B patients inside an
all-I-SPY2 dataset dropped from F1 ≈ 0.98 to **0.077**.

**The probe doubles as proof the pipeline is correct.** A broken pipeline could not reach
0.9978. It is now run on every new dataset before anything is trusted.

**Mitigations applied:** single-source default for the classifier phase; the 80 mm
physical window (equalises the resampling factor, removing the resolution signature); an
identical framing rule for masks and boxes; and the probe reported beside every pooled
result.
**Mitigations proposed and never done:** ComBat harmonisation, adversarial de-biasing, and
the cohort-based federated partition.

---

## 8. ABLATION STUDIES

### The 2×2 — normalisation × freezing (two seeds per cell)

| configuration | normalisation | augmentation | freezing | **test AUC** | train acc | **gap** |
|---|---|---|---|---:|---:|---:|
| **R18** | min–max | 100% | none | **0.6078 ± 0.026** | 0.645 | 0.173 |
| **FREEZE_R18** | min–max | 100% | layer3 | **0.6159 ± 0.003** | 0.599 | 0.162 |
| **CC_R18** | chanclip | 100% | none | **0.5830 ± 0.017** | 0.572 | 0.135 |
| **CCHALF_R18** | chanclip | **50%** | none | **0.5680 ± 0.019** | **0.994** | **0.512** |
| **CCHALF_FREEZE_R18** | chanclip | **50%** | layer3 | **0.5784 ± 0.030** | 0.839 | 0.393 |

**All AUC differences are inside the noise floor.** Two things are real:
1. **Freezing stabilises** — seed spread 0.026 → 0.003, nearly ten-fold. But **per seed
   the train/test gap moved in opposite directions**, so "reduces overfitting" is *not*
   supported; "stabilises the result" is. And `layer3` frees only **6.1%** of parameters —
   the honest test is `layer4` (25%), never run.
2. **Halving augmentation was a disaster** — train acc 0.57 → 0.99, gap tripled, best
   epochs moved to 59 and 26. **The current augmentation is what holds the model back from
   memorising.**
3. **`chanclip` lost** by 0.025 on both seeds, despite winning the authors' own seven-way
   normalisation benchmark.

### The production freezing ablation (2026-08-03, 10 epochs, seed 42)

| run | freeze | best epoch | train acc | **test AUC** | gap (val / test) |
|---|---|---:|---:|---:|---|
| `freeze_layer3_seed_42` | layer3 | 4 | 0.7665 | **0.6067** | 0.2068 / 0.2627 |
| `freeze_none_seed_42` | none | 4 | 0.7821 | **0.5989** | 0.2560 / 0.2635 |

Same direction, same magnitude, inside noise. It confirmed the configuration choice for
the federated campaign rather than establishing a new fact.

---

## 9. AUTHOR REPRODUCTION — the BreastDCEDL audit

**Phase 6.** The authors' repository was cloned in full (196 commits) and audited file by
file. Report: `docs/BREASTDCEDL_REPRODUCIBILITY_REPORT.md`.

**Five findings, in the order they were established:**

1. **There is no training code, anywhere.** All 16 notebooks and both `.py` files were
   searched for `loss.backward`, `optimizer.step`, `.fit(`, `Trainer(`, `for epoch`,
   `model.train()`, `scheduler`, `state_dict`. Then **every deleted file across 196 commits
   was recovered and searched too** — including two that sound promising and are not. The
   recovered `modeling_ispy2` notebook contains **`torch: 0` occurrences**. The only
   `.fit()` calls anywhere are scikit-learn on tabular features.
   **Consequence: no learning rate, batch size, epoch count, scheduler, weight decay,
   augmentation parameter, seed or checkpoint rule is published.**
2. **The published inference notebook does not work as written.**
   `Image.fromarray(im, mode="RGB")` on a float64 array makes PIL reinterpret the raw
   buffer as bytes. Measured on the authors' own sample patient, **the correlation between
   what the model receives and the actual MRI is 0.0114.** Their results are not wrong —
   the notebook as published is simply not the code that produced them.
3. **The checkpoint predictions shipped in `transformer_models/` do not reproduce the
   article** — AUC 0.5158 and 0.5852 against 0.7201, values spanning 0.478–0.524 with
   sd 0.010: a model outputting ~0.5 for everyone.
4. **The published results ARE verifiable from their own prediction file** — 0.7201
   overall / 0.7801 I-SPY2 / 0.6793 I-SPY1 / 0.5398 DUKE, against 0.72 / 0.78 / 0.68 /
   0.54 published. Accuracy 0.754, sensitivity 0.269, specificity 0.959 against 75% /
   0.27 / 0.95. **Every number matches.**
5. **The famous AUC 0.94 is a tabular model, not the ViT** — on the HR+/HER2− I-SPY2
   subgroup imaging alone gives 0.8886 and the clinical model gives 0.9371. **Do not chase
   0.94.**

**Other issues:** Windows absolute paths throughout; no `requirements.txt`;
`crop_around_voi_cords` redefined **four times** in one notebook with different edge
behaviour; patient counts disagreeing between the paper (176/177) and the prediction file
(175); 916 vs 918 DUKE rows.

**Six reproduction runs, one seed each**, on datasets built with their exact rules:

| run | model | best epoch | our test AUC | published | delta |
|---|---|---:|---:|---:|---:|
| PAPER_subtype_vit | ViT-MAE | 2 | **0.6298** | never attempted | — |
| PAPER_subtype_r18 | ResNet-18 | 68 | 0.6153 | — | — |
| PAPER_her2_vit | ViT-MAE | 3 | 0.5904 | 0.744 | **−0.154** |
| PAPER_pcr_r18 | ResNet-18 | 28 | 0.5667 | 0.72 | **−0.153** |
| PAPER_pcr_vit | ViT-MAE | 65 | 0.5324 | 0.72 | **−0.188** |
| PAPER_her2_r18 | ResNet-18 | 1 | 0.4351 | — | below chance |

**On their binary tasks we fall 0.15–0.19 short — two to three times the noise floor. On
the 3-class task the two pipelines tie.** The pattern (best epochs 1, 2, 3, 28, 65 with
`train_acc` reaching 0.999) points at wrong hyperparameters for the task — which is
exactly what the authors do not publish.

**Adopted from them:** the MinCrop geometry, the official split, RGB phase fusion, the
phase-selection rule, median slice aggregation for HER2, `vit_mae_base` as a comparison
model. **Rejected:** the `cv2.resize` branch (verified NOT used for the released data),
the `Image.fromarray` defect (corrected, recorded as `fromarray_fix: true`), and
regenerating MinCrop at all.

**Never done, and it costs minutes:** running their released ViT weights
(`raw_dataset_BreastDCEDL/BreastDCEDL_models.tar.gz`, 343 MB, 85,800,194 parameters) through
our preprocessing. That is the decisive test separating "our pixels are wrong" from "the
deficit is all training".

---

## 10. NVIDIA FLARE DEVELOPMENT

### 10.1 The first federated system (archived)

`unused/legacy_projects/federated_breast_classification/` (7.7 GB) — the binary TripleNeg
pipeline and its NVFLARE harness. It copied `model.py` to **28 places** and needed a
`sync_model.py` to keep them consistent. That is where the lesson came from:

> **FedAvg only averages correctly if every site builds an identical network.**

A second harness, `federated_breastdcedl/`, pointed at BreastDCEDL. Its first run is in
`unused/old_runs/resultados_federado/` — test 1 finished, test 2 stopped at round 8.

### 10.2 The current infrastructure (built 2026-08-03)

Built to a 15-section specification, and the design principle was stated up front:
**`production/` holds no second definition of any hyperparameter.** `config/experiments.py`
is the single source of truth; `production/jobs/` is generated from it;
`production/config/` is a snapshot nothing reads back; `production/scripts/` are wrappers.

Delivered:
* `project.yml` — `api_version: 3`, server (ports 8002/8003), four hospital clients,
  `admin@ips.pt` as `project_admin`, four builders.
* PKI workspace `production/workspace/breast_fl_project/prod_00/` with startup kits for
  all six participants.
* **13 job folders** (test01–test13), each a generated `job.py` plus a README.
* Four partitions, built 2026-08-03T23:03:03–23:03:33 UTC, seed 42, patient-level,
  **hardlinked** (verified by inode, 136/136). *(Six today — the two RQ2 partitions were
  added 2026-08-05; see §11.5.)*
* Distribution figures — 3 overviews + 13 per-test, `.pdf` and `.png`.
* `verify_production.py` — **198 pre-flight checks** *(219 today; §13 item 4)*.
* `start_federation.sh` — server first, then poll the admin port, then the hospitals, each
  a separate OS process with `OMP_NUM_THREADS=1`.
* Per-participant logging (`server.log`, `hospital_N.log`, `admin.log`, `timeline.log`).
* `build_final_summary.py` (~1500 lines) — 8 comparison tables, 8 figure types, and
  `summary.{csv,xlsx,json,md,pdf}` plus 9 LaTeX tables.
* A 25 KB README documenting 22 points.

**Naming decision:** the experiment folders were renamed to `testNN_*` (`test01_centralized`
… `test13_fedopt_skewed`) at the user's instruction.

**Configuration decisions taken in this phase, and the reason for each:**

| decision | reason |
|---|---|
| the **PyTorch** recipe, not the generic one | `nvflare.recipe.FedAvgRecipe` accepts `model` only as a dict; the PT recipe takes the built `nn.Module` |
| `key_metric = "val_balanced_accuracy"` | a small site's validation split can be missing a class, making macro-AUC NaN. And **never** training accuracy |
| budget matching, 30 rounds × 1 = 30 epochs | otherwise RQ1 reads a difference in compute as a difference in federation |
| closed-form cosine LR from `current_round` | a client holds no state between rounds; dropping the schedule or restarting it each round would turn RQ1 into a comparison of schedules |
| the reported loss **excludes** the proximal term | including it would make FedAvg and FedProx losses incomparable across the curves RQ3 is read from |
| four hospitals always provisioned | so a difference between two results can never be a difference in PKI |
| `admin@ips.pt` | NVFLARE validates the admin name against a full e-mail regex; `admin` and `admin@ips` both exit `INVALID_ARGS` |

### 10.3 The admin-rename request

The user asked to rename `admin@ips.pt` → `admin` *"if NVFLARE allows this
configuration"*. **It does not.** NVFLARE 2.8 validates admin names against
`^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$`, hard-coded in
`nvflare/apis/utils/format_check.py`. Provisioning with `name: admin` exits `INVALID_ARGS`.
**The identity stays `admin@ips.pt`.**

---

## 11. FEDERATED EXPERIMENTS

### 11.1 Campaign A — 2026-07-31, 4-class subtype, 1,488 patients (ARCHIVED)

`unused/old_runs/results_01_to_08/01_federated_tests_1to9/`. Real NVFLARE production
infrastructure — PKI startup kits, aggregation server on 8002/8003, each hospital an
independent process, jobs submitted through the admin API, **42.7 MB of weights per
round**, 20 rounds × 1 local epoch, 100% client participation. RTX 4090, 00:50–03:19 UTC.
Evaluated on 224 validation patients; trivial baseline 0.388.

| Test | config | clients | algorithm | accuracy | bal acc | macro F1 | **macro AUC** |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | centralised | 1 | — | 0.3259 | 0.3520 | 0.3188 | 0.6020 |
| 2 | 50/50 | 2 | FedAvg | 0.3750 | 0.3012 | 0.3002 | 0.5828 |
| 3 | 50/50 | 2 | FedProx | 0.3705 | 0.2916 | 0.2905 | 0.5854 |
| 4 | 33×3 | 3 | FedAvg | 0.3929 | 0.3199 | 0.3216 | 0.6050 |
| 5 | 33×3 | 3 | FedProx | 0.3839 | 0.3114 | 0.3085 | 0.6076 |
| 6 | 25×4 | 4 | FedAvg | 0.3705 | 0.3227 | 0.3237 | 0.5763 |
| **7** | 25×4 | 4 | **FedProx** | 0.3929 | 0.3294 | 0.3331 | **0.6181** |
| 8 | 50/20/10/10 | 4 | FedAvg | 0.3259 | 0.2587 | 0.2585 | 0.5747 |
| 9 | 50/20/10/10 | 4 | FedProx | 0.3616 | 0.2971 | 0.3006 | 0.5880 |

**Findings recorded then:** federated matched centralised (−0.03 to +0.05, no consistent
direction); **test 7 beat centralised**; quantity skew cost 0.065 macro-F1 (6→8) while
barely moving AUC — the model kept its ranking ability and lost decision calibration; and
**the model saturates after round 1**, which already contains 99.3% of the final macro-F1,
so ~90% of communication traffic is wasted.
**Two caveats recorded at the time:** all nine partitions were statistically IID
(TV-distance ≈ 0.00), and all nine ran on the **contaminated 4-source** dataset.

### 11.2 Campaign B — 2026-08-01/02, binary TripleNeg, ResNet-50 (ARCHIVED)

`unused/old_runs/results_01_to_08/08_federated_final/`. 50 rounds × 1 local epoch,
FedProx μ = 0.01, I-SPY2, 99 test patients, trivial baseline 0.6263. Queue started
2026-08-01 23:51 UTC, finished 2026-08-02 12:18 UTC; per-test wall clocks 4,771–6,545 s.

| # | configuration | hospitals | algorithm | **macro-AUC** | TN recall |
|---:|---|---:|---|---:|---:|
| **1** | **centralised** | — | — | **0.6874** | 48.6% |
| 5 | balanced | 3 | FedProx | 0.6194 | 40.5% |
| 9 | skewed | 4 | FedProx | 0.6011 | 27.0% |
| 4 | balanced | 3 | FedAvg | 0.5985 | 40.5% |
| 8 | skewed | 4 | FedAvg | 0.5968 | 29.7% |
| 7 | balanced | 4 | FedProx | 0.5929 | 40.5% |
| 3 | balanced | 2 | FedProx | 0.5824 | 40.5% |
| 6 | balanced | 4 | FedAvg | 0.5815 | **51.4%** |
| 2 | balanced | 2 | FedAvg | 0.5776 | 35.1% |

**RQ1 — no.** Centralised 0.6874 against 0.5776–0.6194: a drop of **0.068 to 0.110**, at
or above the noise floor.
**RQ2 — no detectable effect**, consistent with the skew being quantity-only.
**RQ3 — FedProx won 4 of 4 paired comparisons** (+0.005, +0.021, +0.011, +0.004). Each
inside the noise floor; 4/4 in one direction is a **trend, not a fact**.
**Two secondary findings:** the effect is **all-or-nothing** (2, 3 and 4 hospitals gave the
same result — *contradicting* an earlier lung-segmentation project where degradation was
progressive); and **federation hurts the clinically important class** (TripleNeg recall
fell from 48.6% to 27–40% in eight of nine configurations).

Test 6 initially **failed** at collection after 5,826 s (`collect_results.py` had
`resnet18` hard-coded) and was re-run.

### 11.3 Campaign C — 2026-08-04, tests 01–09

**NVIDIA RTX 4000 Ada (20 GB), RunPod, CUDA 12.8, torch 2.8.0, NVFLARE 2.8.0.**
**47.9 minutes for the whole matrix. Zero failures. One run per job, seed 42.**

Timeline from the logs: the centralised baseline finished `2026-08-03T23:48:43Z`; test06
was submitted `2026-08-04T00:35:44Z` and finished `00:41:01Z`; test09 was submitted
`00:53:14Z` and took `0:06:24`.

| test | algorithm | hosp | best round | time (s) | accuracy | bal acc | macro F1 | **macro AUC** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test01 | centralized | 1 | — (epoch 4) | 268 | **0.5299** | 0.4503 | 0.4523 | **0.6068** |
| test02 | fedavg | 2 | 25 | 313 | 0.4030 | 0.3742 | 0.3744 | 0.5594 |
| test03 | fedprox | 2 | 29 | 351 | 0.4328 | 0.4025 | 0.4001 | 0.5917 |
| test04 | fedavg | 3 | 27 | 313 | 0.4851 | 0.4198 | 0.4231 | 0.5990 |
| test05 | fedprox | 3 | 28 | 333 | 0.4590 | 0.4127 | 0.4116 | 0.5958 |
| test06 | fedavg | 4 | 0 | 317 | 0.4776 | 0.4522 | 0.4378 | **0.6531** |
| test07 | fedprox | 4 | 0 | 335 | 0.4739 | 0.4393 | 0.4362 | 0.6075 |
| test08 | fedavg | 4 skew | 21 | 323 | 0.4888 | 0.4259 | 0.4292 | 0.5982 |
| test09 | fedprox | 4 skew | 2 | 387 | 0.4515 | 0.4210 | 0.4197 | 0.6250 |

*(These are the values in `final_summary/summary.csv` as regenerated 2026-08-05. The
2026-08-04 build of the same table differed by ≤0.0005 macro-AUC because the test-set
evaluation was re-run locally afterwards; see `PROJECT_CONTEXT.md` §10.1.)*

**The honest reading, and it is the one that must be carried forward:**
* **No comparison in this table is attributable.** Noise floor 0.067; spread 0.094.
* **Three federated runs scored ABOVE the centralised baseline** (test06, test07,
  test09; test04 is 0.008 below) — the signature of noise dominating, not of federation
  outperforming pooled training.
* **What it supports:** federated training produces models in the **same range** as
  centralised training on this task. (Reframed later as an equivalence claim once all 13
  runs existed — see §11.5 and `PROJECT_CONTEXT.md` §1.5.)
* **Accuracy is the real finding.** Every federated run lands **below** the trivial
  baseline of 0.5112; only the centralised run clears it. The models rank patients better
  than chance (AUC 0.56–0.65) but **decide** worse than a constant rule.
* **HER2+ per-class AUC 0.5079 in the centralised run — chance.**
* Two patterns worth repeating with more seeds and **not worth claiming**: more hospitals
  scored *higher*, not lower; and FedAvg vs FedProx **flips sign** across configurations.

**Note the disagreement with Campaign B:** on the binary task federation cost 0.068–0.110;
here three federated runs beat the baseline. **This is an open question, not a resolved
one**, and the resolution is more seeds.

**What this campaign could not answer.** All four partitions were stratified, so the
class-share spread between hospitals never exceeded 0.43 pp. Tests 08/09 therefore varied
*quantity* and nothing else, and RQ2 had no defensible answer at the close of Campaign C.
That gap is what Campaign D was built to fill.

### 11.4 FedOpt — cancelled 2026-08-04, removed from the table 2026-08-05

Added after 01–09 completed, at the user's request, to run **on the MacBook CPU**, and
numbered test10–test13. Same partitions, clients, seed and rounds; only the server's
update rule differed (SGD lr 1.0, momentum 0.6, client mu = 0).

* **test12 and test13 failed immediately** — `TypeError: FedOptRecipe.__init__() got an
  unexpected keyword argument 'key_metric'`. Fixed with `common.pop("key_metric", None)`.
  **Consequence: FedOpt has no server-side model selection — it keeps the LAST round,
  while FedAvg/FedProx keep the best.**
* **test10 ran on CPU** — job `7e92f496-290f-4db2-aaf1-84e36347e3f8`, submitted
  2026-08-04T10:36:55Z, reached **round 19 of 30**.
* The user then said *"poide cancelar os teste4s"*. **Cancelled.** Partial output is
  preserved at `unused/reference_implementations/fedopt_cancelled_2026-08-04/`; nothing
  was deleted.

**On 2026-08-05 the four rows were deleted from `experiments.py` entirely**, at the
user's instruction (*"conseguru ignorar 10 ao 13 e remonear 14-17"*), and the ids were
reused by the RQ2 campaign. The reasoning is written into the source at the point where
the rows used to be: nothing completed, so nothing can be reported, and four permanently
blank rows in a results table invite the same question in every chapter. **No FedOpt
number appears anywhere in the reported campaign** — verified against
`all_experiments.csv`, `summary.csv` and `manifest.json`, each of which lists exactly 13
experiments.

⚠ **Anything written between 2026-08-04 and 2026-08-05 that says "test10 = FedOpt" is
from that window.** The ids now mean the cohort pair.

### 11.5 Campaign D — 2026-08-05, tests 10–13, THE RQ2 PAIR

**A second rented RunPod host, same image: RTX 4000 Ada, CUDA 12.8, torch 2.8.0,
NVFLARE 2.8.0. Four sequential jobs finishing 01:16:51, 01:25:10, 01:33:40 and
01:42:13 — ~8 minutes each, ~34 minutes end to end, zero failures, seed 42.**

**Why it exists.** RQ2 asks what non-IID heterogeneity costs, and Campaign C could not
answer it: its partitions were stratified to within 0.43 pp, so what tests 08/09 varied
was quantity. The user proposed the fix directly — *"se eu fizesse um teste com 3
hospitais um só com duke outro só cada set e ver como fica"* — and then, crucially,
asked for the control in the same breath: the same three site sizes with the datasets
mixed. That second half is what turns a demonstration into a measurement.

**The design.** Two partitions holding identical site sizes (642 / 101 / 784 patients),
identical client count, rounds, local epochs, seed, model and evaluation. In
`3_clients_cohort` each site *is* one real cohort (DUKE / I-SPY1 / I-SPY2, assigned in
sorted order); in `3_clients_sizematched` the same three sites hold a stratified draw
from all three. Measured class-share spread: **27.45 pp against 0.32 pp**. Each partition
ran once with FedAvg and once with FedProx.

**The renumbering.** The four runs were first numbered test14–test17, in the order
cohort-FedAvg, sizematched-FedAvg, cohort-FedProx, sizematched-FedProx. After FedOpt was
dropped they were renumbered into the freed 10–13, and then reordered again so each
*partition* owns a consecutive FedAvg/FedProx pair (10/11 cohort, 12/13 control), which
is how 02–09 are numbered. Result folders, `experiments.py`, `all_experiments.csv`,
`summary.csv` and the job definitions were all renamed together;
**`per_client_metrics.csv` was not**, and still carries the test14–17 ids (see §12,
bug 29).

| test | partition | algorithm | best round | time (s) | accuracy | bal acc | **macro AUC** |
|---|---|---|---:|---:|---:|---:|---:|
| test10 | one cohort each | fedavg | 7 | 469 | 0.4291 | 0.3582 | **0.5426** |
| test11 | one cohort each | fedprox | 16 | 482 | 0.4590 | 0.4105 | 0.5678 |
| test12 | size-matched | fedavg | 17 | 471 | 0.4478 | 0.4183 | 0.5836 |
| test13 | size-matched | fedprox | 27 | 485 | 0.4664 | 0.3885 | 0.5882 |

**What it produced.**

1. **The first consistent RQ2 answer.** Cohort-native costs **−0.041** under FedAvg and
   **−0.020** under FedProx. Neither clears the noise floor alone; both point the same
   way, where the quantity-skew pairs had disagreed in sign (−0.055 and +0.018).
2. **The minority class collapses.** HER2+ recall 0.321 → 0.113 and its AUC to 0.4728,
   below chance. Not visible in any aggregate metric.
3. **FedProx finally has something to correct** — +0.025 on the cohort partition against
   +0.005 on the control, and it restores HER2+ recall to 0.283. On the stratified
   partitions its effect had flipped sign four times out of four.
4. **Convergence slows under heterogeneity.** Test10 reaches 90.9% of its best aggregated
   validation AUC at round 1 and needs five rounds to reach 95%, where most stratified
   runs need one. An independent signature of the same effect, read off the round curves.
5. **A confound that was created by the same run.** `class_weight_scope` is `"local"`, so
   on a 27.45 pp prior spread the three sites optimised measurably different objectives.
   Part of the measured cost may be that mismatch rather than heterogeneity as such —
   which is now the highest-value follow-up (`PROJECT_CONTEXT.md` §21 item 2).

**The user's judgement on the design, recorded because it was right:** the matched
control was their idea, not an addition made afterwards, and it is the reason the result
is quotable at all.

---

## 12. BUGS DISCOVERED, IN THE ORDER THEY WERE FOUND

Full PROBLEM/CAUSE/DETECTION/SOLUTION/VALIDATION write-ups are in
`PROJECT_CONTEXT.md` §17. Chronologically:

1. **`StratifiedKFold` on slices, not patients** (radiomics phase) — the original leak.
2. **The dataset-source shortcut** — Luminal B F1 0.98; source probe 0.967 vs subtype
   0.589. The central finding.
3. **A claimed gain 0.616 → 0.654 that was a changed validation split** — only 34 of 224
   patients in common; ~70% of the "gain" was an easier split.
4. **`_replace_head` inserted a `Dropout` INTO an existing `Sequential`**, shifting
   classifier indices so checkpoints failed to load.
5. **The "fix" for (4) deleted the Dropout instead**, leaving `dropout` a dead config
   field for every torchvision backbone while `results.json` kept recording `dropout: 0.5`.
6. **`torch.load` default `weights_only=True`** made checkpoints unreadable.
7. **`train_paper.py` read `model_name` while the CLI passed `--model`** — 10 runs
   silently trained the wrong architecture.
8. **`TRIVIAL_BASELINE_ACC` hard-coded to 40/99** — every `results.json` carried 0.404 when
   the truth was 0.5112.
9. **Variable `idx` shadowed** (DCE phases vs slice midpoint) — DCE phase indices silently
   erased.
10. **`startswith("p")` matched `pid`** — tried to average patient identifiers.
11. **`FedAvgRecipe` exported the model as a dotted path** — the server built a ResNet-18
    while the clients built a ResNet-50; the run completed and the numbers were
    meaningless.
12. **The server selected on TRAINING accuracy** — it picked whichever global model let
    clients memorise their own shard best (99%+).
13. **`collect_results.py` had `"resnet18"` hard-coded** — federated runs trained
    ResNet-50 for 50 rounds then crashed at evaluation.
14. **`cls_trainer.py` hard-coded dropout and weight decay** — federated clients were
    regularised differently from the baseline they were compared against.
15. **`pkill -f 'pattern'` matched the ssh command running it** and killed its own shell.
16. **Apple MPS corrupts weights to NaN** — non-deterministic, same seed gave 0.6312 and
    0.6832; `torch.mps.synchronize()` made it deterministic but not finite. Recorded here
    for a long time as **root cause never established**. ★ **Root-caused 2026-08-05** —
    see bug 30.
17. **`nc` absent from the RunPod container** — `start_federation.sh` polled the admin port
    with it, the check failed **silently**, and hospitals never started.
18. **NVFLARE could not JSON-serialise a torchvision ResNet** — `self._norm_layer` is a
    *class*, and without a fix NVFLARE would have rebuilt a **default 1000-class** model on
    the server against 3-class clients.
19. **`FedOptRecipe` rejects `key_metric`** — tests 12 and 13 failed 0.1 min after launch.
20. **`start_federation.sh` regex rejected test10+** — `^(test0[1-9]|_scratch)$`.
21. **`setsid` does not exist on macOS**; **zsh aborts a command list on a failed glob**, so
    a log `rm` never ran and two runs appended to one file.
22. **`fmt_frame` left a real `nan` in an object column**, which reached `str.join` and
    raised.
23. **`rounds.csv` is written to `<exp>/sites/`, not `<exp>/`** — the round-evolution
    figure was **silently empty**.
24. **`fig5` plotted the reciprocal resampling factor** (0.52× instead of 1.96×).
25. **`verify_production` only *built* jobs, never exported them** — **two failures had
    slipped past**.
26. **FedAvg was compared against a *mean* of configurations** instead of its paired
    FedProx run.
27. **README patient counts were guessed** (392/392 etc.) and were corrected from measured
    data (393/391 · 262/262/260 · 198/196/195/195).
28. **A false claim in a docstring** — it said workers caused a SIGKILL, when
    `effective_num_workers` already returns 0 on macOS. Corrected.

**Found on 2026-08-05:**

29. **`per_client_metrics.csv` went stale across the renumbering.** It still carries the
    pre-renumbering ids `test14`–`test17` and holds no rows for tests 02–09. **Cause:**
    the last summary rebuild ran with `--no-client-eval`, which regenerates `summary.csv`
    but leaves the per-client file untouched. **Detected** by reading the file against
    `experiments.py`, where those ids do not exist. **Fix (not yet applied):** re-run
    `build_final_summary.py` without the flag; every input still exists. **Consequence
    until then:** `PROJECT_CONTEXT.md` §10.2 is the only surviving record of the tests
    02–09 per-hospital numbers.
30. ★ **The Apple-MPS NaN, root-caused after two false fixes.** **Cause:**
    `x.to(device, non_blocking=True)` — an asynchronous host-to-device copy, which is only
    safe when the source is pinned, and this project pins only on CUDA. On MPS the copy
    returned before finishing while the DataLoader reused the buffer, so the network
    trained on partially overwritten batches. **Detection:** bisection between a passing
    and a failing path over a **full** epoch. Two earlier attempts declared it fixed and
    were wrong — the first tested 61 steps of a 508-step epoch, the second changed the
    `GradScaler` and never re-ran `run()`. The user's *"isso voce nao pesquisou direito
    como resolver"* was correct both times. **Fix:** `non_blocking = device.type ==
    "cuda"` in `src/core/training.py`, `src/core/evaluation.py` and
    `src/federated/common/training.py`; `get_device()` rebuilt as a CUDA → MPS → CPU
    cascade; the AMP scaler gated on `is_enabled()`; `apply_mps_workaround()` — dead code,
    defined but never called — wired into `run()`. **Validation:** one full epoch, MPS
    loss 1.1539 / acc 0.4372 in 231 s against CPU 1.1502 / 0.4237 in 589 s.
31. **Two files still carry the old MPS diagnosis.**
    `src/federated/common/models.py::get_device` defaults to `allow_mps=False` with a
    docstring saying MPS "is BROKEN here", and `requirements.txt` repeats it. Harmless in
    effect — the campaign ran on CUDA — but the stated reason is now wrong. **OPEN.**
32. **A 76 GB stale duplicate of the whole repository** exists at
    `.../federated-breast-classification`, created when a `mv` moved the project into an
    existing folder of that name. It looks complete and is not: its notebook 03 is the old
    12-cell version and it predates the MPS fix and Campaign D. Left in place by the
    user's choice. **OPEN, and it is a trap rather than a backup.**
33. **Two notebook headings were not renumbered.** `04_evaluate_run.ipynb` opens with
    `# 06 —` and `05_compare_experiments.ipynb` with `# 07 —`. Cosmetic. **OPEN.**
34. **The notebook epoch display read as an off-by-one.** The progress bar showed
    `epoch 003` while the newest completed line said `epoch 002` — correct behaviour
    (the bar labels the epoch being trained, the line the one that finished) displayed
    ambiguously. **Fixed 2026-08-05:** the bar now says `training` and the summary line
    `done`, with matching zero-padding. A genuine `epoch + 1` off-by-one in the bar label
    had been fixed earlier.

**Still open:** the misleading `"cohorts": ["spy2"]` field in the centralised results;
`authors_subtype` blocked by `Config`; federated class weights computed locally (**now
material — tests 10/11 ran that way on divergent priors**); `fig2_tumour_size_by_cohort`
(status of the cosmetic fixes **NOT VERIFIED**); `per_client_metrics.csv` (bug 29); the
two stale MPS statements (bug 31); the duplicate repository (bug 32); and an uncommitted
working tree. **Closed since the previous version:** the stale `COHORT_DIRS` path (fixed
in the reorganisation) and version control (the repository is now under git, though its
history is two commits made outside the working sessions).

---

## 13. FIXES — the structural ones, not the one-liners

The individual fixes are listed with their bugs. What matters for the history is the
**five structural changes** that made whole classes of bug inexpressible:

1. **One declarative experiment table.** `src/federated/config/experiments.py` holds the
   thirteen experiments, six partitions, `TrainingConfig` and `FederationConfig`. Jobs are
   **generated** from it (`generate_jobs.py`, with a `--check` mode); `deployment/config/`
   is a snapshot nothing reads back; `deployment/scripts/` are wrappers. Bugs 11, 13 and 14
   all had one cause — two copies of a setting drifting apart — and that cause is now
   removed. **This is also what made Campaign D cheap:** adding two partitions and four
   experiments was an edit to one file, and the renumbering that followed touched that
   file plus generated artefacts rather than thirteen hand-written configs.
2. **An architecture fingerprint (`2d3031acc2075813`) checked at the server and every
   client**, plus `strict=True` on every checkpoint load. Bug 5 taught that a parameter
   count proves nothing (`Dropout` has no parameters — both builds total 11,187,671) and
   that `strict=False` "succeeds" while leaving the classifier at random init, which on
   this task still reads as a plausible near-chance result.
3. **The head-construction invariant.** *Never insert into an existing `Sequential`.* The
   final `Linear` is only ever replaced in place; a backbone that already ships a `Dropout`
   has it **retuned**, not stacked. At `dropout = 0` the key layout is exactly
   torchvision's. Verified: all seven checkpoints load `strict=True`.
4. **219 pre-flight checks** (`verify_production.py`) that write nothing and must pass
   before any federation starts — structure, `project.yml` against `federation.py`, PKI,
   partitions against requested shares, budget equality, FedProx mu per job, unique result
   names, and job **export**, not merely build. It grew from 198 to 219 with Campaign D,
   and one check had to be **taught about non-stratified partitions**: the stratification
   assertion now reads `Partition.stratified` and, for a partition declared
   non-stratified, asserts the class spread is **at least** 0.05 — so a cohort partition
   that silently came out stratified would fail rather than pass.
5. **The repository reorganisation, 2026-08-05.** Seven folders, one purpose each
   (`raw_dataset_BreastDCEDL/`, `dataset/`, `src/`, `deployment/`, `results/`, `docs/`,
   `notebooks/`), a README in every one, and a root that holds only `README.md` and
   `requirements.txt`. Three things it fixed structurally rather than cosmetically: the
   root `config.py` was renamed `src/dataset_config.py` so it can no longer shadow
   `src/federated/config/`; `RAW_DIR` now points at the imaging release rather than at the
   authors' code clone, and the fallback resolver that had been hiding that breakage was
   deleted; and the raw dataset became reproducible from a `download_dataset.py` that
   reads the Zenodo file list at run time and prints manual instructions on **every**
   failure path.

Plus the standing rules: patient-level splitting enforced by a verification step; accuracy
never quoted without the trivial baseline computed from the split; 0.067 treated as the
noise floor with a `within_noise_floor` column on every table; one log file per
participant; hardlinked data verified by inode; and **move, never delete**.

---

## 14. CURRENT STATUS — 2026-08-05

**Finished:** the classifier phase (21 runs, 13 architectures, 5 data configurations); the
dataset (`multi_subtype_80mm`, built, audited, documented, figured); the source probe; the
2×2 ablation; the BreastDCEDL reproducibility audit; the NVFLARE production infrastructure;
**all thirteen dissertation experiments across two campaigns**; the aggregated final
summary; the repository reorganisation and its documentation; and the MPS root cause.

**Running:** nothing. No process, no job, no rented GPU. Both hosts were released.

**Verified:** MinCrop geometry 767/767 · Duke depth 12/12 · cropped-not-resized 228/228 ·
Duke tumour presence by enhancement physics (96%, 4.69×) · hardlinks 136/136 by inode ·
partitions reconcile to 2,063 patients / 16,378 images across all six · budget equality ·
**all 219 pre-flight checks pass, re-run 2026-08-05** · all seven checkpoints load
`strict=True` · pipeline integrity (0 divergences in 20,028 rows) · **every dataset count,
every class spread and the whole RQ3 convergence table recomputed from the files on
2026-08-05** · zero patients in more than one split and zero with more than one label.

**Immediately outstanding:** regenerate `per_client_metrics.csv` without
`--no-client-eval`; the authors' released weights through our preprocessing; the two
stale MPS statements; the misleading `"cohorts"` field; a decision about the 76 GB
duplicate repository; committing the working tree; and `fig2_tumour_size_by_cohort`,
whose cosmetic fixes are **NOT VERIFIED**.

**The state of each research question:**

| | current answer | confidence |
|---|---|---|
| **RQ1** | **Yes, by equivalence.** Centralised 0.6068 against a federated mean of 0.5927 over twelve runs — gap 0.0141, **4.8× smaller** than the measured 0.067 margin, every run inside it | **Moderate, and it is a positive claim.** What limits it is that the margin is applied to point estimates rather than intervals; three seeds would give the strong form. Campaign B's binary task disagreed (federation cost 0.068–0.110) and that disagreement is unresolved. |
| **RQ2** | **Real heterogeneity costs performance**: −0.041 (FedAvg) and −0.020 (FedProx) between cohort-native sites and a size-matched control, and **HER2+ recall collapses 0.321 → 0.113** | **Moderate in direction, none in magnitude.** Both differences are inside the noise floor; what supports the claim is that two independent comparisons agree (p = 0.25 under the null). Confounded by `class_weight_scope = "local"` on divergent priors. The earlier quantity-skew answer ("no detectable effect") stands and is now correctly labelled as a statement about quantity skew. |
| **RQ3** | **Communication: ~87% of a 30-round schedule is wasted** — the global model is at 94–98% of its best after one round on stratified partitions, and visibly slower under heterogeneity. **FedProx: +0.025 where sites genuinely differ against +0.005 on the matched control** | **Moderate for communication** (a large, consistent effect across eight runs). **Low for FedProx** — the gain is inside the noise floor, but the mechanism and the measurement agree, which the stratified partitions never did. |
| **RQ4** | FedProx under genuine heterogeneity is the one mitigation where mechanism and measurement agree, and it partially restores HER2+ recall (0.113 → 0.283). The security measures are implemented and verified by 219 checks | **Low.** The class-weight-scope experiment — the direct privacy-versus-performance measurement — has still never been run, and is now doubly motivated. |

**The single sentence a reader should take away:** the infrastructure objective (OBJ3) is
fully met and demonstrated; RQ1 and RQ2 now have defensible answers of the right shape —
an equivalence claim and a consistent direction — and both are **single-seed**, which is
the one limitation that governs the entire results chapter.

---

## 15. FUTURE WORK, IN PRIORITY ORDER

Detail, with hypothesis and expected interpretation for each, is in
`PROJECT_CONTEXT.md` §21.

**Reordered 2026-08-05.** Item 3 of the previous list — the cohort partition — was run
and is now Campaign D. FedOpt was formally dropped rather than completed. Seed repetition
moves to first, because the project now has **two** claims that rest on the noise floor
rather than merely being limited by it: RQ1's equivalence and RQ2's direction.

1. **Three seeds on all thirteen experiments.** ~3.5 h on a rented GPU. Converts RQ1 from
   "the points fall inside the margin" to "the interval falls inside the margin", and
   tests whether the RQ2 direction replicates. Nothing else improves the results chapter
   as much per GPU-hour.
2. **Local vs global class weights on the cohort partition.** The real RQ4 experiment — a
   measured privacy-versus-performance trade-off — and it now also **disambiguates RQ2**,
   because tests 10/11 ran with local scope on a 27.45 pp prior spread. ~30 min.
3. **Authors' released ViT weights through our preprocessing.** Inference only, minutes.
   Separates "our pixels are wrong" from "the deficit is all training".
4. **`--stratify none`** — label skew without cohort identity, the missing rung between
   quantity skew and the cohort partition. Tells you whether the RQ2 effect is priors or
   scanner. Implemented, never run. ~30 min.
5. **`--freeze-until layer4`** — the honest freezing test, 25% of parameters instead of
   6.1%.
6. **Ensemble the existing runs** — free.
7. **FedOpt on the cohort partition, only if the model-selection asymmetry is solved
   first.** `FedOptRecipe` rejects `key_metric`, so FedOpt keeps the last round while the
   others keep the best of thirty; reporting a number without resolving that would be a
   methodological error rather than a caveat.
8. **Reacquire MAMA-MIA** for its 22 real hospital IDs inside I-SPY2 — federated learning
   across real sites, within one cohort, which removes the source confound *and* keeps
   heterogeneity. This would be the strongest version of the whole thesis.
9. **ComBat harmonisation + re-run the source probe** — does harmonisation lower the probe
   below 0.90 without lowering the subtype AUC?

**Explicitly not recommended:** GAN augmentation (best published gain ~0.01 against a
0.067 noise floor, and a subtype-conditioned generator would amplify the scanner
signature); threshold tuning on validation (failed four times); Focal / Class-Balanced
loss (the 2.25:1 ratio is below the band where they help); regenerating MinCrop; and
claiming any improvement from a single run.

---

## Appendix — the timeline in one table

| when | what |
|---|---|
| *(date not verified)* | Phase 1 — `radiomic_ai`, PyRadiomics + Random Forest. Slice-level CV leak. |
| *(date not verified)* | Phase 2 — 4-source, 4-class catalogue (1,488 patients, 206,888 slices). Luminal B F1 ≈ 0.98. |
| *(date not verified)* | The pipeline audit. Seven checks pass. **Source probe 0.967 vs subtype 0.589.** Catalogue abandoned. |
| *(date not verified)* | Phase 3 — I-SPY2 only, three classes. 0.62 macro-AUC. Preprocessing ablation: seven interventions, all fail. |
| *(date not verified)* | `06_ispy2_final` — three seeds, 0.6218 ± 0.014, class ordering matches the literature for the first time. |
| *(date not verified)* | Phase 5 — migration to BreastDCEDL. pCR gives 0.501; target moves to 3-class subtype. |
| *(date not verified)* | MinCrop geometry reverse-engineered and verified 767/767. **DUKE becomes usable.** |
| **2026-07-31** | Campaign A — nine federated tests, 4-class, real NVFLARE, RTX 4090. Centralised 0.6020. |
| **2026-08-01** | Binary TripleNeg ladder — ResNet-50/DenseNet, augmentation, batch, lr, linear probe (0.6813). |
| **2026-08-01 → 08-02** | Campaign B — nine federated tests, binary TripleNeg, ResNet-50. Centralised 0.6874 vs 0.5776–0.6194. |
| **2026-08-02** | Handover documents written. Phase 6 — the BreastDCEDL reproducibility audit. |
| **2026-08-03** | Phase 7 — refactor into two isolated pipelines; `unused/` archive created. `multi_subtype_80mm` built (15:09). Dropout regression found and fixed. NVFLARE production infrastructure built. Partitions built (23:03). Freezing ablation (23:31, 23:33). **Centralised baseline, 30 epochs, no early stopping (23:48).** |
| **2026-08-04 00:00–00:59** | **Campaign C — the eight federated jobs. 47.9 minutes, zero failures.** |
| **2026-08-04 01:12** | `final_summary/` generated — 8 comparison tables, 9 LaTeX tables, all figures, `summary.{csv,xlsx,json,md,pdf}`. |
| **2026-08-04 09:56** | CPU smoke test on the MacBook. |
| **2026-08-04 10:33–11:00** | FedOpt attempt on CPU. test12/13 fail on `key_metric`; test10 reaches round 19 of 30; **cancelled by the user**. |
| **2026-08-04 18:00–19:32** | Scientific dataset & preprocessing report written (`DATASET_REPORT.md`); 12 report figures generated and refined. |
| **2026-08-04** | These two documents written from a full inspection of the repository. |
| **2026-08-04 → 08-05** | Preprocessing walkthrough figures and flowchart built for the methodology chapter; the normalisation figure rewritten after the first version was found to compare two identical histograms. NVFLARE configuration and training-parameter reports written and verified against source rather than prose (which caught an omitted gradient-clipping step and a wrong early-stopping claim). |
| **2026-08-05 00:52** | The two RQ2 partitions built — `3_clients_cohort` and `3_clients_sizematched`, 642/101/784 patients each, class spread 27.45 pp against 0.32 pp. |
| **2026-08-05 ~01:09–01:42** | **Campaign D — tests 10–13 on a second rented RTX 4000 Ada. Four sequential jobs, ~34 minutes end to end, zero failures.** First consistent RQ2 answer; HER2+ recall collapses 0.321 → 0.113. |
| **2026-08-05 02:43–11:04** | Results pulled back, FedOpt removed from `experiments.py`, tests renumbered 14–17 → 10–13 and then reordered so each partition owns a consecutive FedAvg/FedProx pair. `final_summary` rebuilt — **with `--no-client-eval`, which is how `per_client_metrics.csv` went stale**. |
| **2026-08-05 12:00–17:42** | **The repository reorganisation:** seven folders, per-folder READMEs, `src/dataset_config.py`, `RAW_DIR` fixed, `download_dataset.py` written, `.gitignore` rewritten, notebooks renumbered 01–05 and rebuilt to expose every configuration option. Full verification pass over links, DOIs, paths and imports — which found a wrong DOI (HydraMix-Net cited for Bussola's data-leakage paper) and a preprint title used for a published article. |
| **2026-08-05 18:00–19:00** | **The Apple-MPS root cause** — `non_blocking=True` from unpinned memory — found by bisection after two premature "fixed" claims. MPS now trains finite and 2.5× faster than CPU. |
| **2026-08-05** | These two documents re-verified against the files and extended. Every count, spread and convergence figure quoted here was recomputed on this date. |
