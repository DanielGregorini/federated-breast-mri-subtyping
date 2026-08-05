# PROJECT CONTEXT — Master's thesis, complete technical record

**Self-contained context document.** Written so that a researcher — or another AI
conversation — with no access to any previous discussion can pick this project up and
continue it without losing anything. Failed experiments and retired decisions are
recorded as carefully as successes, because in this project several of them are the
most valuable results.

**Last updated: 2026-08-05.** Written after inspecting the actual repository, not from
memory. Companion document: `docs/PROJECT_HISTORY.md` (chronological history).

**What changed on 2026-08-05** — read this first if you knew the previous version:
the campaign grew from **9 experiments to 13**; FedOpt was **removed entirely** and
tests 10–13 now hold the **cohort-heterogeneity pair** that gives RQ2 its first real
answer; the whole repository was **reorganised** (`src/`, `docs/`, `deployment/`,
`results/`, `notebooks/`) so almost every path in the previous version is stale; the
repository is now **under git**; and the Apple-MPS NaN whose cause was recorded as
"never established" has been **root-caused and fixed**.

---

## 0. How to read this document

| marker | meaning |
|---|---|
| *(unmarked)* | Verified against a file in the repository during the writing of this document. The path is given. |
| **[PAPER]** | Reported by an external publication, cited. |
| **[MEASURED]** | A conclusion from our own experiments, with the measurement. |
| **NOT VERIFIED** | Stated somewhere in the project but not confirmable from the files present today. |
| **INFORMATION NOT FOUND** | Asked for, and does not exist anywhere in the repository. |

**Repository root:** `/Users/daniel/Developer/tese/federated-breast-mri-subtyping`
(76 GB; the code itself is under 1 MB).

**⚠ THERE IS A SECOND, STALE COPY ON DISK.**
`/Users/daniel/Developer/tese/federated-breast-classification` is a snapshot taken
2026-08-05 ~13:17 and abandoned. It looks complete and is not. Its
`notebooks/03_train_centralized.ipynb` is the old 12-cell version, its
`src/core/training.py` predates the MPS fix, and it holds no tests 10–13 results.
**Everything in this document refers to `federated-breast-mri-subtyping`.** Verify
which tree you are in before editing anything; comparing file sizes of
`notebooks/03_train_centralized.ipynb` (27,678 bytes live vs 5,119 stale) is the
quickest test.

**The repository IS under git** as of 2026-08-05 — two commits (`9e7dc5e Initial
commit`, `89f0c0c inicial`), both made outside the working sessions, and the five
notebooks are currently modified and uncommitted. The earlier per-session commit
history was replaced by those two and is gone. Nothing has been lost from the working
tree, but the history is not a record of how the project was built.

### The one thing a newcomer must internalise

**A good result on this task is a bug report until proven otherwise.** The project lost
months to a number that looked excellent and was an artefact (§9). Run the source
probe, split by patient, quote the trivial baseline, use at least three seeds, and treat
anything below 0.067 macro-AUC as noise. Everything here that reads as overcaution was
learned by being wrong first.

---

## 1. MASTER'S THESIS CONTEXT

### 1.1 Topic

A master's dissertation comparing **centralised against federated deep learning on
NVIDIA FLARE**, using **breast-cancer molecular-subtype classification from DCE-MRI** as
the vehicle.

**The classifier is the means, not the contribution.** This framing matters and was
arrived at the hard way. Months were spent trying to make the classifier better before
it became clear that its ceiling is a property of the data, and that the dissertation's
questions do not depend on beating it.

### 1.2 Main research problem

Medical images cannot leave the hospital that produced them. Patient privacy law,
institutional review boards and data-transfer agreements make pooling multi-centre
imaging data slow or impossible. Federated learning promises to train one model across
several hospitals while every image stays where it was acquired — only model weights
travel. **The problem is that the cost of that promise is not well measured on real
deployments**: most published comparisons use simulators, single-process threads, and
artificially partitioned public datasets.

### 1.3 Motivation

1. Breast cancer is the most common cancer in women worldwide, and molecular subtype
   determines treatment (endocrine therapy, HER2-targeted agents, or chemotherapy).
2. DCE-MRI is already acquired for staging and neoadjuvant response monitoring, so a
   subtype predictor built on it adds no new imaging burden.
3. Multi-centre medical imaging is exactly the setting federated learning was designed
   for, and exactly the setting where data cannot be pooled.
4. A **real** NVFLARE deployment (PKI, separate processes, admin API) rather than the
   simulator is what makes the measurement credible.

### 1.4 Objectives, as recorded in this project

| | |
|---|---|
| **OBJ3** | Implement a functional system using NVIDIA FLARE and DL models |
| **OBJ4** | Compare centralised vs federated performance with quantitative metrics |
| **OBJ5** | Evaluate the impact of non-IID data heterogeneity |

**OBJ1 and OBJ2: INFORMATION NOT FOUND.** They are referenced by numbering but their
text does not appear anywhere in the repository. The dissertation document itself is not
in this repository — only these three objectives were ever transcribed into it.

### 1.5 Research questions

| | question | measured by |
|---|---|---|
| **RQ1** | Can federated learning reach performance **comparable** to centralised training? | test01 (centralised) against the mean of tests 02–13, as an equivalence claim |
| **RQ2** | What is the impact of non-IID data heterogeneity? | **test10 vs test12 and test11 vs test13** — the matched cohort pair. Tests 06 vs 08 vary quantity only |
| **RQ3** | Trade-offs between privacy, communication efficiency and performance? | the per-round convergence curves (communication) and each FedAvg/FedProx pair, of which **10 vs 11** is the only one with real drift to correct |
| **RQ4** | What mitigates FL limitations in clinical environments? | test11 (FedProx under genuine heterogeneity), the security measures, and the class-weight-scope experiment that has **not** been run |

Source: `src/federated/config/experiments.py` (module docstring) and
`docs/EXPERIMENTS.md`.

#### RQ1 is an EQUIVALENCE claim, not a failed significance test

This distinction decides how the whole results chapter reads, and getting it wrong is
the easiest way to make a real finding sound like a null result.

A null-hypothesis test that fails to find a difference proves nothing: absence of
evidence is not evidence of absence, and with one seed per configuration it never
could be. An **equivalence test** inverts the burden. It fixes a margin of practical
equivalence *in advance*, then asks whether the observed difference falls inside it —
the two-one-sided-tests framing (Lakens, <https://doi.org/10.1177/1948550617697177>).

**The margin is 0.067 macro-AUC and it was measured, not chosen.** It is the spread
between two runs of a byte-identical configuration differing only in random seed
(0.7023 and 0.6351, campaign B; §8.4). `seed` fixes initialisation, augmentation draws
and the split, but not cuDNN kernel selection, AMP behaviour or DataLoader worker
ordering. That residual is the method's own irreducible variability, so a difference
smaller than it cannot be attributed to anything the experiment varied.

Measured: centralised **0.6068**, the mean of the twelve federated runs **0.5927**, a
gap of **0.0141** — **4.8× smaller than the margin**, with every federated run inside
it (largest single deviation **0.0642**, test10) and three of them scoring *above* the centralised
baseline. The claim that follows is positive and quotable:

> The cost of federation on this task is smaller than the cost of re-running the
> centralised configuration with a different random seed.

**"No difference detected" is therefore a finding, not a failure.** What limits it is
stated plainly and is not a technicality: one seed per experiment means the margin is
applied to point estimates rather than to confidence intervals. Three seeds would
convert "the points fall inside the margin" into "the interval falls inside the
margin", which is the stronger form and is item 1 of §21.

### 1.6 Hard constraints (non-negotiable, set by the supervisor)

* **Real NVFLARE.** PKI provisioning, one OS process per hospital, independent TCP
  ports, jobs submitted through the admin API. **Not the simulator.**
* Centralised baseline and all its tests run **before** the federated experiments.
* Every execution log and result saved so dissertation figures can be regenerated.
* README in English; all code comments in English.

### 1.7 The classification task

**3-class molecular subtype from a 2D DCE-MRI slice, aggregated to one prediction per
patient.**

| index | class | meaning |
|---|---|---|
| 0 | `HRposHER2neg` | hormone-receptor positive, HER2 negative |
| 1 | `TripleNeg` | HR negative and HER2 negative |
| 2 | `HER2pos` | HER2 positive (any HR status) |

Input: a 224×224 RGB PNG where R = pre-contrast, G = early post-contrast, B = late
post-contrast DCE phase. Output: 3 logits. Slice probabilities are averaged per patient
before any metric is computed.

### 1.8 The proposed federated framework

```
                    admin@ips.pt                 (submits jobs, monitors, downloads)
                         │  localhost:8003  (admin port)
                         ▼
                  ┌─────────────┐
                  │   server    │  aggregates updates, selects the global model
                  │  localhost  │  holds NO patient images
                  └──────┬──────┘
                         │  localhost:8002  (fed_learn port)
        ┌────────────┬───┴────────┬────────────┐
        ▼            ▼            ▼            ▼
   hospital_1   hospital_2   hospital_3   hospital_4
    org h1        org h2       org h3       org h4
   own cert     own cert     own cert     own cert
   own data     own data     own data     own data
```

NVIDIA FLARE **2.8.0**, `ProdEnv`, PKI startup kits from `nvflare provision`, one OS
process per participant. Details in §13.

### 1.9 Expected scientific contribution

1. A measurement of the federated-vs-centralised gap on a real medical-imaging task with
   a **real deployment**, not a simulation.
2. A characterisation of how **cohort heterogeneity** — not merely quantity skew —
   affects federated performance.
3. **A reproducibility audit of BreastDCEDL**, which turned out to be a contribution in
   its own right (§12).
4. **A documented catalogue of interventions that do NOT work** on this task, with the
   measurement behind each. Negative results, reported as results.

### 1.10 How the compared concepts relate

* **Centralised learning** — all training data on one machine. It is the *upper bound*
  every federated run is measured against, and the thing that is legally impossible in
  the real setting. Here: test01.
* **Federated learning** — each hospital trains locally on its own patients; only
  weights are sent to a server, which aggregates and sends a new global model back. No
  image ever leaves a site. Here: tests 02–13.
* **FedAvg** (McMahan et al., 2017) — the server replaces the global model with the
  **weighted mean** of the client models, weighted by each client's sample count. The
  baseline aggregation rule. Tests 02, 04, 06, 08.
* **FedProx** (Li et al., 2020) — identical server-side; adds a **client-side** penalty
  `mu/2 · ‖w_local − w_global‖²` to the local loss, keeping a site from drifting away
  from the model it was given. The entire difference from FedAvg is one number,
  `mu = 0.01`. Tests 03, 05, 07, 09, 11, 13.
* **FedOpt** — a **server-side** optimiser. The mean client delta is treated as a
  pseudo-gradient and the server takes an SGD step with it
  (`w_global ← ServerOpt(w_global, mean(w_client − w_global))`). SGD with lr 1.0 and
  momentum 0 *is* FedAvg exactly, so the momentum (0.6) is the whole difference. Clients
  are untouched (mu = 0), which makes FedOpt **orthogonal** to FedProx rather than an
  alternative. **Implemented, never completed, and now removed from the experiment
  table** — see §10.7. `FederationConfig` still carries `fedopt_lr` and
  `fedopt_momentum`, and `recipes.py` still builds it, so reviving it is a matter of
  adding rows back. **No FedOpt result is reported anywhere in this project.**
* **Number of hospitals (2 / 3 / 4)** — with the training pool fixed at 1,527 patients,
  more hospitals means each site holds fewer patients and the server averages more
  divergent models. This tests whether degradation is *progressive* or *all-or-nothing*.
* **Balanced vs skewed** — balanced gives every site the same number of patients
  (25/25/25/25); skewed gives 5:2:1:1 (55.6 / 22.2 / 11.1 / 11.1%). This is **quantity
  skew**.
* **Non-IID data** — data whose distribution differs *between* sites. Kairouz et al.
  (<https://doi.org/10.1561/2200000083>) enumerate the ways it can differ: **quantity
  skew** (sites hold different *amounts* of the same distribution), **label
  distribution skew** (different class priors), **feature distribution skew** (the same
  class looks different at different sites), and concept shift. Quantity skew is
  formally in the taxonomy and is the **weakest entry in it**: the sites still sample
  from one distribution, so the expected local gradient is the same everywhere and only
  its variance differs. It is what tests 02–09 vary, all four of those partitions being
  stratified to within **0.43 percentage points** of class spread (measured;
  `final_summary/cohort/per_client_data.csv`).
* **Cohort-native vs size-matched partitions — the pair that actually answers RQ2.**
  Tests 10/11 give each hospital one complete source cohort; tests 12/13 give the same
  three hospitals the same **642 / 101 / 784** patients drawn from all three cohorts.
  Site sizes, client count, rounds, seed and algorithm are identical, so the *only*
  thing varying between the pair is whether a site's data is cohort-native. That brings
  in label skew (class spread **27.45 pp** against **0.32 pp**), feature skew (Duke's
  tumours are ~5× smaller by volume and its scanner population differs) and quantity
  skew simultaneously — the realistic case. **This is the strongest available test of
  RQ2 in this project, and unlike quantity skew it produced a consistent answer (§14.1).**
* **Medical imaging** — supplies the realism: small patient counts, strong site
  signatures, class imbalance, and a genuine privacy constraint.
* **Privacy / distributed learning** — the reason federation exists. The cost of that
  privacy, expressed in macro-AUC, is what RQ1 measures.

---

## 2. FINAL DATASET

### 2.1 Identity

| | |
|---|---|
| **name** | `multi_subtype_80mm` |
| **path** | `dataset/multi_subtype_80mm/` |
| **built from** | BreastDCEDL **MinCrop** release (Zenodo record 18114231), cohorts I-SPY2 + I-SPY1 + Duke |
| **built by** | `src/core/dataset_builder.py` + `src/pipelines/thesis/preprocessing.py`, driven by `notebooks/03_build_dataset_mine.ipynb` |
| **patients** | **2,063** |
| **images** | **16,378** |
| **classes** | 3 (`HRposHER2neg`, `TripleNeg`, `HER2pos`) |
| **image format** | RGB PNG, **224 × 224**, 8-bit, `optimize=True` |
| **resolution** | constant **0.35714 mm/px** for every image |

Build parameters, from `data/multi_subtype_80mm/config.json` (verbatim):

```json
{"n_slices": 8, "trim_frac": 0.15, "crop_mm": 80.0, "save_size": 224,
 "normalization": "minmax", "chanclip_q": 0.98, "min_tumor_px": 10,
 "roi_basis": "area_max", "cohorts": ["spy2","spy1","duke"],
 "n_patients": 2063, "n_images": 16378, "errors": {}}
```

### 2.2 On-disk layout

```
dataset/multi_subtype_80mm/
├── config.json          the build parameters above
├── metadata.csv         16,378 rows × 35 columns — every image
├── train.csv            12,131 rows
├── val.csv               2,132 rows
├── test.csv              2,115 rows
└── images/
    └── <PID>/slice_XXX.png      one folder per patient (2,063 folders)
```

`filename` in the CSVs is `<PID>/slice_XXX.png`, relative to `images/`.

### 2.3 Counts (all computed directly from `metadata.csv` for this document)

**Patients per cohort**

| cohort | patients | % |
|---|---:|---:|
| I-SPY2 (`spy2`) | 982 | 47.6% |
| Duke (`duke`) | 914 | 44.3% |
| I-SPY1 (`spy1`) | 167 | 8.1% |
| **total** | **2,063** | |

**Images per cohort**

| cohort | images |
|---|---:|
| I-SPY2 | 7,835 |
| Duke | 7,212 |
| I-SPY1 | 1,331 |
| **total** | **16,378** |

**Patients per class** — HRposHER2neg 1,042 · TripleNeg 564 · HER2pos 457
**Images per class** — HRposHER2neg 8,230 · TripleNeg 4,495 · HER2pos 3,653

**Patients, cohort × class**

| cohort | HR+/HER2− | TripleNeg | HER2+ | total |
|---|---:|---:|---:|---:|
| I-SPY2 | 381 | 359 | 242 | 982 |
| Duke | **592** | 161 | 161 | 914 |
| I-SPY1 | 69 | 44 | 54 | 167 |
| **total** | **1,042** | **564** | **457** | **2,063** |

**Images, cohort × class**

| cohort | HR+/HER2− | TripleNeg | HER2+ | total |
|---|---:|---:|---:|---:|
| I-SPY2 | 3,036 | 2,864 | 1,935 | 7,835 |
| Duke | 4,644 | 1,282 | 1,286 | 7,212 |
| I-SPY1 | 550 | 349 | 432 | 1,331 |
| **total** | **8,230** | **4,495** | **3,653** | **16,378** |

**Class composition *within* each cohort (patients)** — the panel to read first:

| cohort | HR+/HER2− | TripleNeg | HER2+ |
|---|---:|---:|---:|
| Duke | **64.8%** | 17.6% | 17.6% |
| I-SPY1 | 41.3% | 26.3% | 32.3% |
| I-SPY2 | 38.8% | 36.6% | 24.6% |

Duke is **26 percentage points** richer in HR+/HER2− than I-SPY2. This is the confound
of §9.

### 2.4 Splits — patient-level, taken from BreastDCEDL

The split comes from the **`split` column of the BreastDCEDL MinCrop metadata**
(`0 = train, 1 = test, 2 = val`). It is **not** re-derived, so it is identical to what
the dataset authors published for this subset.

**Every slice of a patient is in exactly one split.** Splitting by slice would let a
model recognise the patient rather than the disease.

| split | patients | images | HR+/HER2− | TripleNeg | HER2+ | trivial baseline |
|---|---:|---:|---:|---:|---:|---:|
| train | **1,527** | 12,131 | 773 | 410 | 344 | — |
| validation | **268** | 2,132 | 132 | 76 | 60 | 0.4925 |
| test | **268** | 2,115 | 137 | 78 | 53 | **0.5112** |

**By cohort:**

| split | Duke | I-SPY1 | I-SPY2 | total |
|---|---:|---:|---:|---:|
| train | 642 | 101 | 784 | 1,527 |
| validation | 136 | 33 | 99 | 268 |
| test | 136 | 33 | 99 | 268 |

**The trivial baseline of the test set is 0.5112** — always predicting HR+/HER2−.
Accuracy is meaningless without it, and it is quoted beside every accuracy in this
project.

**Note:** this split is **not** the one the BreastDCEDL paper describes (1,099/177/176).
That partition is defined over the **pCR-labelled** subset; ours covers the
**subtype-labelled** subset. The `split` column is the same; the subset differs.

### 2.5 Class imbalance

Ratio **2.25 : 1** (HR+/HER2− 1,042 vs HER2+ 457) at patient level. This is *below* the
"mild" band (1:4–1:10) in the imbalance-strategies literature. Handled by
inverse-frequency class weights counted **per patient**: `[0.658, 1.241, 1.480]`
(printed by `run_centralized.py`, see `deployment/logs/_ablations/`).

### 2.6 Slices per patient

Mean **7.94**, median **8**, range **1–8**. The maximum is 8 by construction; **51
patients** have fewer than 8 because they had fewer tumour-bearing slices after the 15%
trim.

### 2.7 Metadata / CSV structure — all 35 columns

One row per **image**. No missing values anywhere.

**Identity and label**

| column | type | meaning |
|---|---|---|
| `filename` | str | `<PID>/slice_XXX.png`, relative to `images/`, unique |
| `pid` | str | patient identifier (2,063 unique) |
| `cohort` | str | `spy1` · `spy2` · `duke` |
| `split` | str | `train` · `val` · `test` |
| `label` | int | 0, 1, 2 |
| `label_name` | str | `HRposHER2neg` · `TripleNeg` · `HER2pos` |

**Slice position within the lesion**

| column | range | meaning |
|---|---|---|
| `slice_index` | 1–127 | z index in the source volume |
| `slice_order` | 0–7 | position among the 8 kept slices |
| `z_rel` | 0–1 | relative position along the lesion |
| `n_slices_tumor` | 1–150 | tumour-bearing slices the patient had **before** selection |

**DCE phases actually read** (so a fallback substitution is never invisible)

| column | values | meaning |
|---|---|---|
| `phase_pre` | 0 | acquisition index → R channel |
| `phase_early` | 1–3 | acquisition index → G channel |
| `phase_late` | 2–7 | acquisition index → B channel |

**Tumour measurements** — `−1` means *not computable*, **never imputed**; it marks the
Duke rows, which have no voxel mask. Filter with `> 0` before averaging.

| column | range | meaning |
|---|---|---|
| `roi_source` | `mask` (1,149 pts) · `bbox` (914 pts) | voxel mask or bounding box |
| `roi_basis` | `area_max` · `bbox` | how the in-plane box was derived |
| `tumor_pixels` | −1, 0–10,629 | tumour pixels in the saved frame |
| `tumor_area_mm2` | −1, 0–4,437 | tumour area in the saved frame |
| `tumor_fraction` | −1, 0–0.691 | tumour share of the saved frame |

**Geometry**

| column | range | meaning |
|---|---|---|
| `bbox_x/y/w/h` | −107 … 439 | tumour box **in the final 224×224 image** (negatives are legitimate: the tumour extends past the window edge, which is why cropping zero-pads) |
| `box3d_row0/row1/col0/col1` | 0–256 | tumour box in the **source** volume |
| `crop_center_row/col` | 36–222 | centre of the 80 mm window in the source volume |

**Acquisition and preprocessing**

| column | range | meaning |
|---|---|---|
| `xy_spacing` | **0.3125 – 1.4062** | source in-plane spacing, mm/px |
| `slice_thick` | 0.8 – 4.0 | slice thickness, mm |
| `crop_mm` | **80.0** constant | the physical window |
| `crop_px` | **57 – 256** | that window in source pixels |
| `img_size` | **224** constant | output size |
| `mm_per_px` | **0.35714** constant | final resolution |
| `tum_vol` | −1, 0–495 | tumour volume from the source metadata |

**The three columns that carry the whole preprocessing argument:** `xy_spacing` varies
4.5-fold, `crop_px` follows it 57 → 256, and `mm_per_px` comes out **constant** for all
16,378 rows.

**Caution:** every row is an *image*, not a patient. Use `drop_duplicates("pid")` for
patient-level statistics, or patients with 8 slices count eight times.

### 2.8 How the final dataset differs from the originals

| | BreastDCEDL MinCrop (input) | `multi_subtype_80mm` (output) |
|---|---|---|
| format | 3D NIfTI volumes, float64, original DICOM intensities | 2D RGB PNG, uint8 |
| in-plane size | 256 × 256 (native crop) | **224 × 224** |
| resolution | native, 0.312–1.406 mm/px | **constant 0.357 mm/px** |
| slices per patient | every slice of the volume (tumour spans 1–150) | **8 evenly spaced**, 15% trimmed from each end |
| channels | separate 3D volume per DCE phase | **3 phases fused as R / G / B** |
| intensity | raw | **min–max normalised over the whole 4D volume**, then 8-bit |
| framing | whole 256 px crop | **80 mm physical window** centred on the tumour, zero-padded |
| labels | pCR, HR, HER2, `HR_HER2_STATUS`, and more | only `HR_HER2_STATUS` → 3 classes |
| patients | 2,070 (paper) / 2,072 metadata rows | **2,063** |

Reduction: **63,460 tumour-bearing slices → 16,378 images (25.8%)**.

**Inclusion / exclusion ladder**

| step | patients | lost | reason |
|---|---:|---:|---|
| 1. BreastDCEDL MinCrop metadata | 2,072 | — | as released |
| 2. with an `HR_HER2_STATUS` label | 2,067 | 5 | label missing (all I-SPY1) |
| 3. built successfully | **2,063** | 4 | **no DCE files on disk** (all Duke) |

The four excluded Duke patients are `Breast_MRI_700`, `Breast_MRI_728`,
`Breast_MRI_801`, `Breast_MRI_893`. Each has a valid bounding box and `n_xy = 256` and
**zero `*_dce_aqc_*.nii.gz` files** in the downloaded release. The exclusion is missing
imaging, not a pipeline rejection.

**A discrepancy we could not resolve — NOT VERIFIED:** the BreastDCEDL **paper** reports
**916** Duke patients; the released **metadata CSV** contains **918** Duke rows. We could
not determine which is authoritative or what the two extra rows represent. Our 914 built
patients is consistent with 918 − 4.

### 2.9 Older / unused datasets

All under `unused/old_datasets/` (18 GB). All are regenerable from
`raw_dataset_BreastDCEDL/` plus a builder.

| folder | what it was | why retired |
|---|---|---|
| `spy2_subtype_minmax` | first I-SPY2 subtype set, **all** tumour slices | superseded by the 8-slice, physical-crop design |
| `spy2_subtype_chanclip` | the same with per-channel q0.98 clipping | chanclip measured **−0.025** against min-max |
| `spy2_subtype_chanclip_fixed100mm` | crop-size ladder rung | settled by the 80 mm derivation |
| `spy2_subtype_chanclip_fixed120mm` | crop-size ladder rung | same |
| `spy1_subtype_minmax` | I-SPY1 alone | too small to train on; used as a third cohort instead |
| `multi_subtype_80mm_chanclip` | the 2×2 ablation arm | ablation finished; chanclip lost |
| `multi_subtype_80mm_SOURCEPROBE` | cohort-as-label probe | regenerable in seconds; its `images` symlink is now stale |
| `retired_2026-08-04` | datasets retired in the final cleanup | superseded |
| `_datasets_readme` | notes on the old 4-source TIFF catalogues | those catalogues were deleted before this cleanup |

**Datasets investigated and discarded entirely**

| dataset | why discarded |
|---|---|
| `catalogue_full_1488` (DUKE+I-SPY1+I-SPY2+NACT, 4 classes, 206,888 slices) | the source shortcut, macro-AUC 0.967 (§9) |
| `catalogue_lab_balanced_288` | subset of the above; Luminal B still 76% I-SPY2 |
| Full I-SPY2 (54 GB NIfTI) | superseded by MinCrop; deleted by the user |
| MAMA-MIA | downloaded (expert segmentations + 22 real site IDs) then deleted. **Worth reacquiring** — its `site` column is the strongest available upgrade to RQ2 |
| `breast-mri-molecular-cancer-subtype` (Duke .mha, 34 GB) | ships no masks and `dataset.json` filenames do not resolve; superseded by BreastDCEDL |
| NACT | only 50 patients; dropped with the 4-source catalogue |

**A trap worth remembering.** Duke's own "Mol Subtype" field means ER/PR+ **and** HER2+,
which is not the standard Luminal B definition. Duke contains only **3** Luminal B
patients in total, which is why Luminal B was 90% I-SPY2 in the old catalogue. **The
cohorts use different subtype label definitions. Never merge their labels.**

---

## 3. ORIGINAL DATASETS AND SOURCES

### 3.1 The umbrella release — BreastDCEDL

**[PAPER]** BreastDCEDL is a curated, deep-learning-ready dataset of pre-treatment 3D
DCE-MRI from **2,070 breast-cancer patients**, drawn from three TCIA collections. DICOM
data were converted to standardised 3D NIfTI volumes with unified tumour annotations and
harmonised clinical metadata (pCR, HR, HER2).
Fridman, N. et al., *Scientific Data* **13**, 264 (2026);
preprint [arXiv:2506.12190](https://arxiv.org/abs/2506.12190).
Zenodo record **18114231**.

Two versions exist: **MinCrop** (25 GB, tumour-centred 256×256 crops) and **Full**
(206 GB). **This project uses MinCrop**, and regenerating it is explicitly discouraged —
the Zenodo release *is* the authors' output, and regenerating can only introduce
differences.

**[PAPER]** Cohort sizes and exclusions as the authors report them:

| cohort | included | of enrolled | reason for exclusion |
|---|---:|---:|---|
| I-SPY1 | 172 | 221 | only 172 had ≥3 scans |
| I-SPY2 | 982 | 985 | 3 excluded for missing clinical data |
| Duke | 916 | 922 | 6 excluded for missing DCE data |
| **total** | **2,070** | | |

Official split, `test`/`split` column: `0 = train, 1 = test, 2 = validation`.

### 3.2 I-SPY1 / ACRIN 6657

* **[PAPER]** Enrolled 237 women May 2002 – March 2006; 230 met eligibility for locally
  advanced breast cancer with stage T3 primaries of at least 3 cm. Pre-operative DCE-MRI
  for 222 women is public through TCIA.
* **In BreastDCEDL:** 172 patients. **In our dataset: 167** (5 lost for a missing
  `HR_HER2_STATUS` label).
* **MRI type:** 3D DCE-MRI, multiple post-contrast timepoints.
* **ROI:** **full 3D voxel segmentation mask.** "We converted these analysis masks into
  binary 3D files, coding tumor regions as 1 and background tissue as 0."
* **Measured from the released volumes:** in-plane spacing median **0.781 mm**
  (0.391–1.172), slice thickness median **2.3 mm**, median tumour volume 15.19.
* Sources: [TCIA ISPY1 collection](https://www.cancerimagingarchive.net/collection/ispy1/) ·
  [Chitalia et al., expert tumour annotations, PMC9308769](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9308769/).

### 3.3 I-SPY2

* **[PAPER]** An adaptive neoadjuvant trial. The TCIA collection holds 719 patients
  which, with 266 from ACRIN 6698/I-SPY2, form "I-SPY2 Imaging Cohort 1"; enrolled
  2010–2016, randomised to experimental or control arms.
* **In BreastDCEDL:** 982. **In our dataset: 982** (none lost).
* **MRI type:** 3D DCE-MRI, several post-contrast timepoints.
* **ROI:** **full 3D voxel segmentation mask.**
* **Measured:** in-plane spacing median **0.684 mm** (0.312–1.406 — the widest range of
  the three), slice thickness median **2.0 mm**, median tumour volume 14.72.
* Source: [TCIA ISPY2 collection](https://www.cancerimagingarchive.net/collection/ispy2/).

### 3.4 Duke-Breast-Cancer-MRI

* **[PAPER]** Single-institution retrospective collection of 922 biopsy-confirmed
  invasive breast cancers from Duke Hospital, accrued 1 Jan 2000 – 23 Mar 2014.
  Saha, A. et al., *British Journal of Cancer* **119**, 508–516 (2018).
* **In BreastDCEDL:** 916 (paper) / 918 (metadata rows — unresolved, §2.8).
  **In our dataset: 914.**
* **MRI type:** 3D DCE-MRI. The authors' spec for Duke is "timepoints 0, 1, and the
  final acquisition".
* **ROI:** **expert-drawn BOUNDING BOX only**, not a segmentation. "Expert radiologists
  annotated bounding boxes on planes containing the largest tumor area", transformed to
  the standardised alignment. Columns `sraw / eraw / scol / ecol`.
* **Measured:** in-plane spacing median **0.703 mm** (0.508–1.250), slice thickness
  median **2.0 mm**, median tumour volume **2.82 — roughly 5× smaller than either
  I-SPY cohort**.
* All cohorts: "only the largest (primary) tumor was annotated."

### 3.5 The differences that matter

| | Duke | I-SPY2 | I-SPY1 |
|---|---:|---:|---:|
| annotation | **bounding box** | voxel mask | voxel mask |
| HR+/HER2− share | **64.8%** | 38.8% | 41.3% |
| tumour volume, median | **2.82** | 14.72 | 15.19 |
| in-plane spacing, median | 0.703 | 0.684 | 0.781 |
| slice thickness, median | 2.0 | 2.0 | 2.3 |
| study type | single-institution **clinical** series, 14 years | multi-centre neoadjuvant **trial** | multi-centre neoadjuvant **trial** |

The cohorts are **not interchangeable**. Two are trials with eligibility criteria
favouring large tumours; one is a clinical series. *Small tumour → Duke →
HR+/HER2−* is available to a model as a shortcut that has nothing to do with biology.

### 3.6 Receptor definitions are NOT harmonised — [PAPER]

| marker | I-SPY1 | I-SPY2 | Duke |
|---|---|---|---|
| HR positive | ≥10% staining | ≥1% staining | Allred score > 3 |
| HER2 positive | FISH ratio ≥ 2.0 | FISH ratio ≥ 2.0 | FISH ratio > 2.2 |

A patient near the HR threshold could be labelled positive in I-SPY2 and negative in
I-SPY1. We did **not** re-harmonise — that would need the underlying staining
percentages, which the release does not expose. **A limitation of the combined-cohort
design, not a defect in our pipeline.**

### 3.7 Imaging parameters we cannot report — UNCONFIRMED

The BreastDCEDL paper **does not specify** field strength, pulse-sequence names, or
nominal spatial resolution. It states only that conversion preserved "the integrity of
the signal" and "original DICOM acquisition parameters". **We therefore do not report
field strength or sequence for any cohort**, because no source available to us states
them. Everything in §3.2–3.4 marked "measured" was computed from the released volumes.

### 3.8 Sources investigated, and why each mattered

| source | why it was relevant |
|---|---|
| [BreastDCEDL, arXiv:2506.12190](https://arxiv.org/abs/2506.12190) / [Sci Data](https://www.nature.com/articles/s41597-026-06589-6) | **the dataset.** pCR AUC 0.72 overall (0.78 I-SPY2, 0.68 I-SPY1, 0.54 Duke). Min-max + 8-bit; RGB = pre/early/late; phases 0,2,min(last,6) for I-SPY and 0,1,final for Duke. **Publishes no training hyperparameters.** |
| [github.com/naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL) | the authors' code. Cloned in full (196 commits) and audited file by file — see §12. Local copy: `BreastDCEDL/` |
| [Dual-Attention ResNet, arXiv:2510.13897](https://arxiv.org/html/2510.13897) | same authors, HER2 task. THDA-ResNet AUC **0.744** > ViT 0.64–0.67 > CvT 0.61–0.63. Per-channel q0.98 clipping beats global min-max (0.744 vs 0.700). **Median** slice aggregation best. This is the source of the `chanclip` idea we tested and rejected |
| [Zhang et al., PMC8547260](https://pmc.ncbi.nlm.nih.gov/articles/PMC8547260/) | the reference for the 3-class formulation. 0.79/0.91 accuracy within-centre collapsing to **0.52/0.44** cross-centre. Our numbers correspond to their cross-centre line |
| [Ensemble ResNet, PMC12130697](https://pmc.ncbi.nlm.nih.gov/articles/PMC12130697/) | 687 patients, 4 centres. 2D ResNet 0.62–0.76 — the range we land in. Uses WeightedBCE for imbalance |
| [Systematic review, PMC9028183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9028183/) | 106 studies, 12,989 patients: conventional quantitative MRI features "might play a limited role in the prediction of breast cancer subtypes". **This is why 0.61 is not a failure** |
| Spatial Multi-Task Learning, arXiv:2601.07001 | 960 patients. Multi-task +3.9%, multi-scale attention +3.3%, **tumour-core-only is the WORST configuration (−4.9%)** — peritumoral tissue is informative |
| [Peritumoral margin, PMC9263840](https://pmc.ncbi.nlm.nih.gov/articles/PMC9263840/) | 4–6 mm margin optimal; performance declines from 5 to 10 mm. Basis for the 80 mm window derivation |
| [Imbalance strategies review, PMC13029843](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029843/) | explicit table by ratio: mild (1:4–1:10) → weighted loss. Our ratio is 2.25:1, *below* that band. Focal loss "may even decrease performance" at moderate imbalance |
| [LMFLoss, arXiv:2212.12741](https://arxiv.org/pdf/2212.12741) | +2–9% macro-F1 from imbalance-aware losses — but on datasets at 50:1 or worse. Does not transfer |
| [FLamby, arXiv:2210.04620](https://arxiv.org/abs/2210.04620) | cross-silo FL benchmark with published FedAvg/FedProx baselines; reference for the federated protocol |
| [NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare) `examples/advanced/cifar10` | the canonical FedAvg/FedProx/FedOpt/SCAFFOLD comparison, structurally the same shape as this thesis. Adopted: `src/` + per-algorithm-folder layout, the **Recipes API**, `ProdEnv`, the Client API, `PTFedProxLoss` |
| [NVFLARE docs](https://nvflare.readthedocs.io/en/main/) | job structure, recipes, `ProdEnv`, provisioning |
| MAMA-MIA — Synapse `syn60868042`, [PMC11923173](https://pmc.ncbi.nlm.nih.gov/articles/PMC11923173/) | expert segmentations plus **22 real site IDs** inside I-SPY2. Downloaded then deleted; the strongest available upgrade to RQ2 |

---

## 4. PREPROCESSING PIPELINE

Read directly from `src/core/dataset_builder.py`,
`src/pipelines/thesis/preprocessing.py` and `core/data.py`.

### 4.1 The full flow

```
NIfTI 4D (3 DCE phases, float, original DICOM intensities)
   ▼ 1  sanitize            NaN and ±inf -> 0
   ▼ 2  locate ROI          3D mask (I-SPY) or metadata box (Duke)
   ▼ 3  select slices       trim 15% off each end of the z-span,
                            then 8 evenly spaced (np.linspace)
   ▼ 4  MIN-MAX per VOLUME  x' = clip((x - min V)/(max V - min V), 0, 1)
   ▼ 5  CROP                80 mm physical window, side = 80/xy_spacing px,
                            centred on the lesion, zero-padded at the edges
   ▼ 6  8-bit               (x*255) -> uint8
   ▼ 7  RESIZE              PIL LANCZOS -> 224x224
   ▼ 8  PNG                 R = pre, G = early post, B = late post
   ─────────────── at load time, every epoch ───────────────
   ▼ 9  /255 -> [0,1], tensor (3,224,224)
   ▼ 10 AUGMENTATION (TRAIN ONLY)
   ▼ 11 ImageNet normalisation   (x - mean)/std      <- always, and LAST
   ▼ 12 -> model
```

### 4.2 Step by step, with the reason

**1. Raw input / NIfTI structure.** Each DCE phase is a separate 3D NIfTI volume in the
MinCrop release (already cropped to 256×256 in plane, tumour z-range ± 2 slices). Read
with `nibabel` as float32, then `np.nan_to_num(nan=0, posinf=0, neginf=0)`.
*Why:* one NaN propagates through normalisation and turns a whole patient into NaN
silently. Zero is the conservative replacement — a non-finite voxel in air is
background. *Alternative rejected:* dropping affected patients, which would discard whole
cases for a handful of voxels.

**2. DCE phases.** Three phases per patient, indexed by the metadata columns `pre`,
`post_early`, `post_late`, stacked into `(3, Z, Y, X)`.
**[PAPER]** This is the authors' own fusion: "Pre-contrast, early post-contrast, and late
post-contrast acquisitions assigned to the red, green, and blue channels, respectively",
motivated by enabling pretrained models on natively 4D data.
*Why we kept it:* the three-channel layout is what makes ImageNet pretraining usable at
all, and the temporal ordering means the **colour** of a voxel encodes its enhancement
kinetics — the wash-in/wash-out behaviour that is the diagnostic content of a DCE study.
Greyscale repeated three times would discard it.
*Fallback rule:* when a requested index is absent, the **last available acquisition** is
substituted and the index actually used is written into the CSV (`phase_pre`,
`phase_early`, `phase_late`), so no substitution is invisible. This follows the authors'
own Duke specification. *Guard:* if the three phases do not share a shape, the patient is
rejected rather than resampled.

**3–6. ROI extraction, tumour localisation, box vs mask.**
* **Mask cohorts (I-SPY1/2):** the 3D binary mask is reduced to per-slice areas; slices
  with ≥ `min_tumor_px` (10) tumour pixels define the z-extent. The in-plane box is taken
  from **the single slice with the largest tumour area** (`roi_basis = area_max`).
* **Box cohort (Duke):** the box comes from `sraw/eraw/scol/ecol`. The z-extent is **not**
  from `mask_start/mask_end` — those index the *original* volume while MinCrop is already
  cropped in z. It is derived from the MinCrop construction rule (tumour range + 2 slices
  margin per side), so the tumour occupies `z ∈ [2, nz−3]`.
* *Why the largest-area slice and not the 3D union:* because that is what the Duke
  annotation **is** — a box drawn on the plane with the largest tumour area. Using the 3D
  union for I-SPY and a single plane for Duke would introduce a systematic,
  cohort-specific difference in framing, and given the 0.9978 source probe a
  cohort-specific difference is the most expensive error available. The two centres
  differ by 2.3 mm at the median and up to 25.6 mm in the tail.
* *Guard:* patients with `n_xy ≠ 256` are rejected because the box coordinates do not
  map. In this release all 918 Duke rows have `n_xy = 256`, so the guard excluded nobody.
* **Verification:** the box construction was validated against real masks on I-SPY2,
  where both exist — **767 / 767 exact matches** for patients whose original scan was
  already 256×256.

**7–8. Slice selection — how many and which.** Trim **15% of the lesion's z-extent from
each end**, then take **8 evenly spaced slices** by `linspace` over what remains.
Patients with fewer keep all of them.
*Why proportional and not absolute trimming:* a tumour spanning 60 slices loses 9 per
end; one spanning 8 loses 1. "Drop 3" would erase half of a small tumour. The trim is
geometric, so it behaves identically with a mask or a box.
*Why spread and not the N central slices:* neighbouring slices 1–2 mm apart are
near-duplicates and contribute nearly the same gradient. Spreading covers the lesion end
to end for the same number of images.
*[MEASURED] Effect:* the previous behaviour kept **every** tumour slice — mean 38 per
patient, max 150, 21% with fewer than 100 tumour pixels. Two harms: the per-patient mean
was diluted by near-empty slices, and one patient contributed 150 gradient samples
against another's 8. **63,460 → 16,378 images (25.8%).**

**9–12. Tumour-centred cropping, the physical window, resampling, pixel spacing.**
A square window of **80 physical millimetres** centred on the ROI box centre:
```
side_px = max(round(80.0 / xy_spacing), 8)
```
The window may extend past the image; the caller **zero-pads** rather than shrinking —
shrinking would change the scale for peripheral tumours only, and constant scale is the
whole point.

*Why not resize the tumour to fill the frame?* A margin expressed as "15–20% of the
bounding box" — the common choice — makes the lesion fill the frame identically in every
patient, and in doing so **erases tumour size**. Two consequences:
1. **[MEASURED] Tumour size is predictive here.** Size alone is worth macro-AUC
   0.58–0.68 on this task, and it ranks as the most important predictor in the
   BreastDCEDL authors' own feature-importance analysis.
2. **Size is biology.** Tumour extent is part of staging and correlates with subtype; the
   cohorts differ 5-fold in median volume. An 80 mm frame keeps it — a 60 mm tumour
   occupies nine times the frame area of a 20 mm one, as in reality.

*Why millimetres and not pixels?* Because `xy_spacing` ranges **0.312 – 1.406 mm/px**. A
fixed 224-pixel crop would cover 70 mm for one patient and 315 mm for another; the field
of view would then vary by cohort and the degree of magnification would itself identify
the source. After the window and the resize every image sits at ≈**0.357 mm/px**.
*[MEASURED]* resampling factor **1.96× (Duke) / 1.91× (I-SPY2) / 2.20× (I-SPY1)** —
similar across cohorts — against roughly 4× vs 2× for a proportional crop.

*Why 80 mm specifically?* The smallest window containing the whole tumour for the
**median patient of all three cohorts** (I-SPY1 64.8 mm, I-SPY2 61.2 mm, Duke 29.2 mm)
plus ~7 mm of peritumoral tissue per side — within the 4–6 mm margin reported optimal in
the peritumoral-radiomics literature. Fully contains the lesion in **81%** of patients.

**13–15. Intensity normalisation / min-max / ChanClip.** Applied to the **whole 4D
volume before any cropping**: min–max over all phases and all slices jointly.
```
lo, hi = nanmin(volume), nanmax(volume)
volume = clip((volume - lo) / (hi - lo), 0, 1)
```
**[PAPER]** The authors state "All images underwent Min-Max normalization and conversion
to 8-bit format", applied **per slice**.
*[OURS] Why over the volume instead:* per-slice normalisation destroys the intensity
relationship *between* slices and *between* phases. An almost-empty slice is rescaled to
the same brightness as one containing an intensely enhancing lesion, and the enhancement
ratio between pre- and post-contrast — the biological signal — is removed.
*Order matters and is deliberate:* normalising **before** cropping makes the intensity
scale a property of the patient rather than of the window; normalising after would make
brightness depend on how much background the window happened to include.
*Alternative implemented:* `normalize_channel_clip` — clips each phase at its own 98th
percentile before scaling (`chanclip_q = 0.98`).
**[PAPER]** This was the **winning** strategy in the authors' own later benchmark of
seven normalisations (arXiv:2510.13897): AUC 0.744 vs 0.700 for global min–max, their
worst. **[MEASURED] It lost here**: 0.5830 vs 0.6078 on our 3-class task with two seeds —
the opposite direction, though inside the noise floor. **Kept available, not default.**

**16–20. 8-bit, resize, RGB construction, saving.** In order: crop the normalised plane
(zero-padding outside) → `(clip(plane,0,1)*255).astype(uint8)`, transposed to H×W×C →
resize to **224×224** with **PIL LANCZOS**, only if the window side ≠ 224 → save RGB PNG
with `optimize=True`.
**[PAPER]** 8-bit follows the authors. *Why 224:* the native input size of
ImageNet-pretrained torchvision backbones, so no interpolation happens inside the
network. *Why LANCZOS:* the highest-quality PIL resampling filter for the direction
almost every image travels. **UNCONFIRMED:** we did **not** benchmark LANCZOS against
bilinear or bicubic. Defensible on general grounds, not supported by a measurement of
ours.

**21–22. Online normalisation / ImageNet normalisation.** At load time, every epoch:
`/255 → [0,1]`, augment (train only), then **ImageNet mean/std normalisation, always and
LAST**. Doing it last is deliberate: brightness and noise augmentation operate in
`[0,1]` space where they are interpretable.

**23. Data augmentation.** Not part of dataset construction — the PNGs on disk are
unaugmented. Applied at training time to the **training split only**. Profile `default`,
from `src/core/data.py::AugmentConfig`:

| transform | probability | parameters |
|---|---:|---|
| horizontal flip | **0.50** | left/right — plausible as the contralateral breast |
| vertical flip | **0.00** | a cranio-caudal flip produces anatomy that does not exist |
| rotation | 1.00 | ±15° |
| zoom / scale | 1.00 | 0.9–1.1 |
| translation | 1.00 | ±8% |
| brightness | 1.00 | ×U(0.8, 1.2), clamped |
| gaussian noise | 0.25 | σ ~ U(0.005, 0.03), clamped |
| cutout | 0.00 | disabled |

Rotation, zoom and translation are drawn independently but composed into **one** affine
transform, so the image is interpolated once. MixUp exists at batch level with
`mixup_alpha = 0.0` (off). Profiles: `default`, `half` (rotation/scale/translate/
brightness probabilities → 0.5), `none`.
**[MEASURED]** `half` **lost badly**: training accuracy 0.57 → 0.99, train/test gap
0.135 → 0.512, macro-AUC −0.040. **The default augmentation is what holds this model
back from memorising.**

### 4.3 What we deliberately did NOT do

Stated so their absence is not mistaken for an oversight:
* **No bias-field / N4 correction** — the release is already standardised.
* **No skull/chest-wall stripping or breast segmentation** — the 80 mm tumour-centred
  window makes it largely unnecessary.
* **No registration between phases** — assumed co-registered as released; we only verify
  their shapes match.
* **No z-resampling** — slice thickness is left as acquired; only the in-plane scale is
  harmonised.
* **No inter-cohort intensity harmonisation** (e.g. ComBat). **A real limitation given
  the 0.9978 source probe, and a clear candidate for future work.**

### 4.4 Honest caveat on the whole redesign — [MEASURED]

On the same 99 I-SPY2 test patients, this preprocessing scored **0.5837 ± 0.011** against
**0.6201 ± 0.024** for the older pipeline. The difference is inside the 0.067 noise
floor, so the verdict is *"no difference detected"* — but it is **not** the improvement
that was expected. What it did buy is a validation-to-test gap of **+0.015** against
**+0.073**, i.e. a result that generalises more honestly, and much more even per-class
recall `[0.60, 0.50, 0.37]`.

---

## 5. BEFORE/AFTER PREPROCESSING — ALL VISUALISATIONS

**There are two figure families and they live in two folders.** The dataset/report
family below is regenerated by `src/scripts/build_dataset_report_figures.py` into
**`docs/images/report_figures/`** (12 figures × 2 formats, all present, all regenerated
2026-08-05T13:16) and is embedded in `docs/DATASET_REPORT.md`. The preprocessing
walkthrough family is regenerated by `src/scripts/build_preprocessing_walkthrough.py`
into **`docs/images/preprocessing_figures/`** (5 figures × 2 formats, same timestamp)
and is embedded in `docs/PREPROCESSING_AND_IMAGING.md`.

**The preprocessing walkthrough** — built for the dissertation's methodology chapter,
every panel produced by the same functions the dataset builder calls:

| path (add `.pdf` / `.png`) | what it shows | thesis section |
|---|---|---|
| `docs/images/preprocessing_figures/fig_p1_walkthrough` | **the same slice at every step**, raw → ROI → 80 mm crop → resample → normalise → 8-bit → RGB fusion → final 224×224. Labels state only *what was done*; the justification belongs in the text | Preprocessing |
| `.../fig_p2_normalisation` | **normalisation SCOPE**, whole-volume against per-slice. Note: an earlier version of this figure compared raw against normalised histograms, which are identical because min–max is affine and the display windowing is itself a min–max — it was misleading and was replaced | Normalisation |
| `.../fig_p3_slice_selection` | which 8 of the tumour-bearing slices are kept, and the 15% trim at each end | **3.3.4 Slice Selection** |
| `.../fig_p4_load_time` | what happens at load time: ImageNet normalisation and the augmentation pipeline | Training setup |
| `.../fig_p5_flowchart` | the whole pipeline as one flowchart, deliberately simple | Preprocessing overview |

**The dataset report family:**

| path (add `.pdf` / `.png`, all under `docs/images/report_figures/`) | what it shows |
|---|---|
| `fig1_dataset_composition` | 4-panel overview: patients per cohort · images per cohort · patients+images per class · **class composition within each cohort** (the panel to read first — Duke 64.8% HR+/HER2− vs I-SPY2 38.8%) |
| `fig1_p1_patients_per_cohort` | standalone: 982 / 914 / 167 |
| `fig1_p2_images_per_cohort` | standalone: 7,835 / 7,212 / 1,331 |
| `fig1_p3_patients_images_per_class` | standalone: 1,042 / 564 / 457 patients and 8,230 / 4,495 / 3,653 images |
| `fig1_p4_class_within_cohort` | standalone: the normalised class mix per cohort |
| `fig1_p5_train_val_test` | standalone: 1,527 / 268 / 268 |
| `fig1_p6_pixel_spacing` | standalone: the **4.5-fold** in-plane spacing range, per cohort — the number that justifies the physical crop |
| `fig2_tumour_size_by_cohort` | tumour burden by cohort. **Left:** `tum_vol` from the metadata (medians 15.2 I-SPY1 / 14.7 I-SPY2 / 2.8 Duke). **Right:** tumour area on the saved slice — **Duke is absent because it has no voxel mask, so area cannot be computed and is never imputed** |
| `fig3_examples_cohort_class` | **the final 224×224 training images.** One random patient per cohort × class (3 cohorts × 3 classes), labelled with cohort, class and source pixel spacing. These are the actual files the network reads |
| `fig3b_examples_raw` | **the same patients BEFORE any preprocessing** — source slices, early post-contrast phase, greyscale, native resolution. This is the before/after pair with fig3 |
| `fig4_preprocessing_stages` | **the stage-by-stage transformation**, produced by the same functions the builder calls. Rows 1–3: one patient per class from a **mask** cohort. **Row 4: Duke**, where the ROI is a bounding box and no voxel mask exists. Yellow = mask contour, cyan dotted = tumour extent, red = the 80 mm window |
| `fig5_why_physical_window` | **Left:** the resampling factor is similar across cohorts (1.91–2.20×), so magnification does not encode the source. **Right:** the tumour's share of the frame varies — the information a proportional crop would destroy |

**What changed, before → after, and why it was necessary**

| stage | before | after | necessity |
|---|---|---|---|
| framing | whole 256×256 slice, tumour ≈1.5% of the frame | 80 mm window centred on the tumour | the lesion has to dominate the frame or the network spends capacity on chest wall and air |
| scale | 0.312–1.406 mm/px, cohort-dependent | constant 0.357 mm/px | magnification would otherwise identify the cohort |
| channels | one greyscale volume per phase | R/G/B = pre / early / late | makes ImageNet pretraining usable and encodes kinetics as colour |
| intensity | raw DICOM float | min–max over the whole volume, 8-bit | comparable across patients while preserving inter-slice and inter-phase relationships |
| slices | 1–150 per patient (mean 38) | exactly 8, spread | removes near-duplicate gradients and equalises each patient's contribution |

**Federated distribution figures** (`deployment/figures/`, `.pdf` + `.png`):
`overview_cohorts`, `overview_task_and_classes`, `overview_balanced_vs_skewed`, and
`testNN_*_distribution` for all thirteen tests.

**Result figures** (`results/federated/final_summary/`): see §16.

---

## 6. CLASS DEFINITIONS

### 6.1 The three classes

| index | class | receptor status |
|---|---|---|
| 0 | **HR+/HER2−** (`HRposHER2neg`) | hormone-receptor positive, HER2 negative |
| 1 | **Triple Negative** (`TripleNeg`) | HR negative **and** HER2 negative |
| 2 | **HER2+** (`HER2pos`) | HER2 positive (any HR) |

### 6.2 Biology, and why these change management

* **HR+/HER2−** — the "luminal" group. Tumour growth is driven by oestrogen/progesterone
  signalling. Best prognosis of the three. **Treated with endocrine therapy.**
* **Triple Negative** — no hormone receptors and no HER2 amplification. Aggressive,
  younger patients, higher early recurrence. **No targeted agent exists; treated with
  chemotherapy.**
* **HER2+** — amplification of the HER2/ERBB2 receptor tyrosine kinase. Historically the
  worst prognosis; now among the most treatable. **Treated with HER2-targeted agents
  (trastuzumab and successors).**

**[PAPER]** Receptor-defined subgroups differ materially in prognosis and treatment;
all subtypes compared with Luminal A are significantly associated with worse
progression-free survival
([*The Breast*, 2022](https://www.sciencedirect.com/science/article/pii/S0305737222001724);
[*Front. Oncol.* 9:1124, 2019](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2019.01124/full)).
HER2 status is scored under the
[2018 ASCO/CAP guideline](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8742337/).

**These three are the ones that change management.** Predicting them from imaging is
clinically meaningful in a way that predicting an arbitrary partition would not be.

### 6.3 How they were derived from the original labels

**They were not derived.** The three-class target is a **native column of the BreastDCEDL
metadata**: `HR_HER2_STATUS` in `BreastDCEDL_metadata_min_crop.csv`, with values
`HRposHER2neg`, `TripleNeg`, `HER2pos`. Verified. Patients without that column are
dropped (5 patients, all I-SPY1).

Note that the *underlying* receptor thresholds differ per cohort (§3.6) — the column is
harmonised in name, not in definition.

### 6.4 Why not Luminal A vs Luminal B

Luminal A and Luminal B are separated principally by **Ki-67 proliferation index**, which
has no established imaging correlate and is not released with these cohorts. An earlier
phase of this project attempted a four-class split including Luminal A/B and found the
distinction was **~90% carried by cohort identity** rather than biology — Duke
contributes only **3** Luminal B patients in total, so Luminal B was 90% I-SPY2. The same
72 I-SPY2 Luminal B patients inside an all-I-SPY2 dataset dropped from F1 ≈ 0.98 to
**0.077**. The three receptor-defined classes are what the data can actually support.

### 6.5 Binary tasks used in the authors' reproduction

Three other targets exist in the codebase and were used only to compare against
BreastDCEDL's published numbers:

| task | definition | n test | trivial baseline |
|---|---|---:|---:|
| **pCR** | pathological complete response after neoadjuvant therapy, binary | 176 | 0.6989 |
| **HER2** | HER2+ vs rest, binary | 268 | 0.8022 |
| **subtype** | the 3-class task above, on the authors' preprocessing | 268 | 0.5112 |

A fourth, **TripleNeg vs rest**, was the target of the *previous* federated campaign
(archived, §10.6); its trivial baseline on that split was 0.6263.

---

## 7. MODELS

### 7.1 The model actually used — ResNet-18

Defined by `src/src/federated/config/experiments.py::TrainingConfig` and built by
`src/core/models.py::build_model`.

| | |
|---|---|
| architecture | **ResNet-18**, torchvision |
| pretrained | **yes**, ImageNet |
| parameters | **11,178,051** — 10,494,979 trainable (93.89%), 683,072 frozen |
| params + buffers | 11,187,671 |
| architecture fingerprint | `2d3031acc2075813` (checked at server and every client) |
| head | `Sequential(Dropout(0.5), Linear(512, 3))` |
| dropout | **0.5** |
| frozen layers | `freeze_until = "layer3"` → conv1 + bn1 + layer1 + layer2 frozen; `freeze_bn = False` |
| input | 224×224 RGB |
| output classes | 3 |
| loss | cross-entropy, **class-weighted** (inverse frequency, counted per **patient**), `label_smoothing = 0.1` |
| optimiser | **AdamW** |
| learning rate | **1e-4** |
| weight decay | **5e-4** |
| batch size | **24**, with at most **1 slice per patient per batch** |
| epochs | **30** centralised (= 30 rounds × 1 local epoch federated) |
| early stopping | **DISABLED** (`early_stopping_patience = 0`) for the federated campaign, at the user's instruction |
| scheduler | **cosine**, evaluated per round in closed form: `lr(r) = base·(1+cos(π·r/T))/2` |
| seed | **42** — one run per job |
| mixed precision | on where CUDA is available |
| aggregation | slice probabilities averaged per patient (`mean`) |
| model selection | centralised: validation **macro-AUC**; federated: **`val_balanced_accuracy`** on each hospital's held-out patients |

**Why ResNet-18 and not something larger:** measured. Across 1.5M–87.6M parameters,
CNNs, transformers and 3D networks, every architecture landed in the same 0.55–0.63 band.
The ceiling is signal, not capacity, so the cheapest adequate backbone is right — and it
is 6× faster than ResNet-50 (12 min vs 74.5 min).

**Why frozen to `layer3`:** it is what makes the baseline reproducible — seed spread fell
from 0.026 to **0.003**, nearly ten-fold, for +0.008 AUC that means nothing on its own.
**Honest caveat:** on ResNet-18 this frees only 683,072 of 11,178,051 parameters
(**6.1%**), because the weight is almost all in `layer4`. "Stabilises the result" is what
was measured; **"reduces overfitting" is NOT supported** — the two seeds moved the
train/test gap in opposite directions.

**Why one slice per patient per batch:** neighbouring slices of one tumour are
near-duplicates and produce almost the same gradient, which removes the stochastic noise
that makes SGD generalise.

**Why class weights per patient, not per slice:** counting slices conflates how many
patients carry a class with how large their tumours are.

### 7.2 Every model tested

All numbers are **patient-level macro-AUC on held-out test sets**.

| model | params | best result | status | note |
|---|---:|---:|---|---|
| **ResNet-18** | 11.2M | **0.6078 ± 0.026** (3-cohort); 0.6159 ± 0.003 frozen | **USED** | the measured winner; also the only one whose validation did not lie (val−test gap −0.009 against +0.062 and +0.059) |
| ResNet-50 | 23.5M | 0.6044 ± 0.018 | discarded | indistinguishable from ResNet-18 and 6× slower |
| ViT-MAE-base | 85.8M | 0.6298 (authors' pipeline, 1 seed) | tested only | **the authors' own model**. Best single 3-class number, but best epoch was **2 of 32** |
| THDA-ResNet-34 | 21.3M | 0.5710 ± 0.007 | discarded | the authors' best for HER2 (0.744 in their paper); loses here |
| ResNet-101 | 42.5M | 0.5711 | discarded | binary TNBC task |
| ResNet-152 | 58.1M | 0.6430 | discarded | binary TNBC task |
| ConvNeXt-T / S / B | 27.8 / 49.5 / 87.6M | 0.6181 / 0.5815 / 0.5881 | discarded | binary TNBC; larger is worse |
| Swin-T | 27.5M | 0.6609 | discarded | binary TNBC |
| Swin-V2-T | 27.6M | 0.5645 | discarded | binary TNBC; the inversion vs Swin-T is unexplained and inside noise |
| EfficientNet-B0 | 4.0M | 0.6011 | discarded | binary TNBC |
| DenseNet-121 | 7.0M | 0.6709 | discarded | binary TNBC |
| MobileNetV3-S | 1.5M | 0.6011 | discarded | binary TNBC; matches models 20× larger |
| Zhang CNN / ConvLSTM | — | 0.5460 / 0.5691 | discarded | 5-phase sequential |
| R(2+1)D-18 / R3D-18 / MC3-18 | — | 0.5817 / 0.5631 / 0.5393 | discarded | 3D volumetric; also collapse 30,190 slices into 783 volumes |
| **linear probe** (frozen backbone) | **4,098 trainable** | **0.6813** | evidence, not a deliverable | **matched full fine-tuning of 23.5M**, with an overfitting gap of 0.020 against 0.37. The single strongest piece of evidence in the project |

**Conclusion on architecture: it does not matter.** From 1.5M to 87.6M parameters,
across CNNs, transformers and 3D networks, everything lands in the same band. With a
noise floor of 0.067 the ordering distinguishes nothing.

**Discard reasons, summarised:** *not better than ResNet-18 and more expensive* (R50,
ConvNeXt, Swin, ResNet-101/152); *worse* (THDA-ResNet, Zhang CNN/CLSTM, 3D nets);
*equal but not the incumbent* (EfficientNet, DenseNet, MobileNet). Nothing was discarded
for a bug.

**Trained ViT weights from the authors ARE released** on Zenodo
(`BreastDCEDL_models.tar.gz` → `breastdcedl_pcr_vit_model_weights.pth`, 343 MB,
`ViTForImageClassification`, 85,800,194 parameters, head `(2, 768)`). Present locally at
`raw_dataset_BreastDCEDL/BreastDCEDL_models.tar.gz`. **They have never been used** — see
§21, item 1.

### 7.3 Hyperparameters that were swept

| swept | values tried | outcome |
|---|---|---|
| learning rate | 3e-5 / 1e-4 / 3e-4 | **1e-4 best**; 3e-4 worst (0.5728) |
| batch size | 8 / 24 / 64 | 24 already best |
| training length | 10 / 20 / 60 / 200 / 300 epochs | no effect — best epoch lands at 1–5 regardless |
| dropout | 0.2 (old pipeline) → 0.5 (current) | 0.5 is the configuration behind every current number |
| weight decay | 1e-4 (old) → 5e-4 (current) | not isolated |
| seeds | 1 and 42 for the classifier phase; **42 only** for the federated campaign | see the noise floor, §8.4 |

---

## 8. OVERFITTING INVESTIGATION

### 8.1 The centralised baseline's training curve — the clearest evidence

From `results/federated/test01_centralized/seed_42/rounds.csv`, 30 epochs,
no early stopping:

| epoch | train_loss | train_acc | val_acc | val_bal_acc | **val_auc** | val_macro_f1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.1499 | 0.4394 | 0.4851 | 0.3842 | 0.6355 | 0.3712 |
| 1 | 1.0180 | 0.5334 | 0.4216 | 0.3865 | 0.6086 | 0.3712 |
| 2 | 0.9014 | 0.6192 | 0.3843 | 0.3949 | 0.6067 | 0.3650 |
| 3 | 0.8092 | 0.6884 | 0.4963 | 0.4649 | 0.6468 | 0.4618 |
| **4** | 0.7246 | **0.7525** | 0.5448 | 0.4809 | **0.6661** ← best | 0.4855 |
| 5 | 0.6590 | 0.7996 | 0.4366 | 0.4126 | 0.6166 | 0.4108 |
| 9 | 0.4964 | 0.9135 | 0.4664 | 0.4393 | 0.6400 | 0.4307 |
| 14 | 0.4135 | 0.9693 | 0.5037 | 0.4478 | 0.6325 | 0.4486 |
| 19 | 0.3822 | 0.9849 | 0.5000 | 0.4275 | 0.6248 | 0.4266 |
| 24 | 0.3654 | 0.9944 | 0.4888 | 0.4200 | 0.6379 | 0.4203 |
| 29 | 0.3615 | **0.9955** | 0.5037 | 0.4555 | 0.6459 | 0.4580 |

**Read it in one line:** training accuracy climbs from 0.44 to **0.9955** while
validation AUC peaks at **epoch 4** and never improves again. Training loss falls from
1.15 to 0.36; validation stops responding after the fifth pass.

* **Best epoch: 4 of 30.** `train_acc_at_best = 0.7525`.
* **Generalisation gap at best epoch:** val 0.2077, test 0.2226 (recorded in
  `results.json`).
* **Test performance:** macro-AUC 0.6068, accuracy 0.5299, balanced accuracy 0.4503.
* **Early stopping was DISABLED** for this run by explicit instruction, which is why the
  whole curve is visible. Model selection still used the best validation epoch.

### 8.2 Best epochs across the classifier phase — the pattern

Computed from `results/_from_pod/multi/*/history.csv` for this document:

| run | freeze | aug | best epoch / run | train_acc at best | test acc | **train − test gap** | test AUC |
|---|---|---|---:|---:|---:|---:|---:|
| FREEZE_R18_s1 | layer3 | default | 1 / 31 | 0.4376 | 0.4179 | **0.0197** | 0.6178 |
| FREEZE_R18_s42 | layer3 | default | 5 / 35 | 0.7603 | 0.4552 | 0.3051 | 0.6140 |
| R18_s1 | none | default | 2 / 32 | 0.5337 | 0.4030 | 0.1307 | 0.5894 |
| R18_s42 | none | default | 5 / 35 | 0.7560 | 0.5410 | 0.2150 | 0.6263 |
| R50_s1 | none | default | 1 / 31 | 0.4445 | 0.3545 | 0.0900 | 0.6168 |
| R50_s42 | none | default | 67 / 97 | 0.9986 | 0.5187 | 0.4799 | 0.5920 |
| CC_R18_s1 | none | default | 1 / 31 | 0.4263 | 0.3731 | 0.0532 | 0.5953 |
| CC_R18_s42 | none | default | 4 / 34 | 0.7177 | 0.5000 | 0.2177 | 0.5707 |
| CCHALF_R18_s1 | none | **half** | 18 / 48 | 0.9879 | 0.4664 | **0.5215** | 0.5545 |
| CCHALF_R18_s42 | none | **half** | 59 / 89 | **0.9997** | 0.4963 | **0.5034** | 0.5815 |
| CCHALF_FREEZE_R18_s1 | layer3 | **half** | 26 / 56 | 0.9952 | 0.4701 | **0.5251** | 0.5574 |
| CCHALF_FREEZE_R18_s42 | layer3 | **half** | 3 / 33 | 0.6826 | 0.4216 | 0.2610 | 0.5994 |
| SPY2_R18_s1 | none | default | 8 / 38 | 0.8802 | 0.4141 | 0.4661 | 0.5758 |
| SPY2_R18_s42 | none | default | 2 / 32 | 0.5068 | 0.5152 | **−0.0084** | 0.5916 |
| PAPER_subtype_r18_s42 | none | default | 68 / 98 | 0.9995 | 0.5037 | 0.4958 | 0.6153 |
| PAPER_subtype_vit_s42 | none | default | 2 / 32 | 0.4942 | 0.5336 | **−0.0394** | 0.6298 |
| PAPER_pcr_r18_s42 | none | default | 28 / 58 | 0.9957 | 0.6023 | 0.3934 | 0.5667 |
| PAPER_pcr_vit_s42 | none | default | 65 / 95 | 0.9982 | 0.6420 | 0.3561 | 0.5324 |
| PAPER_her2_r18_s42 | none | default | 1 / 31 | 0.5721 | 0.2761 | 0.2960 | 0.4351 |
| PAPER_her2_vit_s42 | none | default | 3 / 33 | 0.6118 | 0.4739 | 0.1380 | 0.5904 |
| SONDA_r18 (source probe) | none | default | 42 / 72 | 0.9982 | 0.9813 | **0.0168** | 0.9978 |

**Best epochs of 1–5 are routine.** `train_acc` reaches 0.99 within tens of epochs on
every configuration that is allowed to run long enough.

### 8.3 The three conclusions

**1. Effective sample size is the PATIENT count, not the slice count.** 1,527 training
patients (12,131 slices). A patient's 8 slices are near-duplicates; the model has ~1,500
independent examples, not 12,000.

**2. It is NOT a capacity problem.** A **linear probe with 4,098 trainable parameters**
on a frozen ImageNet backbone scored **0.6813** — *above* fully fine-tuned 23.5M — with
an overfitting gap of **0.020** against 0.37. If capacity were the bottleneck, the
smallest model in the project could not be the best one.

**3. The ceiling is signal.** Combined with a 106-study systematic review concluding that
MRI has a limited role in subtype prediction, this reframes 0.61 as a **correct answer**
rather than a failure.

### 8.4 Seed variability — the noise floor, 0.067 macro-AUC

Two runs of a **byte-identical** configuration differing only in random seed landed at
**0.7023 and 0.6351 — a gap of 0.067.** `seed` fixes initialisation and the split but
**not** cuDNN kernel selection, AMP behaviour, or DataLoader worker ordering. An earlier
reading of ±0.001 was a lucky pair and is **wrong**.

**Consequence, enforced in the reporting code:** any difference below 0.067 is "no
difference detected". The `within_noise_floor` column exists in every comparison table
for exactly this reason. In the federated campaign the **full spread from worst to best
is 0.093** while the noise floor is 0.067 — so **no comparison in that table is
attributable.**

### 8.5 Validation overestimation

Validation-to-test gaps of **+0.049 to +0.097** on the pooled dataset. **Everything
selected on validation failed to transfer:**

| selection strategy | outcome |
|---|---|
| decision threshold | moved results both ways, up to **−0.13** in recall |
| slice aggregation (7 variants) | mean effect ≈ 0, spread ±0.05 |
| ensemble composition | validation-picked 0.7005 vs all-runs 0.7066 |
| best vs last checkpoint | 2 wins each |

**Rule adopted: report threshold-free AUC at a fixed 0.5 threshold. Do not tune on 99
validation patients.**

### 8.6 Every anti-overfitting intervention measured

| intervention | Δ macro-AUC | verdict |
|---|---:|---|
| **freezing conv1–layer2 (`layer3`)** | **+0.008**, seed spread 0.026 → **0.003** | **kept** — for the variance, not the mean |
| **augmentation at 50% (`half`)** | **−0.040**, train acc 0.57 → **0.99**, gap 0.135 → **0.512** | **rejected** — this is the single clearest overfitting result in the project |
| dropout 0.5 | the current default; never isolated against 0.0 in a controlled pair | see §17.1 — it was **silently disabled by a bug** for a period |
| `chanclip` (per-channel q0.98) | **−0.025** (both seeds same direction) | rejected, despite the authors measuring it best (0.744 vs 0.700) |
| `pclip` (p0.5/p99.5 global) | **−0.034**, lost on all 3 seeds | rejected |
| subtraction + per-patient z-score | −0.021 | rejected |
| balanced sampling | −0.027 | rejected |
| label-noise filter (bbox ≥ 100 px²) | −0.032 | rejected |
| joint 3-phase normalisation | −0.004 | no effect |
| fusing DCE phases as RGB (vs replicated greyscale) | −0.024 | kept anyway — it is what makes the channels meaningful |
| tight tumour crop (4 mm) | +0.015 (inside noise) | rejected |
| 6 mm vs 4 mm margin | −0.021 | rejected |
| multi-task decomposition (3 binary heads + CBAM) | **hurt TripleNeg by 0.040** | rejected |
| 2.5D frame stacking (3 / 9 channels) | −0.015 / −0.014 | rejected |
| 3D volumes (96³) | 0.53–0.58 | rejected |
| patient-aware batch sampler | kept; effect never isolated | **worth isolating** |
| MixUp | in the ladder, never conclusive | `mixup_alpha = 0.0` |
| ensembling 7 runs | free ensemble 0.7066 vs 0.7023 best single | **worth doing — cheap** |
| Focal / CB-Loss / samplers | **never run** | the imbalance ratio (2.25:1) is *below* the band where the literature says they help |
| logit adjustment (train prior) | **never run** | **worth doing** — post-hoc, no retraining, does not violate the no-tuning rule |

### 8.7 Class imbalance as an overfitting factor

Ratio 1.54:1 (I-SPY2 alone) to 2.25:1 (pooled). Handled by inverse-frequency class
weights per patient: `[0.658, 1.241, 1.480]`. **Working only partly:** in the centralised
baseline HER2pos (minority) recall is **0.1887** against 0.7007 and 0.4615, while its AUC
(0.5079) is at chance. In earlier runs the *ranking* was fine and the *decision boundary*
was not; in test01 neither is. Per-class AUC for HER2+ being 0.508 is the single most
important negative number in the federated campaign.

### 8.8 Cohort effects as an overfitting factor

See §9. A model on pooled cohorts can reach a respectable subtype AUC by learning
*scanner and protocol*. That is a form of overfitting to the dataset rather than the
disease, and it does not show up as a train/val gap at all.

---

## 9. COHORT / SOURCE PROBE — the project's central experiment

### 9.1 Why it was created

An earlier phase reported **Luminal B F1 ≈ 0.98** on what the literature calls the
hardest class. That was the warning sign. Suspicion of shortcut learning led to training
the *identical* pipeline with the **cohort** as the label instead of the subtype.

### 9.2 What it predicts and how it was trained

* **Target:** which cohort produced this image (`spy1` / `spy2` / `duke`) — 3 classes.
* **Everything else identical** to the subtype task: same images, same patients, same
  preprocessing, same ResNet-18, same augmentation, same optimiser, same patient-level
  split, same aggregation.
* Dataset: `unused/old_datasets/multi_subtype_80mm_SOURCEPROBE` (an images symlink plus
  relabelled CSVs — regenerable in seconds; the symlink is now stale).
* Split: the same **patient-level** train/val/test as the subtype task (1,527 / 268 /
  268). Run id `SONDA_r18`, seed not varied (1 run).

### 9.3 The result

| dataset | probe macro-AUC | subtype macro-AUC | gap |
|---|---:|---:|---:|
| old 4-source catalogue | 0.967 | 0.589 | 0.378 |
| **current 3-cohort pooled** | **0.9978** | **0.6078** | **0.390** |

Full probe metrics (`results/all_runs_pod.csv` and `results/_from_pod/multi/SONDA_r18/`):

| | value |
|---|---|
| test macro-AUC | **0.9978** |
| validation macro-AUC | 0.9999 |
| test accuracy | **0.9813** (263 of 268 patients) |
| balanced accuracy | 0.9801 |
| trivial baseline | 0.5075 |
| per-class AUC | `[0.9963, 1.000, 0.997]` |
| best epoch / run | 42 / 72 |
| train acc at best | 0.9982 (gap to test 0.0168) |

Per-source accuracy in the earlier audit: **Duke 0.989 · I-SPY1 0.979 · I-SPY2 0.994.**

### 9.4 What it means

**Yes — the model can distinguish Duke / I-SPY1 / I-SPY2 almost perfectly.** Cohort
identity is essentially fully recoverable from the pixels.

**Why that is a confound.** Cohort identity is strongly correlated with the label: Duke
is 64.8% HR+/HER2− against I-SPY2's 38.8%, and Duke's tumours are ~5× smaller by volume.
A model can therefore reach a respectable subtype AUC by learning *scanner and protocol*
rather than biology. The shortcut is literally *"small tumour → probably Duke → probably
HR+/HER2−"*.

**Mechanism, demonstrated.** In the old 4-class catalogue Luminal B was 90% I-SPY2
because Duke contains only 3 Luminal B patients. The same 72 I-SPY2 Luminal B patients
inside an all-I-SPY2 dataset dropped from F1 ≈ 0.98 to **0.077**.

**How to read a probe score:** ≥0.90 the result is contaminated · ~0.70 report it beside
the result · ~0.50 pooling is safe.

**The probe doubles as proof the pipeline is correct.** A broken pipeline — wrong crop
coordinates, mislabelled patients, corrupted channels — could not reach 0.9978. This is
why it is run on every new dataset.

### 9.5 Mitigation — what was done, and what was only proposed

**Done:**
* **Single-source datasets** were made the default for the classifier phase
  (`spy2only_80mm`). This is the only reliable fix, and it costs half the data.
* **The 80 mm physical window** was designed partly to *stop adding* cohort signal: it
  equalises the resampling factor (1.91–2.20× across cohorts instead of 4× vs 2×) and
  removes the resolution signature that a fixed-pixel crop would encode.
* **The framing rule was made identical for a mask and for a box** (largest-area plane
  for both), so the annotation asymmetry does not become a framing asymmetry.
* **The probe is reported beside every pooled result**, in the dataset report, the
  production README, `config/experiments.py` and this document.

**Done since the previous version of this document:**
* **The cohort-based federated partition (`--by-cohort`) was built and run** — tests
  10–13, 2026-08-05. It turns the confound into the experiment: one real cohort per
  hospital, against a size-matched control (§14.1). **The probe must be quoted beside
  test10 in particular**, because with one cohort per site "identify the cohort, then
  use that cohort's prior" becomes available to the *aggregated* model. The
  counter-argument belongs in the same paragraph: within any single client of test10 the
  cohort is constant, so it carries no discriminative information locally and can only
  re-emerge after aggregation.

**Proposed and STILL NOT done:**
* **Inter-cohort intensity harmonisation (ComBat or similar).** Never implemented. The
  one intervention that would attack the confound at its source.
* **Adversarial / gradient-reversal de-biasing on the cohort label.** Never implemented.
* **Re-running the probe on a harmonised dataset.** The measurement that would tell you
  whether harmonisation removed the shortcut or removed the signal.

### 9.6 Scientific implications

1. **Any pooled-cohort result on this dataset must be reported with the probe.** Without
   it, a reader cannot tell biology from scanner.
2. **It invalidated an entire earlier phase** (4 sources, 4 classes, 1,488 patients,
   206,888 slices) and now governs every design decision.
3. **It is a contribution in itself** — a controlled demonstration of shortcut learning
   in multi-cohort medical imaging, with the negative control (the same 72 patients
   inside a single-cohort dataset) that most papers omit.
4. **For the federated arm specifically:** because the partitions are *stratified*, every
   hospital holds a mix of all three cohorts, so the shortcut is available to every site
   **equally**. It inflates absolute numbers without creating heterogeneity *between*
   sites. That is why it does not invalidate the RQ1/RQ3 comparisons — but it does mean
   the absolute macro-AUCs are optimistic.

---

## 10. ALL EXPERIMENTS

Fields that do not exist anywhere are marked. **Common to every experiment in §10.1–10.4:**
dataset `multi_subtype_80mm` (2,063 patients / 16,378 images), 3-class subtype task,
global test set 268 patients / 2,115 images, trivial baseline **0.5112**.

### 10.1 THE FEDERATED CAMPAIGN — 13 experiments, 2026-08-04 and 2026-08-05

**Environment:** NVIDIA **RTX 4000 Ada** (20 GB), RunPod host, CUDA 12.8, torch 2.8.0,
NVFLARE 2.8.0. Tests 01–09 ran 2026-08-03T23:48 → 2026-08-04T00:59, **47.9 minutes,
zero failures**. Tests 10–13 ran on a second rented host and finished at
2026-08-05T01:16:51, 01:25:10, 01:33:40 and 01:42:13 — **four sequential jobs of ~8
minutes each, ~34 minutes end to end, zero failures**. (The campaign start time is not
recorded in `job.json`; 34 min is inferred from the finish times and the per-job wall
clocks of 469–485 s.) **One run per job, seed 42, throughout.**

Every number below is read from `results/federated/final_summary/summary.csv`
(regenerated 2026-08-05T11:03) and cross-checked against each
`results/federated/<name>/test_metrics.json`.

| test | algorithm | hosp | partition | best epoch | best round | time (s) | accuracy | balanced acc | macro P | macro F1 | **macro AUC** |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **test01** | centralized | 1 | — pooled | **4** | — | 267.6 | **0.5299** | 0.4503 | 0.4606 | 0.4523 | **0.6068** |
| **test02** | fedavg | 2 | balanced | — | 25 | 313 | 0.4030 | 0.3742 | 0.3915 | 0.3744 | 0.5594 |
| **test03** | fedprox | 2 | balanced | — | 29 | 351 | 0.4328 | 0.4025 | 0.4028 | 0.4001 | 0.5917 |
| **test04** | fedavg | 3 | balanced | — | 27 | 313 | 0.4851 | 0.4198 | 0.4333 | 0.4231 | 0.5990 |
| **test05** | fedprox | 3 | balanced | — | 28 | 333 | 0.4590 | 0.4127 | 0.4208 | 0.4116 | 0.5958 |
| **test06** | fedavg | 4 | balanced | — | **0** | 317 | 0.4776 | 0.4522 | 0.4646 | 0.4378 | **0.6531** ★ |
| **test07** | fedprox | 4 | balanced | — | **0** | 335 | 0.4739 | 0.4393 | 0.4389 | 0.4362 | 0.6075 |
| **test08** | fedavg | 4 | skewed 5:2:1:1 | — | 21 | 323 | 0.4888 | 0.4259 | 0.4374 | 0.4292 | 0.5982 |
| **test09** | fedprox | 4 | skewed 5:2:1:1 | — | 2 | 387 | 0.4515 | 0.4210 | 0.4365 | 0.4197 | 0.6250 |
| **test10** | fedavg | 3 | **one cohort each** | — | 7 | 469 | 0.4291 | 0.3582 | 0.3523 | 0.3536 | **0.5426** ▼ |
| **test11** | fedprox | 3 | **one cohort each** | — | 16 | 482 | 0.4590 | 0.4105 | 0.4302 | 0.4144 | 0.5678 |
| **test12** | fedavg | 3 | size-matched control | — | 17 | 471 | 0.4478 | 0.4183 | 0.4187 | 0.4153 | 0.5836 |
| **test13** | fedprox | 3 | size-matched control | — | 27 | 485 | 0.4664 | 0.3885 | 0.3925 | 0.3884 | 0.5882 |

★ best · ▼ worst. `training_time_kind`: test01 = *sum of per-epoch compute*; tests 02–13
= *job wall clock including orchestration*. **They are not the same quantity and must
not be compared.** Tests 10–13 took ~50% longer per job than 02–09 on the same hardware;
the campaigns ran on two separately rented hosts and the difference is **NOT VERIFIED**
as anything other than host variation.

**⚠ Three files disagree about test01's macro-AUC in the fourth decimal.** The
authoritative value is in `results/federated/test01_centralized/seed_42/results.json`:
**0.6067918080145278 → 0.6068**. `final_summary/summary.csv` agrees (0.6068);
**`all_experiments.csv` says 0.6069**, and `docs/RESULTS.md` and the root `README.md`
repeat that. A 0.0001 rounding artefact that changes nothing, but quote 0.6068 in the
dissertation and correct the other three when convenient.

**A discrepancy you will notice if you read the older version of this document.** The
previous §10.1 quoted test02 as 0.5598/0.4067 and test06 as 0.6527; the files now say
0.5594/0.4030 and 0.6531. `test_metrics.json` and `predictions_test.csv` for every test
carry mtimes of 2026-08-05 04:06–06:56, well after the runs themselves (2026-08-04
01:17 for test02) — the **test-set evaluation was re-run locally** from the saved
`global_model.pt` after the campaign. Differences are ≤0.0005 macro-AUC and at most one
patient in a confusion matrix, consistent with re-inference on different hardware
flipping a borderline argmax. **The exact cause of the individual flips is NOT
VERIFIED.** The current files are authoritative; nothing in any conclusion moves.

**Per-class AUC, recall and confusion matrices** (rows = true, cols = predicted, order
`[HRposHER2neg, TripleNeg, HER2pos]`), from `test_metrics.json`:

| test | per-class AUC | per-class recall | confusion |
|---|---|---|---|
| test01 | 0.6238 / 0.6886 / **0.5079** | 0.7007 / 0.4615 / 0.1887 | `[[96,20,21],[30,36,12],[33,10,10]]` |
| test02 | 0.5569 / 0.6285 / **0.4928** | 0.4745 / 0.3462 / 0.3019 | `[[65,26,46],[22,27,29],[27,10,16]]` |
| test03 | 0.6091 / 0.6461 / 0.5199 | 0.4818 / 0.4615 / 0.2642 | `[[66,34,37],[23,36,19],[28,11,14]]` |
| test04 | 0.6191 / 0.6468 / 0.5311 | 0.6423 / 0.3718 / 0.2453 | `[[88,21,28],[29,29,20],[30,10,13]]` |
| test05 | 0.6030 / 0.6355 / 0.5488 | 0.5839 / 0.3333 / 0.3208 | `[[80,25,32],[24,26,28],[24,12,17]]` |
| test06 | 0.6667 / 0.6845 / **0.6082** | 0.5839 / 0.2821 / 0.4906 | `[[80,14,43],[30,22,26],[19,8,26]]` |
| test07 | 0.6066 / 0.6588 / 0.5572 | 0.5620 / 0.3974 / 0.3585 | `[[77,29,31],[26,31,21],[24,10,19]]` |
| test08 | 0.6139 / 0.6580 / 0.5227 | 0.6350 / 0.3974 / 0.2453 | `[[87,19,31],[32,31,15],[28,12,13]]` |
| test09 | 0.6394 / 0.6811 / 0.5545 | 0.5328 / 0.3718 / 0.3585 | `[[73,20,44],[26,29,23],[25,9,19]]` |
| **test10** | 0.5473 / 0.6078 / **0.4728** | 0.5766 / 0.3846 / **0.1132** | see `test_metrics.json` |
| **test11** | 0.5723 / 0.6323 / 0.4988 | 0.5766 / 0.3718 / 0.2830 | see `test_metrics.json` |
| **test12** | 0.5923 / 0.6567 / 0.5019 | 0.5109 / 0.4231 / 0.3208 | see `test_metrics.json` |
| **test13** | 0.6196 / 0.6488 / 0.4963 | 0.6496 / 0.3462 / 0.1698 | see `test_metrics.json` |

**Job IDs and timestamps** (from `results/federated/<name>/job.json`), e.g. test06:
`9efefb2e-42c6-41bc-922e-fb98271f568d`, submitted `2026-08-04T00:35:44+00:00`, finished
`2026-08-04T00:41:01+00:00`, status `FINISHED:COMPLETED`, 4 clients, 30 rounds ×
1 local epoch, `fedprox_mu = 0.0`, `key_metric = val_balanced_accuracy`.
test09: `0bf42b80-2e29-46c2-9091-697a9e4b8e15`, duration `0:06:24`.
test01 `results.json` records `finished: 2026-08-03T23:48:43+00:00`.
Tests 10–13 finished 01:16:51, 01:33:40, 01:25:10 and 01:42:13 on 2026-08-05.

**Result / log / figure paths for every test** (post-reorganisation):

| what | path |
|---|---|
| per-test results | `results/federated/testNN_*/` |
| centralised output | `results/federated/test01_centralized/`**`seed_42/`** — `best_model.pt`, `results.json`, `rounds.csv` (30 epochs), `predictions_test.csv`, `report_test.txt`. **Note the extra `seed_42/` level**, which the federated tests do not have |
| federated global models | `results/federated/testNN_*/global_model.pt` — **44,789,067 bytes each, identical size for all twelve** |
| per-round metrics | `results/federated/testNN_*/sites/rounds.csv` (all federated tests; **note the `sites/` level**) |
| per-site client log | `results/federated/testNN_*/sites/train.log` |
| per-patient predictions | `results/federated/testNN_*/predictions_test.csv` |
| aggregated deliverable | `results/federated/final_summary/` |
| logs | `deployment/logs/testNN/` — `server.log`, `hospital_N.log`, `admin.log`, `timeline.log`, `pids` |

**Interpretation — the honest reading, and it must be carried forward.**

* **RQ1 is answered as equivalence** (§1.5): centralised 0.6068 vs federated mean
  0.5927, gap **0.0141** against a 0.067 margin. Positive claim, single-seed caveat.
* **No single pairwise comparison inside the table is attributable.** The full spread
  worst-to-best is now **0.1105** (test06 0.6531 − test10 0.5426) against a 0.067 noise
  floor, so the extremes are separated but no adjacent pair is.
* **Three federated runs scored ABOVE the centralised baseline** — test06 +0.0463,
  test09 +0.0182, test07 +0.0007 — and a fourth, test04, is 0.0078 below it. Signature
  of noise dominating, **not** of federation outperforming pooled training. (Earlier
  drafts said "four above"; only three are strictly above.)
* **The cohort pair is the exception that does support a claim** — not because any one
  difference clears the noise floor, but because **two independent comparisons agree in
  direction** (§14.1).
* **Accuracy is still the real finding.** Every federated run lands **below** the trivial
  baseline of 0.5112; only the centralised run clears it (0.5299). The models rank
  patients better than chance (AUC 0.54–0.65) but **decide** worse than always predicting
  the majority class.
* **HER2+ per-class AUC was 0.5079 centralised and 0.4728 in test10 — at and below
  chance.** Under real heterogeneity the minority class goes first.
* Patterns worth repeating with more seeds and **not worth claiming yet:** among the
  stratified runs more hospitals scored *higher*, not lower (2h 0.56/0.59 · 3h 0.60/0.60
  · 4h 0.65/0.61); and on those same partitions FedAvg vs FedProx **flips sign**
  (+0.033, −0.003, −0.045, +0.026).

### 10.2 Per-hospital results (federated tests only)

Every hospital evaluated with the final global model on its **own local validation
split**.

**⚠ THE CURRENT `per_client_metrics.csv` IS STALE AND INCOMPLETE — READ THIS BEFORE
USING IT.** `results/federated/final_summary/per_client_metrics.csv` (mtime
2026-08-05T02:43) contains **only twelve rows**, covering the cohort experiments under
their **pre-renumbering names `test14`–`test17`**:

| file says | is actually |
|---|---|
| `test14_fedavg_cohort` | **test10** |
| `test15_fedavg_sizematched` | **test12** |
| `test16_fedprox_cohort` | **test11** |
| `test17_fedprox_sizematched` | **test13** |

The rows for tests 02–09 are **not in the current file**. The last rebuild ran
`build_final_summary.py --no-client-eval`, which regenerates `summary.csv` (mtime
11:03) but leaves `per_client_metrics.csv` untouched — so the ids were never renamed
and the earlier tests were never re-scored. The table below for tests 02–09 is
therefore transcribed from the **2026-08-04 build of that file, which no longer exists
on disk**; the same is true of the copy kept at
`/private/tmp/claude-501/.../2fc77b2f-.../scratchpad/final_summary_pod_backup`, which
was checked and holds the identical twelve `test14`–`test17` rows.

**Nothing is lost and this is cheaply fixed.** Every input still exists — each
`global_model.pt` and each `deployment/data/partitions/<partition>/hospital_N/val.csv`
— so re-running `python src/scripts/build_final_summary.py` **without**
`--no-client-eval` regenerates all 39 rows under the correct ids. Do that before
quoting per-hospital numbers in the dissertation.

| test | site | train pat / img | val pat | accuracy | bal acc | macro F1 | **macro AUC** |
|---|---|---|---:|---:|---:|---:|---:|
| test02 | hospital_1 | 612 / 4,870 | 152 | 0.4803 | 0.4497 | 0.4475 | 0.6294 |
| test02 | hospital_2 | 611 / 4,849 | 152 | 0.4605 | 0.4405 | 0.4342 | 0.6117 |
| test03 | hospital_1 | 612 / 4,870 | 152 | 0.4868 | 0.4583 | 0.4545 | 0.6496 |
| test03 | hospital_2 | 611 / 4,849 | 152 | 0.4539 | 0.4126 | 0.4117 | 0.5752 |
| test04 | hospital_1 | 408 / 3,245 | 102 | 0.4510 | 0.4108 | 0.4100 | 0.6314 |
| test04 | hospital_2 | 408 / 3,249 | 102 | 0.4510 | 0.4286 | 0.4231 | 0.6398 |
| test04 | hospital_3 | 406 / 3,206 | 101 | 0.5347 | 0.4762 | 0.4761 | 0.5995 |
| test05 | hospital_1 | 408 / 3,245 | 102 | 0.4608 | 0.4377 | 0.4295 | 0.6522 |
| test05 | hospital_2 | 408 / 3,249 | 102 | 0.4510 | 0.4308 | 0.4296 | 0.6188 |
| test05 | hospital_3 | 406 / 3,206 | 101 | 0.5248 | 0.4639 | 0.4646 | 0.6312 |
| test06 | hospital_1 | 306 / 2,439 | 77 | 0.4805 | 0.4375 | 0.4385 | 0.6470 |
| test06 | hospital_2 | 305 / 2,423 | 77 | 0.4675 | 0.4327 | 0.4316 | 0.6362 |
| test06 | hospital_3 | 305 / 2,426 | 76 | 0.5395 | 0.4766 | 0.4760 | 0.6472 |
| test06 | hospital_4 | 305 / 2,399 | 76 | 0.4211 | 0.3746 | 0.3777 | 0.5715 |
| test07 | hospital_1 | 306 / 2,439 | 77 | 0.4805 | 0.4412 | 0.4399 | 0.6434 |
| test07 | hospital_2 | 305 / 2,423 | 77 | 0.4416 | 0.3972 | 0.3971 | 0.5845 |
| test07 | hospital_3 | 305 / 2,426 | 76 | 0.4737 | 0.3837 | 0.3708 | 0.6348 |
| test07 | hospital_4 | 305 / 2,399 | 76 | 0.4211 | 0.3967 | 0.3943 | 0.5781 |
| test08 | hospital_1 | 678 / 5,395 | 170 | 0.5118 | 0.4720 | 0.4682 | 0.6387 |
| test08 | hospital_2 | 273 / 2,171 | 67 | 0.5373 | 0.5196 | 0.5062 | **0.7519** |
| test08 | hospital_3 | 136 / 1,073 | 34 | 0.4412 | 0.4126 | 0.4159 | **0.5237** |
| test08 | hospital_4 | 135 / 1,066 | 34 | 0.5882 | 0.5583 | 0.5502 | 0.6713 |
| test09 | hospital_1 | 678 / 5,395 | 170 | 0.5824 | 0.5411 | 0.5428 | 0.6990 |
| test09 | hospital_2 | 273 / 2,171 | 67 | 0.5672 | 0.5503 | 0.5412 | 0.7447 |
| test09 | hospital_3 | 136 / 1,073 | 34 | 0.5294 | 0.4888 | 0.4942 | 0.6748 |
| test09 | hospital_4 | 135 / 1,066 | 34 | 0.5294 | 0.5016 | 0.4973 | 0.6843 |

**Caution:** local validation sets are 34–170 patients. The spread across hospitals in
tests 08/09 (0.5237 to 0.7519) is dominated by having 34 patients in a split, not by the
skew. `per_client` in the per-experiment `metrics.json` is `{}` — the per-client numbers
live only in `per_client_metrics.csv`.

**Per-hospital results for the cohort pair (tests 10–13)**, read from the current
`per_client_metrics.csv` with the ids translated as above. These are the rows that
matter for RQ2, because they show *where* the aggregate difference comes from:

| test | site | cohort held | val pat | trivial baseline | accuracy | bal acc | macro F1 | **macro AUC** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| test10 fedavg cohort | hospital_1 | DUKE | 128 | 0.6641 | 0.4531 | 0.4183 | 0.3875 | 0.6170 |
| test10 fedavg cohort | hospital_2 | I-SPY1 | 19 | 0.4211 | 0.3684 | 0.3167 | 0.2571 | **0.4794** |
| test10 fedavg cohort | hospital_3 | I-SPY2 | 157 | 0.3885 | 0.4522 | 0.4255 | 0.4214 | 0.5804 |
| test12 fedavg mixed | hospital_1 | mixed | 128 | 0.5078 | 0.4375 | 0.4006 | 0.3993 | 0.6345 |
| test12 fedavg mixed | hospital_2 | mixed | 20 | 0.5000 | 0.6500 | 0.6667 | 0.6530 | **0.7389** |
| test12 fedavg mixed | hospital_3 | mixed | 156 | 0.5064 | 0.4359 | 0.4047 | 0.4023 | 0.5908 |
| test11 fedprox cohort | hospital_1 | DUKE | 128 | 0.6641 | 0.4375 | 0.3890 | 0.3641 | 0.6068 |
| test11 fedprox cohort | hospital_2 | I-SPY1 | 19 | 0.4211 | 0.5263 | 0.4972 | 0.4845 | 0.6274 |
| test11 fedprox cohort | hospital_3 | I-SPY2 | 157 | 0.3885 | 0.4076 | 0.3834 | 0.3728 | 0.5840 |
| test13 fedprox mixed | hospital_1 | mixed | 128 | 0.5078 | 0.4922 | 0.4412 | 0.4411 | 0.6146 |
| test13 fedprox mixed | hospital_2 | mixed | 20 | 0.5000 | 0.7500 | 0.7333 | 0.7313 | **0.7778** |
| test13 fedprox mixed | hospital_3 | mixed | 156 | 0.5064 | 0.4167 | 0.3417 | 0.3377 | 0.6000 |

**Three things in that table are worth the dissertation's space.**

1. **The small site is where heterogeneity bites hardest.** hospital_2 holds 19–20
   validation patients either way. Cohort-native (I-SPY1 only) it scores **0.4794**
   under FedAvg — *below chance*; the same site with the same 20 patients drawn from all
   three cohorts scores **0.7389**. With n = 20 neither number is stable, but the
   direction is the same under FedProx (0.6274 vs 0.7778).
2. **FedProx rescues that site and costs the large one.** From test10 to test11,
   hospital_2 goes 0.4794 → 0.6274 while hospital_1 goes 0.6170 → 0.6068 and hospital_3
   0.5804 → 0.5840. That is exactly the behaviour the proximal term is designed for:
   it stops the small, distributionally-odd site from being pulled apart, at a small
   cost to the site that dominates the average.
3. **The trivial baselines differ per site under the cohort partition** — 0.6641 at
   DUKE against 0.3885 at I-SPY2 — which is the class skew made concrete. A per-site
   accuracy is uninterpretable without its own baseline here, and the size-matched
   control has all three sites at ~0.50 by construction.

### 10.3 THE CLASSIFIER PHASE — 21 runs (`results/classifier/all_runs_pod.csv`)

**Environment: RunPod GPU (RTX 4090 for most; the exact GPU per run is NOT recorded).**
**Training time per run is NOT recorded in this CSV — INFORMATION NOT FOUND** except
where noted in prose (ResNet-18 ≈12 min, ResNet-50 ≈74.5 min).
Config: 100 epochs max, early stopping patience 30, AdamW lr 1e-4, wd 5e-4, dropout 0.5,
label smoothing 0.1, batch 24, cosine, `backbone_lr_scale` 0.1, `max_slices_per_patient_
per_batch` 1, aggregation mean, monitor `patient_auc`.

| run | dataset | model | best epoch / run | val AUC | **test AUC** | test acc | test bal acc | baseline | n test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **SONDA_r18** (source probe) | `..._SOURCEPROBE` | resnet18 | 42 / 72 | 0.9999 | **0.9978** | 0.9813 | 0.9801 | 0.5075 | 268 |
| PAPER_subtype_vit_s42 | `paper_subtype` | vit_mae_base | 2 / 32 | 0.6722 | **0.6298** | 0.5336 | 0.4708 | 0.5112 | 268 |
| FREEZE_R18_s1 | `multi_subtype_80mm` | resnet18 | 1 / 31 | 0.6572 | 0.6178 | 0.4179 | 0.3984 | 0.5112 | 268 |
| R50_s1 | `multi_subtype_80mm` | resnet50 | 1 / 31 | 0.6612 | 0.6168 | 0.3545 | 0.4252 | 0.5112 | 268 |
| R18_s42 | `multi_subtype_80mm` | resnet18 | 5 / 35 | 0.6748 | 0.6263 | 0.5410 | 0.4238 | 0.5112 | 268 |
| PAPER_subtype_r18_s42 | `paper_subtype` | resnet18 | 68 / 98 | 0.6547 | 0.6153 | 0.5037 | 0.4147 | 0.5112 | 268 |
| FREEZE_R18_s42 | `multi_subtype_80mm` | resnet18 | 5 / 35 | 0.6550 | 0.6140 | 0.4552 | 0.4020 | 0.5112 | 268 |
| CCHALF_FREEZE_R18_s42 | `..._chanclip` | resnet18 | 3 / 33 | 0.6623 | 0.5994 | 0.4216 | 0.3897 | 0.5112 | 268 |
| CC_R18_s1 | `..._chanclip` | resnet18 | 1 / 31 | 0.6723 | 0.5953 | 0.3731 | 0.4067 | 0.5112 | 268 |
| R50_s42 | `multi_subtype_80mm` | resnet50 | 67 / 97 | 0.6852 | 0.5920 | 0.5187 | 0.4237 | 0.5112 | 268 |
| SPY2_R18_s42 | `spy2only_80mm` | resnet18 | 2 / 32 | 0.5900 | 0.5916 | 0.5152 | 0.4895 | 0.4040 | 99 |
| PAPER_her2_vit_s42 | `paper_her2` | vit_mae_base | 3 / 33 | 0.5817 | 0.5904 | 0.4739 | 0.5370 | 0.8022 | 268 |
| R18_s1 | `multi_subtype_80mm` | resnet18 | 2 / 32 | 0.6867 | 0.5894 | 0.4030 | 0.4098 | 0.5112 | 268 |
| CCHALF_R18_s42 | `..._chanclip` | resnet18 | 59 / 89 | 0.6437 | 0.5815 | 0.4963 | 0.4211 | 0.5112 | 268 |
| SPY2_R18_s1 | `spy2only_80mm` | resnet18 | 8 / 38 | 0.6081 | 0.5758 | 0.4141 | 0.4061 | 0.4040 | 99 |
| CC_R18_s42 | `..._chanclip` | resnet18 | 4 / 34 | 0.6735 | 0.5707 | 0.5000 | 0.4082 | 0.5112 | 268 |
| PAPER_pcr_r18_s42 | `paper_pcr` | resnet18 | 28 / 58 | 0.5793 | 0.5667 | 0.6023 | 0.5168 | 0.6989 | 176 |
| CCHALF_FREEZE_R18_s1 | `..._chanclip` | resnet18 | 26 / 56 | 0.6610 | 0.5574 | 0.4701 | 0.3831 | 0.5112 | 268 |
| CCHALF_R18_s1 | `..._chanclip` | resnet18 | 18 / 48 | 0.6567 | 0.5545 | 0.4664 | 0.3998 | 0.5112 | 268 |
| PAPER_pcr_vit_s42 | `paper_pcr` | vit_mae_base | 65 / 95 | 0.5898 | 0.5324 | 0.6420 | 0.4969 | 0.6989 | 176 |
| PAPER_her2_r18_s42 | `paper_her2` | resnet18 | 1 / 31 | 0.6113 | **0.4351** | 0.2761 | 0.4564 | 0.8022 | 268 |

**Grouped by configuration (mean ± sd over seeds):**

| configuration | n seeds | **test macro-AUC** |
|---|---:|---:|
| SONDA (cohort as label) | 1 | **0.9978** |
| PAPER_subtype_vit | 1 | 0.6298 |
| **FREEZE_R18** | 2 | **0.6159 ± 0.0027** |
| PAPER_subtype_r18 | 1 | 0.6153 |
| **R18** | 2 | **0.6078 ± 0.0261** |
| R50 | 2 | 0.6044 ± 0.0175 |
| PAPER_her2_vit | 1 | 0.5904 |
| SPY2_R18 (I-SPY2 only, n_test = 99) | 2 | 0.5837 ± 0.0112 |
| CC_R18 (chanclip) | 2 | 0.5830 ± 0.0174 |
| CCHALF_FREEZE_R18 | 2 | 0.5784 ± 0.0297 |
| CCHALF_R18 | 2 | 0.5680 ± 0.0191 |
| PAPER_pcr_r18 | 1 | 0.5667 |
| PAPER_pcr_vit | 1 | 0.5324 |
| PAPER_her2_r18 | 1 | 0.4351 |

Per-run confusion matrices, per-class AUC/recall and full training histories:
`results/classifier/_from_pod/multi/<run>/{results.json, history.csv,
internal_test_patient_predictions.csv, val_patient_predictions.csv, train.log,
stdout.log, config.json}`.
Checkpoints (7): `results/classifier/checkpoints/*.pt`.
**Note:** `CC_*`, `CCHALF_*` and `SONDA` checkpoints were not retained.

### 10.4 The production freezing ablation — 2026-08-03

`results/federated/_ablations/`. Both runs: 10 epochs, seed 42, same pooled
dataset, run through `run_centralized.py`.

| run | freeze_until | best epoch | train acc at best | val AUC | **test AUC** | test acc | gap (val / test) |
|---|---|---:|---:|---:|---:|---:|---|
| `freeze_layer3_seed_42` | layer3 | 4 | 0.7665 | 0.6592 | **0.6067** | 0.5037 | 0.2068 / 0.2627 |
| `freeze_none_seed_42` | none | 4 | 0.7821 | 0.6582 | **0.5989** | 0.5187 | 0.2560 / 0.2635 |

Also present: `cpu_check_seed_42` (a 1-epoch smoke test used to confirm the CPU path on
the MacBook; not a scientific result). Logs in `deployment/logs/_ablations/`.

**Caveat found while writing this document:** both `_ablations` and `test01` record
`"cohorts": ["spy2"]` in their `config.json` while training on the **pooled** dataset. The
`cohorts` field is inert when a prepared dataset path is given (the data comes from the
CSVs, and the logs confirm 1,527 pooled training patients across all three cohorts), but
**the recorded field is misleading and should be fixed before the write-up.**

### 10.5 The two ARCHIVED federated campaigns

Both are in `unused/old_runs/results_01_to_08/`. **They are archived because their
DATASET is superseded, not because the science is wrong.**

**(a) `01_federated_tests_1to9/` — 2026-07-31, 4-class subtype, 1,488 patients**

Real NVFLARE production infrastructure (PKI, admin API, 42.7 MB of weights per round),
20 rounds × 1 local epoch, ResNet-18, 224 px, batch 32, AdamW lr 1e-4, wd 1e-4, dropout
0.2, class-weighted loss, label smoothing 0.1, seed 42, FedProx μ = 0.01. RTX 4090,
00:50–03:19 UTC. Evaluated on 224 validation patients; trivial baseline 0.388.

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

Findings recorded then: federated matched centralised (−0.03 to +0.05, no consistent
direction); test 7 **beat** centralised; quantity skew cost 0.065 macro-F1 (6→8) while
barely moving AUC; **the model saturates after round 1**, which already contains 99.3% of
the final macro-F1 — ~90% of communication traffic is wasted.
**Two caveats recorded at the time:** all nine partitions were statistically IID
(TV-distance ≈ 0.00), and all nine ran on the **4-source contaminated** dataset, so any
difference may measure the robustness of the source shortcut rather than of the method.

**(b) `08_federated_final/` — 2026-08-01/02, BINARY TripleNeg-vs-rest, ResNet-50**

50 rounds × 1 local epoch, FedProx μ = 0.01, I-SPY2, 99 test patients, trivial baseline
0.6263. From `FED_SUMMARY.txt` and the archived READMEs:

| # | configuration | hospitals | algorithm | **macro-AUC** | TN recall |
|---:|---|---:|---|---:|---:|
| **1** | **centralised** (ResNet-50, 50 epochs) | — | — | **0.6874** | 48.6% |
| 5 | balanced | 3 | FedProx | 0.6194 | 40.5% |
| 9 | skewed | 4 | FedProx | 0.6011 | 27.0% |
| 4 | balanced | 3 | FedAvg | 0.5985 | 40.5% |
| 8 | skewed | 4 | FedAvg | 0.5968 | 29.7% |
| 7 | balanced | 4 | FedProx | 0.5929 | 40.5% |
| 3 | balanced | 2 | FedProx | 0.5824 | 40.5% |
| 6 | balanced | 4 | FedAvg | 0.5815 | **51.4%** |
| 2 | balanced | 2 | FedAvg | 0.5776 | 35.1% |

The centralised run's `last.pt` scored 0.6600 against `best.pt` 0.6874. Test 6 initially
**failed** at collection (`collect_results.py` had `resnet18` hard-coded) after 5,826 s
and was re-run. Per-test wall clocks: 4,771–6,545 s each.

**Findings from that campaign:**
* **RQ1 — no.** Centralised 0.6874 against 0.5776–0.6194: a drop of **0.068 to 0.110**,
  at or above the noise floor.
* **RQ2 — no detectable effect.** Balanced 0.5815/0.5929 vs skewed 0.5968/0.6011,
  consistent with the skew being quantity-only.
* **RQ3 — FedProx won 4 of 4 paired comparisons** (+0.005, +0.021, +0.011, +0.004). Each
  individually inside the noise floor; 4/4 in one direction is a **trend, not a fact**.
* **The effect is all-or-nothing:** 2, 3 and 4 hospitals gave the same result. This
  *contradicts* an earlier lung-segmentation project where degradation was progressive.
* **Federation hurts the clinically important class:** TripleNeg recall fell from 48.6%
  to 27–40% in eight of nine configurations. Papers usually report only the aggregate.

**Note the disagreement between (b) and §10.1:** on the binary task federation cost
0.068–0.110; on the current 3-class task three federated runs *beat* the baseline. The
difference is that the current campaign has one seed per job and a spread (0.093) barely
above the noise floor (0.067). **This is a real open question, not a resolved one.**

### 10.6 Other archived campaigns (`unused/old_runs/results_01_to_08/`)

| folder | what it holds |
|---|---|
| `02_preprocessing_ablation` | seven centralised runs on one 224-patient split: **A** baseline (replicated channels) **0.6537** · B RGB fusion 0.6300 · C balanced sampling 0.6270 · D fused+balanced 0.6418 · E old pipeline 0.6432 · F label-noise filter 0.6108 · G joint 3-phase norm 0.6396. **Everything tested failed.** Also documents that an earlier claimed gain 0.616→0.654 was **wrong** — the validation split had changed (only 34 of 224 patients in common) |
| `03_lab_balanced_4class` | 288 patients, 72/class. macro-AUC **0.5895**, accuracy 0.3864 vs baseline 0.2500. **All 13 patients predicted Luminal B were I-SPY2** |
| `04_lab_balanced_3class` | same patients, Luminal A+B merged — the cleanest controlled comparison in the project. Accuracy rose 0.3864 → 0.4545 and the model got **worse** (baseline rose 0.25 → 0.50, so accuracy−baseline went +13.6 pp → **−4.5 pp**) |
| `05_pipeline_audit` | **the original source-shortcut proof.** Seven integrity checks passed (0 divergences in 20,028 rows; zero train/val patient overlap). Swapped-bbox hypothesis ruled out by measurement (mean intensity 92.5 vs 50.5, air fraction 1.9% vs 21.6%). Source probe: **accuracy 0.8864, macro-AUC 0.9667** against subtype 0.3864 / 0.5895 |
| `06_ispy2_final` | I-SPY2-only, phases 0/2/5, tight crop, 679/97/195 split, **3 seeds**: macro-AUC **0.6218 ± 0.014**. First result on a held-out test set the model never saw. **The class ordering matched the literature for the first time** (TripleNeg 0.678 > LumB 0.642 > LumA 0.602 > HER2 0.566). Confounder-to-subtype gap fell **0.378 → 0.077**. `stacking/` holds the 2.5D ablation (0.6066 ± 0.008 and 0.6083 ± 0.028, both below base) |
| `07_ispy2_breastdcedl` | the I-SPY2 BreastDCEDL ladder |
| `breastdcedl_project_resultados` | subtype / authors / models / rep / binary / slices40 ladders |
| `breastdcedl_project_experiments` | two early local training logs |
| `resultados_federado` | the first federated attempt on BreastDCEDL — test 1 finished, test 2 stopped at round 8 |

Also in `08_federated_final/SUMMARY.txt`, the binary-task hyperparameter ladder:
ResNet-50 10/20/200 epochs → 0.6639 / 0.6652 / **0.7023**; DenseNet-121 → 0.6744 /
0.6465 / 0.6765; augmentation 0.0 → 0.6260, 1.8 → 0.6364; lr 3e-4 → 0.5728, 3e-5 →
0.6473; batch 8 → 0.6264, 64 → 0.6238; 6 mm margin → 0.6443; **linear probe `last.pt`
→ 0.6813**; multi-task 3 binary heads + attention → mean AUC 0.5907 and 3-class
composed 0.5490.

### 10.7 FedOpt — IMPLEMENTED, CANCELLED, AND REMOVED FROM THE EXPERIMENT TABLE

**⚠ The ids `test10`–`test13` no longer mean FedOpt.** For roughly one day they did;
they now hold the cohort-heterogeneity pair (§14.1). This section records what the
FedOpt attempt was and why nothing from it is reported, because any older note, log
filename or figure that says "test10 = FedOpt" is from that window.

FedOpt was added after 01–09 completed and was to run **on the MacBook CPU**. Same four
partitions, same clients, same seed 42, same 30 rounds; only the **server's** update
rule differed (`fedopt_lr = 1.0`, `fedopt_momentum = 0.6`; client `mu = 0`). Pairs were
10↔02, 11↔04, 12↔06, 13↔08.

| old id | old name | partition | outcome |
|---|---|---|---|
| test10 | `test10_fedopt_2h` | 2_clients_balanced | **partial** — job `7e92f496-290f-4db2-aaf1-84e36347e3f8` submitted 2026-08-04T10:36:55Z on **CPU**, reached **round 19 of 30**, cancelled by the user |
| test11 | `test11_fedopt_3h` | 3_clients_balanced | job folder + README generated; never completed |
| test12 | `test12_fedopt_4h` | 4_clients_balanced | **failed at launch** — `TypeError: FedOptRecipe.__init__() got an unexpected keyword argument 'key_metric'` (§17.10) |
| test13 | `test13_fedopt_skewed` | 4_clients_skewed | same failure |

**Why it was removed rather than left as four empty rows.** Nothing completed, so
nothing can be reported; and a results table carrying four permanently blank
experiments invites the reader to ask what happened to them in every chapter. The
rationale is written into `src/federated/config/experiments.py` at the point where the
rows used to be, so the removal cannot be mistaken for an oversight.

**The asymmetry that made FedOpt unusable here, worth keeping if it is ever revived:**
`FedOptRecipe` rejects `key_metric`, so `common.pop("key_metric", None)` was added.
FedOpt therefore has **no server-side model selection — it keeps the LAST round, while
FedAvg and FedProx keep the best of thirty.** The three algorithms would not be measured
the same way, and any future FedOpt number must carry that caveat.

**Confirmed: no FedOpt result appears anywhere in the reported campaign.**
`results/federated/all_experiments.csv`, `final_summary/summary.csv` and
`final_summary/manifest.json` each list exactly 13 experiments, all `fedavg`,
`fedprox` or `centralized`. The partial output is archived at
`unused/reference_implementations/fedopt_cancelled_2026-08-04/`; nothing was deleted.

| id | name | partition | status |
|---|---|---|---|
| test10 | `test10_fedopt_2h` | 2_clients_balanced | **partial** — job `7e92f496-290f-4db2-aaf1-84e36347e3f8` submitted 2026-08-04T10:36:55Z on **CPU** (MacBook), reached **round 19 of 30**, then cancelled by the user. `results/test10_fedopt_2h/sites/rounds.csv` holds 40 rows |
| test11 | `test11_fedopt_3h` | 3_clients_balanced | job folder + README generated; never completed |
| test12 | `test12_fedopt_4h` | 4_clients_balanced | **first attempt FAILED** — `TypeError: FedOptRecipe.__init__() got an unexpected keyword argument 'key_metric'`. Fixed (see §17.10); never re-run |
| test13 | `test13_fedopt_skewed` | 4_clients_skewed | same failure, same fix; never re-run |

**Important consequence of the fix:** `FedOptRecipe` does not accept `key_metric`, so
`common.pop("key_metric", None)` was added. **FedOpt therefore has no server-side model
selection — it keeps the LAST round, while FedAvg/FedProx keep the best.** Any FedOpt
number must be reported with that stated.

The user's instruction was **"poide cancelar os teste4s"** (cancel the tests). Partial
output is preserved; nothing was deleted.

### 10.8 Fields that do not exist for any experiment

* **Training loss / validation loss per experiment as a headline number** — the
  per-round/per-epoch CSVs have `train_loss` and `slice_loss`, but no single "final
  train/val loss" field is recorded. Use `rounds.csv` / `history.csv`.
* **Per-run wall-clock time for the 21 classifier runs** — INFORMATION NOT FOUND.
* **GPU model per classifier run** — INFORMATION NOT FOUND (RunPod, RTX 4090 for most).
* **Date/time for the 21 classifier runs** — INFORMATION NOT FOUND in
  `all_runs_pod.csv`; some `train.log` files under `_from_pod/` carry timestamps.
* **Multiple seeds for any federated experiment** — by explicit instruction, one run per
  job at seed 42.

---

## 11. THE 2×2 ABLATION — normalisation × freezing

Run on `multi_subtype_80mm` (min–max) and `multi_subtype_80mm_chanclip` (per-channel
q0.98), each with `freeze_until = none` and `= layer3`, **two seeds each (1 and 42)**.
The augmentation rung (`half`) was added on the chanclip arm.

| configuration | normalisation | augmentation | freezing | seeds | **test AUC (mean ± sd)** | train acc at best (mean) | **train − test gap (mean)** |
|---|---|---|---|---:|---:|---:|---:|
| **R18** (baseline) | min–max | default 100% | none | 2 | **0.6078 ± 0.026** | 0.645 | 0.173 |
| **FREEZE_R18** | min–max | default 100% | **layer3** | 2 | **0.6159 ± 0.003** | 0.599 | 0.162 |
| **CC_R18** | **chanclip** | default 100% | none | 2 | **0.5830 ± 0.017** | 0.572 | 0.135 |
| **CCHALF_R18** | chanclip | **half (50%)** | none | 2 | **0.5680 ± 0.019** | **0.994** | **0.512** |
| **CCHALF_FREEZE_R18** | chanclip | **half (50%)** | **layer3** | 2 | **0.5784 ± 0.030** | 0.839 | 0.393 |

Per-seed values are in §8.2. Everything else held fixed: ResNet-18, ImageNet pretrained,
dropout 0.5, AdamW lr 1e-4, wd 5e-4, label smoothing 0.1, batch 24, cosine, 100 epochs
max with patience 30, 268-patient test set, trivial baseline 0.5112.

### What it demonstrated

**All AUC differences are inside the 0.067 noise floor.** Nothing here ranks. But two
things are real:

**1. Freezing stabilises.** Seed spread fell from **0.026 to 0.003** — nearly ten-fold.
The AUC gain (+0.008) means nothing on its own; the variance reduction is what matters,
because a federated experiment that has to attribute a 0.07 gap to federation cannot
afford a baseline that moves by 0.04 on its own. **This is why `freeze_until = "layer3"`
is the federated campaign's configuration.**
**Caveat, and it must be stated:** per seed the train/test gap moved in *opposite*
directions (0.0197 vs 0.3051 for FREEZE; 0.1307 vs 0.2150 for baseline). With two seeds,
"reduces overfitting" is **not supported** — "stabilises the result" is.
**Second caveat:** freezing to `layer3` removes only **683,072 of 11,178,051 parameters
(6.1%)**; the honest test is `layer4` (~25%), and it has **never been run**.

**2. Halving augmentation was a disaster.** Training accuracy rose from 0.57 to **0.99**
and the train/test gap **tripled** (0.135 → 0.512). Best epochs moved to 59 and 26 (from
2 and 5). **The current augmentation is what holds the model back from memorising.**

**3. `chanclip` lost**, by 0.025, on both seeds in the same direction — despite being the
*winning* normalisation in the authors' own seven-way benchmark (0.744 vs 0.700 for
global min–max on binary HER2). It did not transfer to our 3-class task on our
preprocessing. Kept available, not default.

**The independent variables did not interact usefully:** freezing helped slightly on both
normalisations; chanclip hurt under both freezing settings.

---

## 12. THE AUTHORS' REPRODUCTION — BreastDCEDL

Full audit: `docs/BREASTDCEDL_REPRODUCIBILITY_REPORT.md`.
Repository: [github.com/naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL),
cloned in full (196 commits) at `BreastDCEDL/` and audited file by file.

### 12.1 What is in their repository

| file | what it holds |
|---|---|
| `DUKE/crop_spy2_spy1.ipynb` | ★★ MinCrop generation for I-SPY1/I-SPY2 — **not referenced in the README** |
| `DUKE/duke_crop.ipynb` | ★★ MinCrop generation for Duke — **not referenced in the README** |
| `DUKE/duke_convert_dicom_to_nifti.ipynb` | DICOM → NIfTI |
| `BrestDCEDL_vit_predict.ipynb` | ★ the **only** deep-learning code — inference only |
| `df_pcr_pred_test_article.csv` | ★ the article's per-patient predictions (175 patients) |
| `utils/data_utils.py` | 16 functions, all reading and plotting |
| `transformer_models/BreastDCEDL_vit_pcr_predictions.csv` | ⚠ does **NOT** reproduce the article |
| `BreastDCEDL_metadata.csv` / `_min_crop.csv` | 2,070 rows each |

### 12.2 Finding 1 — THERE IS NO TRAINING CODE, ANYWHERE

All 16 notebooks and both `.py` files were searched for `loss.backward`,
`optimizer.step`, `.fit(`, `Trainer(`, `for epoch`, `model.train()`, `scheduler`,
`state_dict`. Then **every deleted file across 196 commits was recovered and searched
too** — including `BreastDCEDL_modeling_with_nifti_files.ipynb` and
`ISPY2/modeling_ispy2_with_nifti_files.ipynb`, which sound promising and are not. The
recovered `modeling_ispy2` notebook contains **`torch: 0` occurrences**. The only
`.fit()` calls anywhere are scikit-learn `RandomForestClassifier` and
`GradientBoostingClassifier` on tabular features.

**Consequence: no learning rate, batch size, epoch count, scheduler, weight decay,
augmentation parameter, seed or checkpoint rule is published.** Any "reproduction" of
this work reproduces the **data pipeline**, not the training procedure. This must be
stated whenever their numbers are quoted.

### 12.3 Finding 2 — their published inference notebook does not work as written

`get_jpg_im` calls `Image.fromarray(im, mode="RGB")` where `im` is **float64**
(`read_nifti` returns `get_fdata()`, always float64, never cast). Passing `mode=`
explicitly makes PIL reinterpret the raw buffer as bytes instead of converting.
**Measured on the authors' own sample patient `ACRIN-6698-102212`, the correlation
between what the model receives and the actual MRI is 0.0114.**
Figure: `unused/old_figures/.../13_bug_fromarray.png`.

This does **not** mean their published results are wrong — their prediction file
reproduces them exactly. It means **the notebook as published is not the code that
produced them.**

### 12.4 Finding 3 — the checkpoint predictions in the repo do not reproduce the article

| column | correlation with the article's | AUC on 175 test patients |
|---|---:|---:|
| `pred_pcr_vnew` (the article) | — | **0.7201** |
| `DL_pred_pcr_mean` | 0.05 | 0.5158 |
| `DL_pred_pcr_min` | 0.14 | 0.5852 |

The repository's values span 0.478–0.524 with standard deviation 0.010 — a model
outputting ~0.5 for everyone.

### 12.5 Finding 4 — the published results ARE verifiable, from their own prediction file

Recomputed from `df_pcr_pred_test_article.csv`:

| cohort | n | recomputed | published |
|---|---:|---:|---:|
| overall | 175 | **0.7201** | 0.72 |
| I-SPY2 | 99 | **0.7801** | 0.78 |
| I-SPY1 | 35 | **0.6793** | 0.68 |
| Duke | 41 | **0.5398** | 0.54 |

At threshold 0.5: accuracy 0.754, sensitivity 0.269, specificity 0.959 — against 75%,
0.27, 0.95 published. **Every number matches.**

### 12.6 Finding 5 — the famous AUC 0.94 is a TABULAR model, not the ViT

On the HR+/HER2− I-SPY2 subgroup (n = 40, 12.5% pCR rate): imaging alone gives
**0.8886**; the clinical model (`rf_pred_proba`) gives **0.9371**. The paper's wording —
"our approach" — is ambiguous; the data is not. **Do not chase 0.94.**

### 12.7 Other reproducibility issues found

Windows absolute paths throughout (`G:\My Drive\breast_mri`); no `requirements.txt`;
`crop_around_voi_cords` is redefined **four times** in one notebook with different edge
behaviour; patient counts disagree between the paper (176/177) and the prediction file
(175); paper says 916 Duke patients, metadata has 918 rows.

### 12.8 AUTHOR'S PIPELINE vs OUR PIPELINE

| | **AUTHOR'S** (`src/pipelines/reference/preprocessing.py`) | **OURS** (`src/pipelines/thesis/preprocessing.py`) |
|---|---|---|
| slices per patient | **4**, `range(max(idx−2, first), min(idx+2, last))` — asymmetric | **8 evenly spaced**, 15% trimmed each end |
| crop | **224 px fixed**, centred on the ROI | **80 mm physical window**, side = 80/spacing px |
| effective field of view | **158–175 mm, varies by cohort** — itself a cohort cue | constant 80 mm |
| final resolution | varies by patient | **constant 0.357 mm/px** |
| normalisation | **min–max per SLICE**, joint over the 3 channels | **min–max per VOLUME** (all phases, all slices) |
| resize | `Resize(224)` then ImageNet mean/std | LANCZOS → 224, then ImageNet mean/std at load |
| channels | RGB = pre / early post / late post | **identical** |
| phase indices | 0, 2, min(last, 6) for I-SPY; 0, 1, final for Duke | same rule, with the index used written to the CSV |
| cohorts | all three pooled | all three pooled (`multi_subtype_80mm`) or I-SPY2 only |
| split | official `split` column, 175–176 test patients (pCR subset) | official `split` column, 268 test patients (subtype subset) |
| augmentation | **UNKNOWN — not published** | documented in §4.2 |
| model / training | **UNKNOWN — no training code exists** | ResNet-18, §7.1 |

**One deliberate deviation on the authors' side:** the `Image.fromarray` defect is
**corrected**, recorded as `fromarray_fix: true` in the dataset's `config.json`. A
faithful reproduction of a defect reproduces nothing.

**Adopted from them:** the MinCrop geometry (verified 767/767); the official `test`
split; RGB fusion of pre/early/late; the phase-selection rule; median slice aggregation
for HER2; `vit_mae_base` as a comparison model.
**Rejected:** the `cv2.resize` branch in `crop_spy2_spy1.ipynb` (verified NOT used for
the released data — all 228 patients with `n_xy > 256` were cropped, not resized); the
`Image.fromarray` defect; **regenerating MinCrop at all**.

### 12.9 The reproduction runs — six, one seed each

Datasets built with the authors' exact rules (`authors_pcr`, `authors_her2`,
`authors_subtype`):

| run | model | best epoch | **our test AUC** | **published** | delta |
|---|---|---:|---:|---:|---:|
| PAPER_subtype_vit | ViT-MAE | 2 | **0.6298** | *never attempted by them* | — |
| PAPER_subtype_r18 | ResNet-18 | 68 | 0.6153 | — | — |
| PAPER_her2_vit | ViT-MAE | 3 | 0.5904 | 0.744 | **−0.154** |
| PAPER_pcr_r18 | ResNet-18 | 28 | 0.5667 | 0.72 | **−0.153** |
| PAPER_pcr_vit | ViT-MAE | 65 | 0.5324 | 0.72 | **−0.188** |
| PAPER_her2_r18 | ResNet-18 | 1 | 0.4351 | — | below chance |

### 12.10 What was and was not reproduced

**Successfully reproduced:**
* The **MinCrop geometry**, to the voxel: metadata box == true mask box for **767/767**
  patients with `n_xy == 256`; tumour occupies `z ∈ [2, nz−3]` for 749/767 (97.7%); Duke
  `volume_depth == z_span + 4` for 12/12; **228/228** patients with `n_xy > 256` were
  cropped, not resized. The z convention was later confirmed against their own
  `crop_around_voi_cords(..., slice_padding=2, output_size=256)`.
* **Their published AUCs**, exactly, from their own prediction file (§12.5).
* Their **data** pipeline end to end.

**NOT reproduced:**
* **Their training results.** On their binary tasks we fall **0.15–0.19 short — two to
  three times the noise floor.** On the 3-class task the two pipelines **tie**
  (0.6153–0.6298 vs 0.6078 ± 0.026).
* **Their training procedure**, because it does not exist in the repository.

**The pattern points at a cause:** best epochs of 1, 2, 3, 28, 65 with `train_acc`
reaching 0.999. The models memorise and validation never follows. That is the signature
of **wrong hyperparameters for the task** — and hyperparameters are exactly what the
authors do not publish.

**Caveat on the HER2 comparison:** our HER2 test set is 268 patients across three
cohorts; their HER2 paper uses I-SPY pooled **without** Duke (885/132/132). Indicative,
not direct. The pCR comparison is aligned: 176 vs their 175.

**The decisive test has NOT been run.** Running their **released ViT weights** through
**our** preprocessing would separate "our pixels are wrong" from "the deficit is all
training". The weights are on disk (`raw_dataset_BreastDCEDL/BreastDCEDL_models.tar.gz`).
Target: correlation with `pred_pcr_vnew`, AUC 0.7201 on 175 patients. **Inference only,
minutes of compute.** This is the single highest-value pending item.

---

## 13. FEDERATED LEARNING INFRASTRUCTURE (NVIDIA FLARE)

### 13.1 Directory structure

**Post-reorganisation (2026-08-05).** The federation used to live under one
`federated/` folder with its own `production/` subtree; the code now sits in `src/` with
everything else, and the deployment and its output are split into `deployment/` and
`results/`.

```
src/federated/
├── config/
│   ├── experiments.py     ★ THE SINGLE SOURCE OF TRUTH — 13 experiments, 6 partitions,
│   │                        TrainingConfig, FederationConfig, every path
│   ├── federation.py      ★ THE ONLY FILE THAT KNOWS A HOST OR PORT
│   └── README.md
├── common/                the training/eval code the clients AND the centralised run use
│   ├── models.py          FederatedClassifier + architecture fingerprint
│   ├── data.py            loaders, class weights
│   ├── training.py        the trainer (delegates to src/core/training.py)
│   ├── evaluation.py      patient-level metrics
│   └── thesis.py          the bridge that imports src/core
└── federation/
    ├── recipes.py         build_recipe() — FedAvg / FedProx / FedOpt
    └── client.py          the NVFLARE client loop (flare.init/receive/send)

src/scripts/               ★ all real logic lives here
├── prepare_data.py              global test/val, HARDLINKED
├── partition_data.py            per-hospital splits (--by-cohort, --stratify none)
├── verify_data.py               split integrity
├── generate_jobs.py             writes deployment/jobs/ from experiments.py
├── verify_production.py         ★ 219 pre-flight checks
├── run_experiment.py            submits ONE job through the admin API
├── run_all_experiments.py       the whole matrix
├── run_centralized.py           test01 — NOT an NVFLARE job
├── collect_results.py           scores every model on the ONE global test set
├── build_final_summary.py       ★ ~1500 lines — all tables, figures, reports
├── build_distribution_report.py
├── build_dataset_report_figures.py · build_preprocessing_walkthrough.py
├── audit_dataset.py             verifies hardlinks by INODE
└── snapshot_config.py

deployment/                ★ THE DEPLOYMENT LAYER
├── project.yml            NVFLARE provisioning — participants, ports, builders
├── config/                resolved_config.{json,md} — a SNAPSHOT; nothing reads it back
├── data/
│   ├── global/{test.csv, val.csv, images/}          identical for ALL experiments
│   └── partitions/        2_clients_balanced · 3_clients_balanced · 4_clients_balanced
│                          4_clients_skewed · 3_clients_cohort · 3_clients_sizematched
│                            each: hospital_N/{train,val}.csv + images/ (hardlinks)
├── datasets/              all_distributions.{csv,json}, global_splits.csv,
│                          dataset_audit.json, testNN_*_distribution.csv (13)
├── figures/               overviews + 13 per-test distribution figures (.pdf/.png)
├── jobs/testNN_*/         13 folders, each README.md + job.py — GENERATED
├── logs/testNN/           server.log, hospital_N.log, admin.log, timeline.log, pids
├── scripts/               thin wrappers: provision.sh distributions.sh verify.sh
│                          start.sh run.sh stop.sh collect.sh summary.sh
└── workspace/breast_fl_project/prod_00/   ★ the PKI startup kits — GITIGNORED
    ├── server/            startup/{server.crt, server.key, fed_server.json, start.sh}
    ├── hospital_1..4/     startup/{client.crt, client.key, fed_client.json, start.sh}
    └── admin@ips.pt/      startup/{client.crt, client.key, fed_admin.json}

results/federated/         ★ THE OUTPUT LAYER
├── testNN_*/              job.json, sites/rounds.csv, sites/train.log,
│                          predictions_test.csv, test_metrics.json, global_model.pt
├── _ablations/            freeze_layer3_seed_42 · freeze_none_seed_42
├── all_experiments.csv    13 rows
└── final_summary/         ★ the aggregated deliverable

docs/{ARCHITECTURE,DEPLOYMENT,EXPERIMENTS,NVFLARE_CONFIGURATION,...}.md
```

**`deployment/` holds no second definition of any hyperparameter.** `jobs/` is
generated, `config/` is a snapshot, `scripts/` are wrappers. Three bugs in this project's
history had one cause — two copies of a setting drifting apart — and each produced a run
that completed with meaningless numbers.

### 13.2 Participants

| participant | type | org | role |
|---|---|---|---|
| `server` | server | ips | aggregates updates, selects the global model, **holds no patient images** (it does hold the global test set — a benchmarking decision, §15) |
| `hospital_1` | client | h1 | Site 1 — also the large site in the skewed split |
| `hospital_2` | client | h2 | Site 2 |
| `hospital_3` | client | h3 | Site 3 |
| `hospital_4` | client | h4 | Site 4 |
| `admin@ips.pt` | admin | ips | `project_admin` — submits jobs, monitors, downloads results |

### 13.3 Ports

| port | name | purpose |
|---|---|---|
| **8002** | `fed_learn_port` | hospitals connect here to receive tasks and return model updates |
| **8003** | `admin_port` | the admin identity submits and monitors jobs |

Kept separate because that is what lets a hospital firewall expose only the first.
Declared once in `config/federation.py` and mirrored into `project.yml`;
`verify_production.py` **fails if the two ever disagree**. Clients dial `localhost:8002`
(verified in `fed_client.json`). The pre-flight check confirms the host resolves and both
ports are free before a federation starts.

### 13.4 How NVFLARE provisioning works

`nvflare provision -p project.yml -w workspace` reads `project.yml` and writes one
**startup kit** per participant: a folder holding that participant's certificate, private
key, configuration and `start.sh`. **Mutual TLS between sites is built on those
certificates.** This is what makes the deployment real rather than simulated.

Builders used (all four verified present):
`WorkspaceBuilder` (folder layout) · `StaticFileBuilder` (fed_server.json /
fed_client.json / fed_admin.json) · `CertBuilder` (root CA + one cert/key per
participant) · `SignatureBuilder` (signs the kits so tampering is detectable).
`api_version: 3` (2.8 accepts 3 or 4).

**Two constraints that cost time if discovered late:**
1. **The admin name must be a full e-mail address with a TLD.** NVFLARE validates it
   against `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$`, hard-coded in
   `nvflare/apis/utils/format_check.py`. `admin` and `admin@ips` both exit `INVALID_ARGS`.
   **This project uses `admin@ips.pt`, and it could not be renamed to `admin`** — the user
   asked, and NVFLARE does not allow it. It must also match
   `config/federation.py::ADMIN_USER` exactly, or job submission later looks for a startup
   kit that was never generated.
2. **Provisioning never overwrites.** Each run creates the next `prod_NN` beside the
   previous one. Server and clients must all start from the *same* `prod_NN` or the TLS
   handshake fails with an error that never mentions provisioning. Everything resolves it
   through `config/federation.py::workspace_dir()`, which always picks the highest.
   Currently: **`prod_00`**.

**Four hospitals are provisioned even though tests 02–05 use two or three.** The smaller
experiments use a subset of the same kits, so a difference between two results can never
be a difference in PKI.

### 13.5 How the four hospitals are simulated on one computer

**Not by threads and not by the simulator.** `scripts/start_federation.sh` launches each
participant's own `start.sh` from its own startup kit with `nohup`, so each becomes an
**independent OS process** with its own Python interpreter, memory, certificate and log
file, authenticating over a real TCP connection on localhost.

```bash
./deployment/scripts/start.sh 4 test06     # server + hospital_1..4, logs -> logs/test06/
```

The script starts the server first and **polls the admin port** until it accepts
connections before launching any hospital — a client that starts first retries with a
backoff and delays round 0 by up to a minute. `OMP_NUM_THREADS=1` is exported for every
child: five PyTorch processes each opening a full thread pool spend most of their time
context-switching (**measured: 2 concurrent jobs → 0.074 epochs/s; 7 → 0.058**).

**NVFLARE's three environments, and which this project uses:**

| env | what it is | used here |
|---|---|---|
| `SimEnv` | clients as **threads in one process**, no PKI, no network | **never for a reported number** |
| `PocEnv` | separate processes, throwaway certificates | smoke tests only |
| **`ProdEnv`** | separate processes, real PKI startup kits, own ports, jobs through the admin API | **every reported result** |

### 13.5b The execution environment — where the code was written and where it ran

**Two machines, and the dissertation must not conflate them.**

| | development | execution |
|---|---|---|
| machine | Apple MacBook (Apple Silicon), macOS Darwin 25.5.0 | rented RunPod cloud GPU host |
| accelerator | Apple MPS — **usable again since 2026-08-05** (§17.12) | **NVIDIA RTX 4000 Ada, 20 GB** for the federated campaigns; **RTX 4090** for most of the classifier phase |
| CUDA / torch | — | CUDA 12.8, torch 2.8.0 |
| NVFLARE | 2.8.0 | 2.8.0 |
| what ran there | all code, documentation, figures, notebooks, the CPU smoke test, the cancelled FedOpt attempt | **every reported result** |

**Exactly which GPU served which classifier run is INFORMATION NOT FOUND** —
`all_runs_pod.csv` records no hardware column. The vCPU, RAM and disk of the RunPod
instances are **NOT VERIFIED from this repository**; the values used in the
dissertation's environment table came from the RunPod console at rental time and are
recorded in no file here. If that table has to be defensible, re-check it against the
RunPod billing history rather than against this document.

**The one thing worth saying about the topology:** all five processes of a federation ran
on a single host. That is faithful about PKI, process isolation, the admin API, the
network protocol and data isolation on disk; it is **not** faithful about latency,
bandwidth, or the failure modes of a real WAN — which is why the communication result of
§15.6 is a statement about payload volume rather than about wall-clock cost.

### 13.6 How the infrastructure represents real hospitals

Only model weights cross the boundary; **no image ever leaves a hospital**. Each site
reads only its own `data/partitions/<partition>/<hospital>/{train,val}.csv` and its own
`images/`. Moving a hospital to its own machine changes **three things and nothing else**:
1. In `project.yml`, the server's `default_host` becomes the coordinating centre's DNS
   name or public IP, reachable on 8002 and 8003.
2. Each hospital gains its own `default_host` if it must be reachable inbound.
3. Re-provision, then copy each hospital's startup kit to its own machine.

Mirror the same addresses in `config/federation.py`. **Job definitions, client code, model
and data layout are untouched** — which is the whole reason addresses live in those two
files and nowhere else. Each hospital needs: its startup kit, this repository, and its own
partition folder. It never needs another hospital's data.

### 13.7 Data isolation and patient-level partitioning

* **Every slice of a patient goes to exactly one hospital.** Splitting by slice would let
  a model recognise the *patient* rather than the disease.
* Deterministic — **seed 42**, recorded in each `partition.json` with the source path and
  build timestamp, so any split can be reproduced exactly. Partitions built
  2026-08-03T23:03:03 – 23:03:33 UTC.
* Each hospital additionally holds out **20%** of its own patients as a *local* validation
  split (`local_val_fraction = 0.2`). That produces the metric the server selects on, and
  it is local by construction — a hospital cannot validate on another's patients.
* The **global test set is held out before any partitioning** and is never trained on.
* Training pool: **1,527 patients, 12,131 images.** Every partition is a complete cover of
  that same pool — verified: hospitals + global val + global test reconcile exactly to
  **2,063 patients and 16,378 images in all four partitions**.
* **The images are never regenerated here.** `prepare_data.py` and `partition_data.py`
  **HARDLINK** the existing PNGs, so the federated layout and the source dataset are the
  same inodes on disk. `audit_dataset.py` verifies this **by inode** — 136/136 in the last
  audit.

### 13.8 How the server aggregates

**FedAvg** — the global model is replaced by the sample-weighted mean of the client
models. **FedProx** — *identical* server side; the difference is entirely client-side
(`mu/2·‖w−w_global‖²` added to the local loss). **FedOpt** — the mean delta is treated as
a pseudo-gradient and the server takes an SGD step (lr 1.0, momentum 0.6).

`recipes.build_recipe` **refuses to build a FedProx job with `mu <= 0`**, `client.py`
refuses to apply a proximal term without global weights to anchor to, and
`verify_production.py` checks the mu actually passed to each job. A client that receives
the coefficient and ignores it would be running FedAvg while the results table says
FedProx, and nothing would warn you.

The reported loss **excludes** the proximal term — including it would make FedAvg and
FedProx losses incomparable across the very curves RQ3 is read from.

**Model selection:** `key_metric = "val_balanced_accuracy"`, computed on each hospital's
held-out patients. **Never training accuracy** — a previous iteration reported that to the
server, which then selected whichever global model let clients memorise their own shard
best (99%+, and no information at all). Balanced accuracy rather than macro-AUC because a
site holding 39 patients can draw a validation split missing a class, and macro-AUC is
then NaN.

**The learning-rate schedule.** A federated client is re-instantiated each round and holds
no state, so a `CosineAnnealingLR` object cannot survive to be stepped. Because cosine
annealing is a closed-form function of the step index and the server sends
`current_round` with every model, the client evaluates it directly:
`lr(r) = base·(1 + cos(π·r/T))/2`, which is exactly what `CosineAnnealingLR(T_max=T)`
holds at epoch `r` centrally. Both arms follow the same curve; the client keeps no state.

### 13.9 How jobs are launched, and the admin interface

Jobs are **not** hand-written JSON. `scripts/generate_jobs.py` writes
`deployment/jobs/testNN_*/job.py` from `src/federated/config/experiments.py`;
`generate_jobs.py --check` fails if any job has drifted from the table.

`scripts/run_experiment.py <testNN>` then:
1. verifies the partition (`verify_data.py`),
2. builds the recipe (`federation/recipes.py`),
3. opens a **secure admin session** as `admin@ips.pt` against `localhost:8003`,
4. submits the job, receives a UUID job id,
5. monitors until `FINISHED:COMPLETED`,
6. writes `results/<name>/job.json` with the id, timings and status.

Everything the script prints is timestamped into `logs/<test>/admin.log`, because terminal
output scrolls away and cannot be attached to a thesis.

The **PyTorch** recipe is used, not the generic one:
`nvflare.recipe.FedAvgRecipe` accepts `model` only as a dict and rejects an `nn.Module`;
`nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe` takes the built module, which is what
makes the server's copy the same object the clients construct.

### 13.10 Reproducing an experiment

```bash
cd federated

# once
./deployment/scripts/provision.sh
./deployment/scripts/distributions.sh
./deployment/scripts/verify.sh                 # must pass before anything starts

# the centralised baseline (NOT an NVFLARE job)
python scripts/run_centralized.py --seed 42

# one federated experiment
./deployment/scripts/start.sh 4 test06         # server + 4 hospitals
./deployment/scripts/run.sh test06             # submit through the admin API
./deployment/scripts/stop.sh

# after the runs
./deployment/scripts/collect.sh                # score every model on the one test set
./deployment/scripts/summary.sh                # build results/final_summary/
```

Client count per test: 2 for test02–03, 3 for test04–05, 4 for test06–09.

### 13.11 Verification

`src/scripts/verify_production.py` runs **219 pre-flight checks** and writes nothing. It
covers: the `production/` structure; `project.yml` (api_version, project name, both ports
against `federation.py`, all four clients, the admin e-mail regex, all four builders); the
provisioned workspace and every certificate; the dataset partitions against their
requested shares (to within one patient); the budget equality (30 rounds × 1 epoch = 30
centralised epochs); the FedProx mu actually passed to each job; unique result folder
names; and — added after two failures slipped past — that every job **exports** to JSON,
not merely builds.

---

## 14. FEDERATED EXPERIMENT MATRIX

**Held fixed across all thirteen** (this is the whole design): ResNet-18 ImageNet
pretrained, freeze `layer3`, dropout 0.5, AdamW lr 1e-4 / wd 5e-4, label smoothing 0.1,
batch 24, 1 slice per patient per batch, class weights inverse-frequency per patient
scope `local`, cosine schedule, **seed 42**, 224×224 RGB, dataset `multi_subtype_80mm`,
global test set 268 patients. The centralised baseline and the federated clients run
**literally the same trainer** — `src/federated/common/training.py` delegates to
`src/core/training.py` — so the gap RQ1 measures is federation rather than a difference
in code.

**Budget matching:** 30 rounds × 1 local epoch against 30 centralised epochs. The model
sees the data the same number of times on both sides. Without that, RQ1 would read a
difference in compute as a difference in federation. `verify_production.py` asserts it.

Reproduced from `src/federated/config/experiments.py`, which is the **single source of
truth** — ids, partition names, ratios and algorithms all come from `EXPERIMENTS` and
`PARTITIONS` there, and every job under `deployment/jobs/` is generated from it by
`generate_jobs.py`.

| Test | folder | clients | partition (ratio → shares) | patients per hospital | algorithm | mu | rounds × local epochs | RQ |
|---|---|---:|---|---|---|---:|---|---|
| **01** | `test01_centralized` | 1 | — (all pooled) | 1,527 | — | — | 30 epochs | RQ1 reference |
| **02** | `test02_fedavg_2h` | 2 | `2_clients_balanced` (1:1 → 50.0/50.0%) | 393 / 391 | FedAvg | — | 30 × 1 | RQ1 |
| **03** | `test03_fedprox_2h` | 2 | `2_clients_balanced` | 393 / 391 | FedProx | 0.01 | 30 × 1 | RQ3 |
| **04** | `test04_fedavg_3h` | 3 | `3_clients_balanced` (1:1:1 → 33.3% each) | 262 / 262 / 260 | FedAvg | — | 30 × 1 | RQ1 |
| **05** | `test05_fedprox_3h` | 3 | `3_clients_balanced` | 262 / 262 / 260 | FedProx | 0.01 | 30 × 1 | RQ3 |
| **06** | `test06_fedavg_4h` | 4 | `4_clients_balanced` (1:1:1:1 → 25.0% each) | 198 / 196 / 195 / 195 | FedAvg | — | 30 × 1 | RQ1 + RQ2 IID control |
| **07** | `test07_fedprox_4h` | 4 | `4_clients_balanced` | 198 / 196 / 195 / 195 | FedProx | 0.01 | 30 × 1 | RQ3 |
| **08** | `test08_fedavg_skewed` | 4 | `4_clients_skewed` (**5:2:1:1** → 55.6/22.2/11.1/11.1%) | 435 / 175 / 87 / 87 | FedAvg | — | 30 × 1 | RQ2 — **quantity skew only** |
| **09** | `test09_fedprox_skewed` | 4 | `4_clients_skewed` | 435 / 175 / 87 / 87 | FedProx | 0.01 | 30 × 1 | RQ4 |
| **10** | `test10_fedavg_cohort` | 3 | `3_clients_cohort` (**642:101:784** → 42.0/6.6/51.3%), `stratified=False` | 642 / 101 / 784 — **DUKE / I-SPY1 / I-SPY2** | FedAvg | — | 30 × 1 | **RQ2 primary**, against test12 |
| **11** | `test11_fedprox_cohort` | 3 | `3_clients_cohort` | 642 / 101 / 784, one cohort each | FedProx | 0.01 | 30 × 1 | **RQ3 strongest**, against test10 |
| **12** | `test12_fedavg_sizematched` | 3 | `3_clients_sizematched` (**642:101:784**), stratified | 642 / 101 / 784, cohorts mixed | FedAvg | — | 30 × 1 | **RQ2 control** |
| **13** | `test13_fedprox_sizematched` | 3 | `3_clients_sizematched` | 642 / 101 / 784, cohorts mixed | FedProx | 0.01 | 30 × 1 | RQ3, against test12 |

**On the numbering.** Tests 10–13 are ordered so each *partition* owns a consecutive
FedAvg/FedProx pair, exactly as 02–09 are ordered: 10/11 is the cohort split, 12/13 its
control. The matched pairs for RQ2 are therefore **10 vs 12** and **11 vs 13** — across
the pair, not adjacent within it. (An earlier layout numbered them 10/12 cohort and
11/13 mixed; that was swapped so the tables read consistently. Result folders,
`experiments.py`, `all_experiments.csv` and `summary.csv` were all renamed together, but
`per_client_metrics.csv` was not — see §10.2.)

Counts for tests 01–09 are read from `deployment/datasets/all_distributions.csv`, not
from the nominal percentages: 1,527 does not divide evenly, so the splitter allocates
the remainder to the earlier sites. Counts for 10–13 come from the cohorts themselves
(the training split holds 642 DUKE, 101 I-SPY1 and 784 I-SPY2 patients), which is why
that partition's "ratio" is written as absolute patient counts.

**On "50/20/10/10":** that sums to 90, not 100. It is a **5:2:1:1 ratio**, which
normalises to 55.6 / 22.2 / 11.1 / 11.1. `Partition` stores the ratio and normalises in
code, so the dissertation's wording and the program's behaviour agree instead of silently
differing by ten percent.

**Per-hospital train/val counts** (from `final_summary/summary.md`, `cohort/per_client_data.csv`):

| partition | site | train pat | train img | val pat | val img | class % (HR+/TN/HER2+) |
|---|---|---:|---:|---:|---:|---|
| 2c balanced | hospital_1 | 612 | 4,870 | 152 | 1,214 | 50.7 / 26.8 / 22.5 |
| 2c balanced | hospital_2 | 611 | 4,849 | 152 | 1,198 | 50.6 / 26.8 / 22.6 |
| 3c balanced | hospital_1 | 408 | 3,245 | 102 | 816 | 50.5 / 27.0 / 22.5 |
| 3c balanced | hospital_2 | 408 | 3,249 | 102 | 814 | 50.5 / 27.0 / 22.5 |
| 3c balanced | hospital_3 | 406 | 3,206 | 101 | 801 | 50.7 / 26.8 / 22.4 |
| 4c balanced | hospital_1 | 306 | 2,439 | 77 | 614 | 50.7 / 26.8 / 22.5 |
| 4c balanced | hospital_2 | 305 | 2,423 | 77 | 616 | 50.5 / 26.9 / 22.6 |
| 4c balanced | hospital_3 | 305 | 2,426 | 76 | 608 | 50.5 / 26.9 / 22.6 |
| 4c balanced | hospital_4 | 305 | 2,399 | 76 | 606 | 50.5 / 26.9 / 22.6 |
| 4c skewed | hospital_1 | 678 | 5,395 | 170 | 1,358 | 50.6 / 26.8 / 22.6 |
| 4c skewed | hospital_2 | 273 | 2,171 | 67 | 534 | 50.5 / 26.7 / 22.7 |
| 4c skewed | hospital_3 | 136 | 1,073 | 34 | 266 | 50.7 / 27.2 / 22.1 |
| 4c skewed | hospital_4 | 135 | 1,066 | 34 | 268 | 51.1 / 26.7 / 22.2 |
| **3c cohort** | hospital_1 **DUKE** | 514 | 4,067 | 128 | 1,007 | **66.3 / 16.0 / 17.7** |
| **3c cohort** | hospital_2 **I-SPY1** | 82 | 652 | 19 | 152 | **41.5 / 26.8 / 31.7** |
| **3c cohort** | hospital_3 **I-SPY2** | 627 | 5,005 | 157 | 1,248 | **38.9 / 35.9 / 25.2** |
| 3c size-matched | hospital_1 | 514 | 4,096 | 128 | 1,020 | 50.6 / 26.8 / 22.6 |
| 3c size-matched | hospital_2 | 81 | 640 | 20 | 160 | 50.6 / 27.2 / 22.2 |
| 3c size-matched | hospital_3 | 628 | 4,976 | 156 | 1,239 | 50.6 / 26.9 / 22.5 |

Every row above is read from `results/federated/final_summary/cohort/per_client_data.csv`.
Each site's local 20% validation split is carved out of its own patients, so
514 + 128 = 642 for hospital_1 and so on.

**⚠ THE LIMITATION THE DISSERTATION MUST STATE ABOUT TESTS 02–09.** All four of those
partitions are **stratified**: every hospital keeps the global class ratio. Recomputed
from the patient counts above, the maximum class-share spread across hospitals is
**0.43 percentage points** for the skewed partition (and 0.32 pp for the size-matched
one). So between hospitals only the **quantity** of data varies. **Tests 08 and 09 are
quantity skew, not genuine non-IID label heterogeneity.** The normalised class panel in
every distribution figure is what makes it visible — flat bars mean quantity-only. A
reader who misses this will over-claim what tests 08/09 measure.

**What quantity skew alone does and does not demonstrate.** It *does* test whether
FedAvg's sample-weighted averaging behaves when one site dominates the mean and three
sites contribute few, noisy updates — a real engineering property of the aggregation
rule. It does *not* test heterogeneity in any distributional sense: each site's expected
local gradient is the same, only the variance differs, so a proximal term has nearly
nothing to pull against. That is why FedProx's effect flipped sign across those four
configurations, and why RQ2 had no defensible answer until tests 10–13 existed.

**Evaluation procedure, identical for all thirteen:** the selected global model is scored
on the **same** global test set (268 patients, 2,115 images), slice probabilities
averaged per patient first. `collect_results.py` does this for every experiment, so a
difference between two rows can never be a difference in evaluation.

### 14.1 Tests 10–13 — the matched pair that answers RQ2

This is the design contribution of the second campaign, and it is worth stating
carefully because its whole value lies in what is held constant.

**Both partitions hold the same three site sizes: 642, 101 and 784 patients.** In
`3_clients_cohort` each site *is* one real cohort. In `3_clients_sizematched` the same
three sites are filled with a stratified draw from all three cohorts. Client count,
rounds, local epochs, seed, model, hyperparameters, global test set and evaluation code
are identical. **The only quantity that varies is whether a site's data is
cohort-native.**

| | one cohort per site | size-matched control |
|---|---|---|
| partition | `3_clients_cohort` (`stratified=False`) | `3_clients_sizematched` (stratified) |
| site sizes | 642 / 101 / 784 | 642 / 101 / 784 |
| class shares | 66.4/15.9/17.8 · 41.6/26.7/31.7 · 38.9/35.8/25.3 | ~50.6/26.9/22.5 at all three |
| **class-share spread** | **27.45 pp** | **0.32 pp** |
| tumour size | DUKE ~5× smaller by volume than I-SPY2 | same mix everywhere |
| what else differs | scanner population, acquisition protocol, annotation type | nothing |

Both spreads were recomputed for this document from the patient counts in
`per_client_data.csv` (train + val rows combined), not taken from any earlier note.

**The result.**

| Algorithm | one cohort per site | cohorts mixed | difference |
|---|---:|---:|---:|
| FedAvg (test10 vs test12) | 0.5426 | 0.5836 | **−0.0410** |
| FedProx (test11 vs test13) | 0.5678 | 0.5882 | **−0.0204** |

**Both point the same way: real heterogeneity costs performance.** That consistency is
what the quantity-skew comparison never produced — there the two pairs disagreed in sign
(−0.0549 and +0.0175), the signature of noise dominating. Under a null hypothesis, two
independent comparisons both landing in the predicted direction has probability 0.25:
suggestive, not conclusive, and both differences are still inside the 0.067 noise floor,
so **the magnitude is not established**.

**The finding that is not in the aggregate.** Recall on the minority HER2+ class:

| test | HR+/HER2− | Triple Negative | HER2+ |
|---|---:|---:|---:|
| test10 — one cohort per site | 0.577 | 0.385 | **0.113** |
| test12 — cohorts mixed | 0.511 | 0.423 | **0.321** |

HER2+ recall collapses from 32% to 11%, and that class's AUC falls to **0.4728** — below
chance. **Under genuine heterogeneity the minority class goes first**, and a paper
reporting only aggregate metrics would not show this. FedProx partially recovers it
(0.113 → 0.283 in test11), which is the clearest RQ4 evidence the project has.

**Report the source probe beside test10, always.** With one cohort per site, "identify
the cohort, then use that cohort's prior" becomes available to the *aggregated* model in
a way it is not under stratified partitions. Worth noting as the honest counter-argument:
within any single client of test10 the cohort is **constant**, so it carries no
discriminative information locally and cannot be learned as a shortcut there — it can
only re-emerge after aggregation.

**Rebuild either partition with:**
```bash
python src/scripts/partition_data.py --by-cohort --only 3_clients_cohort --hardlink
python src/scripts/partition_data.py --only 3_clients_sizematched --hardlink
```
A third partitioner, `--stratify none` (label skew without cohort identity), is
implemented and **has never been run**.

---

## 15. EVALUATION

### 15.1 The rules the tooling enforces

1. **Every metric is per PATIENT.** Slice probabilities are averaged into one prediction
   per patient before anything is computed. Slice-level numbers are recorded
   (`slice_accuracy`, `slice_loss`, `n_slices`) **only as an overfitting signal**.
2. **Macro-AUC is the headline metric**, on the global test set, identical for all nine.
3. **Accuracy is never quoted without the trivial baseline of the same split.** It is not
   a constant: **0.5112** on this test set. It is computed from the test CSV at evaluation
   time, never hard-coded (that bug happened — §17.7).
4. **The noise floor is 0.067 macro-AUC.** Differences smaller than that are not results,
   and every comparison table carries a `within_noise_floor` column saying so.
5. **One seed is not a result.**

### 15.2 Metrics computed, for every experiment

Reported **per class and macro-averaged**, all three classes:

| metric | where |
|---|---|
| accuracy | `metrics.json → global_test.accuracy` |
| **balanced accuracy** | `global_test.balanced_accuracy` |
| macro precision / recall / F1 | `macro_precision`, `macro_recall`, `macro_f1` |
| per-class precision / recall / F1 | `per_class_precision`, `per_class_recall`, `per_class_f1` |
| **ROC-AUC one-vs-rest, per class** | `per_class_auc` |
| **macro AUC** | `auc` |
| confusion matrix | `confusion` (3×3) and `confusion_global.csv` |
| class counts | `class_counts` |
| trivial baseline | `trivial_baseline_accuracy` |
| ROC curves | `curves.json` + `figures/roc_global.{pdf,png}` and `roc_hospital_N.*` |
| **PR curves** | `figures/pr_global.{pdf,png}` and `pr_hospital_N.*` — **yes, available** |
| training/validation loss and accuracy per round | `sites/rounds.csv` / `seed_42/rounds.csv` |
| sklearn text report | `report_test.txt` (centralised) |

Balanced accuracy = macro recall by construction; both are reported because reviewers ask
for different ones.

### 15.3 Why macro metrics matter here

The class distribution is **1,042 / 564 / 457** patients (2.25:1) and on the test set
**137 / 78 / 53**. Plain accuracy is dominated by HR+/HER2−: **always predicting it scores
0.5112**, which is *higher* than every one of the twelve federated models achieved. Macro-averaging gives
each class equal weight, so a model that ignores HER2+ entirely cannot hide. This is
exactly what happened: test01 has accuracy 0.5299 (above baseline) but balanced accuracy
0.4503 and HER2+ recall 0.1887.

**Macro-AUC additionally is threshold-free**, which matters because four separate
attempts to tune a decision threshold on validation all failed to transfer (§8.5).

### 15.4 Which dataset is used for final testing, and whether it is shared

**One global test set, shared by every experiment: `deployment/data/global/test.csv`, 268
patients, 2,115 images, trivial baseline 0.5112.** It is held out before any partitioning
and is never trained on by anybody.

**The server holds it.** In a production federation the server would hold nothing; here it
holds a held-out set because the thirteen experiments must be compared on identical
ground. **This is a benchmarking decision, not a claim about deployment**, and the
dissertation must say so.

**The server's *validation* signal is different and does not come from that set.** The
metric the server selects the global model on (`val_balanced_accuracy`) is computed by
each **client**, on that client's own 20% local validation split, and reported upward as
a scalar. The server never sees an image. Under the stratified partitions those local
splits are all draws from the same distribution; **under the cohort partition they are
not**, so in tests 10–13 the server is selecting on a mean of three metrics measured on
three genuinely different populations — a property of federated model selection worth
one sentence in the dissertation.

**Client-specific evaluation also exists** but is secondary: `per_client_metrics.csv`
reports the final global model's performance on each site separately. Those splits are
19–170 patients and are **not** comparable across experiments. **⚠ The current file is
stale and incomplete — see §10.2 and §17.14b.** Honest statement of what is in the
repository today: per-client metrics exist for the four cohort/size-matched experiments
only, under their old ids, and must be regenerated before use.

### 15.6 Communication — the RQ3 measurement

Each global model file is **44,789,067 bytes**, identical for all twelve federated runs
(11,178,051 parameters plus buffers, fp32). That is what crosses the boundary per client
per round **in each direction**, so a 30-round run with *n* clients moves
`2 × 30 × n × 44.8 MB`: **2.6 GB** at 2 clients, **3.9 GB** at 3, **5.3 GB** at 4.

**Almost none of it is necessary.** Averaged across sites, the aggregated global model's
validation AUC after **one** communication round is already 94–98% of its best value on
the stratified partitions. Recomputed from every `sites/rounds.csv` for this document
(column `agg_val_auc`, mean across clients per round):

| test | partition | round-1 / best | reaches 95% at round | reaches 99% at round |
|---|---|---:|---:|---:|
| 02 | 2c balanced | 0.960 | 1 | 2 |
| 03 | 2c balanced | 0.944 | 2 | 2 |
| 04 | 3c balanced | 0.960 | 1 | 3 |
| 05 | 3c balanced | 0.967 | 1 | 2 |
| 06 | 4c balanced | 0.973 | 1 | 3 |
| 07 | 4c balanced | 0.983 | 1 | 4 |
| 08 | 4c skewed | 0.963 | 1 | 3 |
| 09 | 4c skewed | 0.953 | 1 | 2 |
| **10** | **3c cohort** | **0.909** | **5** | **9** |
| **11** | **3c cohort** | 0.945 | 2 | 2 |
| **12** | 3c size-matched | 0.956 | 1 | 4 |
| **13** | 3c size-matched | 0.942 | 4 | 13 |

Stopping at round 4 would have saved **~87%** of the traffic on tests 02–09 with no
measurable loss. **Note the cohort rows:** test10 needs five rounds to reach 95% where
almost every stratified run needs one, and test13 needs thirteen rounds to reach 99%.
Slower convergence under heterogeneity is the expected behaviour of FedAvg on non-IID
clients and is a second, independent signature of the effect §14.1 measures — obtained
from the round curves rather than from the final score.

**What is NOT measured:** wall-clock communication time, bandwidth, and any
compression/quantisation scheme. Only payload size and round count. The 87% figure is
therefore a statement about *what was sent*, not about *what it cost in seconds*.

### 15.5 Model selection differs between the two arms — stated, not hidden

| arm | selects on | why |
|---|---|---|
| centralised (test01) | validation **macro-AUC** | the classifier phase's rule; AUC is well defined on 268 patients |
| federated (02–13) | **`val_balanced_accuracy`**, averaged over hospitals | a site holding 19 patients can draw a validation split missing a class, and macro-AUC is then NaN. Under the cohort partition hospital_2 holds exactly 19 validation patients, so this is not a hypothetical |
| FedOpt (never reported) | **nothing** — `FedOptRecipe` rejects `key_metric`, so the LAST round would be kept | the reason FedOpt is not comparable and was dropped (§10.7) |

Both of the first two are computed on held-out patients, which is the part that matters.
Neither is training accuracy.

---

## 16. FILES AND RESULTS — the complete map

**Repository root:** `/Users/daniel/Developer/tese/federated-breast-mri-subtyping`
(76 GB total, under git since 2026-08-05).
**Do not confuse it with the stale copy** at `.../federated-breast-classification` —
see §0.

**The layout below is the result of the 2026-08-05 reorganisation.** Every path in the
previous version of this document (`src/config.py`, `federated/production/…`,
`BreastDCEDL/` at the root, `src/data/multi_subtype_80mm/`, `src/docs/report_figures/`,
notebooks numbered 01/03/05/06/07) is **stale**. The rule the reorganisation follows:
the root holds only `README.md`, `requirements.txt` and seven folders; everything else
belongs to one of them.

### 16.1 Top level

| PATH | PURPOSE | CONTENT | size |
|---|---|---|---:|
| `README.md` | project entry point | 10 KB, links to every folder README and to the six documents | |
| `requirements.txt` | dependencies | `nvflare>=2.8,<3`, `torch>=2.2`, `torchvision`, `numpy`, `pandas`, `pillow`, `scikit-learn`, `matplotlib`. **⚠ its comment about Apple MPS is stale — see §17.12** | |
| `raw_dataset_BreastDCEDL/` | **the ZENODO RELEASE — raw imaging.** Never written to | `BreastDCEDL_{DUKE,ISPY1,ISPY2}_min_crop/`, `BreastDCEDL_metadata_min_crop.csv`, `BreastDCEDL_models.tar.gz` (released ViT weights), `BreastDCEDL_dataset.pdf`, `BreastDCEDL_demo_data/`, `README.md`, **`download_dataset.py`** | 35 GB |
| `dataset/` | **the processed 2-D dataset** | `README.md` + `multi_subtype_80mm/{config.json, metadata.csv, train.csv, val.csv, test.csv, images/}` | 979 MB |
| `src/` | **all the code** | see 16.2 | 900 KB |
| `deployment/` | **the running system** | see 16.3 | 76 MB |
| `results/` | every run that was kept | `classifier/` (phase 1) and `federated/` (phases 2–3) — see 16.3 | 1.0 GB |
| `docs/` | all documentation and every figure | 15 Markdown documents + `images/` | 22 MB |
| `notebooks/` | the pipeline as notebooks | `01_dataset_analysis` · `02_build_dataset` · `03_train_centralized` · `04_evaluate_run` · `05_compare_experiments` | 4.9 MB |
| `unused/` | everything archived, nothing deleted | see 16.4 — **gitignored** | 38 GB |

**Where the authors' code went.** `BreastDCEDL/` is no longer at the root; the clone is
archived at `unused/reference_implementations/BreastDCEDL_authors_repo/`. The naming
trap it used to create is gone: the only `BreastDCEDL` name in the active tree is
`raw_dataset_BreastDCEDL/`, which is the **data** release.

**What git ignores, and why** (`.gitignore`): `unused/` (not part of the deliverable);
`raw_dataset_BreastDCEDL/*` **by content**, with `!README.md` and
`!download_dataset.py` named back in, because git does not descend into an ignored
directory and the downloader would otherwise vanish with the imaging; `dataset/*/images/`
(the 16,378 PNGs are regenerable, the manifests that define them are tracked);
`deployment/workspace/` (**PKI startup kits hold private keys**); `*.pt`/`*.pth`;
and the hardlinked per-hospital image trees. **⚠ `.gitignore` still carries rules for
the pre-reorganisation layout** (`/federated_breast_flare/…`,
`/federated_breast_classification/…`) which now match nothing — harmless, but confusing
to read, and the stale duplicate repository is **not** ignored.

### 16.2 `src/` — all the code

| PATH | PURPOSE | CONTENT |
|---|---|---|
| `dataset_config.py` | the single configuration object for the classifier phase | `Config`, `TASKS`, `PIPELINE`/`TASK`/`MODEL`, `RAW_DIR`, `COHORT_DIRS`, `DATA_DIR`, `RESULTS_DIR`. **Renamed from `config.py`** so it cannot shadow `src/federated/config/`. `COHORT_DIRS` now points at `raw_dataset_BreastDCEDL/` — the long-standing stale path is **fixed** |
| `core/models.py` | model factory | `build_model`, `_new_head`, `_retune_dropout_before`, 13 architectures |
| `core/data.py` | dataset + augmentation | `AugmentConfig`, `PROFILES`, `apply_augment`, patient-aware sampler |
| `core/dataset_builder.py` | **the builder** | the `Record` dataclass (the 35 CSV columns), the whole PNG-generation loop |
| `core/training.py` | **the shared trainer** | `get_device()` (CUDA → MPS → CPU cascade), `describe_device()`, `train_one_epoch(..., progress=)`, `run(cfg, ..., progress=)` |
| `core/evaluation.py` | patient-level metrics | aggregation, macro-AUC, trivial baseline |
| `core/reporting.py` | auto-reporting | 24 files per run |
| `core/experiment.py` | run orchestration | |
| `pipelines/reference/preprocessing.py` | **the authors' rules**, each citing its source | 4 slices, 224 px, per-slice min-max |
| `pipelines/thesis/preprocessing.py` | **this thesis's rules**, each citing its measurement | 8 slices, 80 mm, per-volume min-max |
| `federated/config/experiments.py` | **THE SINGLE SOURCE OF TRUTH** | `TrainingConfig`, `FederationConfig`, 6 `PARTITIONS`, 13 `EXPERIMENTS`, path constants |
| `federated/config/federation.py` | provisioning-level settings | participants, ports, organisations |
| `federated/common/{models,data,training,evaluation}.py` | the client-side layer | thin delegations to `core/`, plus the architecture fingerprint and the non-finite-update guard |
| `federated/common/thesis.py` | the import bridge to `core/` | resolves the sibling package |
| `federated/federation/client.py` | the NVFLARE client | local train → validate → send; refuses to transmit a non-finite update |
| `federated/federation/recipes.py` | FedAvg / FedProx / FedOpt recipe construction | |
| `scripts/run_centralized.py` | the centralised baseline (not an NVFLARE job) | |
| `scripts/partition_data.py` | patient-level splitting | `--by-cohort`, `--stratify none`, `--only`, `--hardlink` |
| `scripts/prepare_data.py` | builds the global test/val split | hardlinks, never copies |
| `scripts/generate_jobs.py` | generates all 13 jobs from `experiments.py` | `--check` mode |
| `scripts/run_experiment.py` · `run_all_experiments.py` | submit and monitor via the admin API | |
| `scripts/collect_results.py` | pulls each global model, scores it on the global test set | `model_provenance()` reads `meta_props.current_round` |
| `scripts/build_final_summary.py` | **the aggregated deliverable** | `--no-client-eval` skips per-hospital scoring (§10.2) |
| `scripts/verify_production.py` | **219 pre-flight checks**, writes nothing | re-run 2026-08-05: **all 219 pass** |
| `scripts/verify_data.py` · `audit_dataset.py` | split and dataset integrity | |
| `scripts/build_dataset_report_figures.py` | regenerates `docs/images/report_figures/` | |
| `scripts/build_preprocessing_walkthrough.py` | regenerates `docs/images/preprocessing_figures/` | |
| `scripts/build_distribution_report.py` · `snapshot_config.py` | distribution figures · config snapshot | |

### 16.3 `deployment/` and `results/` — the running system and its output

| PATH | PURPOSE | CONTENT |
|---|---|---|
| `deployment/project.yml` | **the NVFLARE provisioning file** | server (`fed_learn_port: 8002`, `admin_port: 8003`), `hospital_1..4` with orgs `h1..h4`, `admin@ips.pt` as `project_admin` |
| `deployment/workspace/breast_fl_project/` | the PKI startup kits | **private keys — gitignored** |
| `deployment/jobs/testNN_*/` | 13 generated jobs | generated from `experiments.py`; never hand-edited |
| `deployment/data/global/{test,val}.csv + images/` | **the official global test set** | 268 + 268 patients |
| `deployment/data/partitions/<partition>/hospital_N/{train,val}.csv + images/` | per-hospital data | 6 partitions; images are **hardlinks** to `dataset/multi_subtype_80mm/images/` |
| `deployment/datasets/` | split manifests and provenance | `all_distributions.{csv,json}`, `global_splits.csv`, `dataset_audit.json`, 13 × `testNN_*_distribution.csv` |
| `deployment/logs/testNN/` | per-participant logs | `server.log`, `hospital_N.log`, `admin.log`, `timeline.log`, `pids` |
| `deployment/figures/` | one distribution figure per experiment | 13/13 present, asserted by `verify_production.py` |
| `deployment/config/resolved_config.{json,md}` | a configuration **snapshot** | generated 2026-08-05T14:01; nothing reads it back |
| `deployment/scripts/` | operator entry points | `provision.sh`, `start.sh`, `run.sh`, `stop.sh`, `collect.sh`, `summary.sh`, `verify.sh`, `distributions.sh` — all delegate to `src/scripts/` |
| `results/classifier/all_runs_pod.csv` | **the 21-run classifier table** | §10.3 |
| `results/classifier/checkpoints/` | 7 checkpoints | `FREEZE_R18_s{1,42}`, `R18_s{1,42}`, `R50_s1`, `SPY2_R18_s{1,42}` |
| `results/classifier/_from_pod/multi/<run>/` | per-run raw output | `results.json`, `history.csv`, `config.json`, `train.log`, `stdout.log`, prediction CSVs |
| `results/federated/test01_centralized/seed_42/` | the centralised baseline | `best_model.pt`, `results.json`, `rounds.csv` (30 epochs), `predictions_test.csv`, `report_test.txt`. **The `seed_42/` level exists only here** |
| `results/federated/testNN_*/` (02–13) | each federated run | `job.json`, `global_model.pt` (44,789,067 B), `test_metrics.json`, `predictions_test.csv`, `sites/rounds.csv`, `sites/train.log` |
| `results/federated/_ablations/` | the production freezing ablation | `freeze_layer3_seed_42/`, `freeze_none_seed_42/` |
| `results/federated/all_experiments.csv` | 13 rows, one per experiment | the compact results table |
| **`results/federated/final_summary/`** | **THE AGGREGATED DELIVERABLE** | see below |
| `.../final_summary/summary.{csv,xlsx,json,md,pdf}` | the whole campaign in five formats | 13 rows; regenerated 2026-08-05T11:03 |
| `.../final_summary/comparisons/` | 8 CSVs | `centralized_vs_fedavg`, `centralized_vs_fedprox`, `fedavg_vs_fedprox`, `fedavg_vs_fedprox_paired`, `2_hospitals`, `3_hospitals`, `4_hospitals_balanced`, `4_hospitals_skewed`. **No `3_hospitals_cohort` comparison file exists** — the RQ2 pair is not yet a generated table |
| `.../final_summary/tables/` | 9 LaTeX tables | `main_results.tex` + one per comparison |
| `.../final_summary/experiments/<name>/` | per-experiment detail | 13 folders, `metrics.json`, `confusion_global.csv`, `curves.json`, predictions, `figures/` |
| `.../final_summary/figures/` | 5 cross-experiment figures × 2 formats | `bar_metrics_all`, `roc_all_experiments`, `metric_comparison_heatmap`, `training_evolution`, `federated_round_evolution` |
| `.../final_summary/cohort/` | dataset context | `partitions.csv` (6 partitions), `per_client_data.csv`, `class_distribution.{pdf,png}` |
| `.../final_summary/per_client_metrics.csv` | per-hospital metrics | **stale and incomplete — §10.2** |
| `.../final_summary/manifest.json` | what was generated and when | `n_complete: 13`, `incomplete: {}` |

**One log-per-participant, never a shared file**, because two participants appending to
one log interleave mid-line under load and the result cannot be reconstructed.
`timeline.log` timestamps federation-level events so the order of events *across*
participants is recoverable.

### 16.4 `unused/` — the archive (34 GB, nothing deleted)

| PATH | PURPOSE |
|---|---|
| `old_datasets/` (18 GB) | 9 superseded datasets — §2.9 |
| `old_dataset_builders/breastdcedl_project_src/` | the previous `src/`. Holds `prepare_multicohort.py` (the builder that produced every current dataset), `run_queue.py`, `collect_2x2.py`, `make_source_probe.py`, `deploy.sh`. **This is still the code running on the GPU pod** |
| `old_training/` | `train_gpu.py`, `models.py`, `run_queue.py`, 4 collectors. **All 21 runs in `all_runs_pod.csv` were produced by this code** |
| `old_notebooks/` | notebooks 01–11 of the previous structure. **Notebook 11 documents the Duke bbox validation (767/767) and carries the before/after crop figures** |
| `old_runs/` (7.8 GB) | 8 campaigns — §10.5, §10.6 |
| `old_figures/` | every figure from the previous notebooks, including `13_bug_fromarray.png` |
| `legacy_projects/` (8.4 GB) | `breastdcedl_project` (shell), `federated_breast_classification` (the first federated system, 7.7 GB), `federated_breastdcedl`, `radiomic_ai` (the original PyRadiomics+RF app), `test_autores` |
| `old_docs/` | two handovers (2026-08-01, 2026-08-02) and four older Portuguese notes — **the narrative record of how the design was reached** |
| `unused/README.md` | explains why each folder was retired, with the measurement behind it |

---

## 17. BUGS AND PROBLEMS DISCOVERED

Every one of these is documented so it is not reintroduced. **Note the recurring
signature: "the run completes and the numbers look plausible."** That is why the project
now uses an architecture fingerprint, `strict=True` on every checkpoint load, a
declarative experiment table, and a 219-check pre-flight.

### 17.1 The dropout regression — a fix that removed the wrong thing

**PROBLEM.** `core/models.py::build_model` accepted a `dropout` argument and **ignored
it** for every torchvision backbone — resnet, efficientnet, convnext, mobilenet,
densenet, vit_b_16, swin. Verified: `build_model("resnet18", 3, pretrained=False,
dropout=0.0)` and `dropout=0.5` returned identical `state_dict` keys and a bare
`Linear(512, 3)`, with **zero `Dropout` modules in the graph**. Only `THDAResNet` and
`HFClassifier` honoured it. Meanwhile every `results.json` kept recording
`dropout: 0.5`. Additionally, **no checkpoint could be loaded**: all seven in
`results/checkpoints/` store `fc.1.weight` / `fc.1.bias` (i.e.
`Sequential(Dropout, Linear)`), so `strict=True` raised *Missing key(s): fc.weight,
fc.bias / Unexpected key(s): fc.1.weight, fc.1.bias*.

**CAUSE.** An earlier bug — `_replace_head` inserting a `Dropout` **into** an existing
`Sequential`, shifting classifier indices — was "fixed" by **deleting the Dropout**
instead of making the head layout consistent. That left `dropout` a dead config field.

**DETECTION.** While wiring the federated project up, a checkpoint load failed. Two
things had hidden it for months:
* **`Dropout` has no parameters.** Both builds total **11,187,671** params+buffers for
  ResNet-18, so a parameter count in a run log proves nothing about the head. Only the
  `fc.1.*` key names distinguish them.
* **`strict=False` "succeeds"** and leaves `fc` at random init — confirmed, the weight
  tensor is bit-for-bit unchanged after loading. On this task that still produces a
  plausible near-chance macro-AUC.

**SOLUTION.** `core/models.py` now honours `dropout` for **every** backbone, via two
helpers:
```python
def _new_head(in_features, num_classes, dropout):
    head = nn.Linear(in_features, num_classes)
    return nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head

def _retune_dropout_before(seq, i, dropout):
    if i > 0 and isinstance(seq[i - 1], nn.Dropout):
        seq[i - 1].p = dropout
        return True
    return False
```
**The invariant:** the head ends with exactly one `Dropout(p)` feeding the final
`Linear`; **never insert into an existing `Sequential`.** The final `Linear` is only ever
replaced in place, and where a backbone already ships a `Dropout` feeding it
(efficientnet, mobilenet) that `Dropout` is **retuned** to `p` rather than stacked with a
second one. At `dropout = 0` no `Sequential` is created and the key layout is exactly
torchvision's. `src/federated/common/models.py::_attach_dropout_head`, which existed only to
work around this from outside, was **removed** — with core honouring the field, wrapping
again would apply dropout twice.

**VALIDATION.** All seven checkpoints load with `strict=True`, the ResNet-18 ones at
11,187,671 params+buffers, and the architecture fingerprint is stable.
**Blast radius: none — checked, not assumed.** All 21 runs in `all_runs_pod.csv` were
produced by `unused/old_training/`, whose `build_model` *did* wrap the head as
`Sequential(Dropout, Linear)`; every one of their `config.json` files carries the old
field names (`model_name`, `aug_profile`, `backbone_lr_scale`), and no run anywhere was
written by the current `core/` schema. **No recorded number was trained without dropout
and nothing needs re-running.** The regression was latent — the first affected run would
have been the next one. (The old code applied dropout to resnet, densenet and swin but
**not** to efficientnet, convnext, mobilenet or vit, so "restore what the old code did"
would have re-created the inconsistency. It is uniform now.)

### 17.2 The dataset-source shortcut ★ the central finding

**PROBLEM.** Luminal B reached F1 ≈ 0.98 on what the literature calls the hardest class.
**CAUSE.** The model was answering "is this I-SPY2?" instead of "which subtype is this?".
Luminal B was 90% I-SPY2 because Duke contains only 3 Luminal B patients.
**DETECTION.** Trained the identical pipeline to predict the **source**: macro-AUC 0.967
against subtype 0.589 (old catalogue); **0.9978 vs 0.6078** (current).
**SOLUTION.** Single-source datasets became the default; pooled datasets must be reported
with the probe beside them; the 80 mm physical window removes the resolution signature;
the framing rule is identical for masks and boxes.
**VALIDATION.** The same 72 I-SPY2 Luminal B patients inside an all-I-SPY2 dataset
dropped to F1 **0.077**. Seven pipeline integrity checks passed (0 divergences in 20,028
rows), so it is not a labelling bug. Full detail: §9.

### 17.3 Server / client architecture mismatch

**PROBLEM.** `FedAvgRecipe` exported the model as `{"path": "model.ClassifierNet"}`, so
**the server built a ResNet-18 while the clients built a ResNet-50.** The run completed
and the numbers were meaningless.
**CAUSE.** A dotted path with the wrong defaults, plus hand-edited generated JSON.
**DETECTION.** Results looked wrong; investigation traced it to the exported job config.
**SOLUTION.** The **PyTorch** recipe is used (`nvflare.app_opt.pt.recipes.fedavg`), which
takes the built `nn.Module`; the server and every client build from **the same config
object**; an **architecture fingerprint** (`2d3031acc2075813`) is checked at both ends;
jobs are generated from `config/experiments.py`, never hand-edited.
**VALIDATION.** `generate_jobs.py --check` and `verify_production.py` both fail on drift.

### 17.4 The server selected on TRAINING accuracy

**PROBLEM.** `key_metric = "accuracy"` was fed by **training** accuracy, so the server
selected whichever global model let clients memorise their own shard best (99%+).
**SOLUTION.** `key_metric = "val_balanced_accuracy"`, computed on each hospital's
held-out patients. Balanced accuracy rather than macro-AUC because a small site's
validation split can be missing a class, making macro-AUC NaN.

### 17.5 `collect_results.py` had `"resnet18"` hard-coded

**PROBLEM.** Federated runs trained ResNet-50 for 50 rounds and then **crashed at
evaluation** (test6 of the archived campaign, after 5,826 s).
**SOLUTION.** The architecture is read from the run's config. **VALIDATION.** The
campaign was re-run and completed.

### 17.6 `cls_trainer.py` hard-coded dropout and weight decay

**PROBLEM.** Federated runs used different regularisation from the centralised baseline
they were compared against — so the measured gap was partly regularisation, not
federation.
**SOLUTION.** One shared `TrainingConfig`; `src/training.py` delegates to the same
`src/core/training.py` the centralised run uses.

### 17.7 `TRIVIAL_BASELINE_ACC` hard-coded to 40/99

**PROBLEM.** Every `results.json` carried 0.404 when the truth for that split was 0.5112.
**SOLUTION.** The baseline is computed from the test CSV at evaluation time. It is not a
constant and never will be.

### 17.8 `train_paper.py` read `model_name`, the CLI passed `--model`

**PROBLEM.** **10 runs silently trained the wrong architecture.**
**SOLUTION.** Argument names unified; the resolved config is written into every
`results.json`.

### 17.9 `torch.load` default `weights_only=True`

**PROBLEM.** Checkpoints became unreadable after training on a newer torch.
**SOLUTION.** Explicit `weights_only=False` where our own checkpoints are loaded.

### 17.10 `FedOptRecipe` does not accept `key_metric`

**PROBLEM.** `TypeError: FedOptRecipe.__init__() got an unexpected keyword argument
'key_metric'` — tests 12 and 13 failed 0.1 min after launch.
**SOLUTION.** `common.pop("key_metric", None)` in `federation/recipes.py`.
**CONSEQUENCE THAT MUST BE REPORTED:** **FedOpt has no server-side model selection — it
keeps the LAST round, while FedAvg/FedProx keep the best.** Any FedOpt-vs-FedAvg
comparison is therefore not like-for-like.

### 17.11 NVFLARE could not serialise a torchvision ResNet

**PROBLEM.** `TypeError: Object of type type is not JSON serializable` when submitting a
job.
**CAUSE.** A torchvision ResNet stores `self._norm_layer = nn.BatchNorm2d` — **a class,
not an instance** — and NVFLARE's `_get_args` records constructor arguments raw. Worse,
without the fix NVFLARE would have rebuilt a **default 1000-class** model on the server
against 3-class clients.
**SOLUTION.** `src/federated/common/models.py::FederatedClassifier` — a thin wrapper that stores
its constructor arguments as plain instance attributes so `_get_args` can recover them,
and delegates `state_dict` / `load_state_dict` to the inner network:
```python
class FederatedClassifier(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=3, pretrained=True,
                 dropout=0.5, freeze_until="layer3", freeze_bn=False):
        super().__init__()
        self.model_name = model_name; self.num_classes = num_classes
        self.pretrained = pretrained; self.dropout = dropout
        self.freeze_until = freeze_until; self.freeze_bn = freeze_bn
        self.net = build_model(SimpleNamespace(...), num_classes)
    def state_dict(self, *a, **kw):  return self.net.state_dict(*a, **kw)
    def load_state_dict(self, sd, *a, **kw): return self.net.load_state_dict(sd, *a, **kw)
```
Adapted from the user's legacy `ClassifierNet`, adding the attribute storage the legacy
version lacked. **VALIDATION.** All nine jobs submitted and completed.

### 17.12 Apple MPS corrupts weights — ★ ROOT-CAUSED AND FIXED, 2026-08-05

This entry previously ended "root cause: NOT VERIFIED — the failure was contained, not
explained." It is now explained, and the explanation is worth reading because the
diagnostic method is reusable.

**PROBLEM.** Training on Apple MPS produced **NaN loss** and
`isfinite(parameters) == False` partway through the first epoch, with `train_acc` stuck
around 0.90 — an impossible accuracy for epoch 1 on this task, which was the tell.
Non-deterministic: the same seed gave 0.6312 and 0.6832.

**CAUSE — `x.to(device, non_blocking=True)` from unpinned memory.** The training loop
issued asynchronous host-to-device copies. `non_blocking=True` is only safe when the
*source* is pinned, and this project pins only on CUDA
(`pin_memory=torch.cuda.is_available()`). On MPS the copy therefore returned before it
had finished while the DataLoader was free to reuse the source buffer, so the network
trained on **partially overwritten batches**. Nothing raises; the weights simply drift
to NaN a few hundred steps in.

**DETECTION — bisection between a passing and a failing path.** Two earlier attempts
declared MPS fixed and were wrong: the first because the probe ran 61 steps of a
508-step epoch, the second because a `GradScaler` change was accepted without re-running
`run()`. What worked was holding everything else identical and comparing a path that
passed against a path that failed, one difference at a time, over a **full** epoch.

**SOLUTION** — three files, `src/core/training.py`, `src/core/evaluation.py`,
`src/federated/common/training.py`:
```python
non_blocking = device.type == "cuda"
x = x.to(device, non_blocking=non_blocking)
y = y.to(device, non_blocking=non_blocking)
```
Asynchronous where it is safe and helps, synchronous everywhere else. Alongside it:
`get_device()` became a CUDA → MPS → CPU cascade with `PYTORCH_ENABLE_MPS_FALLBACK=1`
set before use; the AMP scaler is gated on `scaler.is_enabled()` so the non-CUDA path
takes a plain `backward()`; and `apply_mps_workaround()` — which had been dead code,
defined but never called by `run()` — was wired in.

**VALIDATION**, one full epoch, same seed and data:

| device | loss | train_acc | time |
|---|---:|---:|---:|
| MPS, `non_blocking=True` | **nan** | 0.9425 | — |
| MPS, blocking copies | **1.1539** | 0.4372 | 231 s |
| CPU, same code | 1.1502 | 0.4237 | 589 s |

MPS is now finite, 2.5× faster than CPU, and agrees with CPU to 0.004 loss.

**CONSEQUENCE, unchanged:** anything trained locally on the Mac *before* this fix is
suspect. It does not touch any reported result — the whole campaign ran on CUDA.

**⚠ TWO PLACES STILL CARRY THE OLD DIAGNOSIS AND SHOULD BE CORRECTED.**
`src/federated/common/models.py::get_device` still defaults to `allow_mps=False` and its
docstring still says MPS "is BROKEN here"; `requirements.txt` still says "Apple MPS
corrupts this network's weights to NaN … so src/models.py falls back to CPU there". The
federated arm banning MPS is harmless (it runs on CUDA) but the stated reason is now
wrong. The non-finite-update guard in `federation/client.py` should stay regardless —
it defends against any diverged client, not just this one cause.

### 17.13 Cohort leakage / the Duke bounding-box limitation

**PROBLEM.** Duke has **no voxel mask**, only a bounding box on the largest-tumour plane.
Using the 3D union for I-SPY and a single plane for Duke would have introduced a
systematic cohort-specific difference in framing — the most expensive kind of error here.
**SOLUTION.** The in-plane box is taken from **the largest-area slice for every cohort**,
so mask and box are treated identically. `tumor_pixels` / `tumor_area_mm2` /
`tumor_fraction` are **−1** for Duke and **never imputed**.
**VALIDATION.** The box construction matched the true mask box for **767/767** I-SPY2
patients with `n_xy == 256`. Physics check that the Duke crops actually contain tumour
(no mask to overlay), enhancement (G−R) centre vs periphery:

| cohort | ROI source | centre | periphery | ratio | centre > periphery |
|---|---|---:|---:|---:|---:|
| spy2 | mask | 30.14 | 7.32 | 4.12× | 95% |
| spy1 | mask | 34.97 | 8.84 | 3.96× | 90% |
| **duke** | **box** | 23.54 | 5.02 | **4.69×** | **96%** |

**Also:** 33% of Duke patients have a box side < 32 px. With a proportional crop that is
a 4–11× upsample and the blur itself becomes a cohort cue — solved by the 80 mm physical
window.

### 17.14 Preprocessing / configuration inconsistencies

| problem | status |
|---|---|
| `config.py::COHORT_DIRS` pointed at the authors' code clone, which holds the metadata CSV but **not** the imaging volumes | **FIXED 2026-08-05.** The file is now `src/dataset_config.py` (renamed so it cannot shadow `src/federated/config/`), `RAW_DIR = REPO_ROOT / "raw_dataset_BreastDCEDL"`, and the fallback resolver that was hiding the breakage is gone. The builder would run today |
| `test01` and `_ablations` record `"cohorts": ["spy2"]` while training on the pooled dataset | **OPEN.** The field is inert when a prepared dataset path is given (`results.json` records `source: …/multi_subtype_80mm` and the logs confirm 1,527 pooled patients) but the record is misleading and should be corrected before the results chapter cites it |
| `authors_subtype` exists and is trained, but `Config` refuses the combination | **OPEN.** Needs an explicit `allow_cross_pipeline` flag |
| `notebooks/04_evaluate_run.ipynb` and `05_compare_experiments.ipynb` still open with the headings `# 06 — …` and `# 07 — …` | **OPEN, cosmetic.** The files were renumbered 01–05; two internal titles were not. Confusing when a reader matches a heading to a filename |
| `.gitignore` still holds rules for the pre-reorganisation layout (`/federated_breast_flare/…`, `/federated_breast_classification/…`) | **OPEN, harmless.** They match nothing. The stale duplicate repository (§0) is **not** ignored and is not inside this tree, so git does not see it either way |

### 17.14b The per-client metrics file went stale across the renumbering ⚠ OPEN

**PROBLEM.** `final_summary/per_client_metrics.csv` holds twelve rows under the
**pre-renumbering ids `test14`–`test17`** and no rows at all for tests 02–09.
**CAUSE.** The last rebuild ran `build_final_summary.py --no-client-eval`, which
regenerates `summary.csv` but leaves the per-client file untouched — so it kept both the
old ids and the old, narrower experiment set.
**DETECTION.** Reading the file while writing this document; the ids do not exist in
`experiments.py`.
**SOLUTION (not yet applied).** Re-run `build_final_summary.py` **without**
`--no-client-eval`. Every input still exists — each `global_model.pt` and each
hospital's `val.csv` — so all 39 rows regenerate under the correct ids.
**Until then, §10.2 is the only surviving record of the tests 02–09 per-hospital
numbers.**

### 17.14c The notebook epoch display read as an off-by-one ⚠ FIXED, cosmetic

**PROBLEM.** During training the progress bar showed `epoch 003/…` while the last
printed summary line still read `epoch 002/…`, which looks like the bar is an epoch
ahead.
**CAUSE.** It is not: the bar labels the epoch being *trained*, the summary line is
printed only once that epoch has *finished*, so during epoch 3 the newest completed line
is necessarily epoch 2. An earlier genuine off-by-one (`epoch + 1` in the bar label) had
already been fixed; this was the display remaining ambiguous.
**SOLUTION.** The bar now reads `epoch 003/030 training` and the summary line
`epoch 002/030 done`, with identical zero-padding on both sides of the slash.
**VALIDATION.** `src/core/training.py` parses; no tooling reads that log line.

### 17.15 Federated class weights computed locally ⚠ OPEN, and it is RQ4 material

**PROBLEM.** `class_weight_scope = "local"` — each hospital computes class weights from
its **own** rows. Clients therefore optimise slightly different objectives, and FedAvg
averages models trained on different losses.
**STATUS — this stopped being hypothetical on 2026-08-05.** Harmless in the stratified
partitions (the weights agree to three decimals), but tests 10 and 11 **ran on the
cohort partition with `class_weight_scope = "local"`**, where hospital_1 sees 66.4%
HR+/HER2− and hospital_3 sees 38.9%. Those two runs therefore averaged models trained on
**measurably different objectives**, and that is a confound inside the RQ2 result: part
of what tests 10/11 measure as "the cost of heterogeneity" may be the cost of
*optimising three different losses*, not of the data being heterogeneous as such.
Running the same pair with `scope = "global"` separates the two and is item 2 of §21.
**BOTH OPTIONS ARE IMPLEMENTED.** `partition_data.py` writes `global_class_weights` into
every site manifest, computed once from the pooled training split, so no site has to see
another site's data to use them. `local` = realistic, divergent objectives; `global` =
one objective, one leaked vector of class counts. **This is exactly RQ4.**

### 17.16 Reporting and tooling bugs

| PROBLEM | CAUSE | SOLUTION |
|---|---|---|
| `to_markdown` raised | `tabulate` not installed | hand-rolled `md_table` |
| `ValueError: truth value of a DataFrame is ambiguous` in `discover()` | `df or default` on a DataFrame | explicit `if seed_rounds is not None` |
| FedAvg compared against a **mean** of configurations | the comparison table had no pairing key | added `fedavg_vs_fedprox_paired`, keyed on partition |
| table headers clipped; integers rendered `29.0000` | no column formatting | `COLUMN_LABELS`, `column_label()`, `fmt_frame()`, `INTEGER_COLUMNS` |
| a real `nan` in an object column reached `str.join` and raised | `fmt_frame` did not fill NaN before stringifying | `show.astype(object).where(pd.notna(show), "")` |
| the round-evolution figure was **silently empty** | `rounds.csv` is written to `<exp>/sites/`, not `<exp>/` | `discover()` now checks both locations |
| fig5 plotted the **reciprocal** resampling factor (0.52× instead of 1.96×) | inverted expression | `224.0 / (80.0 / xy_spacing)` |
| variable `idx` shadowed (DCE phases vs slice midpoint) | name collision | **DCE phase indices were silently erased**; renamed |
| `startswith("p")` matched `pid` | over-broad prefix match | tried to average patient identifiers; fixed |
| `verify_production` only **built** jobs, never exported them | missing check | export-to-JSON check added — **two failures had slipped past** |

### 17.17 Server / client / process-management bugs

| PROBLEM | CAUSE | SOLUTION |
|---|---|---|
| hospitals never started; the check failed **silently** | `start_federation.sh` polled the admin port with `nc`, absent from the RunPod container | `port_open()` using a Python socket |
| `start_federation.sh` rejected test10+ | regex `^(test0[1-9]\|_scratch)$` | `^(test[0-9]{2}\|_scratch)$` |
| `pkill -f 'pattern'` **killed the ssh command running it** | the pattern matched the command's own argv | kill supervisors by PID first; NVFLARE's `sub_start.sh` supervisors otherwise respawn their children |
| `setsid` not found | it does not exist on macOS | removed from the macOS path |
| two runs appended to one log file | zsh aborts a command list on a failed glob, so the `rm` of the old log never ran | guarded |

### 17.18 Overfitting, imbalance and normalisation — recorded as problems

Covered in full in §8. In brief: best epochs of 1–5 with `train_acc` → 0.99 are routine;
the effective sample size is the **patient** count; **halving augmentation tripled the
gap**; **ImageNet intensity mismatch** is real (min-max PNGs have mean intensity
**31.5/255** where ImageNet expects ~115; after ImageNet normalisation a batch has mean
−1.43, std 0.56; `chanclip` reaches 98.8/255 **and still lost** by 0.025) but is **not
the bottleneck**; class weighting fixes the ranking, not the decision boundary.

### 17.19 Run-to-run variance — the noise floor

**PROBLEM.** Two byte-identical configurations differing only in seed scored **0.7023 and
0.6351** — a gap of **0.067**.
**CAUSE.** `seed` fixes initialisation and the split but not cuDNN kernel selection, AMP,
or DataLoader worker ordering.
**DETECTION.** An earlier reading of ±0.001 was a lucky pair and is **wrong**.
**SOLUTION.** 0.067 is enforced as the noise floor in the reporting code; every
comparison table carries `within_noise_floor`.
**IMPACT.** This **invalidates most single-run comparisons made earlier in the project**,
including the entire architecture benchmark.

---

## 18. IMPORTANT SCIENTIFIC FINDINGS

### POSITIVE RESULTS

1. **A real NVFLARE federation was deployed and ran the full matrix.** Thirteen
   experiments across two campaigns, PKI, separate processes, admin API, **zero
   failures** — 47.9 minutes for tests 01–09 and ~34 minutes for 10–13. Not the
   simulator. This satisfies OBJ3 outright.
2. **RQ1 answered as an equivalence claim.** Centralised 0.6068 against a federated mean
   of 0.5927 over twelve runs — a gap of **0.0141, 4.8× smaller than the 0.067
   equivalence margin**, with every federated run inside the margin. Stated positively:
   *the cost of federation on this task is smaller than the cost of re-running the
   centralised configuration with a different seed.* Single-seed caveat stands (§1.5).
2b. **RQ2 has a real answer for the first time, from a matched pair.** With site sizes,
   client count, rounds, seed and algorithm held identical and only cohort-nativeness
   varying, cohort-native sites cost **−0.041 (FedAvg)** and **−0.020 (FedProx)** macro-AUC.
   Neither difference clears the noise floor alone; what makes it a finding is that
   **two independent comparisons agree in direction**, where the quantity-skew pairs
   disagreed in sign. And the aggregate hides the real damage: **HER2+ recall collapses
   0.321 → 0.113** and its AUC falls to 0.4728, below chance (§14.1).
2c. **RQ3 has a strong communication result.** Averaged across sites, the global model's
   validation AUC after **one** communication round is already 94–98% of its best value
   on the stratified partitions; 30 rounds were used and four would have sufficed. At
   44.8 MB per client per round per direction, **~87% of the traffic bought nothing
   measurable**. Under genuine heterogeneity convergence is visibly slower — test10
   reaches only 90.9% of its best at round 1 and needs 5 rounds to reach 95%, against 1
   round for most stratified runs — which is itself a heterogeneity signature.
2d. **FedProx behaves as designed, but only where there is drift to correct.** +0.025
   macro-AUC on the cohort partition against +0.005 on its size-matched control — five
   times larger where the sites genuinely differ, and it partially restores the collapsed
   HER2+ recall (0.113 → 0.283). On the stratified partitions its effect flipped sign
   across all four configurations, which is what "nothing to correct" looks like.
3. **The MinCrop geometry of BreastDCEDL was reverse-engineered and verified to the
   voxel** — 767/767 exact matches. This made Duke usable for the first time and grew the
   dataset from 982 to 2,063 patients.
4. **The Duke bounding box was proven to contain tumour by physics**, without a mask:
   enhancement at the centre is 4.69× the periphery, and the centre exceeds the periphery
   in 96% of patients — *better* than either mask cohort.
5. **The 80 mm physical window works as designed.** Every image sits at a constant 0.357
   mm/px, and the resampling factor is equalised across cohorts (1.91–2.20×) instead of
   4× vs 2×.
6. **Freezing to `layer3` reduced seed spread ten-fold**, 0.026 → 0.003. This is what
   makes the federated baseline trustworthy.
7. **BreastDCEDL's published results are verifiable** from their own prediction file —
   every number recomputes exactly (0.7201 / 0.7801 / 0.6793 / 0.5398).
8. **The full tooling chain is reproducible**: one declarative experiment table, generated
   jobs, **219 pre-flight checks (all passing, re-run 2026-08-05)**, hardlink-verified
   data, and a one-command summary generator producing csv/xlsx/json/md/pdf plus LaTeX
   tables and all figures.
9. **The Apple-MPS failure was root-caused**, after twice being declared fixed and not
   being: asynchronous host-to-device copies from unpinned memory. MPS now trains finite
   and 2.5× faster than CPU (§17.12). Not a scientific result, but it is the reason the
   notebooks can be run on the author's own machine.

### NEGATIVE RESULTS — none of these are hidden, and several are the contribution

1. **The ceiling is 0.55–0.63 macro-AUC for 3-class subtype, and NOTHING moves it.**
   Twenty-one runs, five data configurations, thirteen architectures. Not preprocessing,
   not architecture, not normalisation, not augmentation, not freezing, not field of view.
2. **Every one of the twelve federated runs scored BELOW the trivial accuracy baseline**
   of 0.5112 (0.4030–0.4888). The models rank patients better than chance but decide
   worse than a constant rule.
3. **HER2+ is at chance, and worse under heterogeneity.** Per-class AUC 0.5079 in the
   centralised run (recall 0.1887) and **0.4728 in test10** — below chance — with recall
   0.1132. This is the single most clinically pointed negative result in the project.
3b. **The RQ2 effect size is not established.** Both cohort-vs-control differences
   (−0.041, −0.020) sit inside the noise floor. Direction is supported by agreement
   across two algorithms; magnitude is not supported at all, and the document should
   never quote one without the other.
3c. **Tests 08/09 do not measure heterogeneity.** Their partitions are stratified to
   within 0.43 pp, so what they vary is quantity. Reporting them as a non-IID result
   would be an over-claim, and this is stated in `experiments.py` at the definition
   itself so it cannot drift out of the documentation.
4. **Halving augmentation tripled overfitting** — train acc 0.57 → 0.99, gap 0.135 →
   0.512, AUC −0.040.
5. **`chanclip` lost by 0.025** despite being the winning normalisation in the dataset
   authors' own seven-way benchmark. The literature did not transfer.
6. **`pclip` lost by 0.034** on all three seeds.
7. **The new preprocessing did not beat the old one** — 0.5837 ± 0.011 against
   0.6201 ± 0.024 on the same 99 test patients. "No difference detected", and *not* the
   improvement expected.
8. **Multi-task decomposition hurt TripleNeg by 0.040**, against a literature claim of
   +3.9%.
9. **3D and 2.5D both lost** (0.53–0.58 and −0.015).
10. **Balanced sampling (−0.027) and the label-noise filter (−0.032) both lost.**
11. **We fall 0.15–0.19 short of BreastDCEDL's published binary results**, two to three
    times the noise floor — and their training code does not exist to compare against.
12. **BreastDCEDL's published inference notebook feeds the model byte-garbage**
    (correlation with the actual MRI: **0.0114**), and the prediction file shipped in
    `transformer_models/` does not reproduce their paper (AUC 0.52 against 0.72).
13. **Four validation-based selection strategies all failed to transfer** — threshold,
    slice aggregation, ensemble composition, best-vs-last checkpoint.
14. **RQ2 has never been properly tested.** The skew is quantity-only.

### IMPORTANT OBSERVATIONS

1. **The ceiling is SIGNAL, not capacity.** A 4,098-parameter linear probe scored
   **0.6813** — above fully fine-tuned 23.5M — with an overfitting gap of 0.020 against
   0.37. If capacity were the bottleneck this could not happen.
2. **The literature agrees.** A 106-study, 12,989-patient systematic review concludes
   conventional quantitative MRI features "might play a limited role in the prediction of
   breast cancer subtypes". Zhang et al. report 0.79/0.91 within-centre collapsing to
   **0.52/0.44 cross-centre** — our numbers correspond to their cross-centre line.
   **0.61 is a correct answer, not a failure.**
3. **The cohorts are trivially separable** (probe 0.9978 vs subtype 0.6078). Any pooled
   result must be reported with the probe.
4. **The noise floor is 0.067 macro-AUC**, which invalidates most single-run comparisons
   ever made on this task — including many in this project's own history.
5. **The effect of federation was all-or-nothing on the binary task** (2, 3 and 4
   hospitals gave the same result), *contradicting* an earlier lung-segmentation project
   where degradation was progressive.
6. **Federation hurt the clinically important class** on the binary task: TripleNeg recall
   fell from 48.6% to 27–40% in eight of nine configurations. Papers usually report only
   the aggregate.
7. **The federated model saturates after round 1** — on the archived campaign round 1
   already contained 99.3% of the final macro-F1, meaning ~90% of communication traffic
   is wasted. A direct argument about the communication/performance trade-off (RQ3).
8. **Architecture does not matter** across 1.5M–87.6M parameters, CNNs, transformers and
   3D networks.
9. **The current augmentation is what holds the model back from memorising** — this is the
   only regulariser with a measured effect.

### UNRESOLVED QUESTIONS

1. **Is the 0.15–0.19 deficit against the authors entirely training hyperparameters?**
   The decisive test — their released weights through our preprocessing — has never been
   run and costs minutes.
2. **Does federation cost anything on the current dataset?** The equivalence claim says
   the cost is below the margin; the binary campaign said 0.068–0.110, *outside* it.
   Same infrastructure, different task, half the data. One seed per job cannot resolve
   which regime this task is in, and the most likely explanation — that it depends on
   having enough patients per site — is a hypothesis, not a measurement.
3. **How much of the RQ2 effect is heterogeneity and how much is the local class-weight
   scope?** Tests 10/11 varied both at once, because `class_weight_scope = "local"` on a
   27.45 pp prior spread means the three sites optimised different losses. **This is the
   most important open question about the newest result**, and §21 item 2 answers it.
4. **Is the RQ2 effect label skew or feature skew?** The cohort partition changes class
   priors, tumour size and scanner population simultaneously. `--stratify none` would
   separate the first from the other two; it is implemented and has never been run.
5. **Does freezing genuinely reduce overfitting?** Two seeds moved the gap in opposite
   directions. And `layer4` (25% of parameters) has never been tested — only `layer3`
   (6.1%).
6. **Would ComBat or adversarial de-biasing reduce the source signature without
   destroying the signal?** Never attempted. The probe result (0.9978 vs 0.6078) means
   the absolute height of every pooled number in this project is unquantifiably
   optimistic.
7. **Do the 918 Duke metadata rows vs 916 in the paper matter?** Unresolved.

**Resolved since the previous version:** *"Would a cohort-based partition change RQ2's
answer?"* — yes, it gave RQ2 its first consistent answer (§14.1). *"Why does MPS corrupt
weights?"* — asynchronous copies from unpinned memory (§17.12).

---

## 19. IMPORTANT REFERENCES

**Every entry below was actually consulted in this project.** Nothing here is invented;
where a claim could not be sourced it is marked UNCONFIRMED in the relevant section.

### Datasets

| source | supported |
|---|---|
| **BreastDCEDL** — [Zenodo 18114231](https://zenodo.org/records/18114231) · TCIA | the entire dataset; MinCrop release; `BreastDCEDL_models.tar.gz` (released ViT weights, unused) |
| **Duke-Breast-Cancer-MRI** — [TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) | the Duke cohort, its bounding-box annotation, its clinical series character |
| **I-SPY1 / ACRIN 6657** — [TCIA](https://www.cancerimagingarchive.net/collection/ispy1/) | the I-SPY1 cohort and its enrolment criteria |
| **I-SPY2** — [TCIA](https://www.cancerimagingarchive.net/collection/ispy2/) | the I-SPY2 cohort |
| **MAMA-MIA** — Synapse `syn60868042` | downloaded then deleted; its `site` column (22 real hospitals inside I-SPY2) is the strongest available upgrade to RQ2 |

### Papers

| paper | supported |
|---|---|
| Fridman, N. et al., **BreastDCEDL**, *Sci Data* **13**, 264 (2026) · [arXiv:2506.12190](https://arxiv.org/abs/2506.12190) | the dataset spec, the RGB phase fusion, min-max + 8-bit, the phase-selection rule, the published pCR AUCs we audit in §12 |
| Fridman, N. et al., **THDA-ResNet / normalisation benchmark**, [arXiv:2510.13897](https://arxiv.org/html/2510.13897) | the `chanclip` idea (q0.98 per channel, 0.744 vs 0.700), median slice aggregation, the THDA-ResNet architecture we tested |
| Saha, A. et al., *Br J Cancer* **119**, 508–516 (2018) | the Duke collection |
| Chitalia et al., I-SPY1 expert annotations — [PMC9308769](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9308769/) | I-SPY1 tumour annotations |
| Zhang et al. — [PMC8547260](https://pmc.ncbi.nlm.nih.gov/articles/PMC8547260/) | **the reference for the 3-class formulation**; the within- vs cross-centre collapse (0.79/0.91 → 0.52/0.44) that our numbers match |
| Ensemble ResNet, 4 centres — [PMC12130697](https://pmc.ncbi.nlm.nih.gov/articles/PMC12130697/) | the 0.62–0.76 range for 2D ResNets on this task; WeightedBCE for imbalance |
| Systematic review, 106 studies — [PMC9028183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9028183/) | **why 0.61 is not a failure** |
| Spatial Multi-Task Learning — arXiv:2601.07001 | the multi-task idea (tested, hurt); **tumour-core-only is the worst configuration (−4.9%)**, which is why peritumoral tissue is kept |
| Peritumoral margin — [PMC9263840](https://pmc.ncbi.nlm.nih.gov/articles/PMC9263840/) · Sci Rep 2025 | the 4–6 mm optimal margin, basis for the 80 mm derivation |
| Multidimensional ROI, 3 centres — [PMC12065080](https://pmc.ncbi.nlm.nih.gov/articles/PMC12065080/) | ROI design |
| Imbalance strategies review — [PMC13029843](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029843/) | **why weighted loss and nothing more exotic**: mild (1:4–1:10) → weighted loss; focal loss "may even decrease performance" |
| LMFLoss — [arXiv:2212.12741](https://arxiv.org/pdf/2212.12741) | why imbalance-aware losses do not transfer to a 2.25:1 ratio |
| MAMA-MIA — [PMC11923173](https://pmc.ncbi.nlm.nih.gov/articles/PMC11923173/) | the site-ID upgrade path |
| FLamby — [arXiv:2210.04620](https://arxiv.org/abs/2210.04620) | cross-silo FL benchmark; reference for the federated protocol |
| **FedAvg** — McMahan et al., 2017 | the aggregation rule of tests 02/04/06/08 |
| **FedProx** — Li et al., 2020 | the proximal term of tests 03/05/07/09; μ = 0.01 |
| Clinical implications of intrinsic molecular subtypes — [*The Breast*, 2022](https://www.sciencedirect.com/science/article/pii/S0305737222001724) | the prognostic rationale for the three classes |
| Deciphering HER2 Breast Cancer Disease — [*Front. Oncol.* 9:1124, 2019](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2019.01124/full) | HER2 biology |
| 2018 ASCO/CAP HER2 guideline — [PMC8742337](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8742337/) | the HER2 scoring standard the cohort definitions refer to |

### Repositories and documentation

| source | supported |
|---|---|
| [naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL) | §12 in its entirety — the MinCrop generation notebooks, the inference notebook, the prediction files. Cloned locally at `BreastDCEDL/` |
| [NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare), `examples/advanced/cifar10` | the `src/` + per-algorithm-folder layout, the Recipes API, `ProdEnv`, the Client API, `PTFedProxLoss` |
| [NVFLARE documentation](https://nvflare.readthedocs.io/en/main/) | job structure, recipes, `ProdEnv`, provisioning, the admin API |

---

## 20. CURRENT PROJECT STATUS

### WHAT IS FINISHED

* **The classifier phase.** 21 runs, five data configurations, thirteen architectures.
  Characterised and closed.
* **The dataset.** `multi_subtype_80mm` built, audited, documented (`DATASET_REPORT.md`,
  `DATASET_DOCUMENTATION.md`, `DATASET_SPEC.md`) and figured.
* **The source probe.** 0.9978. Run and reported.
* **The 2×2 ablation.** Complete, with two seeds per cell.
* **The BreastDCEDL reproducibility audit.** Complete, every claim verified against files.
* **The NVFLARE production infrastructure.** `project.yml`, PKI workspace, 13 generated
  jobs, **6 partitions**, distribution figures, **219-check verification**, local
  multi-process deployment, per-participant logging, results organisation.
* **All thirteen dissertation experiments.** Tests 01–09 on 2026-08-04 (47.9 min) and
  tests 10–13 on 2026-08-05 (26 min). **Zero failures in either campaign.**
* **The final summary.** `results/federated/final_summary/` — 8 comparison tables,
  9 LaTeX tables, per-experiment confusion/ROC/PR figures, 5 cross-experiment figures,
  and `summary.{csv,xlsx,json,md,pdf}` covering all 13.
* **The repository reorganisation and its documentation.** `src/`, `docs/`,
  `deployment/`, `results/`, `notebooks/`, per-folder READMEs, a standalone dataset
  document, and a stdlib-only Zenodo downloader with a manual-instructions fallback on
  every failure path.
* **The MPS root cause** (§17.12), which makes the notebooks runnable on the author's
  own machine.

### WHAT IS CURRENTLY RUNNING

**Nothing.** No process is running, no job is queued, no GPU is rented. Both RunPod
hosts were released after their results were pulled back.

### WHAT HAS BEEN VERIFIED

| verification | result |
|---|---|
| MinCrop geometry against real masks | **767/767** |
| Duke `volume_depth == z_span + 4` | 12/12 |
| patients with `n_xy > 256` were cropped, not resized | **228/228** |
| Duke crops contain tumour (enhancement physics) | 96% centre > periphery, 4.69× ratio |
| federated data are the **same inodes** as the source PNGs | **136/136** by inode |
| partitions reconcile to the full dataset | **2,063 patients / 16,378 images**, all six partitions |
| per-hospital share vs requested share | within one patient, all partitions |
| budget equality (30×1 = 30 epochs) | asserted by `verify_production.py` |
| pre-flight checks | **all 219 pass** — re-run 2026-08-05 for this document |
| all seven checkpoints load `strict=True` | pass, 11,187,671 params+buffers |
| architecture fingerprint at server and clients | `2d3031acc2075813`, stable |
| pipeline integrity (old audit) | 7 checks, **0 divergences in 20,028 rows**, zero train/val patient overlap |
| every dataset count in this document | **recomputed from `dataset/multi_subtype_80mm/metadata.csv` on 2026-08-05** |
| patient-level splitting | **0 patients in more than one split; 0 patients with more than one label** — recomputed 2026-08-05 |
| class spreads quoted for the RQ2 pair | **recomputed** from `per_client_data.csv`: 27.45 pp cohort, 0.32 pp control, 0.43 pp skewed |
| the RQ3 convergence table | **recomputed** from every `sites/rounds.csv`; reproduces `RESULTS.md` exactly (round-1 fractions 0.944–0.983, tests 02–09) |
| MPS vs CPU agreement after the fix | loss 1.1539 vs 1.1502 over a full epoch |

### WHAT STILL NEEDS TO BE DONE

**Immediate / cheap / high value**
1. **Regenerate `per_client_metrics.csv` without `--no-client-eval`** (§17.14b). One
   command; restores 39 correctly-named rows. Do this before writing the results chapter.
2. **Run the authors' released ViT weights through our preprocessing.** Inference only,
   minutes. Still the highest-value pending scientific item.
3. **Correct the two stale MPS statements** in `src/federated/common/models.py` and
   `requirements.txt` (§17.12).
4. **Fix the misleading `"cohorts": ["spy2"]` field** in the centralised results.
5. **Decide what to do with the 76 GB stale duplicate repository** at
   `.../federated-breast-classification` (§0). It is not a backup — it is a trap.
6. **Commit the working tree.** The repository is under git but the notebooks are
   modified and uncommitted, and the history is two commits made outside the sessions.
7. **Fix the two notebook headings** that still say `# 06` and `# 07`.
8. **Finish `fig2_tumour_size_by_cohort`** — status of the cosmetic fixes is
   **NOT VERIFIED**; the figure was regenerated 2026-08-05T13:16 but the specific
   complaints (Duke absent from the right panel without explanation, no legend on the
   left panel, colliding median labels, literal backticks in the title) were not
   re-checked for this document.

**Scientific**
9. **More seeds.** Every federated number is one run. With a 0.067 noise floor, the
   campaign supports the equivalence claim and the *direction* of the RQ2 effect, and
   nothing sharper.
10. **Local vs global class weights under the cohort partition** — now doubly motivated,
    because tests 10/11 ran with `local` scope on genuinely divergent priors (§17.15).
11. **`--freeze-until layer4`** — the honest freezing test (25% of parameters vs 6%).
12. **Ensemble the existing runs** — free, and the literature reports +0.10.
13. **`--stratify none`** — label skew *without* cohort identity, the missing middle
    rung between quantity skew and the cohort partition.

### WHAT EXPERIMENTS ARE NEXT

See §21, in priority order.

### WHAT QUESTIONS REMAIN OPEN

See §18, "UNRESOLVED QUESTIONS". The four that matter most for the dissertation:
**(a)** does federation cost anything here, with enough seeds to tell; **(b)** how much
of the measured cohort-heterogeneity cost is heterogeneity and how much is the local
class-weight scope; **(c)** is the deficit against the authors training or data;
**(d)** how much of the absolute performance level is the cohort shortcut.

### THE STATE OF THE DISSERTATION DOCUMENT ITSELF

**NOT VERIFIED from this repository** — the LaTeX source is not in this tree and was not
inspected. From the working sessions rather than from files: the methodology sections on
preprocessing, slice selection, the deep-learning model, the development environment and
the experimental scenarios have been drafted; the results chapter is the immediate next
writing task; the breast-cancer and classification background chapter has a topic outline
but no prose. Treat this paragraph as a note to be replaced by whoever has the `.tex`.

---

## 21. FUTURE EXPERIMENTS, IN PRIORITY ORDER

**Reordered 2026-08-05.** What used to be item 3 — the cohort partition — has been run
and is now tests 10–13; FedOpt has been dropped rather than completed. Seed repetition
moves to **item 1**, and the reason is worth stating: the project now has two claims
that *depend* on the noise floor rather than merely being limited by it. RQ1 is an
equivalence claim, which is only as strong as the interval around the point estimate;
RQ2 rests on two differences that are both *inside* the noise floor and are believed
because they agree in direction. Three seeds converts both from "consistent with" to
"bounded by", and no other pending experiment improves the dissertation as much per hour
of GPU time.

### 1. Repeat the campaign with three seeds

* **Hypothesis:** the equivalence claim survives interval estimation, and the RQ2
  direction survives replication.
* **Changes:** seeds 42, 1, 2 for every one of the thirteen experiments. **Nothing else.**
* **Fixed:** dataset, model, partitions, protocol, evaluation.
* **Expected interpretation:** a mean ± sd per configuration. For RQ1, whether the
  centralised−federated confidence interval falls inside ±0.067 — the strong form of the
  equivalence test. For RQ2, whether the cohort-vs-control gap keeps its sign in all
  three seeds; if it does, the direction is established even though the magnitude may
  still be inside the noise floor. Anything that does not survive is not reportable.
* **Supports:** **RQ1 and RQ2**, and it is what makes the whole results chapter
  defensible. ~3.5 h on a rented GPU for all 39 runs.

### 2. Local vs global class weights on the cohort partition

* **Hypothesis:** with a 27.45 pp spread in class priors, local weights make the three
  sites optimise measurably different objectives and FedAvg averages models trained on
  different losses; global weights fix that at the cost of leaking one vector of class
  counts to the server.
* **Changes:** `class_weight_scope` `local` → `global`, on `3_clients_cohort` only.
  Both options are already implemented; `partition_data.py` writes
  `global_class_weights` into every site manifest.
* **Fixed:** everything else, including seed and rounds.
* **Expected interpretation:** this is a **measured privacy-versus-performance
  trade-off**, which is exactly the shape RQ4 asks for and the only one the project can
  currently offer. It also **disambiguates the RQ2 result**: if global weights recover a
  substantial part of the −0.041, then part of what tests 10/12 measured was the
  objective mismatch rather than heterogeneity itself. A nil difference is equally clean.
* **Supports:** **RQ4**, and it strengthens **RQ2** either way. ~30 min.

### 3. Authors' released weights through our preprocessing

* **Hypothesis:** our pixels are correct and the 0.15–0.19 deficit is entirely training
  hyperparameters.
* **Changes:** nothing is trained. Load `breastdcedl_pcr_vit_model_weights.pth`, run
  inference on our `authors_pcr` test images.
* **Fixed:** the weights, the test patients (175/176).
* **Expected interpretation:** if AUC ≈ 0.7201 and predictions correlate with
  `pred_pcr_vnew`, our preprocessing is vindicated and the deficit is training. If not,
  the preprocessing differs and the reproduction section must say so.
* **Supports:** the reproducibility contribution (contribution 3), and the credibility of
  every number in §10.3.
* **Cost:** minutes.

### 4. `--stratify none` — label skew without cohort identity

* **Hypothesis:** the missing middle rung. Tests 10–13 change class priors *and* tumour
  size *and* scanner population at once; this changes only the class priors.
* **Changes:** `partition_data.py --stratify none` on three sites; FedAvg and FedProx.
  The partitioner exists and has never been run.
* **Fixed:** site sizes, cohort mixture, everything else.
* **Expected interpretation:** if label skew alone reproduces most of the −0.041, the
  RQ2 result is a *label-prior* effect and the source-signature confound is not doing the
  work; if it reproduces little of it, feature skew (scanner, tumour size) is the larger
  term. Either answer sharpens the RQ2 chapter considerably.
* **Supports:** **RQ2**. ~30 min.

### 5. `--freeze-until layer4`

* **Hypothesis:** freezing 25% of the parameters (not 6%) genuinely reduces overfitting,
  not merely variance.
* **Changes:** one config field; 3 seeds.
* **Fixed:** everything else.
* **Expected interpretation:** if the train/test gap falls consistently across seeds,
  "reduces overfitting" becomes supportable — which it currently is not.
* **Supports:** the methodology chapter and the choice of baseline.

### 6. Ensemble the existing runs

* **Hypothesis:** averaging the per-patient probabilities of the existing checkpoints
  gains what the literature reports (+0.10 in some studies; +0.004 measured here so far).
* **Changes:** nothing is trained — average `predictions_test.csv` across runs.
* **Fixed:** everything.
* **Expected interpretation:** a cheap upper bound on what this data supports.
* **Supports:** the ceiling argument (contribution 4).

### 7. FedOpt on the cohort partition — *if* it is revived at all

* **Status:** dropped from the campaign (§10.7). Listed here only because the code still
  exists and the question is legitimate.
* **Hypothesis:** server momentum helps under heterogeneity, where a plain weighted mean
  does not — which is a **different and more interesting** hypothesis than the one the
  cancelled runs tested, since those used the stratified partitions where there was
  nothing to help with.
* **Changes:** the server's update rule only (lr 1.0, momentum 0.6, client mu = 0), on
  `3_clients_cohort` against test10.
* **Blocker to solve first:** `FedOptRecipe` rejects `key_metric`, so FedOpt keeps the
  **last** round while FedAvg and FedProx keep the best of thirty. Either find the
  supported selection mechanism in NVFLARE 2.8, or re-score every arm at its last round
  so the comparison is like-for-like. **Reporting a FedOpt number without resolving this
  would be a methodological error, not a caveat.**
* **Supports:** **RQ3/RQ4** — but behind items 1–4 in every respect.

### 8. Reacquire MAMA-MIA

* **Hypothesis:** 22 *real* hospital IDs inside I-SPY2 give the most realistic federated
  partition available anywhere in this project.
* **Changes:** the partition source becomes the `site` column.
* **Fixed:** everything else.
* **Expected interpretation:** federated learning across real sites, within one cohort —
  which removes the source-signature confound *and* keeps heterogeneity.
* **Supports:** **RQ2**, and it would be the strongest version of the whole thesis.

### 9. Inter-cohort intensity harmonisation (ComBat) + re-run the source probe

* **Hypothesis:** harmonisation lowers the source probe below 0.90 without lowering the
  subtype AUC.
* **Changes:** one preprocessing step; rebuild the dataset; re-run probe **and** subtype.
* **Expected interpretation:** probe ≫ subtype is the current state. If harmonisation
  closes the gap, pooled results become defensible; if it destroys the subtype signal too,
  that is itself a finding about what the model was using.
* **Supports:** the central confound (contribution 2).

### Explicitly NOT recommended

* **GAN augmentation** — best published gain ~0.01 against a 0.067 noise floor, and a
  subtype-conditioned generator would amplify the scanner signature.
* **Threshold tuning on the validation set** — failed four separate times.
* **Focal / Class-Balanced loss** — the imbalance ratio (2.25:1) is below the band where
  the literature says they help.
* **Regenerating MinCrop** — the Zenodo release *is* the authors' output.
* **Claiming any improvement from a single run.**

---

## 22. LESSONS LEARNED

### Mistakes that cost the most

1. **Trusting a good result.** Luminal B F1 0.98 should have triggered suspicion
   immediately, not months later. **A result that beats the literature on the hardest
   class is a bug report.**
2. **Believing a single seed.** An entire architecture benchmark (0.57–0.70 across
   backbones) distinguishes nothing once the noise floor is known.
3. **Trusting validation on ~100 patients.** Four separate selection strategies all failed
   to transfer.
4. **Copying code into 28 places.** The old federated project copied `model.py` to every
   participant and needed a `sync_model.py`. **FedAvg only averages correctly if every
   site builds an identical network.**
5. **Editing generated JSON.** That is how the server/client architecture mismatch
   happened.
6. **Not pulling results back immediately.** Three RunPod pods were lost in one session;
   the results of roughly ten experiment ladders exist only as comments in a script.

### Ideas that worked

* **The source probe.** Cheap, decisive, and it doubles as proof the pipeline is correct.
  **Run it on any new dataset before trusting anything.**
* **Declarative experiment tables.** When every experiment is a row, drift between
  experiments is not expressible.
* **Verifying reverse-engineered geometry against the real files.** 767/767 is a fact;
  "it should map" is a guess.
* **Generating notebooks from readable Python.** Reviewable in a diff.
* **Physical rather than proportional crops.** Removed a cohort cue *and* preserved a
  genuine predictor.
* **Writing the reason for a decision beside the code, with the number.**
* **Hardlinking rather than copying data**, and verifying it by inode.

### Best practices now adopted

1. Patient-level macro-AUC as the headline; slice-level only as an overfitting signal.
2. Never quote accuracy without the trivial baseline, computed from the split.
3. Split by patient, enforced by a verification step that refuses to pass.
4. Treat differences below 0.067 as noise.
5. One seed is not a result.
6. Compare only on the same test set.
7. Report "no difference detected" as a finding.
8. Move, never delete.
9. `strict=True` on every checkpoint load.
10. One log file per participant, never a shared one.

---

## 23. EXECUTIVE SUMMARY

**What this project is.** A master's dissertation measuring whether **federated learning
matches centralised training** on a real medical-imaging task, deployed on NVIDIA FLARE
with real PKI and separate processes rather than a simulator. Breast-cancer
molecular-subtype classification from DCE-MRI is the vehicle, not the contribution.

**Where it stands.** The classifier phase is **complete and characterised**: 3-class
subtype sits at **0.55–0.63 patient-level macro-AUC** against chance 0.50, and nothing
moves that ceiling. The federated phase is **complete**: **thirteen** experiments, real
NVFLARE, two campaigns, zero failures. Centralised **0.6068** against a federated mean of
**0.5927** — a gap of **0.0141 against a measured 0.067 equivalence margin**, which is
the positive form of RQ1's answer. Tests 10–13 added the matched cohort pair that finally
gives RQ2 an answer: cohort-native sites cost **−0.041 (FedAvg)** and **−0.020
(FedProx)**, consistent in direction across both algorithms, with **HER2+ recall
collapsing 0.321 → 0.113**. Every number is **one seed**, which is the binding
limitation on all of it.

**The five biggest discoveries.**
1. **The dataset-source shortcut.** A model trained on pooled cohorts answers "which
   scanner produced this?" instead of "which subtype is this?" — **macro-AUC 0.9978
   predicting cohort against 0.6078 predicting subtype**. This invalidated an entire
   earlier phase and now governs every design decision. It also proves the pipeline is
   correct: a broken pipeline could not reach 0.9978.
2. **The ceiling is signal, not capacity.** A frozen backbone with **4,098 trainable
   parameters** scored 0.6813 — above fully fine-tuned 23.5M — with an overfitting gap of
   0.020 against 0.37. With a 106-study review concluding MRI has a limited role here,
   0.61 is a correct answer rather than a failure.
3. **BreastDCEDL's training procedure is not reproducible.** No deep-learning training
   code exists anywhere in their repository — verified across all 16 notebooks, both `.py`
   files, and every deleted file across 196 commits. Their published inference notebook
   feeds the model byte-garbage (correlation with the actual MRI **0.0114**), and the
   prediction file they ship does not reproduce their paper (AUC 0.52 vs 0.72). Their
   **data** pipeline, however, is fully verifiable: MinCrop geometry matches to the voxel
   (767/767) and their published AUCs recompute exactly from their own prediction file.
4. **The noise floor is 0.067 macro-AUC**, measured between two byte-identical
   configurations differing only in seed. This invalidates most single-run comparisons
   made earlier in the project and is now enforced in the reporting code.
5. **Every federated run scored below the trivial accuracy baseline** (0.5112) while
   ranking patients better than chance (AUC 0.54–0.65), and HER2+ per-class AUC was
   **0.5079 centralised and 0.4728 under cohort heterogeneity — at and below chance**.
   The models rank but do not decide, and the minority class is where federation hurts.

**The four remaining challenges.** (a) Only one seed per federated job — which now limits
two claims rather than one, since RQ1's equivalence and RQ2's direction both rest on it.
(b) Tests 10/11 ran with **local** class-weight scope on genuinely divergent priors, so
part of the measured heterogeneity cost may be an objective mismatch rather than
heterogeneity itself. (c) The 0.15–0.19 deficit against the authors is unexplained and
the decisive test costs minutes. (d) The absolute level of every pooled number is
inflated by the cohort shortcut, and no harmonisation has been attempted.

**Next steps, in order.** 1) Three seeds on all thirteen experiments. 2) Global vs local
class weights on the cohort partition. 3) Authors' weights through our preprocessing.
4) `--stratify none`, the missing rung between quantity skew and cohort identity.
5) `layer4` freezing. Full detail in §21.





