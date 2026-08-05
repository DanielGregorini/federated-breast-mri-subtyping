# BreastDCEDL — Reproducibility Report

Full audit of [github.com/naomifridman/BreastDCEDL](https://github.com/naomifridman/BreastDCEDL)
(196 commits, full history, analysed at `git_BreastDCEDL/BreastDCEDL`), the Zenodo
release, and the two papers.

**Bottom line up front.** The *data* pipeline is fully reproducible and I verified it
against the real files. The *training* pipeline does not exist in the repository — not in
any notebook, not in `utils/`, and not in any deleted file recoverable from git history.
The published inference notebook, run as written, feeds the model byte-garbage rather
than MRI. The article's own prediction file, however, reproduces every published number
exactly.

---

## 1. Repository structure

```
BreastDCEDL/
├── README.md, LICENSE, .zenodo.json
├── BreastDCEDL_metadata.csv              full version, 2070 rows
├── BreastDCEDL_metadata_min_crop.csv     MinCrop version, 2070 rows
├── df_pcr_pred_test_article.csv          ★ the article's per-patient predictions (175)
│
├── BrestDCEDL_demo.ipynb                 visualisation only
├── BrestDCEDL_demo_on_local_data_min_crop.ipynb   visualisation only
├── BrestDCEDL_zenodo_demo.ipynb          visualisation only
├── BrestDCEDL_vit_predict.ipynb          ★ the ONLY deep-learning code
├── plot_pCR_article_results.ipynb        figures + a sklearn model on clinical data
│
├── DUKE/
│   ├── crop_spy2_spy1.ipynb              ★★ MinCrop generation for I-SPY1/I-SPY2
│   ├── duke_crop.ipynb                   ★★ MinCrop generation for DUKE
│   ├── duke_convert_dicom_to_nifti.ipynb DICOM → NIfTI
│   ├── duke_tcia_metadata.ipynb          metadata harmonisation
│   ├── research_duke.ipynb               exploration
│   ├── duke_modeling_with_niftii_files.ipynb   sklearn on tabular features
│   ├── TCIA_metadata/*.xlsx              6 clinical spreadsheets
│   └── data_samples/                     2 patients, DICOM + NIfTI
│
├── ISPY1/, ISPY2/                        metadata notebooks + TCIA spreadsheets + samples
├── utils/data_utils.py                   16 functions — reading and plotting only
├── transformer_models/BreastDCEDL_vit_pcr_predictions.csv   ⚠ does NOT reproduce the article
└── images/                               22 article figures
```

`★★` marks the two files that matter most and that **the README does not reference**.

### Files not referenced in the README but required

| file | why it is required |
|---|---|
| `DUKE/crop_spy2_spy1.ipynb` | the only definition of how MinCrop is generated for I-SPY |
| `DUKE/duke_crop.ipynb` | the same, for DUKE |
| `DUKE/duke_convert_dicom_to_nifti.ipynb` | DICOM → NIfTI conversion |
| `df_pcr_pred_test_article.csv` | the only artefact that reproduces the published results |

---

## 2. Every preprocessing step, as implemented

### 2.1 MinCrop generation — `crop_around_voi_cords()`

Signature: `crop_around_voi_cords(arr_shape, voi, slice_padding=2, output_size=256)`
with `voi = [(mask_start, mask_end), (sraw, eraw), (scol, ecol)]` read from the metadata.

| step | operation |
|---|---|
| **z range** | `ss = max(0, mask_start − 2)`, `es = min(D−1, mask_end + 2)` |
| **in-plane** | square window of **exactly 256 pixels**, centred on the VOI, shifted to keep the whole VOI inside, clamped to the image; expanded only if the VOI itself exceeds 256 |
| **resize** | **none** |
| **normalisation** | **none** — original DICOM intensities preserved as float64 |

**Verified against the real data:**

| check | result |
|---|---|
| metadata box == true mask box when `n_xy == 256` | **767/767 (100%)** |
| tumour occupies `z ∈ [2, nz−3]` | 749/767 (97.7%) |
| DUKE: `volume_depth == z_span + 4` | 12/12 |
| patients with `n_xy > 256` were cropped, not resized | **228/228** |

The last row matters: `crop_spy2_spy1.ipynb` contains a `cv2.resize(img, (256,256), INTER_AREA)`
branch for `n_xy != 256`, but the **published data used the crop path for all of them**. I
tested every affected patient in both I-SPY cohorts. Anyone who reproduces via the resize
branch will get different pixels.

### 2.2 Model input preparation — `BrestDCEDL_vit_predict.ipynb`

```python
def minmax(imc):
    im = imc.copy()
    if im.max() == 0: return im
    return (im - im.min()) / (im.max() - im.min())

def to_rgb(a, b, c):
    return minmax(np.stack([a, b, c], axis=2))     # per SLICE, joint across channels

def get_jpg_im(pid, acq, k):
    im = to_rgb(acq[0][k], acq[1][k], acq[2][k])
    return Image.fromarray(im, mode="RGB")          # ⚠ see §6

_val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
```

| step | value |
|---|---|
| phases | `acq[0], acq[1], acq[2]` — the first three files in numeric order |
| RGB | R = pre, G = early post, B = late post |
| **normalisation scope** | **per slice**, jointly over the three channels |
| slice selection | `idx=(f+l)//2; range(max(idx-2,f), min(idx+2,l))` → **4 slices**, asymmetric |
| crop | **224 pixels** around the ROI centre (`safe_crop_around_roi`) |
| resize | `Resize(224)` — no-op on an already-224 crop |
| final normalisation | ImageNet mean/std |
| aggregation | **not in the notebook** — it returns a list of per-slice probabilities |

Note the phase selection depends on what is in the folder. The `data_samples` contain
exactly the three chosen acquisitions (for `ACRIN-6698-102212`: indices 0, 2, 6, matching
the paper's "SER timepoints 0, 2, min(last, 6)"), so `acq[0..2]` picks them correctly.
Point the same code at a folder holding all acquisitions and it silently uses 0, 1, 2.

### 2.3 Train/validation/test split

The `test` column of the metadata: `0 = train, 1 = test, 2 = validation`. The article's
test set is `test == 1` and contains **175 patients** (I-SPY2 99, DUKE 41, I-SPY1 35).
The paper says 176/177 in places; the file says 175.

### 2.4 Label generation

Read directly from the harmonised metadata: `pCR`, `HR`, `HER2`, `HR_HER2_STATUS`,
`TripleNeg`, `HER2pos`, `HRposHER2neg`. No derivation in code.

### 2.5 Data augmentation

**Not present anywhere.** Figure 2C of the paper shows "examples of image augmentation
techniques used during model training", but no augmentation code exists in the repository
and no parameters are published.

---

## 3. What the repository already contains

| asset | present? | notes |
|---|---|---|
| Preprocessed dataset | ✗ in repo | MinCrop is on Zenodo (25 GB), Full is 206 GB |
| MinCrop images | ✗ | 2 sample patients per cohort only, under `*/data_samples/` |
| Cached PNGs | ✗ | none — the pipeline works from NIfTI |
| Metadata files | ✓ | both CSVs, 2070 rows, complete |
| Ready-to-train dataset | ✗ | no image/label folder structure exists |
| Trained checkpoints | ✓ **on Zenodo** | `BreastDCEDL_models.tar.gz` → `breastdcedl_pcr_vit_model_weights.pth`, 343 MB, 85,800,194 params, head `(2, 768)` |
| Configuration files | ✗ | none of any kind |

**Answer to "should I use their data instead of generating it?" — yes, use the Zenodo
MinCrop.** Regenerating it requires the 206 GB Full version plus the two crop notebooks,
which contain Windows absolute paths (`G:\My Drive\breast_mri`) and were run
interactively. The Zenodo MinCrop *is* the authors' output, and I verified its geometry
matches the code. Regenerating it can only introduce differences.

---

## 4. The models the authors used

### 4.1 pCR — Vision Transformer

| component | value | source |
|---|---|---|
| architecture | `ViTForImageClassification` | notebook |
| pretrained weights | `facebook/vit-mae-base` | notebook |
| parameters | 85,800,194 | measured from the checkpoint |
| input size | 224 × 224 × 3 | notebook |
| classes | 2 (head `(2, 768)`) | measured |
| loss | cross-entropy | HER2 paper, named only |
| optimizer | AdamW | **imported but never used** in the notebook |
| learning rate | **NOT PUBLISHED** | — |
| weight decay | **NOT PUBLISHED** | — |
| scheduler | **NOT PUBLISHED** | — |
| batch size | **NOT PUBLISHED** | — |
| epochs | **NOT PUBLISHED** | — |
| early stopping | **NOT PUBLISHED** | — |
| augmentation | **NOT PUBLISHED** | figure only |
| validation strategy | **NOT PUBLISHED** | `test == 2` exists but is never used in code |
| test strategy | `test == 1`, 175 patients | verified |
| seed | **NOT PUBLISHED** | — |

### 4.2 HER2 — THDA-ResNet ([arXiv:2510.13897](https://arxiv.org/html/2510.13897))

Reported AUC 0.744, with per-channel q0.98 clipping and median slice aggregation. **No
code for it exists in this repository at all.**

### 4.3 Clinical model — sklearn

`GradientBoostingClassifier(n_estimators=1000, max_depth=None, min_samples_split=2,
random_state=42)` on tabular features, `StandardScaler` beforehand. This is the only
model in the repository whose training code is complete and runnable. Note
`class_weight='balanced'` is present but **commented out**.

---

## 5. Verification of the published results

Recomputed from `df_pcr_pred_test_article.csv` (column `pred_pcr_vnew`):

| cohort | n | recomputed AUC | published |
|---|---:|---:|---:|
| overall | 175 | **0.7201** | 0.72 |
| I-SPY2 | 99 | **0.7801** | 0.78 |
| I-SPY1 | 35 | **0.6793** | 0.68 |
| DUKE | 41 | **0.5398** | 0.54 |

At threshold 0.5: accuracy **0.754**, sensitivity **0.269**, specificity **0.959** —
against the published 75%, 0.27, 0.95. Every number matches.

**The 0.94 headline is a tabular model, not the ViT.** On the HR+/HER2− I-SPY2 subgroup
(n=40, 12.5% pCR rate):

| predictor | AUC |
|---|---:|
| ViT, imaging only (`pred_pcr_vnew`) | 0.8886 |
| **clinical model (`rf_pred_proba`)** | **0.9371** |

---

## 6. Reproducibility problems found

### 6.1 No training code exists — confirmed exhaustively

Searched all 16 notebooks and both `.py` files for `loss.backward`, `optimizer.step`,
`.fit(`, `Trainer(`, `for epoch`, `model.train()`, `scheduler`, `state_dict`. Then
recovered every deleted file from 196 commits of history, including
`BreastDCEDL_modeling_with_nifti_files.ipynb` and
`ISPY2/modeling_ispy2_with_nifti_files.ipynb`.

**Result: the only `.fit()` calls anywhere are sklearn `RandomForestClassifier` and
`GradientBoostingClassifier` on tabular data.** The recovered `modeling_ispy2` notebook
contains `torch: 0` occurrences.

### 6.2 The published inference notebook does not work as written

`get_jpg_im` calls `Image.fromarray(im, mode="RGB")` where `im` is **float64** in [0,1]
— `read_nifti` returns `nib.load(...).get_fdata()`, which is always float64, and there is
no cast anywhere.

Passing `mode=` explicitly makes PIL reinterpret the raw buffer as bytes instead of
converting. A (256, 256, 3) float64 array is 1,572,864 bytes; PIL reads the first 196,608
of them as RGB.

Measured on the authors' own sample patient `ACRIN-6698-102212`:

> **correlation between what the model receives and the actual MRI: 0.0114**

See `breastdcedl_project/figures/13_bug_fromarray.png`.

This does not mean the published results are wrong — `df_pcr_pred_test_article.csv`
reproduces them exactly. It means **the notebook as published is not the code that
produced them.**

### 6.3 The checkpoint predictions in the repository do not reproduce the article

`transformer_models/BreastDCEDL_vit_pcr_predictions.csv` holds predictions for all 1452
patients with `mean` and `min` aggregations:

| column | correlation with `pred_pcr_vnew` | AUC on the 175 test patients |
|---|---:|---:|
| `pred_pcr_vnew` (article) | — | **0.7201** |
| `DL_pred_pcr_mean` | 0.05 | 0.5158 |
| `DL_pred_pcr_min` | 0.14 | 0.5852 |

The repository's values span 0.478–0.524 with standard deviation 0.010 — a model
outputting ~0.5 for everyone. Whatever produced that file, it is not the article's model.

### 6.4 Other issues

* Windows absolute paths hard-coded throughout the crop notebooks (`G:\My Drive\...`,
  `C:\Users\naomi\Downloads\...`).
* No `requirements.txt`, no pinned versions.
* `crop_around_voi_cords` is redefined **four times** in `crop_spy2_spy1.ipynb` with
  different behaviour at the edges. Which definition was in scope when the data was
  generated is not determinable from the notebook.
* Patient counts disagree between the paper (176/177) and the prediction file (175).

---

## 7. Pipeline flowcharts

### Data flow (reproducible)

```
DICOM (TCIA)
    │  duke_convert_dicom_to_nifti.ipynb        [DUKE only]
    ▼
NIfTI float64, full resolution              ← "Full version", 206 GB
    │  crop_around_voi_cords(voi, slice_padding=2, output_size=256)
    │    z:  [mask_start−2, mask_end+2]
    │    xy: 256 px square centred on the VOI
    │    no resize, no normalisation
    ▼
MinCrop NIfTI, 3 phases, 256×256            ← Zenodo, 25 GB   ★ START HERE
```

### Inference flow (published, but defective — see §6.2)

```
MinCrop volumes (3 phases)
    ▼  select 4 slices: idx=(f+l)//2, range(max(idx-2,f), min(idx+2,l))
    ▼  to_rgb  →  min-max per SLICE over the 3-channel stack
    ▼  Image.fromarray(..., mode="RGB")        ⚠ defect
    ▼  safe_crop_around_roi(cx, cy, crop_size=224)
    ▼  Resize(224) → ToTensor → Normalize(ImageNet)
    ▼  ViTForImageClassification, softmax[:, 1]
    ▼  list of 4 per-slice probabilities   (aggregation not published for pCR)
```

### Training flow

```
                        ⛔ DOES NOT EXIST
```

---

## 8. What is missing, inferable, or impossible

| category | items |
|---|---|
| **Missing but inferable** | slice aggregation for pCR (the HER2 paper says median; the repo's unused CSV offers mean/min) · validation usage (`test == 2` exists) |
| **Missing and NOT inferable** | learning rate · batch size · epochs · scheduler · weight decay · augmentation parameters · early-stopping criterion · seed · checkpoint-selection rule · class-imbalance handling |
| **Cannot be reproduced** | the training run. Not partially — at all. |
| **Can be reproduced exactly** | the MinCrop geometry (verified) · the test split · the published metrics (verified from their prediction file) |

---

## 9. Recommendation

1. **Use the Zenodo MinCrop.** Do not regenerate it. It is the authors' own output and its
   geometry is verified.
2. **Verify your image pipeline against their checkpoint.** Load
   `breastdcedl_pcr_vit_model_weights.pth`, run your preprocessing, and correlate against
   `pred_pcr_vnew` on the 175 test patients. Target: AUC 0.7201. This is the only available
   objective check that your pixels are right, and it costs inference only.
3. **State plainly in the dissertation** that the training procedure is not published and
   that your hyperparameters are your own. This is a documented fact about the source, not
   a limitation of your work.
4. **Do not chase 0.94.** It is a 40-patient subgroup scored by a tabular model on clinical
   variables. The imaging result is 0.72 overall, 0.78 on I-SPY2, 0.54 on DUKE.
