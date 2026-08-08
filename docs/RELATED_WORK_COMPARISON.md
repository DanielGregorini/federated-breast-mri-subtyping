# Comparator papers for Chapter 4

Papers to read and cite when discussing the results of tests 01–09. Ranked by how
directly their experimental setting can be compared with ours.

**Our setting, for reference.** 3-class molecular subtype (HR+/HER2−, TripleNeg, HER2+)
from DCE-MRI; ResNet-18; centralised baseline + FedAvg + FedProx (mu = 0.01); 2, 3 and 4
hospitals; stratified balanced and 5:2:1:1 quantity-skewed partitions; patient-level
macro-AUC 0.5594–0.6531 federated against 0.6068 centralised; trivial baseline 0.5112;
single seed; NVIDIA FLARE, real PKI deployment.

## Verification status

Every number below was taken from the source indicated. Three levels are marked:

| mark | meaning |
|---|---|
| **[FULL]** | read from the full text or the paper's own results table |
| **[ABS]** | read from the abstract or record page only; full tables not retrieved |
| **[BLOCKED]** | publisher returned HTTP 403; details come from the indexed abstract only, tables **not** verified |

Where a paper does not report a metric, this is stated as **NOT REPORTED** rather than
filled in. No value in this document was estimated, converted or inferred.

---

# TIER 1 — direct comparators

## 1. Pan et al. (2026) — Federated breast cancer detection, synthetic ultrasound augmentation **[FULL]**

**The single most structurally comparable protocol found.**

1. **Citation.** Pan, H., Hong, Z., Durak, G., Xu, Z., Bagci, U. *Federated Breast Cancer
   Detection Enhanced by Synthetic Ultrasound Image Augmentation.* arXiv:2506.23334v3 (2026).
2. **Link.** https://arxiv.org/abs/2506.23334
3. **Dataset.** BUS-BRA (1,268 benign / 607 malignant = 1,875 images), BUSI (437/210 = 647),
   UDIAT (109/54 = 163). Total 2,685 images. Patient counts NOT REPORTED.
4. **Modality.** Breast ultrasound.
5. **Task.** Binary, benign vs malignant.
6. **Architecture.** ImageNet-pretrained DenseNet-121.
7. **FL algorithms.** FedAvg and FedProx (mu = 0.03).
8. **Clients.** 3 (one per dataset — a natural, not synthetic, split).
9. **Heterogeneity.** Non-IID by construction: three different acquisition sources, and
   heavily quantity-skewed (1,875 / 647 / 163 images).
10. **Metrics.** Accuracy and AUC, per client and averaged. Per-class metrics NOT REPORTED.
    F1 NOT REPORTED.
11–13. **Results (average over the three clients):**

| Method | Avg accuracy | Avg AUC |
|---|---:|---:|
| **Centralised** | 0.8967 | **0.9391** |
| FedAvg | 0.8730 | 0.9206 |
| FedAvg + DCGAN | 0.8928 | 0.9237 |
| FedAvg + DDPM | 0.8997 | 0.9362 |
| **FedProx** | 0.9021 | **0.9429** |
| FedProx + DCGAN | 0.8963 | 0.9538 |
| FedProx + DDPM | 0.9117 | 0.9574 |

Per-client AUC, centralised: BUS-BRA 0.9199, BUSI 0.9510, UDIAT 0.9463.
Per-client AUC, FedAvg: 0.9191, 0.9048, 0.9380. Training: 100 rounds x 1 local epoch,
batch 32. **Number of seeds NOT REPORTED.**

14. **Comparable?** Yes, structurally — though the task is binary and the modality is
    ultrasound. The protocol shape is nearly identical to ours (rounds x 1 local epoch,
    FedAvg vs FedProx vs centralised, natural non-IID client split).
15. **What to discuss.** Their plain **FedProx (0.9429) exceeds their centralised baseline
    (0.9391)**, and FedAvg (0.9206) falls below it. That is the same pattern we observe —
    federated runs straddling the centralised baseline rather than sitting below it — and
    it is the cleanest published precedent for our test06/test09 results. Also note they
    report **no seeds**, exactly the limitation we declare; use this to argue that
    single-run FL comparisons are common in the literature and that our explicit noise
    floor is a methodological improvement, not a deficiency.

## 2. Ali et al. (2026) — Federated vs centralised mammography across film–digital domain shift **[FULL]**

**The methodological model for how our Chapter 4 should be argued.**

1. **Citation.** Ali, Y., Müller, J., Weinmann, A., Gregori, J. *Performance of federated
   versus centralized learning for mammography classification across film–digital domain
   shift.* Frontiers in Digital Health, 8:1715858 (2026).
2. **DOI.** https://doi.org/10.3389/fdgth.2026.1715858
3. **Dataset.** CBIS-DDSM (scanned film): 2,864 train/val images (1,683 benign, 1,181
   malignant), 1,403 test. VinDr-Mammo (digital): 16,370 train/val (14,463 benign, 1,907
   malignant), 8,184 test. Totals 19,234 train/val and 9,587 test. Patient counts
   NOT REPORTED.
4. **Modality.** Mammography (film vs full-field digital).
5. **Task.** Binary, benign vs malignant.
6. **Architecture.** ResNet-50 and Swin V2-T, both ImageNet-pretrained.
7. **FL algorithms.** FedAvg, FedProx, SCAFFOLD, FedBN.
8. **Clients.** 2, cross-silo.
9. **Heterogeneity.** Non-IID film–digital domain shift, with **both quantity skew and
   label skew** — and, critically, **homogeneous single-domain controls** run alongside.
10. **Metrics.** AUROC, average precision, accuracy@0.5, precision@0.5, recall@0.5,
    F1@0.5, precision at recall 0.90, ROC and PR curves.
11–13. **Results:**

| Setting | Centralised | Federated (FedAvg) |
|---|---:|---:|
| ResNet-50, CBIS test | AUC 0.73 | AUC 0.62 |
| ResNet-50, VinDr test | AUC 0.96 | AUC 0.95 |
| ResNet-50, combined test | NOT REPORTED separately | AUC 0.93 |
| ResNet-50, VinDr-only homogeneous control | AUC 0.96 | AUC 0.97 |
| ResNet-50, CBIS-only homogeneous control | AUC 0.73 | AUC 0.75 |

Swin V2-T FedAvg: CBIS AUC ~0.53–0.62, VinDr ~0.91–0.95 (ranges as stated in the
abstract). FedProx and SCAFFOLD gave "marginal improvements or no consistent benefit"
over FedAvg; FedBN "did not consistently improve minority-domain performance".
**Three fixed random seeds, mean ± SD, plus 95% bootstrap percentile intervals
(1,000 replicates).**

14. **Comparable?** Yes for methodology and for the ResNet + FedAvg/FedProx axis; the task
    is binary and the modality is mammography.
15. **What to discuss.** Three things. (a) Their **homogeneous controls** are exactly the
    design our `--by-cohort` partition would give us, and their finding that federation
    costs almost nothing when domains match but costs 0.11 AUC on the minority domain when
    they do not, is the strongest available argument that our stratified partitions cannot
    answer RQ2. (b) **FedProx gave no consistent benefit over FedAvg** — the same
    sign-flipping we see across our four paired comparisons. (c) Their use of three seeds
    plus bootstrap intervals is the standard our single-seed campaign should be measured
    against; cite this when justifying the three-seed repeat as future work.

## 3. Jiménez-Sánchez et al. (2023) — Memory-aware curriculum federated learning **[FULL]**

**Closest match on scale and on the centralised-versus-federated gap.**

1. **Citation.** Jiménez-Sánchez, A., Tardy, M., González Ballester, M.A., Mateus, D.,
   Piella, G. *Memory-aware curriculum federated learning for breast cancer classification.*
   Computer Methods and Programs in Biomedicine, 229:107318 (2023).
2. **DOI.** https://doi.org/10.1016/j.cmpb.2022.107318 · preprint arXiv:2107.02504
3. **Dataset.** Three vendor cohorts: **Hologic 1,460 subjects** (730 benign/normal, 730
   malignant), **Siemens 410** (287/123), **GE 852** (421/431). Total 2,722 subjects.
4. **Modality.** Mammography (2,048 px, minimal downsampling).
5. **Task.** Binary, normal/benign vs biopsy-confirmed malignant.
6. **Architecture.** ResNet-22 (their Table IV), plus a domain discriminator.
7. **FL algorithms.** A FedAvg-style federated baseline ("Fed") plus their variants
   Fed-CL, Fed-Align, Fed-Align-CL. FedProx NOT EVALUATED.
8. **Clients.** 3, one per vendor.
9. **Heterogeneity.** Genuine non-IID: different vendors, and strong label skew
   (Siemens is 30% malignant, GE is 51%).
10. **Metrics.** AUC and PR-AUC, per site and averaged. **Median of 5 runs**, with
    p-values between methods (their Table V). Accuracy, balanced accuracy and per-class
    metrics NOT REPORTED.
11–13. **Results (their Table III, median of 5 runs):**

| Method | Hologic AUC | Siemens AUC | GE AUC | **Avg AUC** | Avg PR-AUC |
|---|---:|---:|---:|---:|---:|
| Single (local, same-site test) | 0.83 | 0.83 | 0.85 | — | — |
| Fed (standard federated) | 0.78 | 0.65 | 0.83 | **0.75** | 0.77 |
| Fed-CL | 0.80 | 0.63 | 0.81 | 0.75 | 0.78 |
| Fed-Align | 0.79 | 0.69 | 0.85 | 0.78 | 0.80 |
| Fed-Align-CL (proposed) | 0.84 | 0.70 | 0.83 | 0.79 | 0.82 |
| **Mix (pooled, non-privacy-preserving oracle)** | 0.83 | 0.86 | 0.82 | **0.84** | 0.85 |

Cross-site transfer (train one, test another) ranges 0.59–0.73 AUC.

14. **Comparable?** Yes — 3 sites, ~2,700 patients (we have 2,063), a ResNet, real
    heterogeneity, and an explicit pooled baseline. Task is binary, modality mammography.
15. **What to discuss.** The **centralised-to-federated gap of 0.09 macro-AUC
    (Mix 0.84 vs Fed 0.75)** is the single most quotable comparator number in this list —
    it is the same order as our full spread (0.0937) and larger than our noise floor. Note
    also that their per-site federated performance collapses on the *smallest and most
    label-skewed* site (Siemens, 410 subjects, 0.65) while the large site holds up
    (Hologic 0.78) — this is the behaviour our tests 08/09 were designed to detect and did
    not, and it supports the argument that our skew was too weak. Finally, their 5 runs
    plus significance testing is a directly citable precedent for our noise-floor argument.

---

# TIER 2 — essential context, less directly comparable

## 4. Ankolekar et al. (2025) — Systematic review, FL in breast/lung/prostate cancer **[ABS]**

1. **Citation.** Ankolekar, A., Boie, S., Abdollahyan, M., et al., on behalf of the OPTIMA
   Consortium. *Advancing breast, lung and prostate cancer research with federated
   learning. A systematic review.* npj Digital Medicine, 8:314 (2025).
2. **DOI.** https://doi.org/10.1038/s41746-025-01591-5 · PMID 40425787 ·
   preprint https://doi.org/10.1101/2024.08.08.24311681
3–13. A review, not a primary study. **Headline finding: federated learning outperformed
   centralised ML in 15 of 25 studies.** Also reports challenges in reproducibility and
   standardisation across the reviewed literature. Per-study tables not retrieved (the
   Nature and medRxiv full texts were not accessible); read the medRxiv PDF directly.
14. **Comparable?** Not a comparator — a framing citation.
15. **What to discuss.** **This is the citation that defuses the most awkward result in
    our campaign.** Four of our federated runs scored above the centralised baseline, which
    reads as implausible in isolation; this review establishes that FL exceeding centralised
    training is reported in the majority of oncology FL studies. Use it to argue that our
    result is consistent with the literature *and* that the literature is itself
    insufficiently controlled — which is precisely why we report a noise floor and decline
    to claim the ordering. Their reproducibility criticism supports the same point.

## 5. Ogier du Terrail et al. (2022) — FLamby cross-silo benchmark **[FULL, with a caveat]**

1. **Citation.** Ogier du Terrail, J., Ayed, S.-S., Cyffers, E., et al. *FLamby: Datasets
   and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings.*
   NeurIPS 2022 Datasets and Benchmarks Track. arXiv:2210.04620.
2. **Link.** https://arxiv.org/abs/2210.04620
3. **Breast-relevant datasets.** **Fed-Camelyon16**: 399 digitised breast biopsy slides
   from 2 hospitals (Radboud UMC, UMC Utrecht), split by the original site metadata into
   **2 clients**. **Fed-TCGA-BRCA**: 1,066 breast cancer patients, tabular, split by tissue
   source site into **6 clients** (51–311 patients each).
4. **Modality.** Camelyon16 is histopathology whole-slide imaging; TCGA-BRCA is tabular.
5. **Task.** Camelyon16: binary slide classification (tumour present/absent).
   TCGA-BRCA: survival prediction.
6. **Architecture.** Camelyon16: weakly-supervised DeepMIL over tiles, features from an
   ImageNet-pretrained ResNet-50. TCGA-BRCA: linear (Cox) model.
7. **FL algorithms.** FedAvg, FedProx, Scaffold, Cyclic, FedAdagrad, FedYogi, FedAdam —
   all at full client participation.
8. **Clients.** 2 (Camelyon16), 6 (TCGA-BRCA).
9. **Heterogeneity.** Natural splits, not synthetic partitions.
10. **Metrics.** AUC for Fed-Camelyon16; C-index for Fed-TCGA-BRCA.
11–13. **CAVEAT: the per-strategy results are presented graphically in their Figure 2 and
    no numeric table appears in the paper, so exact values are NOT QUOTABLE from the PDF.**
    The stated qualitative findings are exact quotations:
    - "No local training or FL strategy is able to reach a performance on par with the
      pooled training, except for Fed-TCGA-BRCA and Fed-Heart-Disease."
    - "For Fed-Camelyon16, Fed-LIDC-IDRI and Fed-IXI, the current results do not indicate
      any benefit in collaborative training."
    - "FedAvg does not reach top performance among FL strategies, except for
      Fed-Camelyon16 and Fed-IXI, it remains a competitive baseline strategy."
    - Their FedProx mu for Fed-Camelyon16 was 0.316228 (their Table 11), searched over
      {1, 0.1, 0.01} — note this is far larger than our fixed mu = 0.01.
14. **Comparable?** Not numerically (different tasks and metrics), but it is the reference
    benchmark for the FedAvg/FedProx/pooled comparison design.
15. **What to discuss.** Two points. (a) FLamby's conclusion that federated strategies
    **fall short of pooled training on every imaging dataset** is the counterweight to the
    npj review — cite both and let the tension stand, because our own campaign cannot
    resolve it. (b) FLamby **tuned mu per dataset** and landed on 0.316 for the breast
    histopathology task; we fixed mu = 0.01 without tuning, which is a limitation worth
    stating explicitly when discussing why our FedProx-vs-FedAvg contrasts flip sign.

## 6. Schmidt et al. (2024) — ACR-NCI-NVIDIA federated learning challenge **[ABS]**

1. **Citation.** Schmidt, K., Bearce, B., Chang, K., Coombs, L., Farahani, K., Elbatel, M.,
   Mouheb, K., Marti, R., Zhang, R., Zhang, Y., Wang, Y., Hu, Y., Ying, H., Xu, Y.,
   Testagrose, C., Demirer, M., Gupta, V., Akünal, Ü., Bujotzek, M., Maier-Hein, K.H.,
   Qin, Y., Li, X., Kalpathy-Cramer, J., Roth, H.R. *Fair evaluation of federated learning
   algorithms for automated breast density classification: The results of the 2022
   ACR-NCI-NVIDIA federated learning challenge.* Medical Image Analysis, 95:103206 (2024).
2. **DOI.** https://doi.org/10.1016/j.media.2024.103206 · PMID 38776844
3. **Dataset.** Three simulated medical facilities with distinct datasets. Image and patient
   counts NOT REPORTED in the accessible record.
4. **Modality.** Mammography.
5. **Task.** BI-RADS breast density classification (ordinal, 4 categories).
6. **Architecture.** Varies by challenge submission; NOT REPORTED in the accessible record.
7. **FL algorithms.** Multiple, submitted by challenge participants; **the challenge ran on
   NVIDIA FLARE**. Specific algorithms NOT REPORTED in the accessible record.
8. **Clients.** 3.
9. **Heterogeneity.** Different systems and class distributions per facility.
10. **Metrics.** Linear kappa. AUC, accuracy, F1 NOT REPORTED in the accessible record.
11–13. Winning submission: **linear kappa 0.653** on the challenge test data and
    **0.413** on an external test set. The paper states results scored "comparably to a
    model trained on the same data in a central location".
14. **Comparable?** Not on metrics (kappa, not macro-AUC), but it is the closest published
    work on **infrastructure**: same framework, same disease area, real challenge conditions.
15. **What to discuss.** Cite it to justify NVIDIA FLARE as the deployment choice and to
    support the claim that a properly run federation lands "comparable to central". The
    drop from 0.653 internal to 0.413 external is also a useful caution about our own
    absolute numbers, which come from a single held-out split of the same cohorts.

## 7. Ogier du Terrail et al. (2023) — Federated learning for TNBC response prediction **[ABS]**

1. **Citation.** Ogier du Terrail, J., Leopold, A., Joly, C., Béguier, C., Andreux, M.,
   Maussion, C., Schmauch, B., Tramel, E.W., Bendjebbar, E., Zaslavskiy, M., Wainrib, G.,
   Milder, M., Gervasoni, J., Guerin, J., Durand, T., Livartowski, A., Moutet, K.,
   Gautier, C., Djafar, I., Moisson, A.-L., Marini, C., Galtier, M., Balazard, F.,
   Dubois, R., Moreira, J., Simon, A., Drubay, D., Lacroix-Triki, M., Franchet, C.,
   Bataillon, G., Heudel, P.-E. *Federated learning for predicting histological response to
   neoadjuvant chemotherapy in triple-negative breast cancer.* Nature Medicine, 29(1):135–146 (2023).
2. **DOI.** https://doi.org/10.1038/s41591-022-02155-w · PMID 36658418
3. **Dataset.** 650 TNBC patients across multiple hospitals (per the search record; the
   full text was not accessible — **verify the per-centre breakdown before citing**).
4. **Modality.** Whole-slide histopathology images, plus clinical data.
5. **Task.** Binary — predict pathological complete response to neoadjuvant chemotherapy.
6. **Architecture.** Weakly-supervised WSI model; details NOT RETRIEVED.
7. **FL algorithm.** Federated training across real hospital firewalls; specific strategy
   NOT RETRIEVED.
8. **Clients.** Multiple real hospitals; exact count NOT RETRIEVED.
9. **Heterogeneity.** Real-world multicentric.
10. **Metrics.** ROC AUC.
11–13. Reported ROC AUC **0.66 on average for the best federated method** (per the search
    record). Pooled/centralised comparison NOT RETRIEVED. Per-centre local baselines are
    described qualitatively: "local ML models relying on whole-slide images can predict
    response to NACT but ... collaborative training of ML models further improves
    performance".
14. **Comparable?** Only thematically — same disease, real federation, but histopathology
    and a different endpoint.
15. **What to discuss.** Two uses. (a) It is the flagship real-world breast-cancer FL study
    and belongs in the introduction to Chapter 4. (b) Its **AUC of 0.66 on a real
    multicentric breast task** is a useful reality check on our 0.6068–0.6531: state-of-the-art
    federated breast oncology on real data lands in the same band, which supports our
    argument that ~0.61 is a correct answer for this class of problem rather than a failure.

---

# TIER 3 — supporting, or unverified

## 8. Tzortzis et al. (2025) — Real-world FL on mammography across three hospitals **[FULL]**

1. **Citation.** Tzortzis, I.N., Gutierrez-Torre, A., Sykiotis, S., et al. *Towards
   generalizable Federated Learning in medical imaging: A real-world case study on
   mammography data.* Computational and Structural Biotechnology Journal, 28:106–117 (2025).
2. **DOI.** https://doi.org/10.1016/j.csbj.2025.03.031
3. **Dataset.** 994 mammograms from 294 patients. Hospital 1 (Hellenic Cancer Society,
   Greece) 490 images / 132 patients; Hospital 2 (University of Novi Sad, Serbia) 455 / 145;
   Hospital 3 (Aristotle University of Thessaloniki) 49 / 17.
4. **Modality.** Digital mammography.
5. **Task.** BI-RADS 1–5 classification (BI-RADS 0 and 6 removed).
6. **Architecture.** A CNN; layer details not fully specified.
7. **FL algorithm.** FedAvg only. FedProx NOT EVALUATED.
8. **Clients.** 3 real hospitals in three institutions.
9. **Heterogeneity.** Real non-IID, plus extreme quantity skew (490 / 455 / 49 images).
10. **Metrics.** F1 per hospital. AUC, accuracy, balanced accuracy NOT REPORTED.
11–13. Centralised: Hospital 1 F1 = 0.689, Hospital 2 F1 = 0.743, Hospital 3 not trained
    centrally (too little data; used for testing only). Federated (50 rounds, 2 local
    epochs): Hospital 1 F1 = 0.550, Hospital 2 F1 = 0.743, Hospital 3 F1 = 0.559. Other
    round/epoch configurations (100x1, 20x5) converged similarly (F1 ≈ 0.549–0.550 for
    Hospital 1). Seeds NOT REPORTED.
14. **Comparable?** Partly — real multi-hospital deployment, per-site evaluation, explicit
    centralised comparison, and a round/epoch budget sweep like ours.
15. **What to discuss.** Their **per-hospital federated F1 varies from 0.550 to 0.743 under
    a single global model**, and their smallest site (17 patients) behaves erratically —
    directly parallel to our per-hospital tables, where the 34-patient sites in tests 08/09
    span macro-AUC 0.5237 to 0.7519. Cite this to support the claim that per-site spread in
    our Section 4 tables is dominated by split size, not by the partition or the algorithm.
    Their finding that 50x2, 100x1 and 20x5 all converge similarly also supports our fixed
    30x1 budget.

## 9. Roth et al. (2020) — Federated learning for breast density classification **[ABS]**

1. **Citation.** Roth, H.R., Chang, K., Singh, P., Neumark, N., Li, W., Gupta, V.,
   Gupta, S., Qu, L., Ihsani, A., Bizzo, B.C., et al. *Federated Learning for Breast Density
   Classification: A Real-World Implementation.* In: Domain Adaptation and Representation
   Transfer, and Distributed and Collaborative Learning (DART/DCL 2020), MICCAI Workshops.
   LNCS vol. 12444, Springer. arXiv:2009.01871.
2. **DOI.** https://doi.org/10.48550/arXiv.2009.01871
3. **Dataset.** Seven clinical institutions worldwide. Per-institution image and patient
   counts NOT REPORTED in the abstract.
4. **Modality.** Mammography.
5. **Task.** BI-RADS breast density classification.
6. **Architecture.** NOT REPORTED in the abstract.
7. **FL algorithm.** NOT REPORTED in the abstract (the work uses NVIDIA Clara).
8. **Clients.** 7.
9. **Heterogeneity.** "Substantial differences among the datasets from all sites
   (mammography system, class distribution, and data set size)."
10. **Metrics.** Relative improvement figures only in the abstract; absolute AUC/accuracy
    NOT REPORTED there.
11–13. Federated models performed **6.3% better on average** than models trained on an
    institution's local data alone, with a **45.8% relative improvement in generalisability**
    when evaluated on other sites' test data. **Centralised/pooled baseline NOT REPORTED.**
    Read the full LNCS chapter for the per-site tables.
14. **Comparable?** Weakly on numbers, strongly as the canonical precedent for a real
    multi-institution breast FL deployment.
15. **What to discuss.** Use it to motivate the multi-hospital design and, in particular,
    the **generalisability** framing: their central claim is that FL helps
    *cross-site transfer* rather than in-site accuracy. That is a lens our campaign does not
    currently apply and is a clean piece of future work to propose — evaluating each of our
    global models on each hospital's data is already possible from our per-hospital tables.

## 10. Ismail et al. (2025) — Comparative FL analysis, multi-class breast ultrasound **[BLOCKED]**

1. **Citation.** *A Comparative Analysis of Federated Learning for Multi-Class Breast Cancer
   Classification in Ultrasound Imaging.* AI (MDPI), 6(12):316 (2025). **Author list not
   verified** — the publisher blocked retrieval; confirm before citing.
2. **DOI.** https://doi.org/10.3390/ai6120316 ·
   record: https://doaj.org/article/153c32a6aee74fbebf1ccad02f767421
3. **Dataset.** BUSI (600 patients), BUS-UCLM (38), BCMID (323).
4. **Modality.** Breast ultrasound.
5. **Task.** Multi-class breast cancer classification (class list not verified).
6. **Architecture.** Five networks compared; MobileNet, ResNet and InceptionNet reported as
   most effective for FL deployment.
7. **FL algorithms.** **FedAvg and FedProx.**
8. **Clients.** **Two- and three-client federations** — the only paper found that sweeps
   client count as we do.
9. **Heterogeneity.** "Varying levels of data heterogeneity"; exact protocol not verified.
10. **Metrics.** Accuracy and **macro-F1** (the same macro framing we use).
11–13. Reported: FL "outperformed local and centralized training"; in the two-client
    federations FL achieved **up to 8% higher accuracy and almost 6% higher macro-F1** than
    local and centralised training. **The per-algorithm results tables were NOT VERIFIED —
    MDPI returned HTTP 403. Retrieve the PDF manually before quoting any number.**
14. **Comparable?** Potentially the second-best match in this list — multi-class, macro-F1,
    FedAvg vs FedProx, 2 and 3 clients, centralised baseline. Verify first.
15. **What to discuss.** If the tables confirm the abstract, this is a second independent
    report of **federated exceeding centralised** on a multi-class breast task, and its
    two- versus three-client contrast maps onto our test02/test03 versus test04/test05.

## 11. (2026) — Physics-aware multi-modal FL, includes DUKE DCE-MRI **[BLOCKED]**

1. **Citation.** *Federated Privacy-Preserving Multi-Modal Deep Learning for Breast Cancer
   Diagnosis: A Physics-Aware Approach.* Diagnostics (MDPI), 16(11):1629 (2026).
   **Author list not verified.**
2. **DOI.** https://doi.org/10.3390/diagnostics16111629
3. **Dataset.** BUSI (n = 780), **DUKE DCE-MRI (n = 922)**, CBIS-DDSM (n = 400), with
   patient-wise stratified five-fold cross-validation.
4. **Modality.** Ultrasound, **DCE-MRI**, mammography.
5. **Task.** Breast cancer diagnosis; class definitions not verified.
6. **Architecture.** Modality-specific deep models with late-fusion inference; not verified.
7. **FL algorithms.** **FedAvg, FedProx, SCAFFOLD, FedNova and FP16-FedAvg.**
8. **Clients.** Not verified.
9. **Heterogeneity.** **IID and non-IID, the latter via a Dirichlet distribution with
   alpha = 0.5.**
10. **Metrics.** Accuracy (with SD), plus per-round training time, communication time,
    latency and cumulative bandwidth. AUC and macro metrics not verified.
11–13. Per-modality accuracy reported as 92.50 ± 1.2% (ultrasound), **90.63 ± 1.5% (MRI)**
    and 92.00 ± 1.3% (mammography), with McNemar tests (p < 0.05) against their baselines.
    **The five-algorithm federated comparison table was NOT VERIFIED — MDPI returned
    HTTP 403.**
14. **Comparable?** Worth chasing: it uses **the same DUKE DCE-MRI cohort that forms 44% of
    our dataset**, and evaluates FedAvg and FedProx under a controlled IID/non-IID contrast.
15. **What to discuss.** Their **Dirichlet alpha = 0.5 non-IID protocol is exactly the
    label-skew mechanism our stratified partitions lack**, and citing it sharpens the
    limitation we state for tests 08/09. Their communication and latency measurements are
    also the natural comparator for our RQ3 discussion, which currently has no
    communication-cost analysis at all.

## 12. (2025) — FL for pre-operative TNBC detection from multiparametric MRI **[ABS, minimal]**

1. **Citation.** *Federated Learning for Pre-operative Detection of Triple-Negative Breast
   Cancer from Multiparametric MRI: Preliminary Results.* In: Product-Focused Software
   Process Improvement (PROFES 2025 workshops), LNCS vol. 15840-ish, Springer.
   **Author list and page range not verified.**
2. **DOI.** https://doi.org/10.1007/978-3-032-12092-2_25 (Springer and ACM both blocked
   retrieval)
3. **Dataset.** Not verified.
4. **Modality.** **Multiparametric MRI including DCE-MRI** — the closest modality match found.
5. **Task.** Triple-negative vs rest — **one of our three classes, as a binary problem.**
6. **Architecture.** Radiomic features into a federated MLP (not a CNN).
7. **FL algorithm.** Federated MLP training; strategy not verified.
8. **Clients.** **5 virtual clients simulating hospitals.**
9. **Heterogeneity.** Not verified.
10–13. **No metric values could be retrieved.** The only reported finding available is
    qualitative: image standardisation markedly improves TNBC classification, highlighting
    the role of preprocessing in federated pipelines.
14. **Comparable?** In setting, the closest paper in this list — DCE-MRI, TNBC, simulated
    hospitals. In evidence, the weakest, because it is a preliminary workshop paper and no
    numbers were retrievable.
15. **What to discuss.** Cite it as the nearest prior work to our exact task and note that
    it is preliminary and radiomics-based, which positions our deep-learning, three-class,
    real-NVFLARE campaign as a genuine step beyond it. Its finding that **preprocessing
    standardisation dominates federated TNBC performance** also corroborates our own
    preprocessing programme and the source-signature probe.

---

# How these map onto our research questions

| RQ | Best comparators | What they let us say |
|---|---|---|
| **RQ1** centralised vs federated | Jiménez-Sánchez (gap 0.09), Pan et al. (FedProx above centralised), FLamby (federated below pooled on all imaging tasks), npj review (FL won in 15/25) | The literature does not agree on the sign of the gap. Our inability to resolve it with one seed is a shared limitation, not an isolated failure. |
| **RQ2** heterogeneity | Ali et al. (homogeneous vs domain-shifted controls), Diagnostics 2026 (Dirichlet alpha = 0.5), Jiménez-Sánchez (vendor split) | All three use real or simulated *label/feature* heterogeneity. None uses quantity skew alone. This is the sharpest evidence that our tests 08/09 under-test RQ2. |
| **RQ3** FedAvg vs FedProx | Ali et al. ("no consistent benefit"), Pan et al. (FedProx > FedAvg on every configuration), FLamby (mu tuned to 0.316) | Our sign-flipping across pairs matches Ali et al.; our untuned mu = 0.01 is a stated limitation FLamby's tuning makes visible. |
| **RQ4** mitigation | Diagnostics 2026 (SCAFFOLD, FedNova), Ali et al. (FedBN), Jiménez-Sánchez (domain alignment + curriculum) | Several stronger mitigations than FedProx exist and are untested here — a concrete future-work list. |

## Non-federated comparators already in the project record

For the *task* ceiling rather than the federation question, the centralised
subtype-classification references already collected remain the right ones —
Zhang et al. (PMC8547260), the four-centre Ensemble ResNet study (PMC12130697), the
106-study systematic review (PMC9028183), and BreastDCEDL itself. They are documented in
`docs/PROJECT_CONTEXT.md` §19 and are not repeated here.
