# Results — Centralised against Federated

Every number here is read from `results/thesis/final_summary/summary.csv` and the
per-experiment `test_metrics.json`. All 13 experiments are scored on the **same** global
test set: 268 patients, 2,115 images, trivial baseline **0.5112**.

---

## 1. How to read these numbers before reading them

**The noise floor is 0.067 macro-AUC.** Two runs of a byte-identical configuration
differing only in random seed scored 0.7023 and 0.6351. `seed` fixes initialisation and
the split, but not cuDNN kernel selection, AMP behaviour, or DataLoader worker ordering.
Any difference smaller than that is *no difference detected*.

**Every experiment is a single run at seed 42.** This is the binding limitation. Where
two independent comparisons agree in direction that is evidence; where one differs by
0.02 it is not.

**Accuracy is meaningless without its baseline.** Always predicting the majority class
scores 0.5112 here — higher than most of the models.

---

## 2. The results

| Test | Configuration | Algorithm | macro-AUC | Bal. acc | Accuracy |
|---|---|---|---:|---:|---:|
| 01 | Centralised | — | **0.6068** | 0.4503 | 0.5299 |
| 02 | 2 hospitals, balanced | FedAvg | 0.5816 | 0.3949 | 0.4328 |
| 03 | 2 hospitals, balanced | FedProx | 0.5917 | 0.4025 | 0.4328 |
| 04 | 3 hospitals, balanced | FedAvg | 0.5990 | 0.4198 | 0.4851 |
| 05 | 3 hospitals, balanced | FedProx | 0.5958 | 0.4127 | 0.4590 |
| 06 | 4 hospitals, balanced | FedAvg | 0.6077 | 0.4327 | 0.4925 |
| 07 | 4 hospitals, balanced | FedProx | 0.6075 | 0.4393 | 0.4739 |
| 08 | 4 hospitals, skewed | FedAvg | 0.5982 | 0.4259 | 0.4888 |
| 09 | 4 hospitals, skewed | FedProx | **0.6152** | 0.4285 | 0.4776 |
| 10 | 3 hospitals, one cohort each | FedAvg | 0.5426 | 0.3582 | 0.4291 |
| 11 | 3 hospitals, one cohort each | FedProx | 0.5678 | 0.4105 | 0.4590 |
| 12 | 3 hospitals, cohorts mixed | FedAvg | 0.5836 | 0.4183 | 0.4478 |
| 13 | 3 hospitals, cohorts mixed | FedProx | 0.5882 | 0.3885 | 0.4664 |

---

## 3. Can federated learning match centralised training?

**Yes, within the margin the method's own variability defines.**

This is an *equivalence* claim, not a failure to detect a difference, and the distinction
matters. A null-hypothesis test that finds no significant difference proves nothing. An
equivalence test fixes a margin of practical equivalence in advance and asks whether the
observed difference falls inside it — the standard two-one-sided-tests framing
(Lakens, <https://doi.org/10.1177/1948550617697177>).

The margin here is **0.067 macro-AUC**, derived from the data rather than chosen: it is
the spread between two byte-identical runs differing only in seed.

| | macro-AUC |
|---|---:|
| Centralised | 0.6068 |
| Federated, mean of 12 | **0.5899** |
| Difference | **0.0169** |

The gap is **4.0 times smaller than the margin**, and every one of the twelve federated
runs falls inside it. The largest single deviation is 0.0641, on test 10. Three federated
runs scored *above* the centralised baseline, by 0.0084, 0.0009 and 0.0007.

> The cost of federation on this task is smaller than the cost of re-running the
> centralised configuration with a different random seed.

**What limits the claim.** One seed per experiment means the margin is applied to point
estimates rather than to confidence intervals. Three seeds would convert this from "the
points fall inside" to "the interval falls inside", which is the stronger form.

**Against the literature.** This sits at the optimistic end of published cross-silo
results, and the reason is probably the data rather than the method. Sheller et al.
found federated training reaching 99% of centralised performance on multi-institutional
brain tumour segmentation (<https://doi.org/10.1038/s41598-020-69250-1>), and the
large-scale EXAM study showed federated training outperforming any single site's local
model on COVID-19 outcome prediction across 20 institutions
(Dayan et al., <https://doi.org/10.1038/s41591-021-01506-3>). Against that, the FLamby
benchmark reports that on several realistic cross-silo medical tasks federated strategies
fall measurably short of pooled training and no single strategy dominates
(Ogier du Terrail et al., <https://arxiv.org/abs/2210.04620>).

**An honest disagreement inside this project.** An earlier campaign on a binary
Triple-Negative task with a single cohort measured federation costing 0.068–0.110 —
*outside* the margin. Same infrastructure, different task and half the data. The
comparable-performance claim is therefore specific to this task at this scale, and the
most likely explanation is that it depends on having enough patients per site.

---

## 4. What does data heterogeneity cost?

The four partitions used by tests 02–09 are **stratified**: the class-share spread
between hospitals never exceeds 0.4 percentage points. They vary *how much* data each
site holds and nothing else. That is **quantity skew**, which Kairouz et al. include in
the taxonomy of non-IID federated data but which is the weakest entry in it — the sites
still sample from the same distribution (<https://doi.org/10.1561/2200000083>).

Tests 10–13 were built to fix that. Both configurations hold **identical** site sizes —
642, 101 and 784 patients — and differ only in whether a site draws from one source
cohort or from all three. The class spread is **27.5 percentage points against 0.3**.

| Algorithm | One cohort per site | Cohorts mixed | Difference |
|---|---:|---:|---:|
| FedAvg (10 vs 12) | 0.5426 | 0.5836 | **−0.041** |
| FedProx (11 vs 13) | 0.5678 | 0.5882 | **−0.020** |

**Both point the same way: real heterogeneity costs performance.** That consistency is
what the quantity-skew comparison never produced. There the two pairs disagreed in sign,
−0.0095 for FedAvg and +0.0077 for FedProx, and both are an order of magnitude smaller
than the noise floor.

Both differences remain inside the noise floor, so the magnitude is not established.
Under the null, two independent comparisons both landing in the predicted direction has a
probability of 0.25 — suggestive, not conclusive.

**The clinically important finding is not in the aggregate.** Recall on the minority
HER2+ class:

| Test | HR+/HER2− | Triple Negative | HER2+ |
|---|---:|---:|---:|
| 10 — one cohort per site | 0.577 | 0.385 | **0.113** |
| 12 — cohorts mixed | 0.511 | 0.423 | **0.321** |

HER2+ recall collapses from 32% to 11%, and that class's AUC falls to 0.472 — below
chance. Under genuine heterogeneity the minority class goes first. Papers that report
only aggregate metrics would not show this.

---

## 5. Privacy, communication and performance

### Communication is the strong result

The global model reaches its plateau almost immediately. The table below reads the
aggregated model's own trajectory: `agg_val_auc` from each site's `rounds.csv`, averaged
across sites, is the global model evaluated at a site *before* that site trains on it.
The round at which it first reaches 95% and 99% of its own best value is what matters.

| Test | Reaches 95% at round | Reaches 99% at round | Its best round |
|---|---:|---:|---:|
| 02 | 2 | 3 | 3 |
| 03 | 2 | 2 | 2 |
| 04 | 1 | 3 | 3 |
| 05 | 1 | 2 | 2 |
| 06 | 1 | 3 | 3 |
| 07 | 1 | 4 | 5 |
| 08 | 1 | 3 | 3 |
| 09 | 1 | 4 | 4 |
| 10 | 5 | 9 | 13 |
| 11 | 2 | 2 | 9 |
| 12 | 1 | 4 | 6 |
| 13 | 4 | 13 | 13 |

Ten of the twelve are within 5% of their best by round 5, and eight of them by round 1.
The two that are not, tests 10 and 13, are both from the cohort comparison, which is the
consistent reading: heterogeneity is also what slows convergence.

Thirty rounds were used. At 44.79 MB per client per round in each direction:

| Hospitals | Per round | 30 rounds | Stopping at round 4 | Saving |
|---|---:|---:|---:|---:|
| 2 | 179.2 MB | 5.37 GB | 0.72 GB | **87%** |
| 3 | 268.7 MB | 8.06 GB | 1.07 GB | **87%** |
| 4 | 358.3 MB | 10.75 GB | 1.43 GB | **87%** |

Across the whole campaign that is **102.12 GB** moved, against a dataset of 973 MB. Most
of it bought nothing measurable. The caveat is that this reads the metric the server
already selects on, so it says when the global model stops improving on held-out client
data rather than proving that an early-stopped model would score the same on the global
test set. No experiment stopped early to check.

### FedProx behaves as designed, but only where there is drift to correct

| Partition | FedAvg | FedProx | FedProx gain |
|---|---:|---:|---:|
| One cohort per site (heterogeneous) | 0.5426 | 0.5678 | **+0.025** |
| Cohorts mixed (control) | 0.5836 | 0.5882 | **+0.005** |

**Five times larger where the sites genuinely differ.** The proximal term of FedProx
exists to stop a client drifting from the model it was given
(Li et al., <https://arxiv.org/abs/1812.06127>), and on the stratified partitions it had
almost nothing to correct. That is exactly what those results show: +0.0101 at two
hospitals, −0.0032 at three, −0.0001 at four, and +0.0170 under quantity skew. The
sign is not stable and every value is far inside the noise floor.

### Privacy is architectural here, not measured

No experiment varies a privacy parameter. There is no differential privacy, no secure
aggregation, and no comparison. The privacy property is structural — no image leaves a
site, only weights and scalar metrics cross the boundary — and verifiable in the code,
but it is not quantified.

The one measurable privacy trade-off that *is* implemented is the scope of the class
weights: computed locally each site optimises a slightly different objective, computed
globally every site shares one objective at the cost of leaking one vector of class
counts to the server. It was never run.

This matters because model updates are not automatically private. Gradient-inversion
attacks can reconstruct training images from shared updates
(Geiping et al., <https://arxiv.org/abs/2003.14053>), which is why a deployment claim
resting on "only weights are shared" needs either differential privacy or secure
aggregation (Bonawitz et al., <https://doi.org/10.1145/3133956.3133982>) before it is a
guarantee rather than an architecture.

---

## 6. What mitigates the limitations

**FedProx under heterogeneity** is the one mitigation tested where the mechanism and the
measurement agree: +0.025 where the sites genuinely differ, and it partially recovers the
collapsed HER2+ recall (0.113 → 0.283).

**The security measures** — PKI provisioning, mutual TLS, patient-level partitioning,
per-site local validation, and a non-finite-weight guard that refuses to transmit a
diverged update — are implemented and verified by the pre-flight checks in `deployment/code/scripts/verify_production.py`,
which refuse to let a federation start if any of them fails.

**Not run, and each would strengthen this section:** global against local class-weight
scope under the cohort partition, which is the direct privacy-versus-performance
measurement; FedOpt, which was implemented and cancelled; and any form of differential
privacy.

---

## 7. What these results do and do not support

**Supported.**
- Federated training reaches performance comparable to pooled training on this task, by
  an equivalence margin derived from the method's own variability.
- Genuine cohort heterogeneity costs performance, consistently in direction across two
  algorithms, and costs the minority class far more than the aggregate suggests.
- FedProx helps materially more under real heterogeneity than under a matched control.
- Roughly 87% of the communication in a 30-round schedule is wasted on this task.

**Not supported.**
- Any specific magnitude for the heterogeneity cost. One seed, differences inside the
  noise floor.
- Any ranking among the balanced configurations. The full spread across the twelve
  federated experiments is 0.0725 against a noise floor of 0.067.
- Any privacy claim beyond the architectural one.

**The one caveat that belongs beside every pooled result.** A probe predicting which
cohort an image came from reaches macro-AUC 0.9978 against 0.6068 for the subtype. The
absolute numbers in this document are optimistic for that reason. In the stratified
partitions the shortcut is available to every site equally, so it inflates the level
without creating heterogeneity between sites — which is why it does not invalidate the
comparisons, only their absolute height.

---

## 8. References

- Sheller, M. J. et al. *Federated learning in medicine: facilitating multi-institutional
  collaborations without sharing patient data.* Scientific Reports 10, 12598 (2020).
  <https://doi.org/10.1038/s41598-020-69250-1>
- Dayan, I. et al. *Federated learning for predicting clinical outcomes in patients with
  COVID-19.* Nature Medicine 27, 1735–1743 (2021).
  <https://doi.org/10.1038/s41591-021-01506-3>
- Rieke, N. et al. *The future of digital health with federated learning.* npj Digital
  Medicine 3, 119 (2020). <https://doi.org/10.1038/s41746-020-00323-1>
- Kairouz, P. et al. *Advances and Open Problems in Federated Learning.* Foundations and
  Trends in Machine Learning 14, 1–210 (2021). <https://doi.org/10.1561/2200000083>
- McMahan, B. et al. *Communication-Efficient Learning of Deep Networks from Decentralized
  Data.* AISTATS (2017). <https://arxiv.org/abs/1602.05629>
- Li, T. et al. *Federated Optimization in Heterogeneous Networks.* MLSys (2020).
  <https://arxiv.org/abs/1812.06127>
- Ogier du Terrail, J. et al. *FLamby: Datasets and Benchmarks for Cross-Silo Federated
  Learning in Realistic Healthcare Settings.* NeurIPS Datasets and Benchmarks (2022).
  <https://arxiv.org/abs/2210.04620>
- Geiping, J. et al. *Inverting Gradients — How easy is it to break privacy in federated
  learning?* NeurIPS (2020). <https://arxiv.org/abs/2003.14053>
- Bonawitz, K. et al. *Practical Secure Aggregation for Privacy-Preserving Machine
  Learning.* ACM CCS (2017). <https://doi.org/10.1145/3133956.3133982>
- Lakens, D. *Equivalence Tests.* Social Psychological and Personality Science 8, 355–362
  (2017). <https://doi.org/10.1177/1948550617697177>
- Fridman, N. et al. *BreastDCEDL.* Scientific Data 13, 264 (2026).
  <https://doi.org/10.1038/s41597-026-06589-6>
