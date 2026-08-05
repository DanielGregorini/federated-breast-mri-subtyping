# The experiments — design, limitations, and how to read them

What the nine runs are, what each one can and cannot answer, and every place where a
deliberate choice could be mistaken for a result.

The protocol itself lives in `config/experiments.py` as a declarative table. This
document is the reasoning behind it.

---

## The design

One centralised baseline against eight federated runs. The federated runs vary the
number of hospitals (2, 3, 4) and how the data is divided between them (balanced vs
skewed), and each configuration runs once with FedAvg and once with FedProx, with
everything else held fixed.

| # | hospitals | split | algorithm | primary question |
|---|---:|---|---|---|
| 01 | — | all pooled | — | the reference |
| 02 | 2 | 50/50 | FedAvg | RQ1 |
| 03 | 2 | 50/50 | FedProx | RQ3 |
| 04 | 3 | 33/33/33 | FedAvg | RQ1 |
| 05 | 3 | 33/33/33 | FedProx | RQ3 |
| 06 | 4 | 25 each | FedAvg | **RQ1 headline**, RQ2 control |
| 07 | 4 | 25 each | FedProx | RQ3 |
| 08 | 4 | 5:2:1:1 | FedAvg | RQ2 |
| 09 | 4 | 5:2:1:1 | FedProx | RQ4 |

| | question | read from |
|---|---|---|
| **RQ1** | Can federated match centralised? | 01 vs 06 |
| **RQ2** | Impact of non-IID heterogeneity? | 06 vs 08 |
| **RQ3** | FedAvg vs FedProx trade-offs? | every even/odd pair |
| **RQ4** | What mitigates FL limitations? | 09, plus class-weight scope |

### A note on 50/20/10/10

The dissertation describes the skewed split that way, which sums to 90 rather than
100. It is a **5:2:1:1 ratio**, and normalising it gives 55.6 / 22.2 / 11.1 / 11.1.
`Partition` stores the ratio and normalises it in code, so the dissertation's wording
and the program's behaviour agree instead of silently differing by ten percent.

---

## What is held fixed, and why that is the whole design

Every experiment shares one `TrainingConfig`: the same model, the same freezing, the
same optimiser, the same augmentation, the same class-weighting rule, the same seed.
The centralised baseline and the federated clients run **literally the same trainer** —
`src/training.py` delegates to `src/core/training.py` — so the gap RQ1
measures is federation rather than a difference in code.

This is not caution. Three separate bugs in the previous iteration of this project
came from per-experiment configs drifting apart:

* the server built a ResNet-18 while the clients built a ResNet-50;
* an evaluation script had an architecture name hard-coded;
* the federated clients used different regularisation from the centralised baseline
  they were compared against.

All three were invisible until the numbers looked wrong.

### Budget matching

30 rounds × 1 local epoch against 30 centralised epochs. The model sees the data the
same number of times on both sides. Without that, RQ1 would read a difference in
compute as a difference in federation.

This makes the baseline **weaker than the headline classifier result** (0.6159 ±
0.003, from a 100-epoch sweep with patience 30), and that is correct rather than a
problem. The number a federated run is compared against must be the one trained on the
same budget. In practice the difference is small, because the best epoch on this task
lands between 1 and 5 — the model exhausts the signal in the first pass.

---

## Deliberate limitations — state these in the dissertation

### 1. Tests 08 and 09 are quantity skew, not non-IID

Every hospital keeps the global class ratio; only the *amount* of data varies. That is
why the previous run of these experiments found **no detectable RQ2 effect** — there
was very little heterogeneity to detect.

Two genuine alternatives are implemented and neither is the default:

```bash
python scripts/partition_data.py --stratify none    # label skew
python scripts/partition_data.py --by-cohort \
    --source ../dataset/mine_subtype_pooled
```

`--by-cohort` gives one real cohort per hospital: DUKE at 64.6% HRposHER2neg against
I-SPY2's 38.8%, with tumours five times smaller by volume and a different scanner
population. That is the strongest available upgrade to RQ2.

**It carries a cost that must be reported with it.** On pooled cohorts the source
probe reaches macro-AUC **0.9978** predicting which cohort an image came from, against
0.6078 for the subtype. A model trained across those sites has a shortcut available:
identify the cohort, then use that cohort's class prior. Report the probe beside the
result, always.

### 2. The server holds the test set

In a production federation it would hold nothing. Here it holds a held-out set because
the nine experiments must be compared on identical ground. **A benchmarking decision,
not a claim about deployment.**

### 3. Model selection differs between the two arms

| | selects on | why |
|---|---|---|
| centralised (01) | validation **macro-AUC**, 99 patients | the classifier phase's rule; AUC is well defined on 99 patients |
| federated (02–09) | **`val_balanced_accuracy`**, per hospital | a site holding 39 patients can draw a validation split missing a class, and macro-AUC is then NaN |

Both are computed on held-out patients, which is the part that matters. Neither is
training accuracy — a previous iteration reported that to the server, which then
selected whichever global model let clients memorise their own shard best (99%+, and
no information at all).

### 4. Only the seeds you actually run

The noise floor is **0.067 macro-AUC**. `run_all_experiments.py` runs the centralised
baseline at two seeds by default and each federated experiment once, which is enough
to see a 0.07 effect and **not** enough to rank FedAvg against FedProx. Say so.

---

## Places where a choice could be mistaken for a result

### The dropout discrepancy — found here, fixed upstream

While wiring this project up, `src/core/models.py::build_model` was
found to accept a `dropout` argument and **ignore it** for every torchvision backbone:
`dropout=0.0` and `dropout=0.5` returned byte-identical architectures, a bare
`Linear(512, 3)`.

The seven checkpoints in `results/checkpoints/` do not look like that. Every one
stores `fc.1.weight` — index 1 of a `Sequential`, with a `Dropout` at index 0. They
were trained by a build that applied dropout, and the code as it stood could not load
them at all. So the configuration that produced FREEZE_R18 = 0.6159 was not the
configuration the classifier code was training: it was that configuration minus its
regularisation, silently, while `results.json` still recorded `dropout: 0.5`.

**`core/models.py` now honours `dropout` across every backbone**, so this project
builds the head through it and adds nothing of its own — the `_attach_dropout_head`
helper that used to sit in `src/models.py` has been **removed**, because with core
honouring the field, wrapping again would apply dropout twice and change the
architecture out from under FedAvg. Verified end to end: all seven checkpoints load
with `strict=True`, the ResNet-18 ones at 11,187,671 parameters and buffers, matching
the recorded value, and the architecture fingerprint is stable.

Three consequences worth carrying into the write-up:

* **No recorded number is affected, and nothing needs re-running.** Checked rather
  than assumed: all 21 runs in `all_runs_pod.csv` were produced by the older
  `unused/old_training/` code, whose `build_model` did wrap the head as
  `Sequential(Dropout, Linear)`. Every one of their `config.json` files carries the
  old field names (`model_name`, `aug_profile`, `backbone_lr_scale`), and no run
  anywhere in the repository was written by the current `core/` schema. The
  regression was latent — the first affected run would have been the next one.
* **The parameter count cannot detect this.** `Dropout` has no parameters, so both
  builds total 11,187,671 params+buffers for ResNet-18. Only the `fc.1.*` key names
  distinguish them. A run log reporting the expected parameter count is not evidence
  that the head is right.
* This is the fourth bug in this project's history whose signature was "the run
  completes and the numbers look plausible". It is the reason for the architecture
  fingerprint in `src/models.py` and for `strict=True` everywhere a checkpoint is
  loaded — `strict=False` here loads the backbone and leaves the classifier at random
  init, which on this task still scores near chance and reads as a result.

### The learning-rate schedule

A federated client is re-instantiated each round and holds no state, so a
`CosineAnnealingLR` object cannot survive to be stepped. Dropping the schedule would
leave the federated arm at a constant rate while the baseline decays; re-creating it
each round would produce a sawtooth. Either turns RQ1 into a comparison of schedules.

Because cosine annealing is a closed-form function of the step index, and the server
sends `current_round` with every model, the client evaluates it directly:

    lr(r) = base * (1 + cos(pi * r / T)) / 2

which is exactly what `CosineAnnealingLR(T_max=T)` holds at epoch `r` centrally. Both
arms follow the same curve; the client keeps no state. The effect is small — the best
epoch lands at 1–5, where cosine has decayed under 3% — but a known asymmetry in the
one comparison the dissertation rests on is not worth leaving in.

### FedProx is client-side

FedProx and FedAvg use the **same recipe and the same aggregation**. The entire
difference is `mu/2 * ||w - w_global||^2` added to the local loss. A client that
receives `mu` and ignores it is running FedAvg while the results table says FedProx,
and nothing warns you — so `federation/recipes.py` refuses to build a FedProx job with
`mu <= 0`, and `src/training.py` refuses to apply a proximal term without the global
weights to anchor to.

The reported loss excludes the proximal term. Including it would make FedAvg and
FedProx losses incomparable across the very curves RQ3 is read from, and would make
the number move with `mu` rather than with the model.

### Class weights: whose frequencies? (RQ4)

`TrainingConfig.class_weight_scope` is `"local"` by default — each hospital weights
its loss by its own class frequencies. Each site therefore optimises a slightly
different objective, and FedAvg averages models trained on different losses.

Under the stratified partitions of tests 02–09 this is harmless; the weights agree to
three decimals. **Under a cohort partition it stops being harmless**, and the choice
becomes RQ4 material:

| scope | what it means | cost |
|---|---|---|
| `local` | realistic — a hospital knows only its own data | sites pull towards different decision boundaries |
| `global` | one shared objective | leaks one vector of class counts to the server |

Both are implemented. `partition_data.py` writes `global_class_weights` into every
site manifest, computed once from the pooled training split, so no site has to see
another site's data to use them.

---

## What the previous run of these nine found

On the **old** binary TripleNeg-vs-rest task with ResNet-50 — superseded data, but the
protocol was the same shape:

* **RQ1 — no.** Centralised 0.6874 against 0.5776–0.6194 federated. A drop of 0.068 to
  0.110, at or above the noise floor.
* **RQ2 — no detectable effect**, consistent with the skew being quantity-only.
* **RQ3 — FedProx won 4 of 4 paired comparisons** (+0.005, +0.021, +0.011, +0.004).
  Every one inside the noise floor. Four out of four in one direction is a **trend,
  not a fact**.

Two secondary findings worth carrying forward:

* The effect is **all-or-nothing**: 2, 3 and 4 hospitals gave the same result. This
  *contradicts* the earlier lung-segmentation project, where degradation was
  progressive.
* **Federation hurts the clinically important class.** TripleNeg recall fell from
  48.6% to 27–40% in eight of nine configurations. Papers usually report only the
  aggregate, which hides this entirely — `collect_results.py` therefore records
  `per_class_recall` in every row.

---

## Checklist before reporting a number

- [ ] `scripts/verify_data.py` passed on this partition
- [ ] the source probe was run if the dataset pools cohorts, and is quoted beside the result
- [ ] accuracy is quoted with the trivial baseline of the same split
- [ ] the metric is patient-level macro-AUC, not slice-level
- [ ] at least two seeds, or the text says "one seed"
- [ ] any difference below 0.067 is reported as "no difference detected"
- [ ] per-class recall is reported, not only the aggregate
- [ ] `python scripts/generate_jobs.py --check` passes, so no job drifted from the table
