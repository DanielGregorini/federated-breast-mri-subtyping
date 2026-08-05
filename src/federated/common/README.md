# `src/` — how to train

Pure PyTorch. **No file in this folder imports `nvflare`.** That invariant is what
lets the centralised baseline and the federated clients run the same code, which in
turn is what makes RQ1 a comparison of federation rather than of code.

`scripts/verify_data.py --check-imports` checks that invariant rather than trusting it.

| module | responsibility |
|---|---|
| `thesis.py` | the bridge to `src/core/` — the only file that knows where the classifier phase lives |
| `models.py` | the shared network, freezing policies, architecture fingerprint |
| `data.py` | per-site loaders, class weights, the trivial baseline |
| `training.py` | the training loop — **one call trains one epoch** — plus the FedProx fork |
| `evaluation.py` | patient-level aggregation and metrics |

## These files are deliberately thin

The model, the augmentation, the patient-aware sampler and the evaluator are
**imported** from `src/core/`, never reimplemented. There is exactly
one definition of this network in the repository.

That is not tidiness. FedAvg averages tensors position by position: if two sites build
networks differing by so much as an inserted Dropout, the averaged weights are
meaningless and nothing warns you. The previous iteration of this project kept 28
copies of `model.py` in step with a `sync_model.py` script — a bug waiting for the one
time somebody forgets to run it.

## The one-epoch shape

`train_one_epoch(model, loader, ...)` returns after a single pass. The centralised
baseline calls it thirty times; a federated client calls it once per round, between
receiving and sending weights. Neither knows about the other.

With `prox_mu == 0` it **delegates** to the classifier phase's own loop, so FedAvg
clients and the centralised baseline run byte-identical training code. FedProx needs a
term that depends on the model parameters rather than the logits, so it cannot be
expressed as a criterion and the loop is forked — differing by exactly two things: a
frozen copy of the received global weights, and `mu/2 * ||w - w_global||^2` added
before the backward pass.

## The architecture fingerprint

`models.architecture_fingerprint(net)` hashes every parameter **name and shape**, and
none of the values — weights are supposed to differ between sites, shapes are not.
The job passes the expected value to every client and a mismatch is fatal.

This exists because of a real failure: a recipe once exported the model with the wrong
default arguments, so the server built a ResNet-18 while the clients built a
ResNet-50. The run completed. The numbers were nonsense.

## Evaluation is per patient, always

Slice probabilities are averaged into one prediction per patient before any metric is
computed. Slices from one patient are near-duplicates, so a slice-level score measures
how well the model recognises the *patient*, not the disease.

The headline metric is **macro-AUC** (one-vs-rest). Accuracy is never reported without
the trivial baseline of the same split beside it — the majority-class rate, computed
from the test CSV rather than hard-coded, because it changes with the cohort.
