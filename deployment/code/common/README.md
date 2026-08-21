# deployment/code/common

The library the federated clients and the operational scripts both import. Pure
PyTorch. No file here imports `nvflare`.

| File | What it does |
|---|---|
| `thesis.py` | The bridge to `src/core/`. Loads `src/dataset_config.py` by explicit file location rather than by module name, so it cannot be shadowed by another module called `config` on the path. |
| `models.py` | The shared network, the freezing policy, and the architecture fingerprint. |
| `data.py` | Per-site loaders, class weights, and the trivial baseline of a split. |
| `training.py` | One call trains one epoch, plus the FedProx variant of the loop. |
| `evaluation.py` | Patient-level aggregation and metrics. |

The model, the augmentation, the patient-aware sampler and the evaluator are imported
from `src/core/`, never reimplemented, so there is exactly one definition of this
network in the repository. FedAvg averages tensors position by position, and two sites
building networks that differ by an inserted Dropout would average into meaningless
weights with nothing to warn you.

## How to use it

Imported, not run.

## The one-epoch shape

`train_one_epoch(model, loader, ...)` returns after a single pass. The centralised
baseline calls it thirty times, and a federated client calls it once per round between
receiving and sending weights. Neither knows about the other.

With `prox_mu == 0` it delegates to `src/core/training.py`, so FedAvg clients and the
centralised baseline run the same code. FedProx needs a term that depends on the model
parameters rather than the logits, so its loop is forked, differing by exactly two
things: a frozen copy of the received global weights, and `mu/2 * ||w - w_global||^2`
added before the backward pass.

## The architecture fingerprint

`models.architecture_fingerprint(net)` hashes every parameter name and shape, and none
of the values. Weights are supposed to differ between sites, shapes are not. The job
passes the expected value to every client and a mismatch is fatal.
