# results

Two kinds of material live here and they are kept apart: the frozen record that the
dissertation reports, and the folders where runs you launch yourself land.

| Folder | What it holds |
|---|---|
| [`thesis/`](thesis/README.md) | **The dissertation record.** The 13 reported experiments, `test01_centralized` through `test13_fedprox_sizematched`, plus `final_summary/`. Read-only in spirit: every number in the dissertation comes from here, so nothing in it should ever be overwritten. |
| [`classifier/`](classifier/README.md) | Your own centralised runs from `notebooks/03_train_centralized.ipynb`, one auto-numbered `test_NNN_*` folder each. Empty until you train something. |
| [`federated/`](federated/README.md) | Your own federated campaigns, collected by `notebooks/07_federated_run.ipynb` or `deployment/scripts/collect.sh`. Empty until you run a federation. |

The classifier phase of the project itself, 21 runs across 13 architectures, was
archived to `unused/old_runs/classifier/` and is not reported in the dissertation.

## Before quoting any number

The noise floor is **0.067 macro-AUC** and the trivial accuracy baseline is **0.5112**.
Every reported experiment is a single run at seed 42. Differences below the noise floor
are reported as *no difference detected*. See [the results document](../docs/RESULTS.md).
