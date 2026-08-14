# results/federated — your own federated campaigns

Where a federation **you** run lands, one folder per experiment, collected by
[`notebooks/07_federated_run.ipynb`](../../notebooks/07_federated_run.ipynb) or by
`deployment/scripts/collect.sh`. Empty until you run one.

The thirteen experiments the dissertation reports are **not** here. They are preserved
unchanged in [`results/thesis/`](../thesis/README.md), and that folder is the only
source for every number in the text. Keeping the two apart means a new campaign can
never overwrite the record it would be compared against.

A collected experiment folder holds the same layout as the record: `job.json`,
`global_model.pt`, `test_metrics.json`, `predictions_test.csv` and `sites/` with the
per-round logs. `src/scripts/build_final_summary.py` aggregates whatever campaign is
in this folder into a `final_summary/` of its own.

Local run folders here are not version-controlled; only this README is.
