# deployment/datasets — split manifests and provenance

Tabulated descriptions of how the data was divided. No images and no model weights.

| File | What it is |
|---|---|
| `all_distributions.csv` / `.json` | patients and slices per class, per hospital, for every partition |
| `global_splits.csv` | the held-out global test and validation splits |
| `dataset_audit.json` | the source dataset's own totals, for reconciliation |

## Where it comes from

Written by `src/scripts/partition_data.py` and `src/scripts/audit_dataset.py` when the
partitions are built, which is what
[`notebooks/06_federated_setup.ipynb`](../../notebooks/06_federated_setup.ipynb) drives.

These are the numbers the dissertation quotes for site sizes and class spread. The
authoritative per-site record is each hospital's own `manifest.json` under
`deployment/data/partitions/<name>/hospital_N/`.
