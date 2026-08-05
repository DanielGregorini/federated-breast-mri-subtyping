# notebooks — the pipeline, numbered in the order it runs

Each notebook is a step. They contain no logic of their own: everything is a call into
`src/core/` and `src/pipelines/`, so a result produced in a notebook and one produced by
a script cannot differ.

| Notebook | What it does |
|---|---|
| `01_dataset_analysis.ipynb` | Explores the raw release before anything is built: cohort sizes, receptor labels, voxel spacing, tumour volume, and how far the three cohorts differ from each other |
| **`02_build_dataset.ipynb`** | **The full preprocessing pipeline.** Turns the NIfTI volumes into the 2-D PNG dataset. Run this one first if you only run one |
| `03_train_centralized.ipynb` | Trains the centralised classifier interactively, for inspecting a run while it happens |
| `04_evaluate_run.ipynb` | Evaluates one finished run in detail: per-class metrics, confusion matrix, curves |
| `05_compare_experiments.ipynb` | Compares runs against each other, with the noise floor applied |

## Running them

Start Jupyter from the repository root, not from this folder:

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Each notebook puts `src/` on the path itself, so the imports work regardless of where
the kernel was started, provided the working directory is `notebooks/`.

## The numbering

It is the order of the pipeline, not the order the notebooks were written. Earlier
versions carried numbers up to 07 with gaps where retired notebooks had been removed;
those are archived and no longer referenced.

For the batch equivalents of steps 3 to 5, use the scripts in
[`src/scripts/`](../src/scripts/README.md) — they are what the reported results were
produced with.
