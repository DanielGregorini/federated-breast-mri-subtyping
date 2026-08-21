# notebooks

The whole pipeline, numbered in the order it runs. Each notebook is one step and
carries its own logic, so you can read what happens without opening anything else.

| Notebook | What it does | What it writes |
|---|---|---|
| `01_raw_dataset_analysis.ipynb` | describes the raw BreastDCEDL release: label availability, the official split, and how far the three cohorts differ | nothing |
| `02_build_dataset.ipynb` | the preprocessing pipeline, NIfTI volumes to a 2-D PNG dataset | `dataset/multi_subtype_80mm/` |
| `03_dataset_analysis.ipynb` | describes the dataset that came out: composition, splits, the trivial baseline of each one, resolution, and example images | nothing |
| `04_train_centralized.ipynb` | the centralised classifier end to end: dataset, sampler, model, training loop, metrics, figures | `results/classifier/test_NNN_*/` |
| `05_evaluate_run.ipynb` | reads one finished run and adds the analyses that need judgement | nothing |
| `06_compare_experiments.ipynb` | every run in one table | `results/classifier/all_experiments.csv` |
| `07_federated_setup.ipynb` | the global test set and the six per-hospital partitions | `deployment/data/` |
| `08_federated_run.ipynb` | drives the twelve federated experiments and collects all thirteen results | `results/federated/` |

Run `02` first. Every notebook after it fails without the dataset, and `01` is the
only one that needs the raw release.

## How to run them

Start Jupyter from the repository root:

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Every notebook resolves paths as `Path.cwd().parent`, so the working directory has to
be `notebooks/`. That is what Jupyter does when you open one from there.

## The shape every notebook follows

1. A title cell saying what the notebook is for and what it needs.
2. One configuration cell holding every path and every constant, each path commented
   with what is expected to be there. Nothing below that cell hard-codes a path or a
   hyperparameter.
3. Alternating markdown and code, with the markdown before each cell saying what the
   cell does.
4. Alternatives left as commented-out lines wherever a different choice is one line
   away.
5. A closing cell listing what was written and where.

## What notebook 08 cannot inline

NVFLARE runs here as a real deployment rather than a simulator. The server and each
hospital are separate operating-system processes with their own certificates, and the
FLARE runtime imports the client script inside each hospital process. A function
defined in a notebook cell exists only in that kernel and cannot be imported by
another process, so the client lives at
[`deployment/code/federation/client.py`](../deployment/code/federation/client.py). The
notebook reads that file and prints the parts that matter.

The participants also have to outlive the kernel, so they are started from their own
startup kits in a terminal. See [`deployment/README.md`](../deployment/README.md).
