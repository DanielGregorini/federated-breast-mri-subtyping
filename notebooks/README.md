# notebooks — the whole pipeline, numbered in the order it runs

Each notebook is one step, and each one carries its own logic rather than calling
into `src/`. Open one and you can read what happens without opening anything else.

| notebook | what it does | writes |
|---|---|---|
| `01_dataset_analysis.ipynb` | describes the raw BreastDCEDL release: label availability, the official split, and how far the three cohorts differ | nothing |
| `02_build_dataset.ipynb` | the full preprocessing pipeline. NIfTI volumes to a 2-D PNG dataset | `dataset/multi_subtype_80mm/` |
| `03_train_centralized.ipynb` | the centralised classifier, end to end: dataset, sampler, model, training loop, metrics, figures | `results/classifier/test_NNN_*/` |
| `04_evaluate_run.ipynb` | reads one finished run and adds the analyses that need judgement | nothing |
| `05_compare_experiments.ipynb` | every run in one table, with the noise floor applied | `results/classifier/all_experiments.csv` |
| `06_federated_setup.ipynb` | the global test set and the six per-hospital partitions | `deployment/data/` |
| `07_federated_run.ipynb` | runs the twelve federated experiments and collects all thirteen results | `results/federated/` |

Run `02` first if you only run one. Notebooks `03` and `06` both fail without it.

## How to run them

Start Jupyter from the repository root, not from this folder:

```bash
jupyter notebook notebooks/02_build_dataset.ipynb
```

Every notebook resolves paths as `Path.cwd().parent`, so the working directory has
to be `notebooks/`. That is what Jupyter does by default when you open a notebook
from here.

## The shape every notebook follows

1. A title cell saying what the notebook is for and what it needs.
2. **One configuration cell** with every path and every constant, each path
   commented with what is expected to be there. Nothing below that cell hard-codes
   a path or a hyperparameter.
3. Alternating markdown and code. The markdown before each cell says what the cell
   does and why the step exists.
4. Alternatives left as commented-out lines wherever a different choice is one line
   away, with the measurement that decided it where one exists.
5. A closing cell listing exactly what was written and where.

## What is still in `src/`, and why

Two things, and only two.

**`src/federated/federation/`** holds the NVFLARE client and recipe. NVFLARE runs
here in ProdEnv, not the simulator: the server and each hospital are separate
operating-system processes with their own X.509 identities talking over mutual TLS,
and the FLARE runtime imports the client script inside each hospital process. A
function defined in a notebook cell exists only in that kernel and cannot be
imported by another process. Notebook 07 explains this where it matters and prints
the relevant code.

**`src/scripts/start_federation.sh`** and its siblings start processes that have to
outlive the notebook kernel. A hospital started from a cell dies when the kernel
restarts, halfway through a 30-round run.

Everything else that used to live under `src/core/` is now inlined in the notebooks.
The modules are still there and still work, and the batch scripts in `src/scripts/`
still drive them. That is deliberate: the reported campaign was produced through
those scripts, and they are kept unchanged so the result stays reproducible exactly
as it was run.

## The numbers the notebooks reproduce

The inlined logic was checked against the recorded results rather than assumed to
match. The metric stack in notebook 03 recomputes the reported centralised macro AUC
as **0.6067918080145278**, identical to twelve decimal places, along with its
accuracy, balanced accuracy, macro F1, confusion matrix and every per-class figure.
The model builder gives **11,178,051** parameters with **10,494,979** trainable and
**683,072** frozen. The splitters in notebook 06 reproduce the built partitions
patient for patient: 642 / 101 / 784 for the cohort split and 848 / 340 / 170 / 169
for the 5:2:1:1 skew, with the same per-site training counts after the local
validation carve.

## One number to read everything against

Two runs of a byte-identical configuration, differing only in the random seed, were
measured **0.067 macro AUC** apart on this task. That is the noise floor. One seed is
not a result, and a difference smaller than 0.067 is not a difference.
