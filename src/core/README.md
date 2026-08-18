# src/core

The dataset builder and the shared trainer. The centralised baseline and every
federated client import from here, so both arms of the experiment run the same code.

| File | What it does |
|---|---|
| `dataset_builder.py` | Turns the raw NIfTI volumes into the 2-D PNG dataset. Reads the volumes, locates the lesion from a mask or a bounding box, delegates the three pipeline-specific decisions to `pipelines/`, and writes the images and the metadata CSV. It refuses to finish if an integrity check fails. |
| `data.py` | The `Dataset`, the augmentation profiles, and `PatientBatchSampler`, which allows at most one slice per patient per batch. |
| `models.py` | `build_model` for thirteen architectures, the head-construction rule, and the layer freezing. |
| `training.py` | The epoch loop. AMP, gradient clipping at max-norm 1.0, the cosine schedule, and device selection. |
| `evaluation.py` | Patient-level metrics. Slice probabilities are averaged into one prediction per patient before anything is computed. |
| `reporting.py` | The per-run report. Metrics, curves, confusion matrices and predictions. |
| `experiment.py` | Orchestration for a single centralised run. |

Nothing here imports `nvflare`. `deployment/code/scripts/verify_data.py --check-imports` checks
that rather than trusting it.

## How to use it

Nothing here is a command. It is imported, either by a script in
[`../scripts/`](../scripts/README.md) or by a federated client through
[`deployment/code/common/`](../../deployment/code/common/README.md).

The shortest path through it is the centralised baseline:

```bash
python deployment/code/scripts/run_centralized.py --seed 42
```

## Two things to know before editing

The final `Linear` is replaced in place rather than appended, and a backbone that
already ships a `Dropout` has it retuned rather than stacked with a second one.
Inserting into an existing `Sequential` shifts the classifier indices, and every
checkpoint saved before the change then fails to load.

Device selection prefers CUDA, then CPU. Apple MPS is opt-in only, because through
this training loop it produces a non-finite loss from the first epoch. `get_device`
records the measurements.
