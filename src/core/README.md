# core — the dataset builder and the shared trainer

Everything both arms of the experiment have in common. The centralised baseline and
every federated client import from here, which is what makes the comparison a
measurement of federation rather than of two different trainers.

| File | What it does |
|---|---|
| `dataset_builder.py` | Turns the raw NIfTI volumes into the 2-D PNG dataset. Reads the volumes, locates the lesion from a mask or a bounding box, delegates the three pipeline-specific decisions to `pipelines/`, writes the images and the metadata CSV, and refuses to finish if any integrity check fails. |
| `data.py` | The `Dataset`, the augmentation profiles, and `PatientBatchSampler`, which allows at most one slice per patient per batch. |
| `models.py` | `build_model` for thirteen architectures, the head-construction invariant, and the freezing logic. |
| `training.py` | The epoch loop: AMP, gradient clipping at max-norm 1.0, the cosine schedule, and device selection. |
| `evaluation.py` | Patient-level metrics. Slice probabilities are averaged into one prediction per patient before anything is computed. |
| `reporting.py` | The per-run report: metrics, curves, confusion matrices and predictions. |
| `experiment.py` | Orchestration for a single centralised run. |

## Two invariants worth knowing before editing

**The head is never built by insertion.** The final `Linear` is replaced in place, and a
backbone that already ships a `Dropout` has it retuned rather than stacked with a second
one. Inserting into an existing `Sequential` shifts the classifier indices, and every
checkpoint saved before the change then fails to load. That happened, and the "fix"
removed dropout entirely for months without anyone noticing, because `Dropout` has no
parameters and the parameter count did not move.

**Device selection prefers CUDA, then CPU.** Apple MPS is opt-in only: through this
training loop it produces a non-finite loss from the first epoch, and the failure
survives torch 2.12 and hides from a reduced standalone probe. `get_device` documents
the measurements.
