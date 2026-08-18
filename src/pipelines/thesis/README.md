# src/pipelines/thesis

The preprocessing this dissertation proposes. `preprocessing.py` supplies the three
decisions `core/dataset_builder.py` asks a pipeline for.

| Rule | Value |
|---|---|
| slices | 8 evenly spaced, trimming 15% from each end of the lesion |
| crop | 80 mm physical window, giving a constant 0.357 mm/px |
| normalisation | min-max over the whole 4-D volume |
| cohorts | I-SPY2 only by default |

Each rule carries the measurement that decided it in the source file.

## How to use it

Name it in a `Config` in `src/dataset_config.py`, then build the dataset with
[`notebooks/02_build_dataset.ipynb`](../../../notebooks/02_build_dataset.ipynb).
