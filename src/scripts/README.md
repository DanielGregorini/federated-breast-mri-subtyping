# src/scripts

The two scripts that regenerate the figures in the documentation. Run them from the
repository root.

| Script | What it writes |
|---|---|
| `build_dataset_report_figures.py` | `docs/images/report_figures/`. Dataset composition, tumour size by cohort, example images before and after preprocessing, and the physical-window rationale. |
| `build_preprocessing_walkthrough.py` | `docs/images/preprocessing_figures/`. The step-by-step walkthrough, the normalisation comparison, slice selection, load-time augmentation, and the pipeline flowchart. |

```bash
python src/scripts/build_dataset_report_figures.py
python src/scripts/build_preprocessing_walkthrough.py
```

Both write PNG. Pass `--pdf` to also write a vector copy of each figure.
`build_preprocessing_walkthrough.py` takes `--pid` to walk through a different
patient.

They read `dataset/multi_subtype_80mm/metadata.csv` and the raw volumes under
`raw_dataset_BreastDCEDL/`, and they call the same functions the dataset builder
calls, so a figure and a number in the text cannot disagree.

The scripts that run a federated campaign are in
[`deployment/code/scripts/`](../../deployment/code/scripts/README.md).
