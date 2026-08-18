# docs/images

Every figure used in the documentation. None is drawn by hand.

| Folder | What it holds | Built by |
|---|---|---|
| `preprocessing_figures/` | the step-by-step walkthrough, the normalisation comparison, slice selection, load-time augmentation, and the pipeline flowchart | `src/scripts/build_preprocessing_walkthrough.py` |
| `report_figures/` | dataset composition, tumour size by cohort, example images before and after preprocessing, and the physical-window rationale | `src/scripts/build_dataset_report_figures.py` |

## How to rebuild them

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
