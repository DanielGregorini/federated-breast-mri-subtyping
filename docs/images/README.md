# docs/images — every figure in the documentation

Each figure exists as both `.png` and `.pdf`. **None is drawn by hand**, and every one is
regenerated from the data by a script in `src/scripts/`. If a figure and a number in the
text disagree, regenerate the figure.

| Folder | What it holds | Built by |
|---|---|---|
| `preprocessing_figures/` | the step-by-step walkthrough, the normalisation comparison, slice selection, load-time augmentation, and the pipeline flowchart | `src/scripts/build_preprocessing_walkthrough.py` |
| `report_figures/` | dataset composition, tumour size by cohort, example images before and after, and the physical-window rationale | `src/scripts/build_dataset_report_figures.py` |

Result figures are not here. They live in
`results/federated/final_summary/figures/`, built by
`src/scripts/build_final_summary.py` and the `build_*_figure(s).py` scripts.

## Where the data comes from

`dataset/multi_subtype_80mm/metadata.csv` and the raw volumes under
`raw_dataset_BreastDCEDL/`. The walkthrough calls the same functions the dataset builder
does, so what it shows is what the network is trained on.
