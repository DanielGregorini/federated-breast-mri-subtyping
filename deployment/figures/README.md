# deployment/figures — data distribution figures

Figures about how the data is split, not about results. Result figures live in
`results/thesis/final_summary/figures/`.

Each exists as both `.png` and `.pdf`.

| Figure | Shows |
|---|---|
| `overview_balanced_vs_skewed` | class composition per hospital, balanced against 5:2:1:1 |
| `overview_cohorts` | the cohort partition against its size-matched control |

## Where they come from

`src/scripts/build_distribution_report.py`, reading
`deployment/data/partitions/*/partition.json`. None is drawn by hand. Regenerate them
after rebuilding a partition or they will describe the previous split.
