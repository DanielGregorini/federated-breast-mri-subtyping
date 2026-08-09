# deployment/jobs — the generated NVFLARE jobs

One folder per experiment, each holding a `job.py` and a README describing what that
experiment is.

**These are generated, not written.** `src/scripts/generate_jobs.py` builds them from
`src/federated/config/experiments.py`, which is the single source of truth. Editing a
`job.py` by hand is how this project once ended up with a server building a ResNet-18
while every client built a ResNet-50, so do not do it.

```bash
python src/scripts/generate_jobs.py            # rebuild all of them
python src/scripts/generate_jobs.py --check    # fail if any has drifted from the table
```

## Where the data comes from

Each job names a partition under `deployment/data/partitions/`, built by
[`notebooks/06_federated_setup.ipynb`](../../notebooks/06_federated_setup.ipynb).

## Where the outputs go

`results/federated/<experiment name>/`, collected by
`src/scripts/collect_results.py`.

`test01_centralized/` is here for completeness and is **not** an NVFLARE job. The
centralised baseline runs through `src/scripts/run_centralized.py`.
