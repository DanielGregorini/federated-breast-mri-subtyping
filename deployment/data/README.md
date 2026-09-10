# deployment/data

The images and manifests the participants read. One folder per hospital, so a site
physically cannot open another site's patients.

```
data/
├── global/                     the held-out sets, held by the server
│   ├── images/<patient>/       PNG slices
│   ├── test.csv                268 patients, 2,115 slices
│   ├── val.csv                 268 patients, 2,132 slices
│   └── manifest.json           sizes, class counts, trivial baseline per split
└── partitions/
    ├── 2_clients_balanced/          seed 42
    ├── 2_clients_balanced_s19/      the same shape, seed 19
    ├── 2_clients_balanced_s50/      the same shape, seed 50
    ├── 3_clients_balanced/  ...
    ├── 3_clients_cohort/    ...
    ├── 3_clients_sizematched/ ...
    ├── 4_clients_balanced/  ...
    └── 4_clients_skewed/    ...
        ├── partition.json      seed, per-site patient and class counts
        └── hospital_N/
            ├── images/<patient>/
            ├── train.csv
            ├── val.csv
            └── manifest.json
```

Six partition shapes at three seeds, so eighteen folders. A folder with no suffix is
seed 42, the split every reported result was measured on.

Images are hardlinks into `dataset/multi_subtype_80mm/`, so each site has its own path
without storing the same immutable PNG many times. Copy the folder with `rsync -aH`
to keep them links; without `-H` the copy expands to one real file per site.

Patients are split by patient, never by slice. Every image of a patient goes to one
site, and no patient appears in two of them or in both a training set and the global
test set. Each hospital holds back 20% of its own patients as a local validation
split, which is what produces the metric the server selects on.

`global/` does not depend on the seed. Which patients are training, validation and
test is read from the `test` column of the BreastDCEDL metadata, not drawn here, so
all three seeds are scored on the same 268 test patients and their results are
directly comparable.

The seed changes two things: which training patients each hospital gets, and which
of its own patients a hospital holds back for local validation. In
`3_clients_cohort` it changes only the second, because a site there is one whole
cohort and that assignment is fixed.

## How to build it

Notebook 07 writes the whole folder:

```bash
jupyter notebook notebooks/07_federated_setup.ipynb
```

The script path does the same in two steps:

```bash
python deployment/code/scripts/prepare_data.py --hardlink
```

Carves `global/` out of the processed dataset.

```bash
python deployment/code/scripts/partition_data.py --hardlink
```

Divides the remaining training patients into the six partitions, each the way its own
row in `config/experiments.py` declares. `--only NAME` builds one of them.
`--by-cohort` forces one complete source cohort per hospital, `--stratify none` lets
the class ratio differ between sites.

Another seed needs `--suffix`, which is what keeps it beside the seed-42 split rather
than over it:

```bash
python deployment/code/scripts/partition_data.py --seed 19 --suffix _s19 --hardlink
python deployment/code/scripts/partition_data.py --seed 50 --suffix _s50 --hardlink
```

The CSVs, `partition.json` and the per-site `manifest.json` are version controlled,
because they are what define a split. The PNGs are not.

## The written record

`partition_data.py` writes the splits. The documentation of them, one folder per
seed, is [`results/thesis/distributions/`](../../results/thesis/distributions/):
per-hospital and per-class counts, the figures, the patient listing saying which
site and split each patient landed in, and a cross-seed comparison table. Build it
with:

```bash
python deployment/code/scripts/build_distribution_report.py
```
