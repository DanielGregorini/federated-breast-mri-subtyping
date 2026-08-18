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
    ├── 2_clients_balanced/
    ├── 3_clients_balanced/
    ├── 3_clients_cohort/
    ├── 3_clients_sizematched/
    ├── 4_clients_balanced/
    └── 4_clients_skewed/
        ├── partition.json      per-site patient and class counts
        └── hospital_N/
            ├── images/<patient>/
            ├── train.csv
            ├── val.csv
            └── manifest.json
```

Images are hardlinks into `dataset/multi_subtype_80mm/`, so each site has its own path
without storing the same immutable PNG many times.

Patients are split by patient, never by slice. Every image of a patient goes to one
site, and no patient appears in two of them or in both a training set and the global
test set. Each hospital holds back 20% of its own patients as a local validation
split, which is what produces the metric the server selects on.

## How to build it

Notebook 06 writes the whole folder:

```bash
jupyter notebook notebooks/06_federated_setup.ipynb
```

The script path does the same in two steps:

```bash
python deployment/code/scripts/prepare_data.py --hardlink
```

Carves `global/` out of the processed dataset.

```bash
python deployment/code/scripts/partition_data.py --hardlink
```

Divides the remaining training patients into the six partitions. `--only NAME` builds
one of them. `--by-cohort` gives each hospital one complete source cohort.
`--stratify none` lets the class ratio differ between sites.

## How to check it

```bash
python deployment/code/scripts/verify_data.py
```

Checks that no patient is in two sites, that no training patient is in the global test
set, and that every local validation split covers all three classes. Exits non-zero if
any check fails. Run it after every rebuild.

Nothing in this folder is version controlled except this README.
