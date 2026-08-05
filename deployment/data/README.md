# `data/` — physically separated hospitals

```
data/
├── global/
│   ├── test/          the held-out test set. Identical for all nine experiments.
│   └── labels.csv
└── partitions/
    ├── 2_clients_balanced/hospital_{1,2}/{train,val}/
    ├── 3_clients_balanced/hospital_{1..3}/{train,val}/
    ├── 4_clients_balanced/hospital_{1..4}/{train,val}/
    └── 4_clients_skewed/hospital_{1..4}/{train,val}/
```

**Each hospital folder holds only that hospital's patients**, as real copies rather
than symlinks. It costs disk and buys two things: the layout is exactly what would be
`rsync`-ed to a real hospital machine, and it is impossible for a bug to let one site
read another's data — the files are not there.

## Three rules, enforced not assumed

1. **Split by patient, never by slice.** Every image of a patient goes to one site.
   Slices from one patient are near-duplicates; a slice-level split would let the
   model recognise the patient instead of the disease.
2. **No patient in two places.** Not across hospitals, and not between any training
   set and the global test set.
3. **Each hospital keeps its own validation split** — 20% of its own patients. This is
   what produces the metric the server selects on, and it is local by construction: a
   hospital cannot validate on another hospital's patients.

`scripts/verify_data.py` checks all three and exits non-zero if any fails. Run it
after every partitioning.

## Why the global test set sits with the server

In a production federation the server usually holds no data at all. Here it holds a
held-out set because the nine experiments must be compared on identical ground —
changing which patients are tested moved macro-AUC by more than any intervention ever
measured in this project. This is a benchmarking decision, not a claim about
deployment, and the dissertation states it as such.

## Regenerating

```bash
python3 scripts/prepare_data.py      # global test set
python3 scripts/partition_data.py    # the four splits
python3 scripts/verify_data.py       # leakage checks
```

Nothing here is version-controlled. Back it up separately.
