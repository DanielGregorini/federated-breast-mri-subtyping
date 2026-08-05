# `config/` — what to run

The single source of truth. Nothing else in the project hard-codes a hyperparameter,
a port, or a split.

| file | holds |
|---|---|
| `experiments.py` | the nine experiments, four partitions, model and hyperparameters |
| `federation.py` | **the only file that knows a host or a port** |
| `training.py` | augmentation and trainer-only settings |

Both are runnable and print a summary:

```bash
python3 config/experiments.py
python3 config/federation.py
```

## Why declarative

Nine experiments differing along three axes is exactly where per-experiment configs
drift apart. In the previous iteration three bugs came from that drift — a server
building a different architecture from its clients, a hard-coded model name in the
evaluation script, and federated clients using different regularisation from the
baseline they were compared against. All three were silent. With every experiment as
a row in one table, that class of bug cannot be expressed.

## Changing the protocol

Edit the table, then regenerate what derives from it:

```bash
python3 scripts/generate_job_readmes.py   # job READMEs
python3 scripts/partition_data.py         # per-hospital splits
```
