# deployment/code/config

What to run. Nothing else in the project hard-codes a hyperparameter, a port or a
split.

| File | What it holds |
|---|---|
| `experiments.py` | `TrainingConfig` (model and hyperparameters), `FederationConfig` (rounds, local epochs, FedProx mu, the selection metric), the six `Partition` definitions and the thirteen `Experiment` rows. |
| `federation.py` | The participant names, the two ports (8002 for clients, 8003 for the admin API) and the resolver that always picks the highest provisioned workspace. |

`federation.py` is mirrored into `deployment/project.yml`, and
Both have to say the same thing.

## How to use it

Both files are runnable and print a summary of what they define:

```bash
python deployment/code/config/experiments.py
python deployment/code/config/federation.py
```

## Changing the protocol

Edit the table, then regenerate everything derived from it:

```bash
python deployment/code/scripts/generate_jobs.py       # the thirteen job folders
python deployment/code/scripts/partition_data.py      # the per-hospital splits
```
