# deployment/config

A snapshot of what the configuration resolved to when the campaign was provisioned.
It is a record for auditing a finished run, and nothing reads it back.

| File | What it is |
|---|---|
| `resolved_config.json` | every hyperparameter, partition and protocol value, machine readable |
| `resolved_config.md` | the same values as tables |

## How to rebuild it

```bash
python deployment/code/scripts/snapshot_config.py
```

Reads `deployment/code/config/experiments.py` and
`deployment/code/config/federation.py` and overwrites both files here. Those two files
are the source of truth, so if the snapshot and the config disagree, the config is
right and the snapshot is stale.
