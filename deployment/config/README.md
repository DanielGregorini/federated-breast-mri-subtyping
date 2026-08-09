# deployment/config — the resolved configuration snapshot

A read-only record of what the configuration resolved to at provisioning time. **Nothing
reads these files back.** They exist so a finished run can be audited months later
without re-deriving anything.

| File | What it is |
|---|---|
| `resolved_config.json` | every hyperparameter, partition and protocol value, machine readable |
| `resolved_config.md` | the same, as a table |

## Where it comes from

`src/scripts/snapshot_config.py`, which reads `src/federated/config/experiments.py` and
`src/federated/config/federation.py`. Those two files are the single source of truth. If
this snapshot and the config disagree, the config is right and the snapshot is stale.
