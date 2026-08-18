# deployment/code

Everything the federation runs. The configuration table, the aggregation recipes, the
client loop, the library they share, and the scripts that drive a campaign.

It sits inside `deployment/` because that is what uses it. `deployment/jobs/testNN/job.py`
imports from the folder beside it, and the FLARE runtime imports `federation/client.py`
inside each hospital process.

| Folder | What it holds |
|---|---|
| [`config/`](config/README.md) | `experiments.py`, the thirteen experiments and every hyperparameter, and `federation.py`, the only file that knows a host or a port. |
| [`federation/`](federation/README.md) | `recipes.py` builds the NVFLARE recipe for one experiment, `client.py` is the loop each hospital runs. |
| [`common/`](common/README.md) | The library the clients and the scripts both import. Pure PyTorch, no `nvflare`. |
| [`scripts/`](scripts/README.md) | The operational scripts. Build the splits, provision, generate the jobs, verify, run, collect, summarise. |

## What it depends on

`common/thesis.py` bridges to [`src/core/`](../../src/README.md), which is where the
model, the augmentation, the patient-aware sampler and the evaluator are defined. The
centralised baseline and every federated client therefore run the same trainer, which
is what makes the comparison a measurement of federation.

It finds `src/` through `$BREAST_CORE_ROOT`, falling back to the repository layout. A
hospital machine that keeps the code somewhere else sets that variable.

## How to use it

Nothing here is imported by name from outside. Both config files are runnable and
print what they define:

```bash
python deployment/code/config/experiments.py
python deployment/code/config/federation.py
```

To run a campaign, start at [`../README.md`](../README.md).
