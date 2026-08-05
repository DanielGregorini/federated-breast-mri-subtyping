# src — all the code

| Folder | What it holds |
|---|---|
| [`core/`](core/README.md) | The dataset builder and the shared trainer. The centralised baseline and every federated client run this same code |
| [`pipelines/`](pipelines/README.md) | The two preprocessing rule sets: `thesis/` is what this dissertation proposes, `reference/` reproduces the dataset authors' published rules |
| [`federated/`](federated/README.md) | Configuration, aggregation recipes, the client loop, and the library the clients share |
| [`scripts/`](scripts/README.md) | Every operational script: build the data, generate the jobs, verify, run, collect, summarise |
| `analysis/` | Exploratory notebooks |
| `dataset_config.py` | Dataset and task configuration: where the raw release is, which cohorts and which task, and the `Config` object the builder takes |

## How the pieces relate

```
dataset_config.py     what to build and from where
        |
        v
core/dataset_builder.py    reads volumes, locates the lesion, writes PNGs
        |
        +---- pipelines/thesis/      which slices, how to crop, how to normalise
        +---- pipelines/reference/   the same three decisions, the authors' way
        |
        v
core/{data,models,training,evaluation}.py     the shared trainer
        |
        +---- run_centralized.py (at the repository root)
        +---- federated/common/  ->  src/federated/federation/client.py
```

`federated/common/thesis.py` is the bridge: it loads `dataset_config.py` by explicit file
location rather than by name, so it cannot be shadowed by another module called `config`
on the path.

## The one rule

`src/federated/config/experiments.py` is the **single source of truth** for every
hyperparameter and every experiment. Jobs are generated from it; the deployment snapshot
is written from it and never read back; the scripts are wrappers around it.

Three separate defects in this project's history had one cause — two copies of a setting
drifting apart. A server built a ResNet-18 while its clients built a ResNet-50; an
evaluation script had the architecture hard-coded; federated clients were regularised
differently from the baseline they were compared against. All three completed normally
and produced meaningless numbers. Generating everything from one table makes that class
of defect inexpressible.
