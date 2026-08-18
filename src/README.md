# src

The dataset and the model. Everything here is about turning MRI volumes into a
trained classifier on one machine, with no notion of hospitals or a network.

The federation imports from here, so both arms of the experiment run the same
trainer. It lives in [`deployment/code/`](../deployment/code/README.md).

| Path | What it holds |
|---|---|
| [`core/`](core/README.md) | The dataset builder and the shared trainer. |
| [`pipelines/`](pipelines/README.md) | The two preprocessing rule sets the dataset builder can be given. |
| [`scripts/`](scripts/README.md) | The two scripts that regenerate the documentation figures. |
| `dataset_config.py` | Where the raw release is, which cohorts and which task, and the `Config` object the builder takes. |

## How the pieces fit

```
dataset_config.py            what to build and from where
        |
        v
core/dataset_builder.py      reads the volumes, locates the lesion, writes PNGs
        |
        +-- pipelines/thesis/      which slices, how to crop, how to normalise
        +-- pipelines/reference/   the same three decisions, the authors' way
        |
        v
core/{data,models,training,evaluation}.py     the shared trainer
        |
        +-- notebooks/03_train_centralized.ipynb    one machine
        +-- deployment/code/common/                 one hospital
```

`deployment/code/common/thesis.py` is the bridge. It loads `dataset_config.py` by
explicit file location rather than by module name, so it cannot be shadowed by another
module called `config` on the path. It finds this folder through `$BREAST_CORE_ROOT`,
falling back to the repository layout.

## How to use it

Nothing here is a command except the two figure scripts. The code is imported, by a
notebook or by a federated client.

The shortest path through it is notebook 03:

```bash
jupyter notebook notebooks/03_train_centralized.ipynb
```

Nothing in this folder imports `nvflare`, and
`deployment/code/scripts/verify_data.py --check-imports` checks that rather than
trusting it.
