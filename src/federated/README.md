# src/federated — the NVIDIA FLARE layer

| Folder | What it holds |
|---|---|
| `config/` | `experiments.py`, the single source of truth for the 13 experiments, 6 partitions and every hyperparameter; and `federation.py`, the only file that knows a host or a port |
| `federation/` | `recipes.py` builds the NVFLARE recipe for one experiment; `client.py` is the client loop the hospitals run |
| `common/` | The library both the clients and the scripts import: `models.py`, `data.py`, `training.py`, `evaluation.py`, and `thesis.py`, the bridge to `../core/` |

## config/

`experiments.py` holds `TrainingConfig` (model and hyperparameters), `FederationConfig`
(rounds, local epochs, FedProx mu, the selection metric), the six `Partition` definitions
and the thirteen `Experiment` rows. Nothing else in the repository defines any of these.

`federation.py` holds the participant names, the two ports (8002 for clients, 8003 for
the admin API) and the resolver that always picks the highest provisioned workspace. It
is mirrored into `deployment/project.yml`, and the pre-flight fails if the two disagree.

## federation/

`recipes.py` builds a `FedAvgRecipe` or a `FedOptRecipe` from an experiment row. Two
things it refuses to do, both because they produced silent wrong results before:

- It will not build a FedProx job with `mu <= 0`. A client that receives the coefficient
  and ignores it runs FedAvg while the results table says FedProx, and nothing warns you.
- It passes the model as a built `nn.Module`, not as a dotted path. A path resolves with
  the wrong defaults, and that is how this project once ran a server building a
  ResNet-18 against clients building a ResNet-50.

`client.py` receives the global weights, trains one local epoch, evaluates on the site's
own held-out patients, and sends the weights back with its metrics. It refuses to send
non-finite weights: FedAvg sums position by position, so one NaN would poison the global
model for every remaining round with nothing in the server log to say which site did it.

## common/

`thesis.py` is the bridge to `../core/`. It loads the root `dataset_config.py` by explicit
file location rather than by name, so it cannot be shadowed by another module called
`config` on the path.
