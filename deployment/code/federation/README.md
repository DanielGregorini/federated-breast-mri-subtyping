# deployment/code/federation

How the training is distributed. Pure NVFLARE, with no knowledge of tumours, subtypes
or DCE phases.

| File | What it does |
|---|---|
| `client.py` | The client loop each hospital runs. Receive the global weights, train one local epoch, evaluate on the site's own held-out patients, send the update back. Also where the FedProx proximal term is applied. |
| `recipes.py` | Turns one `Experiment` row into a FedAvg or FedOpt recipe. |

## What crosses the network

```
flare.init()
while flare.is_running():
    model = flare.receive()          # global weights in
    ... train locally ...
    flare.send(updated_weights)      # weights out
```

Images never appear in that loop. The payload is a state dict, and it is the only
thing that leaves a site.

`client.py` refuses to send non-finite weights. FedAvg sums position by position, so
one NaN would poison the global model for every remaining round.

## How to use it

Neither file is run directly. `client.py` is imported by the FLARE runtime inside each
hospital process, and `recipes.py` is called by `deployment/jobs/testNN/job.py`.

The client needs to be able to find this repository. `start.sh` in each startup kit
inherits the environment it was launched from, so export these first:

```bash
export FEDBREAST_ROOT="$PWD/deployment/code"
export BREAST_CORE_ROOT="$PWD/src"
```

## Two things that are easy to get silently wrong

FedProx is a client-side algorithm. The server sends `mu` with every model, and the
client has to read it, keep a frozen copy of the received global model, and add
`PTFedProxLoss(mu)(local, global)` to its loss. A client that ignores the coefficient
runs FedAvg while the logs say FedProx.

Model selection has to use held-out client data. If clients report training accuracy,
the server selects whichever global model let them memorise their own shard best. The
metric is pinned in `config/experiments.py` as `val_balanced_accuracy`.
