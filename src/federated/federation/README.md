# `federation/` — how to distribute

Pure NVFLARE. No knowledge of tumours, subtypes or DCE phases.

| file | responsibility |
|---|---|
| `client.py` | the FL client: receive → train → evaluate → send |
| `recipes.py` | turns an `Experiment` row into a FedAvg or FedProx recipe |
| `provisioning/project.yml` | participants, ports, PKI builders |
| `provisioning/workspace/` | generated startup kits (gitignored) |

## What crosses the network

```
flare.init()
while flare.is_running():
    model = flare.receive()          # global weights in
    ... train locally ...
    flare.send(updated_weights)      # weights out
```

Images never appear in that loop. The payload is a state dict, and it is the only
thing that leaves a site. That is the privacy claim, made concrete.

## Two things that are easy to get silently wrong

**FedProx is a client-side algorithm.** The server sends a coefficient `mu` with every
model. The client must read it, keep a frozen copy of the received global model, and
add `PTFedProxLoss(mu)(local, global)` to its loss. A client that ignores the
coefficient runs FedAvg while the logs say FedProx, and nothing warns you.

**Model selection must use held-out client data.** The server picks the best global
model by a metric the clients report. If clients report *training* accuracy, the
server selects whichever global model let them memorise their own shard best. The
metric is pinned in `config/experiments.py` as `val_balanced_accuracy`.

## Provisioning

```bash
./scripts/provision.sh
```

Produces one startup kit per participant under `provisioning/workspace/`, each with
its own certificate and private key. Four hospitals are always provisioned; the 2- and
3-client experiments use a subset, so certificates are identical across every test.
