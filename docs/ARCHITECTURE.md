# Architecture

How this project is organised, and why each piece exists where it does.

---

## The one idea the whole layout is built on

**The machine-learning code must not know that federated learning exists, and the
federated-learning code must not know what a breast MRI is.**

Everything else follows. `src/` trains a classifier; it could be run on a laptop with
no NVFLARE installed. `federation/` moves weights between processes; it does not care
whether those weights come from a ResNet or a language model. `jobs/` is the thin
layer that says *which* algorithm trains *which* model over *which* split.

This matters for three concrete reasons:

1. **The centralised baseline and the federated clients run literally the same
   trainer.** If they did not, the gap measured in RQ1 would partly be a difference
   in code rather than a difference in federation.
2. **A bug in the model is found by running `src/` alone**, in seconds, instead of by
   starting a server, four clients and an admin session.
3. **Swapping FedAvg for FedProx is a change to one line of one job file**, because
   the algorithm lives entirely on the NVFLARE side.

---

## Directory map

```
federated/
├── config/            WHAT to run. The single source of truth.
├── src/               HOW to train. Pure ML, no NVFLARE imports.
├── federation/        HOW to distribute. Pure NVFLARE, no domain knowledge.
├── jobs/              ONE folder per experiment. Configuration only.
├── data/              Per-hospital datasets. Physically separated.
├── scripts/           Entry points a human runs.
├── results/           Every run's output, one folder per experiment.
├── analysis/          Notebooks that turn results into dissertation figures.
└── docs/              This file, plus deployment and experiment notes.
```

---

## `config/` — what to run

Three files, and nothing else in the project hard-codes any of their values.

| file | holds | why it is separate |
|---|---|---|
| `experiments.py` | the nine experiments, the four partitions, the model and its hyperparameters | changing the protocol is a change to one table |
| `federation.py` | participant names, hosts, ports, admin identity | **the only file that knows an address**; deploying to real machines touches this and nothing else |
| `training.py` | augmentation profile and other trainer-only settings | keeps `experiments.py` about the *design*, not about pixel jitter |

`experiments.py` is deliberately a **declarative table**, not a set of scripts. In the
previous iteration of this work, three bugs came from per-experiment configs drifting
apart: the server built a ResNet-18 while the clients built a ResNet-50, an evaluation
script had an architecture name hard-coded, and the federated clients used different
regularisation from the centralised baseline they were compared against. All three
were invisible until the numbers looked wrong. When every experiment is a row in one
table, that class of bug is not expressible.

---

## `src/` — how to train

Pure PyTorch. **No file in here imports `nvflare`.** That is the invariant, and
`scripts/verify_data.py --check-imports` checks it rather than trusting it.

```
src/
├── thesis.py      the bridge to src/core/ — the only file that
│                  knows where the classifier phase lives
├── models.py      the shared network, freezing, architecture fingerprint
├── data.py        per-site loaders, class weights, the trivial baseline
├── training.py    one epoch at a time, with the FedProx fork
└── evaluation.py  patient-level aggregation and metrics
```

Flat modules rather than sub-packages: each is one responsibility and one file, and a
folder holding a single module would be structure for its own sake.

**These files are thin on purpose.** The model, the augmentation, the sampler and the
evaluator are *imported* from `src/core/`, not reimplemented. Copying
them would create a second definition that starts drifting on day one — the previous
iteration copied `model.py` into all 28 participant folders and needed a
`sync_model.py` to keep them equal, and FedAvg only averages correctly if every site
builds an identical network. One definition, imported twice.

The training loop is written so that **one call trains one epoch** and returns. That
shape is what lets the same code serve both worlds: the centralised baseline calls it
thirty times in a row, and a federated client calls it once per round, in between
receiving and sending weights. With `prox_mu == 0` it *delegates* to the classifier
phase's loop, so FedAvg clients and the centralised baseline run byte-identical
training code — a fact about the call graph, not a promise in a comment.

`models.py` also produces an **architecture fingerprint**: a hash of every parameter
name and shape, but not of the values. Two sites running the same code produce the
same fingerprint; a site running a stale copy does not. The job passes the expected
value to every client and a mismatch is fatal, because the alternative is a run that
completes and means nothing.

Evaluation deserves its own module because it is where the subtle errors live.
Everything is reported **per patient**, not per slice: slice probabilities are averaged
into one prediction per patient before any metric is computed. Slices from one patient
are near-duplicates, so a slice-level score measures how well the model recognises the
patient, not the disease.

---

## `federation/` — how to distribute

Pure NVFLARE. No knowledge of tumours, subtypes or DCE phases.

```
federation/
├── client.py            the FL client: receive -> train -> evaluate -> send
├── recipes.py           builds the recipe from an Experiment row
└── provisioning/
    ├── project.yml      participants, ports, PKI builders
    └── production/      generated startup kits (gitignored)
```

The output folder is named **`production`** because that is what it holds: the real
deployment, one signed startup kit per participant, the thing that would be copied to
a hospital machine. NVFLARE's simulator and POC modes never write here, and no number
in this dissertation comes from either.

`nvflare provision` never overwrites — each run creates the next `prod_NN` beside the
previous one. That means "the startup kits" is ambiguous unless something picks, and
picking wrongly means a server and a client presenting certificates from two different
runs, which fails at the TLS handshake with an error that never mentions provisioning.
Every script resolves the folder through `config/federation.py::workspace_dir()`,
which always takes the highest.

**`client.py`** is a normal training script wrapped in NVFLARE's Client API:

```
flare.init()
while flare.is_running():
    model = flare.receive()          # global weights from the server
    ... train locally for N epochs ...
    flare.send(updated_weights)      # only weights leave the site
```

Images never appear in that loop. This is the privacy claim made concrete: the payload
crossing the network is a state dict, and it is the only thing that crosses.

**`recipes.py`** turns a row of `config/experiments.py` into an NVFLARE recipe.
FedAvg and FedProx differ by one class and one argument, so the file is short by
design — the complexity belongs to NVFLARE, not to us.

Two details that are easy to get wrong and are therefore fixed here:

- **FedProx is a client-side algorithm.** The server sends a coefficient `mu` with
  every model; the client must read it, keep a frozen copy of the received global
  model, and add `PTFedProxLoss(mu)(local, global)` to its loss. A client that ignores
  the coefficient is running FedAvg while claiming to run FedProx, and nothing warns
  you.
- **Model selection must use held-out client data.** The server picks the best global
  model by a metric the clients report. If clients report training accuracy, the
  server selects whichever global model let them memorise their own shard best. The
  metric name is pinned in `config/experiments.py` as `val_balanced_accuracy`.

---

## `jobs/` — one folder per experiment

```
jobs/test06_fedavg_4h/
├── README.md     objective, clients, split, algorithm, purpose, research question
└── job.py        builds and submits this experiment. Configuration only.
```

Each `job.py` reads its own row from `config/experiments.py` and asks
`federation/recipes.py` to build the recipe. It contains **no hyperparameters and no
model definition** — if it did, the nine jobs would start drifting apart on day one.

The READMEs are **generated** from the experiment table rather than written by hand,
so a change to the protocol cannot leave stale documentation behind.

`test01_centralized` is not an NVFLARE job. It keeps a folder anyway, for symmetry
and because its README belongs beside the other eight.

---

## `data/` — physically separated hospitals

```
data/
├── global/
│   ├── test/          the held-out test set. Identical for all nine experiments.
│   └── labels.csv
└── partitions/
    ├── 2_clients_balanced/
    │   ├── hospital_1/{train,val}/
    │   └── hospital_2/{train,val}/
    ├── 3_clients_balanced/
    ├── 4_clients_balanced/
    └── 4_clients_skewed/
```

**Each hospital folder holds only that hospital's patients**, as real copies rather
than symlinks. It costs disk and buys two things: the layout is exactly what would be
`rsync`-ed to a real hospital machine, and it is impossible for a bug to let one site
read another's data — the files are not there.

Three rules are enforced by `scripts/partition_data.py` and verified by
`scripts/verify_data.py`:

1. **Split by patient, never by slice.** Every image of a patient goes to one site.
2. **No patient appears in two hospitals**, and no training patient appears in the
   global test set.
3. **Each hospital keeps a local validation split** (20% of its own patients), which
   is what produces the metric the server selects on. It is local by construction: a
   hospital cannot validate on another hospital's patients.

The **global test set lives with the server**. In a production federation the server
usually holds no data at all; here it holds a held-out set because the nine
experiments must be compared on identical ground. That choice is a benchmarking
decision, not a claim about deployment, and it is stated as such in the dissertation.

---

## `scripts/` — what a human runs

| script | does |
|---|---|
| `prepare_data.py` | builds the global test set from the source dataset |
| `partition_data.py` | writes the four per-hospital splits |
| `verify_data.py` | leakage checks; refuses to pass if a patient is in two places |
| `provision.sh` | runs `nvflare provision`, producing the PKI startup kits |
| `start_federation.sh` | starts server and N hospitals as separate processes |
| `stop_federation.sh` | stops them, and cleans up orphans |
| `run_experiment.py` | submits one experiment and waits for it |
| `run_all_experiments.py` | the nine, in order, one at a time |
| `collect_results.py` | evaluates every global model on the global test set |

`start_federation.sh` deliberately starts **separate operating-system processes**, not
threads. That is the difference between a simulation and a deployment: each hospital
has its own Python interpreter, its own memory, its own certificate and its own port.
Moving one of them to another machine changes an address, not a design.

---

## `results/` and `analysis/`

`results/<experiment_name>/` holds, per run: the global model checkpoint, the
per-round convergence CSV, per-patient predictions, the resolved configuration, and
the full log. Enough to regenerate every dissertation figure without re-running
anything.

`analysis/` reads `results/` and writes figures. It never trains.

---

## What is deliberately NOT here

- **No `simulator/` folder.** NVFLARE's simulator runs clients as threads in one
  process. It is useful while writing code and useless as evidence, so it is not part
  of the project structure. Every reported number comes from the real deployment.
- **No copy of the model inside each job.** The previous iteration copied `model.py`
  into all 28 participant folders and needed a `sync_model.py` to keep them equal.
  FedAvg only averages correctly if every site builds an identical network, and
  keeping 28 copies in step is a bug waiting to happen. Here the model is defined once
  in `src/models/` and shipped by the recipe.
- **No hand-written `config_fed_server.json`.** Those files are generated by the
  recipe. Editing generated JSON is how the server/client architecture mismatch
  happened last time.
