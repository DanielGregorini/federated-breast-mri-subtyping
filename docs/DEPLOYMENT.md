# Deployment — running the federation, step by step

Every command needed to go from a fresh clone to the nine finished experiments, with
an explanation of what each one does and what you should see when it works.

This is a **real NVFLARE deployment**: PKI certificates, one operating-system process
per hospital, independent TCP ports, jobs submitted through the admin API. It is not
the simulator. That distinction is a hard requirement of this dissertation, and
[step 3](#3-provision-the-pki) is where it becomes concrete.

---

## The short version

```bash
pip install -r requirements.txt          # 0. once
python scripts/prepare_data.py           # 1. global test set
python scripts/partition_data.py         # 2. per-hospital splits
python scripts/verify_data.py            # 2b. refuse if anything leaks
./scripts/provision.sh                   # 3. PKI startup kits
./scripts/start_federation.sh 4          # 4. server + 4 hospitals
python scripts/run_experiment.py test06 # 5. submit one experiment
./scripts/stop_federation.sh             # 6. stop
python scripts/collect_results.py        # 7. score everything
```

Or, for the whole protocol in one command:

```bash
python scripts/run_all_experiments.py
```

The rest of this document explains each step.

---

## 0. Prerequisites

```bash
pip install -r requirements.txt
```

| requirement | why |
|---|---|
| Python 3.10+ | the codebase uses `X \| Y` type syntax |
| `nvflare>=2.8` | the Recipes API and `ProdEnv` used here |
| PyTorch + torchvision | the model |
| a CUDA GPU | **strongly recommended** — see the note below |
| the `src` sibling folder | the model and trainer are defined there, not duplicated here |

Check the installation:

```bash
python -c "import nvflare, torch; print(nvflare.__version__, torch.__version__, torch.cuda.is_available())"
```

### On Apple Silicon

Training falls back to **CPU**, deliberately, and it is slow. This is measured rather
than cautious: on Apple MPS this model corrupts its weights to NaN partway through the
first epoch, non-deterministically, and an explicit per-step `torch.mps.synchronize()`
made it deterministic but still not finite. The same code on CPU and CUDA is exact.

A silently diverged client is worse than a slow one, because FedAvg would average it
into the global model and the server has no way to tell. Use the Mac to build
datasets, verify splits and read results; run the experiments on the CUDA box.

### Where the model comes from

`src/` imports the model, the augmentation and the patient-level evaluator from
`../src/core/`. There is exactly one definition of this network in the
repository, and the centralised baseline and every federated client build it from
that one place. If the classifier phase lives somewhere else:

```bash
export BREAST_CORE_ROOT=/path/to/src
```

---

## 1. Build the global test set

```bash
python scripts/prepare_data.py
```

Copies the held-out splits out of the prepared dataset into `data/global/`:

```
data/global/
├── images/<pid>/slice_NNN.png
├── test.csv      99 patients — the official set, identical for all nine experiments
├── val.csv       99 patients — the centralised baseline's selection set
└── manifest.json
```

**What you should see:**

```
  test    99 patients     791 slices  per-class [40, 40, 19]  trivial 0.4040
  val     99 patients     791 slices  per-class [36, 38, 25]  trivial 0.3838
```

That `trivial 0.4040` is the majority-class rate among patients. Accuracy quoted
without it is meaningless, and it is not a constant — it is 0.404 on I-SPY2 alone and
0.511 on the pooled cohorts.

The split is **not re-drawn**. It is the same split the classifier phase used, so the
federated numbers and the centralised numbers in `all_runs_pod.csv` are measured on
the same 99 patients.

> **Disk.** Add `--hardlink` to link instead of copy. Each site still has its own path
> and still cannot read another's folder; it only avoids storing the same immutable
> PNG many times. Useful on a laptop, irrelevant on the GPU box.

### Why the server holds data at all

In a production federation the aggregation server usually holds nothing. Here it holds
a held-out test set because the nine experiments must be compared on identical ground,
and a test set assembled from per-hospital leftovers would differ between a 2-client
and a 4-client run. **That is a benchmarking decision, not a claim about deployment,**
and the dissertation states it as such.

---

## 2. Partition the training data between hospitals

```bash
python scripts/partition_data.py
```

Writes all four partitions. Each hospital folder is a self-contained dataset:

```
data/partitions/4_clients_balanced/hospital_1/
├── images/<pid>/slice_NNN.png
├── train.csv
├── val.csv            20% of THIS hospital's patients
└── manifest.json
```

**What you should see:**

```
4 hospitals, balanced (25 each)  ->  25.0% / 25.0% / 25.0% / 25.0%
  hospital_1   train  159 pat [62, 57, 40]     val  39 pat  1,580 slices
  hospital_2   train  157 pat [61, 56, 40]     val  39 pat  1,558 slices
  hospital_3   train  156 pat [61, 56, 39]     val  39 pat  1,560 slices
  hospital_4   train  156 pat [61, 56, 39]     val  39 pat  1,555 slices
```

Each hospital holds **only its own patients, as real files**. It costs disk and buys
two things: the layout is exactly what would be `rsync`-ed to a real hospital machine,
and it is impossible for a bug to let one site read another's data — the files are not
there.

### Genuine non-IID, for RQ2

The default partitions are **stratified**: every hospital keeps the global class
ratio, so the only thing that varies is quantity. That is a real limitation, and it
is why the previous run of these experiments found no detectable RQ2 effect.

Two alternatives are implemented:

```bash
python scripts/partition_data.py --stratify none      # label skew
python scripts/partition_data.py --by-cohort \
    --source ../dataset/mine_subtype_pooled   # one cohort per hospital
```

`--by-cohort` is the strongest available upgrade to RQ2: DUKE is 64.6%
HRposHER2neg against I-SPY2's 38.8%, with tumours five times smaller by volume. That
is genuine heterogeneity rather than quantity skew.

> **Read the caveat first.** On pooled cohorts the source probe reaches macro-AUC
> **0.9978** predicting which cohort an image came from, against 0.6078 for the
> subtype itself. A federated result measured there is partly measuring the scanner.
> Run it, report it, and report the probe beside it.

---

## 2b. Verify — this step refuses rather than warns

```bash
python scripts/verify_data.py --check-imports
```

```
======================================================================
PASSED — 78 checks
======================================================================
```

Every check corresponds to a mistake this project or its predecessor actually made:

| check | the failure it prevents |
|---|---|
| no patient in two hospitals | FedAvg averaging two models that memorised the same patient |
| no training patient in the test set | the classic leak — an earlier phase shipped it via `StratifiedKFold` over *slices* |
| one label per patient | a patient whose slices disagree about the diagnosis |
| local val covers all classes | the site reports NaN for the metric the server selects on, and the server silently selects on the remaining sites |
| `src/` does not import `nvflare` | breaks the invariant that the baseline and the clients run the same trainer |

**Run it before every experiment.** A federated result computed on a leaking split is
worse than no result, because it looks fine: the numbers are plausible, the curves
converge, and the conclusion is wrong.

---

## 3. Provision the PKI

```bash
./scripts/provision.sh
```

This is what makes the deployment real. It runs:

```bash
nvflare provision -p production/project.yml \
                  -w production/workspace
```

and produces one **startup kit** per participant — a folder holding that
participant's own certificate, private key and start script:

```
production/workspace/breast_fl_project/prod_00/
├── server/       start.sh, sub_start.sh, fed_server.json, server.crt/.key
├── hospital_1/   ... hospital_4/
├── admin@ips.pt/    the identity that submits jobs
└── ...
```

Every connection between these participants is **mutually authenticated TLS**.
NVFLARE's simulator has none of this, which is why no number in this dissertation
comes from it.

**What you should see:**

```
startup kits in: .../production/breast_fl_project/prod_00
  ok    server
  ok    hospital_1
  ok    hospital_2
  ok    hospital_3
  ok    hospital_4
  ok    admin@ips.pt
```

### Provision once

Four hospitals are provisioned even though tests 02–05 use only two or three. The
smaller experiments use a subset of the same kits, so certificates are identical
across every test and a difference between two results can never be a difference in
PKI.

Re-running `provision.sh` creates the **next** `prod_NN` and leaves the previous one
in place — certificates are never destroyed. But it also means the server and the
clients must all start from the *same* `prod_NN`, or the TLS handshake fails with an
error that never mentions provisioning. Everything in this project resolves that
folder through `config/federation.py::workspace_dir()`, which always picks the
highest, so this cannot go wrong as long as you use the scripts.

### The three NVFLARE environments

| environment | what it is | used here |
|---|---|---|
| `SimEnv` | clients as threads in one process | never — no PKI, no network, not evidence |
| `PocEnv` | separate processes, throwaway certificates | smoke tests only |
| **`ProdEnv`** | real PKI, real ports, admin API | **every reported number** |

---

## 4. Start the federation

```bash
./scripts/start_federation.sh 4      # server + hospital_1..4
./scripts/start_federation.sh 2      # server + hospital_1..2
```

Starts the server, waits for it to accept connections, then starts each hospital —
each as its **own operating-system process**, with its own Python interpreter, its own
memory, its own certificate and its own port.

```
  started server (pid 41234)
  waiting for the server to accept connections... up
  started hospital_1 (pid 41288)
  started hospital_2 (pid 41291)
  ...
```

Under the hood each participant runs its own kit's start script:

```bash
production/workspace/breast_fl_project/prod_00/server/startup/start.sh
production/workspace/breast_fl_project/prod_00/hospital_1/startup/start.sh
```

Ports, from `config/federation.py`:

| port | who connects | purpose |
|---|---|---|
| 8002 | hospitals | receive tasks, return model updates |
| 8003 | admin | submit and monitor jobs |

They are separate so a hospital firewall can expose only the first.

**Order matters.** The server must be accepting connections before a client registers,
or the client retries with a backoff and the first round is delayed by up to a minute.
The script polls the admin port rather than sleeping a fixed amount.

Logs go to `results/_federation_logs/<participant>.log`.

### Checking it is up

```bash
tail -f results/_federation_logs/server.log
tail -f results/_federation_logs/hospital_1.log
pgrep -fl nvflare
```

---

## 5. Submit an experiment

```bash
python scripts/run_experiment.py test06
python scripts/run_experiment.py test06 --dry-run     # build it, do not submit
```

The script re-verifies the data, builds the recipe, and submits **through the admin
API** using the admin identity's certificate — exactly as a coordinating centre would.

```
test06 — test06_fedavg_4h
  objective : Four balanced sites. The primary federated configuration.
  algorithm : fedavg (mu=0.0)
  rounds    : 30 x 1 local epoch
  selection : val_balanced_accuracy
  data verified: 4_clients_balanced
  recipe built: fedavg, 30 rounds, min_clients=4
  admin            : admin@ips.pt
  server           : localhost:8003
submitting...
  job id: 4d232d20-438d-49da-a9f3-74d10a3817d7
```

Ctrl-C **detaches** — the job keeps running on the server. Do not start another
experiment while it holds the GPU.

### Test 01 is not an NVFLARE job

```bash
python scripts/run_centralized.py            # seed 42
python scripts/run_centralized.py --seed 1
```

One machine, all the training data, no server and no clients. It is budget-matched to
the federated arm — 30 epochs against 30 rounds × 1 local epoch — so RQ1 reads a
difference in federation and not a difference in compute.

### What each experiment writes

```
results/test06_fedavg_4h/
├── job.json                 job id, algorithm, partition, status
├── sites/<hospital>/
│   ├── rounds.csv           one row per round: the arriving model AND the sent one
│   └── train.log
├── predictions_test.csv     one row per patient (written by collect_results.py)
└── test_metrics.json
```

`rounds.csv` carries two curves per site and they answer different questions.
`agg_val_*` is the **aggregated** model scored before local training — the convergence
curve of the federation, which is what RQ1 is read from. `post_val_*` describes the
weights actually being sent, which is what the server selects on.

---

## 6. Stop the federation

```bash
./scripts/stop_federation.sh
```

Asks each participant to stop through its own kit's `stop_fl.sh`, waits, then kills
whatever is left — anchored to this project's workspace path.

> **Why not `pkill -f nvflare`.** `pkill -f` matches the process running the pattern.
> A previous version of this project ran exactly that over ssh, killed the ssh session
> issuing the command, and reported success. It also leaves orphaned training children
> holding the GPU, which surfaces as an out-of-memory error in an unrelated run an
> hour later.

---

## 7. Collect and compare

```bash
python scripts/collect_results.py
```

Loads every finished experiment's model into the **shared architecture**, scores it on
the **one** global test set, and writes `results/all_experiments.csv`.

```
experiment algorithm  n_clients  seed  test_auc  test_acc  test_bal  model_used
   test01         -          1    42    0.6xxx    0.5xxx    0.4xxx    selected
   test06    fedavg          4    42    0.5xxx    0.4xxx    0.4xxx    selected
   ...

  centralised mean 0.6xxx (n=2)
  federated   mean 0.5xxx (n=8)
  gap              +0.0xxx  — above the noise floor
```

Evaluation is centralised here rather than done by each run because **the comparison
is the point**: nine experiments scored by nine pieces of code is nine chances for
them to differ. This project has already shipped a `collect_results.py` with
`"resnet18"` hard-coded that silently evaluated runs which had trained a ResNet-50.

### How to read the table

The noise floor on this task is **0.067 macro-AUC**, measured between two runs of a
byte-identical configuration differing only in random seed. Treat a smaller difference
as *"no difference detected"* — that is a finding, and belongs in the dissertation as
one. **One seed is not a result.**

For reference, the previous run of these nine experiments produced a FedAvg-vs-FedProx
difference of 0.004 to 0.021, four times in the same direction. Four out of four is a
trend. Never once outside the noise means it is not a fact.

---

## Running everything

```bash
python scripts/run_all_experiments.py
python scripts/run_all_experiments.py --from test04       # resume
python scripts/run_all_experiments.py --only test06 test07
python scripts/run_all_experiments.py --dry-run
```

Runs the nine in order, **one at a time**, restarting the federation when the required
number of hospitals changes (2 → 3 → 4).

One at a time is measured, not assumed: on one GPU, two concurrent jobs gave 0.074
epochs/s, three gave 0.071 and seven gave 0.058. Past two, CUDA context switching
dominates and everything gets slower together. A federated experiment already runs N
client processes, so one experiment at a time *is* the parallel case.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `not provisioned: ... does not exist` | step 3 not run | `./scripts/provision.sh` |
| provisioning exits with `INVALID_ARGS ... ill-formatted for entity_type=admin` | the admin name is not a full email address — NVFLARE validates it against a regex that requires a TLD, so `admin@ips` is rejected | use `admin@ips.pt`, and keep `project.yml` and `config/federation.py::ADMIN_USER` identical |
| `no startup kit for 'admin@...'` when submitting | `project.yml` and `config/federation.py` disagree about the admin name | make them match, re-provision |
| TLS handshake failure on client start | server and client from different `prod_NN` | stop everything, `./scripts/start_federation.sh N` (it resolves one workspace for all) |
| client exits with `architecture mismatch` | a site is running a stale `src/` | re-sync the repo on that machine; the fingerprint is a hash of parameter names and shapes |
| `cannot find src/` | the classifier phase is elsewhere | `export BREAST_CORE_ROOT=/path/to/src` |
| `cannot locate federated/` | client started outside the repo | `export FEDBREAST_ROOT=/path/to/federated` |
| job submits but no client registers | fewer hospitals started than `min_clients` | `./scripts/start_federation.sh <n>` matching the experiment |
| `NO LOCAL VALIDATION SPLIT` in a client log | partition built without local val | re-run `scripts/partition_data.py` |
| `class_weight_scope='global' but manifest has no global_class_weights` | partition predates the setting | re-run `scripts/partition_data.py` |
| server picks a nonsense model | the key metric is training accuracy | it is pinned to `val_balanced_accuracy` in `config/experiments.py` — check the client is reporting it |
| everything is very slow, Mac | CPU fallback, by design | run on the CUDA box |
| out of memory in an unrelated run | orphaned trainer from a previous experiment | `./scripts/stop_federation.sh`, then `pgrep -fl client.py` |

### Reading a job that failed

```bash
tail -50 results/_federation_logs/server.log
tail -50 results/_federation_logs/hospital_1.log
cat results/<experiment>/sites/hospital_1/train.log
cat results/<experiment>/job.json          # job id and status
```

---

## Deploying to real hospital machines

Three things change, and nothing else does.

1. In `production/project.yml`, the server's `default_host` becomes the
   coordinating centre's DNS name or public IP, reachable by every hospital on ports
   8002 and 8003. Give each client its own `default_host` if it must be reachable.
2. Mirror the same values in `config/federation.py` — **the only file in the Python
   codebase that knows an address**.
3. Re-provision, then copy each hospital's startup kit and its
   `data/partitions/<partition>/<hospital>/` folder to its own machine, and set:

   ```bash
   export FEDBREAST_ROOT=/path/to/federated
   export BREAST_CORE_ROOT=/path/to/src
   export BREAST_SITE_DIR=/path/to/this/hospitals/data
   ```

The job definitions, the client code and the data layout are untouched. That is the
entire reason addresses live in one file.
