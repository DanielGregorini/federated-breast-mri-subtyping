# NVIDIA FLARE — exact configuration used

**Every value in this document was read from a source file or a run artefact in this
repository.** Where the repository does not record something, that is stated explicitly
rather than filled in. Nothing here is inferred from defaults.

Verified 2026-08-04 against `config/`, `federation/`, `production/project.yml`,
`production/logs/` and `production/results/`.

---

## 1. NVIDIA FLARE version

**2.8.0**

| evidence | value |
|---|---|
| `python3 -c "import nvflare; print(nvflare.__version__)"` | `2.8.0` |
| `production/project.yml` | `api_version: 3` |
| `production/README.md:523` | refers to NVFLARE 2.8 admin-name validation |
| campaign record, `docs/PROJECT_CONTEXT.md` | NVFLARE 2.8.0, CUDA 12.8, torch 2.8.0 |

**Not confirmed.** The run logs produced on the RunPod host carry no version string, so
the pod's exact patch level cannot be recovered from the repository. The value above is
the local environment plus the written campaign record.

---

## 2. Simulator configuration

**The simulator was not used for any reported result.**

`federation/recipes.py::build_env` returns `ProdEnv`. `config/federation.py` documents the
three NVFLARE execution environments and which is used:

| environment | what it is | used here |
|---|---|---|
| `SimEnv` | clients as threads in one process; no PKI, no network | **never** for a reported number |
| `PocEnv` | separate processes, throwaway certificates | smoke tests only |
| **`ProdEnv`** | separate processes, real PKI startup kits, own ports, jobs through the admin API | **every reported result** |

```python
def build_env(n_clients: int):
    from nvflare.recipe import ProdEnv
    admin_dir = FED.startup_kit(FED.ADMIN_USER)
    return ProdEnv(startup_kit_location=str(admin_dir), username=FED.ADMIN_USER)
```

Confirmed from `production/logs/test06/server.log` — real listeners, real per-client TLS
connections, separate PIDs:

```
CoreCell - server: creating listener on http://0:8002
CoreCell - server: created backbone external listener for http://0:8002
CoreCell - server: creating listener on http://0:8003
conn_manager - Connection [CN00002 127.0.0.1:8002 <= 127.0.0.1:37334 SSL hospital_4] is created
ClientManager - Client: New client hospital_2@10.129.201.2 joined. Total clients: 1
```

---

## 3. Server configuration

| item | value | source |
|---|---|---|
| name / type / org | `server` / `server` / `ips` | `production/project.yml` |
| `fed_learn_port` | **8002** — clients receive tasks and return updates | `project.yml`, `config/federation.py` |
| `admin_port` | **8003** — admin API submits and monitors jobs | same |
| `default_host` | `localhost` | `project.yml` |
| heartbeat timeout | 600 s | `logs/test06/server.log` |
| workspace | `production/workspace/breast_fl_project/prod_00/server` | `server.log` |
| internal listener | `tcp://localhost:40575` | `server.log` |
| launch command | `python -m nvflare.private.fed.app.server.server_train -m <workspace>/server -s fed_server.json --set secure_train=true org=ips config_folder=config` | `logs/fedopt_overnight.log` |

The two ports are kept separate so a hospital firewall can expose only the first.

The server holds no patient images. It **does** hold the global test set — a benchmarking
decision so that all nine experiments are scored on identical ground, not a claim about
deployment, and the dissertation must say so.

---

## 4. Client configuration

Four clients are provisioned in every case, so certificates are identical across all
experiments and a difference between two results can never be a difference in PKI. The
2- and 3-client experiments use a subset of the same kits.

```yaml
- {name: hospital_1, type: client, org: h1}
- {name: hospital_2, type: client, org: h2}
- {name: hospital_3, type: client, org: h3}
- {name: hospital_4, type: client, org: h4}
```

Each client is a **separate OS process** started from its own PKI startup kit by
`scripts/start_federation.sh`, with `OMP_NUM_THREADS=1` exported for every child.

Runtime, from `results/test06_fedavg_4h/sites/train.log`:

```
=== hospital_3 | data=/root/tese/deployment/data/partitions/4_clients_balanced/hospital_3 ===
hospital_3:
  train  2,426 slices /  305 patients  per-class [154, 82, 69]
  val      608 slices /   76 patients  per-class [39, 20, 17]
device=cuda amp=True fedprox_mu=0.0 local_epochs=1
resnet18: 11,178,051 parameters, 10,494,979 trainable (93.89%), 683,072 frozen
architecture fingerprint 2d3031acc2075813
class weights (local, per patient): [0.66, 1.24, 1.473]
```

Each site holds out **20%** of its own patients locally (`local_val_fraction = 0.2`) to
produce the metric the server selects on. A site cannot validate on another site's
patients.

**Admin identity: `admin@ips.pt`, role `project_admin`.** It could not be renamed to
`admin`: NVFLARE 2.8 validates admin names against
`^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` in
`nvflare/apis/utils/format_check.py`, and both `admin` and `admin@ips` exit
`INVALID_ARGS` at provisioning time.

---

## 5. Federated algorithm / aggregator

| tests | algorithm | recipe class | client mu |
|---|---|---|---:|
| 02, 04, 06, 08 | **FedAvg** | `nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe` | — |
| 03, 05, 07, 09 | **FedProx** | **the same recipe, the same server-side aggregation** | **0.01** |
| 10–13 | **FedOpt** | `nvflare.app_opt.pt.recipes.fedopt.FedOptRecipe` | 0 |

**FedProx changes nothing on the server.** It adds `mu/2 · ‖w_local − w_global‖²` to the
local loss. The entire difference between test06 and test07 is one number passed to the
client. `build_recipe` refuses to build a FedProx job with `mu <= 0`, and refuses a
non-zero mu for `fedavg` or `fedopt`, because a client that receives the coefficient and
ignores it would be running FedAvg while the results table says FedProx.

**FedOpt** is a server-side optimiser: the mean client delta is treated as a
pseudo-gradient and the server takes an SGD step.

```python
FedOptRecipe(
    optimizer_args={"path": "torch.optim.SGD",
                    "args": {"lr": 1.0, "momentum": 0.6},
                    "config_type": "dict"},
    device="cpu",
    **common,
)
```

SGD at lr 1.0 with momentum 0 **is** FedAvg exactly, so the momentum is the whole of the
difference. Clients are untouched (mu = 0), which makes FedOpt orthogonal to FedProx
rather than an alternative.

**The PyTorch recipe is used, not the generic one.** `nvflare.recipe.FedAvgRecipe` accepts
`model` only as a dict and rejects an `nn.Module` outright. Passing a built object is what
makes the server's copy the same object the clients construct — this project once shipped
a server that built a ResNet-18 while every client built a ResNet-50, and the run
completed with meaningless numbers.

**Aggregation weighting:** by `NUM_STEPS_CURRENT_ROUND = n_patients` — patients, not
slices. A site whose patients happen to have larger tumours contributes more slices
without holding more evidence.

**FedOpt has no server-side model selection.** `FedOptRecipe` rejects `key_metric`
(`TypeError` at build time), so `common.pop("key_metric", None)` is applied and it keeps
the **last** round while FedAvg/FedProx keep the **best**. Any FedOpt-vs-FedAvg comparison
is therefore not like-for-like on model selection, and must be reported with that stated.
Tests 10–13 were cancelled before completion.

---

## 6. Number of clients

| tests | clients |
|---|---:|
| 02, 03 | 2 |
| 04, 05 | 3 |
| 06, 07, 08, 09 | 4 |

`min_clients` is set to the partition's `n_clients`. Four are always provisioned.

**100% participation per round, confirmed** from the retained global models:
`meta_props.nr_aggregated` equals the client count in every case — 2, 2, 3, 3, 4, 4, 4, 4.

Test01 (centralised) is **not an NVFLARE job**; it runs through
`scripts/run_centralized.py`.

---

## 7. Communication rounds

```python
num_rounds          = 30
local_epochs        = 1
centralized_epochs  = 30
```

**30 rounds × 1 local epoch = 30 epochs of data**, budget-matched to the centralised
baseline. Without that, RQ1 would read a difference in compute as a difference in
federation. `scripts/verify_production.py` asserts the equality as a pre-flight check.

Confirmed: `sites/rounds.csv` holds rounds **0–29** for every site in every completed
federated test.

---

## 8. Local training configuration

Identical for the centralised and the federated arms — both run literally the same
trainer, because `src/training.py` delegates to
`src/core/training.py`. The gap RQ1 measures is therefore federation, not
a difference in code.

| parameter | value |
|---|---|
| optimiser | AdamW |
| learning rate | 1e-4 |
| weight decay | 5e-4 |
| batch size | 24 |
| dropout | 0.5 |
| label smoothing | 0.1 |
| scheduler | cosine |
| loss | cross-entropy, class-weighted (inverse frequency, counted per **patient**) |
| `class_weight_scope` | `local` (each hospital's own rows) |
| `max_slices_per_patient_per_batch` | 1 |
| patient-level aggregation | mean of slice probabilities |
| seed | 42 (one run per job) |
| mixed precision | True |
| `num_workers` | 8 |
| image size | 224 × 224 RGB |
| classes | 3 |

**The learning-rate schedule is evaluated in closed form.** A federated client is
re-instantiated each round and holds no state, so a `CosineAnnealingLR` object cannot
survive to be stepped. Because the server sends `current_round` with every model, the
client computes

```
lr(r) = base · (1 + cos(π · r / T)) / 2
```

which is exactly what `CosineAnnealingLR(T_max=T)` holds at epoch `r` centrally. Both arms
follow the same curve.

**The reported loss excludes the proximal term.** Including it would make FedAvg and
FedProx losses incomparable across the very curves RQ3 is read from.

---

## 9. Model

**ResNet-18**, torchvision, ImageNet-pretrained.

| item | value |
|---|---|
| total parameters | **11,178,051** |
| trainable | 10,494,979 (93.89%) |
| frozen | 683,072 (`freeze_until = "layer3"` → conv1 + bn1 + layer1 + layer2) |
| `freeze_bn` | False |
| head | `Sequential(Dropout(0.5), Linear(512, 3))` |
| input | 224 × 224 RGB |
| output | 3 logits |
| **architecture fingerprint** | **`2d3031acc2075813`** |

The fingerprint is checked at the server **and** at every client, and every checkpoint
loads with `strict=True`. A parameter count proves nothing about the head — `Dropout` has
no parameters — and `strict=False` "succeeds" while leaving the classifier at random init,
which on this task still reads as a plausible near-chance result.

**The model is passed to the recipe as a built `nn.Module`** — specifically
`src/models.py::FederatedClassifier`, a thin wrapper that stores its constructor arguments
as plain instance attributes and delegates `state_dict` / `load_state_dict` to the inner
network. A bare torchvision ResNet cannot be serialised by NVFLARE (`self._norm_layer` is
a *class*, not an instance: `TypeError: Object of type type is not JSON serializable`) and,
without the wrapper, would be rebuilt on the server as a **default 1000-class** model
against 3-class clients.

---

## 10. Global model saving / checkpointing

**Configured.** `key_metric = "val_balanced_accuracy"`, computed by each client on
**held-out** patients — never training accuracy. A previous iteration of this project
reported training accuracy to the server, which then selected whichever global model let
clients memorise their own shard best (99%+). Balanced accuracy rather than macro-AUC
because a site holding few patients can draw a validation split missing a class, making
macro-AUC NaN.

NVFLARE's PyTorch persistor writes two files, and `collect_results.py::find_global_model`
prefers the first and records which it scored:

```python
GLOBAL_MODEL_NAMES = [
    "best_FL_global_model.pt",   # the model selected by key_metric
    "FL_global_model.pt",        # the last round
]
```

**Confirmed from the retained files themselves.** Each
`production/results/testNN_*/global_model.pt` (44.8 MB) is an NVFLARE persistor file with
keys `model`, `meta_props`, `train_conf`:

| test | `nr_aggregated` | `meta_props.current_round` |
|---|---:|---:|
| test02_fedavg_2h | 2 | 24 |
| test03_fedprox_2h | 2 | 28 |
| test04_fedavg_3h | 3 | 26 |
| test05_fedprox_3h | 3 | 27 |
| test06_fedavg_4h | 4 | 1 |
| test07_fedprox_4h | 4 | 21 |
| test08_fedavg_skewed | 4 | 20 |
| test09_fedprox_skewed | 4 | 1 |

**None is round 29.** Had these been last-round models, every one would read 29. All eight
are therefore the `key_metric`-selected checkpoints.

**Two caveats that belong in the write-up.**

1. The summary's `model_used` column reads `missing` for the federated rows. That is a
   local path artefact, not a missing model: `job.json` records `job_workspace` under
   `/root/tese/...` on the RunPod host, which was released. The scoring that produced
   `test_metrics.json` ran on the pod, where the file resolved.
2. The summary's `best_round` column (25, 29, 27, 28, 0, 0, 21, 2) is derived post-hoc by
   `build_final_summary.py` as a mean across sites, and is **not** the round NVFLARE
   saved. For test07 the two disagree materially — summary 0, persistor 21. **The
   persistor value is the authoritative record of what was scored.**

**Centralised arm:** `results/test01_centralized/seed_42/best_model.pt`, selected on
validation **macro-AUC**, with early stopping disabled so the whole curve is visible.

---

## 11. Communication protocol and data exchanged

**Transport**, from `production/logs/test06/server.log`: backbone external listeners on
`http://0:8002` and `http://0:8003`, an internal listener on `tcp://localhost:40575`, and
per-client connections marked `SSL`. Mutual TLS is built on certificates issued by
`CertBuilder`; the startup kits are signed by `SignatureBuilder` so tampering is
detectable.

**Server → client**, via `flare.receive()`: an `FLModel` carrying `params` (the global
weights) and `current_round`.

**Client → server**, via `flare.send()`:

```python
flare.send(flare.FLModel(
    params  = {k: v.detach().cpu() for k, v in model.state_dict().items()},
    metrics = {"train_accuracy": acc, "train_loss": loss,
               "val_accuracy": ..., "val_balanced_accuracy": ...,
               "val_macro_f1": ..., "val_auc": ...},
    meta    = {"NUM_STEPS_CURRENT_ROUND": n_patients},
))
```

**No image ever leaves a hospital.** Each client reads only
`data/partitions/<partition>/<hospital>/{train,val}.csv` and its own `images/`. Only model
weights and scalar metrics cross the boundary.

**A finite-weight guard runs before every send.** Any non-finite tensor raises `SystemExit`
rather than being transmitted:

> FedAvg sums tensors position by position. One site sending NaN or Inf poisons every
> position it touches, and the global model is NaN from that round on — with nothing in
> the server log saying which site did it, or that anything happened at all. The run
> completes and every number after it is meaningless.

---

## 12. Important configuration parameters

```python
# config/experiments.py :: FederationConfig
num_rounds          = 30
local_epochs        = 1
centralized_epochs  = 30
fedprox_mu          = 0.01
fedopt_lr           = 1.0
fedopt_momentum     = 0.6
key_metric          = "val_balanced_accuracy"
local_val_fraction  = 0.2

# config/experiments.py :: TrainingConfig
model_name = "resnet18" ; pretrained = True ; num_classes = 3 ; image_size = 224
batch_size = 24 ; learning_rate = 1e-4 ; weight_decay = 5e-4
dropout = 0.5 ; label_smoothing = 0.1
optimizer = "adamw" ; scheduler = "cosine"
freeze_until = "layer3" ; freeze_bn = False
class_weighted_loss = True ; class_weight_scope = "local"
max_slices_per_patient_per_batch = 1 ; aggregation = "mean"
seed = 42 ; num_workers = 8 ; mixed_precision = True
```

**Partitions** — patient-level, deterministic (seed 42), hardlinked to the source PNGs and
verified by inode:

| partition | clients | ratio | patients per site |
|---|---:|---|---|
| `2_clients_balanced` | 2 | 1:1 | 393 / 391 |
| `3_clients_balanced` | 3 | 1:1:1 | 262 / 262 / 260 |
| `4_clients_balanced` | 4 | 1:1:1:1 | 198 / 196 / 195 / 195 |
| `4_clients_skewed` | 4 | **5:2:1:1** → 55.6 / 22.2 / 11.1 / 11.1% | 435 / 175 / 87 / 87 |

The dissertation describes the skewed case as "50/20/10/10", which sums to 90 rather than
100. It is a **5:2:1:1 ratio**; `Partition` stores the ratio and normalises in code, so the
wording and the behaviour agree instead of silently differing by ten percent.

**⚠ All four partitions are stratified.** Every hospital keeps the global class ratio —
maximum measured spread across hospitals **0.4 percentage points**. Between hospitals only
the *quantity* of data varies. **Tests 08 and 09 therefore measure quantity skew, not
label or feature non-IID heterogeneity.** This is a deliberate, documented limitation and
must be stated. Two genuine alternatives are implemented and neither is the default:

```bash
python scripts/partition_data.py --stratify none     # label skew
python scripts/partition_data.py --by-cohort         # one real cohort per hospital
```

**Evaluation** is identical for all nine tests: the selected global model is scored on the
same global test set (268 patients, 2,115 images, trivial baseline **0.5112**), with slice
probabilities averaged per patient first.

**Provisioning builders**, all four confirmed present in `project.yml`:
`WorkspaceBuilder` · `StaticFileBuilder` · `CertBuilder` · `SignatureBuilder`.

---

## 13. Relevant files and where they are

| path | role |
|---|---|
| `src/federated/config/experiments.py` | **single source of truth** — 13 experiments, 4 partitions, `TrainingConfig`, `FederationConfig` |
| `src/federated/config/federation.py` | the only file that knows hosts and ports; resolves the current `prod_NN` |
| `deployment/project.yml` | NVFLARE provisioning — participants, ports, builders |
| `src/federated/federation/recipes.py` | `build_recipe()` (FedAvg / FedProx / FedOpt), `build_env()` → `ProdEnv` |
| `src/federated/federation/client.py` | the client loop: `flare.init` / `receive` / `send` |
| `src/federated/common/models.py` | `FederatedClassifier`, `architecture_fingerprint` |
| `src/federated/common/training.py` | delegates to `src/core/training.py` |
| `deployment/workspace/breast_fl_project/prod_00/` | PKI startup kits: `server/`, `hospital_1..4/`, `admin@ips.pt/` |
| `deployment/jobs/testNN_*/job.py` | generated from `experiments.py`; `generate_jobs.py --check` fails on drift |
| `results/federated/testNN_*/` | `job.json`, `global_model.pt`, `test_metrics.json`, `predictions_test.csv`, `sites/rounds.csv`, `sites/train.log` |
| `results/federated/final_summary/` | `summary.{csv,xlsx,json,md,pdf}`, 8 comparison tables, 9 LaTeX tables, figures |
| `deployment/logs/testNN/` | `server.log`, `hospital_N.log`, `admin.log`, `timeline.log`, `pids` |
| `src/scripts/verify_production.py` | 198 pre-flight checks; writes nothing |
| `src/scripts/run_experiment.py` | submits one job through the admin API |
| `src/scripts/collect_results.py` | scores every model on the one global test set |
| `src/scripts/start_federation.sh` | server first, poll the admin port, then the hospitals |

One log file **per participant**, never a shared one: two participants appending to one
log interleave mid-line under load and the result cannot be reconstructed.
`timeline.log` timestamps federation-level events so the order of events *across*
participants is recoverable.

---

## 14. Example job record

`production/results/test06_fedavg_4h/job.json`, verbatim:

```json
{
  "experiment": "test06",
  "name": "test06_fedavg_4h",
  "job_id": "9efefb2e-42c6-41bc-922e-fb98271f568d",
  "algorithm": "fedavg",
  "partition": "4_clients_balanced",
  "n_clients": 4,
  "num_rounds": 30,
  "local_epochs": 1,
  "fedprox_mu": 0.0,
  "key_metric": "val_balanced_accuracy",
  "submitted": "2026-08-04T00:35:44+00:00",
  "status": "FINISHED:COMPLETED",
  "job_workspace": ".../prod_00/admin@ips.pt/transfer/9efefb2e-42c6-41bc-922e-fb98271f568d",
  "finished": "2026-08-04T00:41:01+00:00"
}
```

---

## 15. How to reproduce

```bash
cd federated

# once
./production/scripts/provision.sh
./production/scripts/distributions.sh
./production/scripts/verify.sh          # 198 checks; must pass before anything starts

# the centralised baseline — NOT an NVFLARE job
python scripts/run_centralized.py --seed 42

# one federated experiment
./production/scripts/start.sh 4 test06  # server + 4 hospitals, separate processes
./production/scripts/run.sh test06      # submit through the admin API
./production/scripts/stop.sh

# after the runs
./production/scripts/collect.sh         # score every model on the one test set
./production/scripts/summary.sh         # build results/final_summary/
```

Client count per test: 2 for test02–03, 3 for test04–05, 4 for test06–09.

**Provisioning never overwrites.** Each run creates the next `prod_NN` beside the previous
one. Server and clients must all start from the *same* `prod_NN` or the TLS handshake
fails with an error that never mentions provisioning. Everything resolves it through
`config/federation.py::workspace_dir()`, which always picks the highest. Currently:
**`prod_00`**.
