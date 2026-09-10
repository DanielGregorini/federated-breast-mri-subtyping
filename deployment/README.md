# deployment

Everything NVIDIA FLARE needs in order to run, and everything it writes while running.

The federation is six operating-system processes on one or more machines: an
aggregation server, up to four hospital clients, and an admin identity that submits
jobs. Each has its own X.509 certificate and they talk over mutual TLS. This folder
holds their identities, their data, their jobs and their logs.

No hyperparameter is defined twice. Everything comes from
[`code/config/experiments.py`](code/config/experiments.py).

## What each folder is

| Folder | What it holds |
|---|---|
| `code/` | Everything the federation runs. The configuration table, the recipes, the client loop, the shared library and the operational scripts. |
| `workspace/` | One startup kit per participant, written by `nvflare provision`. Each holds that participant's certificate, private key and start script. Contains real private keys and is never committed. |
| `data/` | The images and manifests each participant reads. `global/` is the shared validation and test set, `partitions/` is the six per-hospital splits at each of the three seeds. |
| `jobs/` | Thirty-six complete NVFLARE jobs: twelve federated experiments at three seeds. Each carries the code the hospitals run. Generated, never written by hand. |
| `project.yml` | The provisioning file. Participants, organisations, ports and builders. |

Those six things are everything the federation needs. Copy this folder to a machine,
start the participants, submit a job. Nothing has to be prepared first.

A run writes into [`results/federated/`](../results/README.md), and creates `logs/`
here for the admin side of each submission.

## The participants

| Name | Type | Role |
|---|---|---|
| `server` | server | Aggregates the client updates and selects the global model. Holds no patient images. |
| `hospital_1` to `hospital_4` | client | Each holds its own patients and never sends an image anywhere. |
| `admin@ips.pt` | admin | Submits jobs, monitors them, downloads results. |

Four hospitals are provisioned even for the two- and three-site experiments, so the
certificates are the same in every test.

The admin name has to be a full e-mail address with a top-level domain. NVFLARE
validates it against a regex, and both `admin` and `admin@ips` fail provisioning with
`INVALID_ARGS`.

## How to run it

Run every command from the repository root.

### 1. Build the data

Notebook 07 writes `data/global/` and the six seed-42 partitions under
`data/partitions/`.

```bash
jupyter notebook notebooks/07_federated_setup.ipynb
```

The other two seeds re-deal the same six shapes. `global/` is not rebuilt: which
patients are training, validation and test comes from the BreastDCEDL metadata and
does not depend on a seed, so all three seeds are scored on the same test set.

```bash
python deployment/code/scripts/partition_data.py --seed 19 --suffix _s19 --hardlink
python deployment/code/scripts/partition_data.py --seed 50 --suffix _s50 --hardlink
```

### 2. Provision the identities

This writes `workspace/breast_fl_project/prod_NN/<identity>/startup/` for every
participant, each with its own certificate, private key and start script.

```bash
bash deployment/code/scripts/provision.sh
```

Provisioning never overwrites. Each run creates the next `prod_NN` beside the
previous one, and `code/config/federation.py` always resolves the highest.
Server and clients have to start from the same one or the TLS handshake fails with an
error that never mentions provisioning.

### 3. Generate the jobs

```bash
python deployment/code/scripts/generate_jobs.py --seed 19 --seed 50
```

Writes the thirty-six job folders under `jobs/` and copies them into the admin's
`transfer/` directory. Each job names its own partition folder and its own seed on
the client command line.

### 4. Start the federation

Open a terminal and point the hospital processes at this repository:

```bash
export BREAST_DATA_ROOT="$PWD/deployment/data"
export BREAST_RESULTS_ROOT="$PWD"
export FEDBREAST_ROOT="$PWD/deployment/code"
export BREAST_CORE_ROOT="$PWD/src"
export OMP_NUM_THREADS=1
```

`BREAST_DATA_ROOT` is how a client finds its own folder, and `BREAST_RESULTS_ROOT`
is where its per-round CSVs go. The job supplies the partition name and NVFLARE
supplies the site name, so those two variables serve every job and nothing has to be
restarted between them. On a machine that holds only its own hospital's data, point
`BREAST_DATA_ROOT` at whatever folder has `partitions/<name>/<site>/` underneath it.

The last two are only for running `federation/client.py` outside a job. A submitted
job carries its own code and ignores them.

Start the server, and wait until port 8003 is accepting connections:

```bash
deployment/workspace/breast_fl_project/prod_00/server/startup/start.sh
```

Then start one hospital per site. Start as many as the experiment needs, from
`hospital_1` upward:

```bash
deployment/workspace/breast_fl_project/prod_00/hospital_1/startup/start.sh
```

Each participant writes its own log inside its own workspace folder. Starting the
server first matters. A client that tries to register before the server is listening
retries with a backoff and delays the first round by up to a minute.

### 6. Submit an experiment

From the admin console, by name:

```
submit_job test10_fedavg_cohort
```

A seed replica is another name in the same list:

```
submit_job test10_fedavg_cohort_s19
```

Or from the shell, which also runs the pre-flight and writes `job.json`:

```bash
python deployment/code/scripts/run_experiment.py test10
``` Add `--dry-run` to
see what would be submitted without submitting it. The number of hospitals running has
to match the experiment, so switching between a 2-site and a 4-site test means
restarting the federation.

### 7. Collect the results

```bash
python deployment/code/scripts/collect_results.py
```

Scores every finished model on the global test set and writes one folder per
experiment into `results/federated/`.

```bash
python deployment/code/scripts/build_final_summary.py
```

Builds `results/federated/final_summary/` with the tables and figures that compare the
experiments against each other.

### 8. Stop the federation

Shut the participants down from the admin console. Each startup kit also ships a
`startup/stop_fl.sh`, which NVFLARE documents as a last resort because it does not
deregister the client from the server.

## Moving a hospital to its own machine

Change the hospital's address in `project.yml` and in
`code/config/federation.py`, re-provision, and copy that hospital's startup kit and
its folders under `data/partitions/` to the other machine. Copy with `rsync -aH`:
the images are hardlinks, and without `-H` every site expands into its own full set
of files.

That machine then sets `BREAST_DATA_ROOT` to wherever it put them. Nothing else
changes.

Full walkthrough, including troubleshooting: [../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)
