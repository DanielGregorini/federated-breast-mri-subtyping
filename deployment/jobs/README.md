# deployment/jobs

One complete NVFLARE job per federated experiment and per seed. Thirty-six folders:
`test02` to `test13` at seed 42, and the same twelve again at seeds 19 and 50,
named with an `_s19` / `_s50` suffix.

Each folder is a job the admin console can submit on its own. Everything the
hospitals run is inside it, so a job can be copied to a hospital machine and
submitted there without this repository.

```
test06_fedavg_4h/
├── README.md                             what this experiment is
├── meta.json                             name, min_clients, deploy_map
└── app/
    ├── config/config_fed_server.json     the controller, the rounds, the model
    │                                     selector and the persistor
    ├── config/config_fed_client.json     the executor and the client command line
    └── custom/                           the code that runs at each hospital
        ├── federation/client.py          receive, train one epoch, evaluate, send
        ├── config/                       experiments.py, federation.py
        ├── common/                       models, data, training, evaluation, thesis
        ├── core/                         the trainer the centralised arm also runs
        └── dataset_config.py
```

No path inside a job folder points outside it. `task_script_path` is
`federation/client.py`, which NVFLARE resolves by walking the job's own `custom/`
directory, and `--results-dir` is relative.

A job names the partition it reads and the seed it trains at on the client command
line in `config_fed_client.json`, so `test06_fedavg_4h_s19` reads
`partitions/4_clients_balanced_s19/` and passes `--seed 19`. Nothing else differs
between a job and its replicas.

Each hospital machine says where its data is, once, with two environment variables
read by `federation/client.py`:

| variable | what it points at |
|---|---|
| `BREAST_DATA_ROOT` | the `deployment/data` folder on that machine |
| `BREAST_RESULTS_ROOT` | where the per-round CSVs are written |

`BREAST_DATA_ROOT` covers every job, because the partition name comes from the job.
`BREAST_SITE_DIR` still overrides it with one exact site folder, but it pins the
machine to a single partition and seed and has to be changed between jobs. Set
neither and the client falls back to this repository's own layout, which works when
the checkout is on the machine and not otherwise.

The centralised baseline has no job folder here. It is not an NVFLARE job: one
machine, all the training data, no server and no clients.
`deployment/code/scripts/run_centralized.py` runs it.

## How to build them

```bash
python deployment/code/scripts/generate_jobs.py --seed 19 --seed 50
```

Writes all thirty-six folders from
[`deployment/code/config/experiments.py`](../code/config/experiments.py), and copies
them into the admin's `transfer/` directory, which is where `submit_job` looks names
up. Without `--seed` it writes the twelve seed-42 jobs only. Build the matching data
first, or a replica job will point at a partition folder that is not there:

```bash
python deployment/code/scripts/partition_data.py --seed 19 --suffix _s19 --hardlink
```

Do not edit a job folder by hand. Change the table and regenerate.

## How to submit one

Start the federation with as many hospitals as the experiment needs, then open the
admin console:

```bash
deployment/workspace/breast_fl_project/prod_00/admin@ips.pt/startup/fl_admin.sh
```

And submit it by name:

```
submit_job test06_fedavg_4h
submit_job test06_fedavg_4h_s19
submit_job test06_fedavg_4h_s50
```

The three run one after another on the same federation. Nothing has to be restarted
between them.

The scripted equivalent, which also runs the pre-flight, writes `job.json` and tees
the admin log:

```bash
python deployment/code/scripts/run_experiment.py test06
```

Output lands in `results/federated/<name>/`, collected by
`deployment/code/scripts/collect_results.py`.
