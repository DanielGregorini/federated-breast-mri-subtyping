# deployment/jobs

One complete NVFLARE job per federated experiment. Twelve folders, `test02` to
`test13`.

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
        ├── pipelines/                    the preprocessing rules
        └── dataset_config.py
```

No path inside a job folder points outside it. `task_script_path` is
`federation/client.py`, which NVFLARE resolves by walking the job's own `custom/`
directory, and `--results-dir` is relative to the repository root the client finds
for itself.

The centralised baseline has no job folder here. It is not an NVFLARE job: one
machine, all the training data, no server and no clients.
`deployment/code/scripts/run_centralized.py` runs it.

## How to build them

```bash
python deployment/code/scripts/generate_jobs.py
```

Writes all twelve folders from
[`deployment/code/config/experiments.py`](../code/config/experiments.py), and copies
them into the admin's `transfer/` directory, which is where `submit_job` looks names
up.

```bash
python deployment/code/scripts/generate_jobs.py --check
```

Rebuilds each job into a temporary folder and compares it file by file. Exits
non-zero if one has drifted from the table. Do not edit a job folder by hand.

## How to submit one

Start the federation with as many hospitals as the experiment needs, then open the
admin console:

```bash
deployment/workspace/breast_fl_project/prod_00/admin@ips.pt/startup/fl_admin.sh
```

And submit it by name:

```
submit_job test06_fedavg_4h
```

The scripted equivalent, which also runs the pre-flight, writes `job.json` and tees
the admin log:

```bash
python deployment/code/scripts/run_experiment.py test06
```

Output lands in `results/federated/<name>/`, collected by
`deployment/code/scripts/collect_results.py`.
