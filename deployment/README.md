# deployment — the running system

What NVIDIA FLARE needs in order to run, and what it produces while running. No
hyperparameter is defined here: `jobs/` is generated from
`src/src/federated/config/experiments.py`, `config/` is a snapshot nothing reads back, and
`scripts/` are thin wrappers.

| Folder | What it holds |
|---|---|
| `workspace/` | The PKI startup kits, one per participant, from `nvflare provision`. **Contains private keys and is never committed.** |
| `jobs/` | Thirteen generated job folders, each a `job.py` and a `README.md` |
| `data/` | The per-hospital datasets: `global/` for the shared validation and test sets, `partitions/` for the six splits. Images are hardlinks into `dataset/` |
| `logs/` | One folder per experiment, one log file per participant, plus `timeline.log` |
| `datasets/` | Split manifests and provenance: `all_distributions.csv`, the per-experiment distribution tables |
| `figures/` | Distribution figures, three overviews and one per experiment |
| `config/` | `resolved_config.{json,md}` — a record of what the configuration was at run time |
| `scripts/` | `provision.sh`, `start.sh`, `run.sh`, `stop.sh`, `verify.sh`, `collect.sh`, `summary.sh` |
| `project.yml` | The NVFLARE provisioning file: participants, ports, builders |

Results are **not** here. They live at the repository root under `results/federated/`,
beside the classifier-phase runs, so a reader looking for results does not have to know
what a deployment is.

## The participants

| Name | Type | Org | Role |
|---|---|---|---|
| `server` | server | ips | Aggregates updates and selects the global model. Holds no patient images |
| `hospital_1` .. `hospital_4` | client | h1 .. h4 | Each holds its own patients and never shares an image |
| `admin@ips.pt` | admin | ips | Submits jobs, monitors them, downloads results |

Four hospitals are provisioned even for the two- and three-site experiments, so a
difference between two results can never be a difference in PKI.

The admin name must be a full e-mail address with a top-level domain: NVFLARE validates
it against a regex in `nvflare/apis/utils/format_check.py`, and both `admin` and
`admin@ips` exit `INVALID_ARGS` at provisioning time.

## Provisioning never overwrites

Each run creates the next `prod_NN` beside the previous one. Server and clients must all
start from the same one, or the TLS handshake fails with an error that never mentions
provisioning. `config/federation.py::workspace_dir()` always picks the highest.
Currently `prod_00`.

Full instructions: [../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)
