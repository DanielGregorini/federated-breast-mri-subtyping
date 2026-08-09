# deployment/logs — what each participant wrote

One folder per experiment. Inside each, one file per participant plus the admin log.
Terminal output scrolls and is lost; these files are the record a run is reconstructed
from months later, and each line carries a UTC timestamp.

| Path | What it is |
|---|---|
| `testNN/server.log` | the aggregation server |
| `testNN/hospital_N.log` | that hospital's own process |
| `testNN/admin.log` | the admin side: what was submitted, when, and what came back |
| `run_all.log` | the whole campaign driven by `run_all_experiments.py` |
| `rq2_run.log` | the cohort comparison, tests 10 to 13 |
| `fedopt_overnight.log` | the FedOpt attempt, implemented and cancelled. Not reported |

## Where they come from

`src/scripts/start_federation.sh` starts each participant with its own log target, and
`src/scripts/run_experiment.py` tees the admin side into `testNN/admin.log`.

Logs are not in version control. They are kept locally because a claim about a real
deployment is only as good as the evidence that it ran.
