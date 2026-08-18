# deployment/logs

What each participant wrote while a run was happening. One folder per experiment, one
file per participant.

| Path | What it is |
|---|---|
| `testNN/server.log` | the aggregation server |
| `testNN/hospital_N.log` | that hospital's own process |
| `testNN/admin.log` | what was submitted, when, and what came back |
| `testNN/timeline.log` | every event in order, across participants, UTC timestamped |
| `testNN/pids` | the process ids started for that run |

Terminal output scrolls and is lost. These files are what a run is reconstructed from
afterwards.

## Where they come from

Each participant's startup kit writes its own log. `deployment/code/scripts/run_experiment.py`
tees the admin side into `testNN/admin.log`.

Logs are not in version control.
