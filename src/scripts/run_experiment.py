#!/usr/bin/env python3
"""Submit one experiment to the running federation and wait for it.

    python scripts/run_experiment.py test06
    python scripts/run_experiment.py test06 --dry-run

Everything this script prints is also appended to
`production/logs/<test>/admin.log`, timestamped. That file is the admin side of the
record the dissertation reconstructs a run from — the server and each hospital write
their own files beside it, from `start_federation.sh`.

The federation must already be running — see `start_federation.sh`. This script
submits through the ADMIN API, exactly as a coordinating centre would, using the
admin identity's own certificate. Nothing here writes a job config by hand: the
recipe builds it, because editing generated JSON is how this project once ended up
with a server building a different architecture from its clients.

TEST 01 IS NOT AN NVFLARE JOB
-----------------------------
It is the centralised baseline. `run_centralized.py` runs it. The dispatch below
sends you there rather than failing with an NVFLARE error about a missing recipe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
# The sibling scripts live beside this file, NOT under federated/. Deriving them
# from PROJECT_ROOT points at a directory that does not exist, and the subprocess
# then fails with "file not found" — which this script reports as a data
# verification failure, a long way from its real cause.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX   # noqa: E402
from config import federation as FED   # noqa: E402
from federation import recipes         # noqa: E402


class _Tee:
    """Write to the terminal AND to the admin log, timestamping each line.

    Terminal output is not evidence: it scrolls, it is lost when the shell closes,
    and it cannot be attached to a dissertation. Every line therefore lands in
    `production/logs/<test>/admin.log` as well, with a UTC timestamp, so the admin
    side of a run is reconstructable months later.
    """

    def __init__(self, stream, path: Path) -> None:
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", buffering=1)
        self._start_of_line = True

    def write(self, text: str) -> int:
        self.stream.write(text)
        for i, part in enumerate(text.split("\n")):
            if i:
                self.fh.write("\n")
                self._start_of_line = True
            if part:
                if self._start_of_line:
                    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.fh.write(f"{stamp}  ")
                    self._start_of_line = False
                self.fh.write(part)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()


def preflight(experiment) -> None:
    """Refuse to submit a job whose data is not there or is not clean.

    A federated run that fails at round 0 because a hospital folder is missing costs
    a start-up cycle; one that succeeds on leaking data costs a chapter.
    """
    partition = EX.PARTITIONS[experiment.partition]
    part_dir = EX.PARTITIONS_DIR / partition.name
    missing = [s for s in partition.client_names
               if not (part_dir / s / "train.csv").is_file()]
    if missing:
        raise SystemExit(
            f"{experiment.id} needs {partition.name}, but these sites have no "
            f"data: {', '.join(missing)}\n"
            "  run: python scripts/partition_data.py")

    if not (EX.GLOBAL_DIR / "test.csv").is_file():
        raise SystemExit("no global test set — run: python scripts/prepare_data.py")

    check = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "verify_data.py"),
         "--only", partition.name],
        capture_output=True, text=True)
    if check.returncode != 0:
        print(check.stdout)
        raise SystemExit(f"data verification failed for {partition.name}. "
                         "Not submitting.")
    print(f"  data verified: {partition.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", help="test02 ... test09, or a folder name")
    p.add_argument("--dry-run", action="store_true",
                   help="build and describe the job without submitting it")
    p.add_argument("--timeout", type=float, default=None,
                   help="seconds to wait before giving up on the job")
    args = p.parse_args()

    experiment = EX.get(args.experiment)

    # A dry run is a check, not a run, so it does not append to the experiment's
    # admin log — otherwise the record of test06 would contain submissions that
    # never happened.
    if not args.dry_run:
        log = EX.LOGS_DIR / experiment.id / "admin.log"
        sys.stdout = _Tee(sys.stdout, log)
        print(f"\n===== run_experiment.py {experiment.id} "
              f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC) =====")

    print("=" * 74)
    print(recipes.describe(experiment))
    print("=" * 74)

    if experiment.kind == "centralized":
        raise SystemExit(
            f"{experiment.id} is the centralised baseline, not an NVFLARE job.\n"
            "  run: python scripts/run_centralized.py")

    preflight(experiment)
    recipe = recipes.build_recipe(experiment)
    print(f"  recipe built: {experiment.algorithm}, "
          f"{EX.FEDERATION.num_rounds} rounds, "
          f"min_clients={experiment.n_clients}")

    if args.dry_run:
        print("\ndry run — nothing submitted.")
        print(f"  clients expected : {', '.join(FED.client_names(experiment.n_clients))}")
        print(f"  train args       : {recipes.train_args_for(experiment)}")
        return

    env = recipes.build_env(experiment.n_clients)
    print(f"  admin            : {FED.ADMIN_USER}")
    print(f"  server           : {FED.SERVER_HOST}:{FED.ADMIN_PORT}")
    print("\nsubmitting...")

    run = recipe.execute(env)
    job_id = run.get_job_id()
    print(f"  job id: {job_id}")

    out_dir = EX.RESULTS_DIR / experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps({
        "experiment": experiment.id, "name": experiment.name,
        "job_id": job_id, "algorithm": experiment.algorithm,
        "partition": experiment.partition, "n_clients": experiment.n_clients,
        "num_rounds": EX.FEDERATION.num_rounds,
        "local_epochs": EX.FEDERATION.local_epochs,
        "fedprox_mu": (EX.FEDERATION.fedprox_mu
                       if experiment.algorithm == "fedprox" else 0.0),
        "key_metric": EX.FEDERATION.key_metric,
        "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print("  waiting for the job to finish (Ctrl-C detaches, it keeps running)")
    try:
        # timeout=0.0 means "no timeout" in NVFLARE's API, not "return immediately".
        # clean_up=False keeps the run workspace — it holds the server and client
        # logs, and deleting them on success is exactly when you most want them
        # later, because a run that finished is a run somebody will ask about.
        workspace = run.get_result(timeout=args.timeout or 0.0, clean_up=False)
    except KeyboardInterrupt:
        print(f"\ndetached. The job is still running as {job_id}.")
        print(f"  reattach with the admin console, or abort it, but do NOT start "
              f"another experiment while it holds the GPU.")
        return

    status = run.get_status()
    print(f"\nstatus: {status}")
    if workspace:
        print(f"job workspace: {workspace}")

    record = json.loads((out_dir / "job.json").read_text())
    record.update({
        "status": str(status),
        "job_workspace": str(workspace) if workspace else None,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (out_dir / "job.json").write_text(json.dumps(record, indent=2))

    if status is not None and "FINISHED" not in str(status).upper():
        raise SystemExit(
            f"job did not finish cleanly ({status}). "
            f"Check results/_federation_logs/ and the job workspace above.")

    print(f"results under: {out_dir}")
    print("next: python scripts/collect_results.py")


if __name__ == "__main__":
    main()
