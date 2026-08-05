#!/usr/bin/env python3
"""Run the nine experiments in order, one at a time.

    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --from test04
    python scripts/run_all_experiments.py --seeds 42 1

ONE AT A TIME, ON PURPOSE
-------------------------
Not because the code could not overlap them, but because it was measured: on one
GPU, two concurrent jobs gave 0.074 epochs/s, three gave 0.071 and seven gave 0.058.
Past two, CUDA context switching dominates and everything gets slower together.
A federated experiment already runs N client processes, so one experiment at a time
IS the parallel case.

THE FEDERATION IS RESTARTED BETWEEN CLIENT COUNTS
-------------------------------------------------
Tests 02-03 need two hospitals, 04-05 need three, 06-09 need four. A server started
for four clients will happily run a two-client job, but then `min_clients` is
satisfied by whichever two register first, which is not necessarily hospital_1 and
hospital_2. The experiments are grouped by client count and the federation is
restarted at each boundary, so every test uses the sites its partition names.

SEEDS — ONE RUN PER JOB, SEED 42
--------------------------------
Set deliberately, and the consequence has to travel with the results.

The noise floor on this task is 0.067 macro-AUC: two byte-identical configurations
differing only in random seed landed that far apart. The FedAvg-vs-FedProx difference
previously measured here was 0.004 to 0.021 — between four and seventeen times
SMALLER than that noise.

So with one seed per job, a difference between two experiments cannot be attributed
to the thing that differs between them. The runs are still worth having: they produce
the deployment, the curves, the per-hospital breakdown and the full metric set. What
they cannot do is rank FedAvg against FedProx, and the summary must not be written as
if they could. `build_final_summary.py` marks any gap under 0.067 as
`within_noise_floor`, which with a single seed will be most of them.

Pass `--seeds 42 1` to restore repeats if that changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX  # noqa: E402

SCRIPTS = PROJECT_ROOT / "scripts"


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print("$ " + " ".join(str(c) for c in cmd))
    started = time.time()
    result = subprocess.run([str(c) for c in cmd])
    mins = (time.time() - started) / 60
    if result.returncode != 0:
        print(f"\n!! {label} FAILED after {mins:.1f} min "
              f"(exit {result.returncode})")
        return False
    print(f"\n{label} finished in {mins:.1f} min")
    return True


def restart_federation(n_clients: int, test_id: str) -> bool:
    """Stop everything, then start a fresh federation whose logs belong to `test_id`.

    RESTARTED PER TEST, NOT PER CLIENT COUNT
    ----------------------------------------
    Tests 06-09 all use four hospitals, so one federation could serve all four. It is
    restarted for each anyway, for one reason: the server and hospital logs have to be
    attributable to a single experiment. A shared federation writes one server.log
    covering four tests, and "what did hospital_3 do during test08" then has to be
    reconstructed by timestamp from an interleaved file.

    A restart costs about twenty seconds against an experiment that takes an hour, and
    it changes nothing about the training: every NVFLARE job is independent, and a
    fresh server also guarantees no state carries from the previous test.
    """
    subprocess.run([str(SCRIPTS / "stop_federation.sh")],
                   capture_output=True)
    time.sleep(3)
    return run([SCRIPTS / "start_federation.sh", n_clients, test_id],
               f"starting federation with {n_clients} hospitals -> logs/{test_id}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", default=None,
                   help="resume from this experiment id")
    p.add_argument("--only", nargs="*", default=None, help="run just these ids")
    p.add_argument("--seeds", nargs="*", type=int, default=[EX.TRAINING.seed],
                   help="seeds for the centralised baseline (default: one run, "
                        "seed 42 — see the SEEDS note in this file's docstring)")
    p.add_argument("--skip-centralized", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a failure instead of stopping")
    args = p.parse_args()

    todo = EX.EXPERIMENTS
    if args.only:
        todo = [EX.get(i) for i in args.only]
    elif args.start:
        ids = [e.id for e in EX.EXPERIMENTS]
        todo = EX.EXPERIMENTS[ids.index(EX.get(args.start).id):]

    print(EX.summary())
    print(f"\nto run: {', '.join(e.id for e in todo)}")
    print(f"seeds (centralised): {args.seeds}")

    if args.dry_run:
        print("\ndry run — nothing executed.")
        return

    failures: list[str] = []
    current_clients = None
    started = time.time()

    for experiment in todo:
        if experiment.kind == "centralized":
            if args.skip_centralized:
                continue
            for seed in args.seeds:
                ok = run([sys.executable, SCRIPTS / "run_centralized.py",
                          "--seed", seed],
                         f"{experiment.id} — centralised baseline, seed {seed}")
                if not ok:
                    failures.append(f"{experiment.id}(seed {seed})")
                    if not args.keep_going:
                        break
            continue

        if not restart_federation(experiment.n_clients, experiment.id):
            failures.append(f"federation({experiment.n_clients} clients)")
            if not args.keep_going:
                break
            continue
        current_clients = experiment.n_clients

        ok = run([sys.executable, SCRIPTS / "run_experiment.py", experiment.id],
                 f"{experiment.id} — {experiment.objective}")
        if not ok:
            failures.append(experiment.id)
            if not args.keep_going:
                break

    subprocess.run([str(SCRIPTS / "stop_federation.sh")], capture_output=True)

    print("\n" + "=" * 74)
    print(f"finished in {(time.time() - started) / 60:.1f} min")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("=" * 74)
        sys.exit(1)
    print("all requested experiments completed")
    print("next: python scripts/collect_results.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
