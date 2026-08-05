#!/usr/bin/env bash
# The four FedOpt experiments on this machine, then the full collection.
#
# Unattended by design: it must finish with results ready to read, or say plainly
# why it did not. Every step logs, nothing is silent, and a failure in one
# experiment does not stop the others (--keep-going).
#
# CPU, because Apple MPS produces NaN through this project's training loop (see
# core/training.py::get_device). ~2.4 h per experiment, ~10 h for the four.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

LOG="$ROOT/production/logs/fedopt_overnight.log"
mkdir -p "$(dirname "$LOG")"
stamp() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

stamp "=== FedOpt: test10..test13 on CPU ==="
stamp "device: $(python3 -c 'import sys;sys.path.insert(0,".");from src import models as M;print(M.get_device())')"

# Refuse to start on a dirty setup rather than discover it at round 12.
if ! python3 scripts/verify_production.py >>"$LOG" 2>&1; then
    stamp "PRE-FLIGHT FAILED — nothing started. See $LOG"
    exit 1
fi
stamp "pre-flight passed"

python3 -u scripts/run_all_experiments.py \
        --only test10 test11 test12 test13 --keep-going >>"$LOG" 2>&1
STATUS=$?
stamp "run_all_experiments exited $STATUS"

# Collect whatever finished, even if something failed — partial results beat none.
stamp "collecting results"
python3 -u scripts/collect_results.py >>"$LOG" 2>&1 || stamp "collect_results failed"
stamp "building final summary"
python3 -u scripts/build_final_summary.py >>"$LOG" 2>&1 || stamp "build_final_summary failed"

DONE=$(ls -d production/results/test1[0-3]_* 2>/dev/null | wc -l | tr -d ' ')
stamp "=== FINISHED: $DONE of 4 FedOpt experiments have a results folder ==="
stamp "summary: production/results/final_summary/summary.md"
