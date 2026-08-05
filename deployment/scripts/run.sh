#!/usr/bin/env bash
# Submit one experiment: ./run.sh test06
# Thin wrapper — execs ../../scripts/run_experiment.py and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$PROJECT_ROOT/scripts/run_experiment.py" "$@"
