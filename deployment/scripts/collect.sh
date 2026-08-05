#!/usr/bin/env bash
# Score finished models on the global test set.
# Thin wrapper — execs ../../scripts/collect_results.py and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$PROJECT_ROOT/scripts/collect_results.py" "$@"
