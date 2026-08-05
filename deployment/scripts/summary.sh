#!/usr/bin/env bash
# Build results/final_summary/.
# Thin wrapper — execs ../../scripts/build_final_summary.py and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$PROJECT_ROOT/scripts/build_final_summary.py" "$@"
