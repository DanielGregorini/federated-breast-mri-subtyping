#!/usr/bin/env bash
# Dataset distribution figures and tables.
# Thin wrapper — execs ../../scripts/build_distribution_report.py and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$PROJECT_ROOT/scripts/build_distribution_report.py" "$@"
