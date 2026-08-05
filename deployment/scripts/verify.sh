#!/usr/bin/env bash
# Full pre-flight verification. Starts nothing.
# Thin wrapper — execs ../../scripts/verify_production.py and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec python3 "$PROJECT_ROOT/scripts/verify_production.py" "$@"
