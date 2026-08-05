#!/usr/bin/env bash
# Stop every participant.
# Thin wrapper — execs ../../scripts/stop_federation.sh and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/stop_federation.sh" "$@"
