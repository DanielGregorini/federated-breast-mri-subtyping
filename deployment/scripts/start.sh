#!/usr/bin/env bash
# Start server + N hospitals: ./start.sh 4 test06
# Thin wrapper — execs ../../scripts/start_federation.sh and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/start_federation.sh" "$@"
