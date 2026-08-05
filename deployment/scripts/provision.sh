#!/usr/bin/env bash
# Generate the PKI startup kits.
# Thin wrapper — execs ../../scripts/provision.sh and adds nothing. See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
exec "$PROJECT_ROOT/scripts/provision.sh" "$@"
