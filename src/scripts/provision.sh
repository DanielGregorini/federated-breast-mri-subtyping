#!/usr/bin/env bash
# Generate the PKI startup kits — one signed identity per participant.
#
#     ./scripts/provision.sh
#
# This is what makes the deployment real rather than simulated. Each participant
# gets its own certificate and private key, and every connection between them is
# mutually authenticated TLS. NVFLARE's simulator has none of this, which is why no
# number in this dissertation comes from it.
#
# WHAT IT WRITES
#     production/workspace/breast_fl_project/prod_NN/
#         server/       start.sh, certificate, fed_server.json
#         hospital_1/   ... hospital_4/
#         admin@ips.pt/ the identity that submits jobs
#
# This is exactly the command the thesis documents:
#     nvflare provision -p production/project.yml -w production/workspace
#
# PROVISION ONCE. Every re-run creates the NEXT prod_NN and leaves the previous one
# in place, so certificates are never destroyed — but it also means the server and
# the clients must all be started from the SAME prod_NN or the TLS handshake fails
# with an error that never mentions provisioning. Everything in this project
# resolves that folder through config/federation.py::workspace_dir(), which always
# picks the highest.
#
# Four hospitals are provisioned even though tests 02-05 use two or three. The
# smaller experiments use a subset of the same kits, so a difference between two
# results can never be a difference in PKI.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$HERE")"
PRODUCTION_DIR="$PROJECT_ROOT/production"
WORKSPACE_DIR="$PRODUCTION_DIR/workspace"
PROJECT_YML="$PRODUCTION_DIR/project.yml"

if ! command -v nvflare >/dev/null 2>&1; then
    echo "nvflare is not on PATH. Install it with:  pip install nvflare" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_YML" ]]; then
    echo "missing $PROJECT_YML" >&2
    exit 1
fi

echo "======================================================================"
echo "PROVISIONING — $(nvflare --version 2>/dev/null || echo nvflare)"
echo "  project : $PROJECT_YML"
echo "  output  : $WORKSPACE_DIR"
echo "======================================================================"

mkdir -p "$WORKSPACE_DIR"
nvflare provision -p "$PROJECT_YML" -w "$WORKSPACE_DIR"

# Find the run that was just created — the highest prod_NN.
LATEST="$(find "$WORKSPACE_DIR" -maxdepth 2 -type d -name 'prod_*' | sort -V | tail -1)"
if [[ -z "$LATEST" ]]; then
    echo "provisioning produced no prod_NN folder" >&2
    exit 1
fi

echo
echo "startup kits in: $LATEST"
MISSING=0

# Read the expected participants from config/federation.py rather than listing them
# here. A hard-coded admin name that drifts from the config produces a provisioning
# run that "succeeds" and a run_experiment.py that then fails looking for a startup
# kit which was never generated — a confusing failure a long way from its cause.
# `while read` rather than `mapfile`: macOS ships bash 3.2, where mapfile does not
# exist, and this script must run on the laptop as well as on the GPU box.
EXPECTED=""
while IFS= read -r line; do
    EXPECTED="$EXPECTED$line"$'\n'
done < <(cd "$PROJECT_ROOT" && python3 -c '
from config.federation import SERVER_NAME, HOSPITALS, ADMIN_USER
print(SERVER_NAME)
for h in HOSPITALS:
    print(h.name)
print(ADMIN_USER)
')

while IFS= read -r participant; do
    [[ -z "$participant" ]] && continue
    if [[ -d "$LATEST/$participant" ]]; then
        printf '  ok    %s\n' "$participant"
    else
        printf '  FAIL  %s (not generated)\n' "$participant"
        MISSING=1
    fi
# A here-string, not a pipe: a piped `while` runs in a subshell, and MISSING would be
# set there and lost, so a failed provisioning would exit 0.
done <<< "$EXPECTED"

if [[ $MISSING -ne 0 ]]; then
    echo >&2
    echo "some participants are missing — check project.yml" >&2
    exit 1
fi

echo
echo "next: ./scripts/start_federation.sh 4"
