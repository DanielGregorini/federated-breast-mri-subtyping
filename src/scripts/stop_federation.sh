#!/usr/bin/env bash
# Stop the server and every hospital, and clean up orphans.
#
#     ./scripts/stop_federation.sh
#
# WHY THIS IS NOT JUST `pkill -f nvflare`
# ---------------------------------------
# Two reasons, both learned the hard way.
#
# 1. `pkill -f 'pattern'` matches the process running the pattern. A previous
#    version of this project ran `pkill -f 'nvflare'` over ssh and killed the ssh
#    session issuing the command, then reported success. Every pattern below is
#    anchored to this project's workspace, and this script excludes its own pid.
#
# 2. A client that is mid-round leaves a child training process behind. Killing the
#    parent alone leaves a Python holding the GPU, and the next experiment then
#    trains on a card that is already full — which shows up as an out-of-memory
#    error in an unrelated run an hour later.
#
# Participants are asked to stop first, and only killed if they do not.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$HERE")"
SELF_PID=$$

WORKSPACE="$(cd "$PROJECT_ROOT" && python3 -c \
    'from config.federation import workspace_dir; print(workspace_dir())' 2>/dev/null || true)"

echo "======================================================================"
echo "STOPPING FEDERATION"
[[ -n "$WORKSPACE" ]] && echo "  workspace : $WORKSPACE"
echo "======================================================================"

# Step 1: the polite way. Each startup kit ships its own stop script, which lets a
# participant finish writing its logs and deregister from the server.
if [[ -n "$WORKSPACE" && -d "$WORKSPACE" ]]; then
    for kit in "$WORKSPACE"/*/; do
        name="$(basename "$kit")"
        if [[ -x "$kit/startup/stop_fl.sh" ]]; then
            echo "  asking $name to stop"
            "$kit/startup/stop_fl.sh" >/dev/null 2>&1 || true
        fi
    done
    sleep 3
fi

# Step 2: anything still alive that belongs to THIS workspace.
kill_matching() {
    local pattern="$1"
    local label="$2"
    local pids
    # -f matches the full command line; exclude this script and its children.
    pids="$(pgrep -f "$pattern" 2>/dev/null | grep -v "^${SELF_PID}$" || true)"
    if [[ -n "$pids" ]]; then
        echo "  terminating $label: $(echo "$pids" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 2
        pids="$(pgrep -f "$pattern" 2>/dev/null | grep -v "^${SELF_PID}$" || true)"
        if [[ -n "$pids" ]]; then
            echo "  force-killing $label: $(echo "$pids" | tr '\n' ' ')"
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
        fi
    fi
}

if [[ -n "$WORKSPACE" ]]; then
    # Anchored to the workspace path, so nothing outside this project is touched.
    kill_matching "$WORKSPACE" "workspace processes"
fi
# The client trainer, anchored to this project's own script path.
kill_matching "$PROJECT_ROOT/federation/client.py" "client trainers"

echo
LEFT="$(pgrep -fl "${WORKSPACE:-__nothing__}" 2>/dev/null | grep -v "^${SELF_PID} " || true)"
if [[ -n "$LEFT" ]]; then
    echo "still running — inspect manually:"
    echo "$LEFT"
else
    echo "federation stopped."
fi
