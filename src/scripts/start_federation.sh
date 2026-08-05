#!/usr/bin/env bash
# Start the server and N hospitals, each as its own operating-system process.
#
#     ./scripts/start_federation.sh 4 test06   # server + hospital_1..4, logs -> logs/test06/
#     ./scripts/start_federation.sh 2 test02   # server + hospital_1..2, logs -> logs/test02/
#     ./scripts/start_federation.sh 4          # logs -> logs/_scratch/
#
# WHY THE TEST ID IS AN ARGUMENT
# ------------------------------
# Logs belong to a PROCESS, not to a job: the server and the hospitals outlive any
# single submission. Tests with different client counts already require a restart, so
# the federation is started once per test and its logs are written straight into that
# test's folder. Passing the id here is what makes `logs/test06/hospital_3.log` the
# hospital_3 log *of test06* rather than of whatever ran last.
#
# SEPARATE PROCESSES, NOT THREADS
# -------------------------------
# This is the difference between a simulation and a deployment. Each hospital gets
# its own Python interpreter, its own memory, its own certificate and its own port.
# Moving one of them to another machine changes an address in
# config/federation.py and federation/provisioning/project.yml, and nothing else.
#
# NVFLARE's own simulator would run all of these as threads inside one process. It
# is faster and it is not evidence, so it is not used for any reported number.
#
# START ORDER MATTERS
# -------------------
# The server must be accepting connections before a client tries to register, or the
# client retries with a backoff and the first round is delayed by up to a minute.
# The wait below polls the admin port instead of sleeping a fixed amount.
#
# THE POLL USES PYTHON, NOT `nc`
# ------------------------------
# `nc` is not installed in the RunPod container this project trains on, and a missing
# `nc` fails silently: the loop spins for its full 60 seconds, the hospitals never
# start, and the only symptom is a line of dots. python3 is already a hard dependency
# of everything here, so it is the portable choice.

set -euo pipefail

N_CLIENTS="${1:-4}"
TEST_ID="${2:-_scratch}"
if ! [[ "$N_CLIENTS" =~ ^[1-4]$ ]]; then
    echo "usage: $0 <n_clients 1-4> [test_id]" >&2
    exit 1
fi
# Any testNN, not just test01-09: the experiment table grew past nine when FedOpt
# was added, and a guard that hard-codes the count silently rejects every new one.
if ! [[ "$TEST_ID" =~ ^(test[0-9]{2}|_scratch)$ ]]; then
    echo "usage: $0 <n_clients 1-4> [testNN]" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$HERE")"

# The one place that resolves prod_NN, so this script and the Python that submits
# jobs can never disagree about which provisioning run they are using.
WORKSPACE="$(cd "$PROJECT_ROOT" && python3 -c \
    'from config.federation import workspace_dir; print(workspace_dir())')"
ADMIN_PORT="$(cd "$PROJECT_ROOT" && python3 -c \
    'from config.federation import ADMIN_PORT; print(ADMIN_PORT)')"

# One folder per test, one file per participant — see production/README.md section
# "Where logs are stored". Never a shared file: two participants appending to one log
# interleave mid-line under load and the result cannot be reconstructed.
LOG_DIR="$PROJECT_ROOT/production/logs/$TEST_ID"
mkdir -p "$LOG_DIR"

# Every line the federation emits is also timestamped into one ordered file, so the
# sequence of events ACROSS participants is recoverable. The per-participant files
# stay authoritative for content; this one is for ordering.
TIMELINE="$LOG_DIR/timeline.log"
stamp() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$TIMELINE"; }
stamp "start_federation.sh n_clients=$N_CLIENTS test=$TEST_ID"

echo "======================================================================"
echo "STARTING FEDERATION — 1 server + $N_CLIENTS hospitals"
echo "  workspace : $WORKSPACE"
echo "  test      : $TEST_ID"
echo "  logs      : $LOG_DIR"
echo "======================================================================"

# Each site's trainer needs to find this project and the classifier phase it shares
# a model with. Exported here so every child process inherits them.
export FEDBREAST_ROOT="$PROJECT_ROOT"
export BREAST_CORE_ROOT="$PROJECT_ROOT"
# Without this, torch opens one thread pool per process and 5 processes on one
# machine spend most of their time context switching. Measured on this project:
# 2 concurrent jobs gave 0.074 epochs/s, 7 gave 0.058.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# True when something is listening on $1. Uses python3 because `nc` is absent from
# the training container — see the note above.
port_open() {
    python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', $1)) == 0 else 1)
" 2>/dev/null
}

start_participant() {
    local name="$1"
    local kit="$WORKSPACE/$name"
    if [[ ! -x "$kit/startup/start.sh" ]]; then
        echo "  FAIL  $name — no startup kit at $kit" >&2
        echo "        run ./scripts/provision.sh" >&2
        exit 1
    fi
    nohup "$kit/startup/start.sh" > "$LOG_DIR/$name.log" 2>&1 &
    local pid=$!
    echo "$pid" >> "$LOG_DIR/pids"
    stamp "started $name pid=$pid kit=$kit"
    echo "  started $name (pid $pid) -> $LOG_DIR/$name.log"
}

start_participant server

printf '  waiting for the server to accept connections'
for _ in $(seq 1 60); do
    if port_open "$ADMIN_PORT"; then
        printf ' up\n'
        break
    fi
    printf '.'
    sleep 1
done
if ! port_open "$ADMIN_PORT"; then
    printf '\n'
    stamp "FAILED: server never opened admin port $ADMIN_PORT"
    echo "server did not open port $ADMIN_PORT — see $LOG_DIR/server.log" >&2
    exit 1
fi

for i in $(seq 1 "$N_CLIENTS"); do
    start_participant "hospital_$i"
done

echo
echo "running processes:"
pgrep -fl 'nvflare' | grep -v 'start_federation' || true
echo
stamp "federation up: server + $N_CLIENTS hospitals"
echo "next: python scripts/run_experiment.py $TEST_ID"
echo "stop: ./scripts/stop_federation.sh"
