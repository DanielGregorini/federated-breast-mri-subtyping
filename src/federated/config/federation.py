"""Who takes part in the federation, and how they reach each other.

THIS IS THE ONLY FILE THAT KNOWS ABOUT HOSTS AND PORTS.

Everything else refers to participants by NAME (`server`, `hospital_1`, ...) and
never by address. That is the whole point: moving from a single laptop to four real
hospital machines is a change to this file and to `provisioning/project.yml`, and
to nothing else.

HOW THE THREE DEPLOYMENT MODES DIFFER
-------------------------------------
NVFLARE offers three execution environments. This project uses the third for every
reported result, and the first two only as development conveniences.

    SimEnv   everything inside one process, threads pretending to be clients.
             Fast, but it is not a deployment: there is no PKI, no network, no
             separate processes. Never used for a reported number.

    PocEnv   separate processes on one machine, but with throwaway certificates
             and a fixed folder layout. Useful to smoke-test a job.

    ProdEnv  the real thing. Each participant is a separate process with its own
             PKI startup kit, listening on its own port, and jobs are submitted
             through the admin API exactly as they would be in a hospital. This is
             what the dissertation requires and what every reported result uses.

The addresses below are localhost today. On real machines, only `default_host` and
the per-participant hosts change — the job definitions, the client code and the
data layout are untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# PROJECT IDENTITY                                                             #
# --------------------------------------------------------------------------- #
PROJECT_NAME = "breast_fl_project"
PROJECT_DESCRIPTION = "Federated breast-cancer molecular-subtype classification (ResNet-18)"
ORGANISATION = "ips"

# --------------------------------------------------------------------------- #
# SERVER                                                                       #
# --------------------------------------------------------------------------- #
# Two ports, and they do different jobs:
#   fed_learn_port  clients connect here to receive tasks and return model updates
#   admin_port      the admin API connects here to submit and monitor jobs
# Keeping them separate is what lets a hospital firewall expose only the first.
SERVER_NAME = "server"
SERVER_HOST = "localhost"      # -> real DNS name or IP when deployed
FED_LEARN_PORT = 8002
ADMIN_PORT = 8003

# --------------------------------------------------------------------------- #
# PARTICIPANTS                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hospital:
    """One federated client — one hospital."""

    name: str
    org: str
    host: str = "localhost"    # -> the hospital's real machine when deployed
    description: str = ""


# Provisioning creates startup kits for the MAXIMUM number of hospitals, and the
# 2- and 3-client experiments simply use a subset. Provisioning once and reusing
# the kits keeps the certificates identical across all nine experiments, so a
# difference between two tests can never be a difference in PKI.
HOSPITALS: list[Hospital] = [
    Hospital("hospital_1", "h1", description="Site 1 — also the large site in the skewed split"),
    Hospital("hospital_2", "h2", description="Site 2"),
    Hospital("hospital_3", "h3", description="Site 3"),
    Hospital("hospital_4", "h4", description="Site 4"),
]

MAX_CLIENTS = len(HOSPITALS)

# --------------------------------------------------------------------------- #
# ADMIN                                                                        #
# --------------------------------------------------------------------------- #
# The admin identity submits jobs, monitors them and downloads results. In a real
# federation this is a person at the coordinating centre, holding their own
# certificate. Scripts authenticate as this user through the Flare API.
#
# The name must be a well-formed email address, including a TLD: NVFLARE validates it
# against `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` and rejects the whole
# provisioning run with INVALID_ARGS otherwise. `admin@ips` fails that check; the
# domain has to be `ips.pt`.
#
# It must also match `project.yml` exactly. If the two disagree, provisioning succeeds
# and `run_experiment.py` then fails looking for a startup kit that was never
# generated — a confusing failure a long way from its cause.
ADMIN_DOMAIN = "ips.pt"
ADMIN_USER = f"admin@{ADMIN_DOMAIN}"
ADMIN_ROLE = "project_admin"

# --------------------------------------------------------------------------- #
# WHERE THE STARTUP KITS LIVE                                                  #
# --------------------------------------------------------------------------- #
# config/ -> federated/ -> src/ -> repository root
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# `nvflare provision -p production/project.yml -w production/workspace` writes here.
# One signed startup kit per participant — the thing that would be copied to a
# hospital's own machine. NVFLARE's SimEnv and PocEnv never write here and never
# produce a reported number.
#
# Layout after provisioning:
#   production/workspace/<PROJECT_NAME>/prod_NN/server/
#                                              /hospital_1/ ... /hospital_4/
#                                              /admin@ips.pt/
#
# Mirrors `config/experiments.py::WORKSPACE_DIR`. The two are kept equal by
# `scripts/verify_production.py`, which fails if they ever diverge — a server and a
# client resolving different workspaces fails at the TLS handshake with an error that
# never mentions provisioning.
PRODUCTION_DIR = PROJECT_DIR / "deployment"
PROJECT_YML = PRODUCTION_DIR / "project.yml"
WORKSPACE_ROOT = PRODUCTION_DIR / "workspace"


def workspace_dir() -> Path:
    """The current provisioned workspace: `production/workspace/<project>/prod_NN/`.

    `nvflare provision` never overwrites. Each run creates the next `prod_NN`
    alongside the previous ones, so certificates from an earlier provisioning stay
    valid and recoverable. The consequence is that "the startup kits" is ambiguous
    unless something picks, and picking wrongly means a server and a client
    presenting certificates from two different runs — which fails at the TLS
    handshake with an error that does not mention provisioning at all.

    So: the HIGHEST prod_NN wins, and every script resolves it through this one
    function. If the nine experiments must share one PKI (they must, or a difference
    between two tests could be a difference in certificates), they share it because
    they all call this.
    """
    root = WORKSPACE_ROOT / PROJECT_NAME
    if not root.is_dir():
        raise SystemExit(
            f"not provisioned: {root} does not exist.\n"
            "  Run scripts/provision.sh — see docs/DEPLOYMENT.md step 3.")

    runs = sorted((d for d in root.iterdir()
                   if d.is_dir() and re.fullmatch(r"prod_\d+", d.name)),
                  key=lambda d: int(d.name.split("_")[1]))
    if not runs:
        raise SystemExit(f"{root} holds no prod_NN folder. Re-run scripts/provision.sh.")
    return runs[-1]


def startup_kit(participant: str) -> Path:
    """One participant's startup kit — the folder holding its certificate and
    its `start.sh`. `participant` is a name, never an address."""
    kit = workspace_dir() / participant
    if not kit.is_dir():
        known = sorted(p.name for p in workspace_dir().iterdir() if p.is_dir())
        raise SystemExit(
            f"no startup kit for {participant!r} in {workspace_dir()}.\n"
            f"  provisioned participants: {', '.join(known)}")
    return kit


# --------------------------------------------------------------------------- #
# DERIVED                                                                      #
# --------------------------------------------------------------------------- #


def hospitals_for(n_clients: int) -> list[Hospital]:
    """The first `n_clients` hospitals.

    Deterministic on purpose: hospital_1 is always the same site with the same
    certificate in every experiment, so the 2-client and 4-client runs differ only
    in how many sites take part.
    """
    if not 1 <= n_clients <= MAX_CLIENTS:
        raise ValueError(f"n_clients must be 1..{MAX_CLIENTS}, got {n_clients}")
    return HOSPITALS[:n_clients]


def client_names(n_clients: int) -> list[str]:
    return [h.name for h in hospitals_for(n_clients)]


def summary() -> str:
    lines = [
        f"FEDERATION — {PROJECT_NAME}",
        f"  server : {SERVER_NAME} @ {SERVER_HOST}  "
        f"fed_learn={FED_LEARN_PORT} admin={ADMIN_PORT}",
        f"  admin  : {ADMIN_USER} ({ADMIN_ROLE})",
        f"  clients: {MAX_CLIENTS} provisioned",
    ]
    for h in HOSPITALS:
        lines.append(f"    {h.name:<12} org={h.org:<4} host={h.host:<12} {h.description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
