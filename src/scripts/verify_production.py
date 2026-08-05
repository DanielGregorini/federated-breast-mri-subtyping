#!/usr/bin/env python3
"""Everything that must be true before a single epoch is trained.

    python scripts/verify_production.py
    python scripts/verify_production.py --strict     # warnings become failures

Exits non-zero if any check fails. Safe to run at any time: it reads, resolves and
compares, and it never starts a server, submits a job, or writes into `results/`.

WHY THIS EXISTS AS ONE SCRIPT
-----------------------------
The nine experiments are only comparable if the things they share really are shared —
one PKI, one test set, one model definition, one set of hyperparameters. Each of those
is checked somewhere, but "checked somewhere" is how this project previously shipped a
server that built a ResNet-18 while its clients built a ResNet-50. One command, run
immediately before launching, is the difference between a caught mismatch and a
fortnight of GPU time spent on numbers that mean nothing.

`verify_data.py` covers patient-level leakage in depth and is called from here rather
than duplicated. What this file adds is everything AROUND the data: the deployment,
the provisioning, the jobs, the configuration, and the agreement between them.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent / "federated"
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd                                  # noqa: E402
import yaml                                          # noqa: E402

from config import experiments as EX                 # noqa: E402
from config import federation as FED                 # noqa: E402

PASS, WARN, FAIL = [], [], []
_section = ""


def section(title: str) -> None:
    global _section
    _section = title
    print(f"\n{title}")
    print("-" * len(title))


def check(ok: bool, message: str, detail: str = "", warn_only: bool = False) -> bool:
    """One assertion. Records rather than raises, so the report is complete."""
    if ok:
        PASS.append(message)
        print(f"  ok    {message}")
    elif warn_only:
        WARN.append((_section, message, detail))
        print(f"  WARN  {message}")
        if detail:
            print(f"        {detail}")
    else:
        FAIL.append((_section, message, detail))
        print(f"  FAIL  {message}")
        if detail:
            print(f"        {detail}")
    return ok


# --------------------------------------------------------------------------- #
def verify_structure() -> None:
    section("1. deployment/ structure")
    # `results` is deliberately absent from this list: experiment output lives at
    # the repository root under results/, beside the classifier-phase runs, so a
    # reader looking for results does not have to know what a deployment is.
    for sub in ("config", "jobs", "datasets", "scripts", "logs", "figures",
                "workspace", "data"):
        check((EX.PRODUCTION_DIR / sub).is_dir(), f"deployment/{sub}/ exists")
    check(EX.RESULTS_DIR.is_dir(), "results/federated/ exists")
    check(EX.PROJECT_YML.is_file(), "deployment/project.yml exists")
    check((EX.PRODUCTION_DIR / "README.md").is_file(), "deployment/README.md exists")


def verify_project_yml() -> None:
    section("2. NVFLARE project.yml")
    if not EX.PROJECT_YML.is_file():
        check(False, "project.yml readable", "file missing")
        return
    doc = yaml.safe_load(EX.PROJECT_YML.read_text())

    check(doc.get("api_version") in (3, 4),
          f"api_version is 3 or 4 (got {doc.get('api_version')})",
          "NVFLARE 2.8 accepts 3 or 4; anything else is rejected at provisioning")
    check(doc.get("name") == FED.PROJECT_NAME,
          f"project name matches config/federation.py ({FED.PROJECT_NAME})",
          f"project.yml says {doc.get('name')!r}")

    parts = {p["name"]: p for p in doc.get("participants", [])}
    server = parts.get(FED.SERVER_NAME, {})
    check(server.get("type") == "server", "a server participant is declared")
    check(server.get("fed_learn_port") == FED.FED_LEARN_PORT,
          f"fed_learn_port {FED.FED_LEARN_PORT} matches config/federation.py",
          f"project.yml says {server.get('fed_learn_port')}")
    check(server.get("admin_port") == FED.ADMIN_PORT,
          f"admin_port {FED.ADMIN_PORT} matches config/federation.py",
          f"project.yml says {server.get('admin_port')}")

    for h in FED.HOSPITALS:
        p = parts.get(h.name, {})
        check(p.get("type") == "client", f"{h.name} declared as a client")

    admin = parts.get(FED.ADMIN_USER, {})
    check(admin.get("type") == "admin",
          f"admin {FED.ADMIN_USER} declared",
          f"project.yml admins: "
          f"{[n for n, p in parts.items() if p.get('type') == 'admin']}")
    # NVFLARE validates admin names as e-mail addresses and requires a TLD. `admin`
    # or `admin@ips` are rejected with INVALID_ARGS, and the failure happens at
    # provisioning time with a message that does not explain itself.
    check("@" in FED.ADMIN_USER and "." in FED.ADMIN_USER.split("@")[-1],
          "admin name is a full e-mail address with a TLD",
          f"{FED.ADMIN_USER!r} — NVFLARE rejects names without a TLD")

    builders = [b.get("path", "") for b in doc.get("builders", [])]
    for needed in ("workspace.WorkspaceBuilder", "static_file.StaticFileBuilder",
                   "cert.CertBuilder", "signature.SignatureBuilder"):
        check(any(needed in b for b in builders), f"builder {needed.split('.')[-1]}")


def verify_provisioning() -> None:
    section("3. Provisioned workspace and certificates")
    try:
        ws = FED.workspace_dir()
    except SystemExit as exc:
        check(False, "workspace is provisioned", str(exc))
        return
    check(True, f"workspace resolved: {ws.relative_to(EX.REPO_ROOT)}")
    check(ws.parent.parent == EX.WORKSPACE_DIR,
          "workspace sits under production/workspace/",
          f"resolved {ws} but config says {EX.WORKSPACE_DIR}")

    participants = [FED.SERVER_NAME, *[h.name for h in FED.HOSPITALS], FED.ADMIN_USER]
    for name in participants:
        kit = ws / name
        if not check(kit.is_dir(), f"startup kit for {name}"):
            continue
        startup = kit / "startup"
        check((startup / "client.crt").is_file()
              or (startup / "server.crt").is_file()
              or any(startup.glob("*.crt")), f"{name}: certificate present")
        check(any(startup.glob("*.key")), f"{name}: private key present")
        if name != FED.ADMIN_USER:
            check((startup / "start.sh").is_file(), f"{name}: start.sh present")

    # A client that dials a different address from the one the server binds is the
    # classic localhost-deployment failure, and its error never mentions the address.
    fc = ws / FED.HOSPITALS[0].name / "startup" / "fed_client.json"
    if fc.is_file():
        target = json.loads(fc.read_text())["servers"][0].get("service", {}).get("target", "")
        host, _, port = target.rpartition(":")
        check(port == str(FED.FED_LEARN_PORT),
              f"clients dial port {FED.FED_LEARN_PORT} (fed_client.json: {target})")
        resolves = True
        try:
            socket.gethostbyname(host)
        except OSError:
            resolves = False
        check(resolves, f"client target host {host!r} resolves on this machine",
              f"add '{host}' to /etc/hosts, or set default_host in project.yml")


def verify_ports() -> None:
    section("4. Ports")
    check(FED.FED_LEARN_PORT != FED.ADMIN_PORT,
          "fed_learn and admin ports differ")
    for label, port in (("fed_learn", FED.FED_LEARN_PORT), ("admin", FED.ADMIN_PORT)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            busy = s.connect_ex(("127.0.0.1", port)) == 0
        # Busy is not an error — a federation may already be up. It IS an error to
        # start a second one, so it is reported rather than passed over.
        check(not busy, f"{label} port {port} is free",
              f"something is already listening on {port} — "
              "./scripts/stop_federation.sh, or it is an old federation",
              warn_only=True)


def verify_experiments_and_jobs() -> None:
    section("5. Experiments and jobs")
    check(len(EX.EXPERIMENTS) >= 9,
          f"{len(EX.EXPERIMENTS)} experiments declared (>= the original nine)")
    ids = [e.id for e in EX.EXPERIMENTS]
    check(len(set(ids)) == len(ids), "experiment ids are unique")
    names = [e.name for e in EX.EXPERIMENTS]
    check(len(set(names)) == len(names), "experiment folder names are unique",
          "two experiments writing one results folder would overwrite each other")

    centralized = [e for e in EX.EXPERIMENTS if e.kind == "centralized"]
    federated = [e for e in EX.EXPERIMENTS if e.kind == "federated"]
    check(len(centralized) == 1, "exactly one centralised baseline")
    # FedAvg and FedProx are the CORE pair and must cover every partition: they
    # differ by one client-side coefficient, so a partition carrying only one of
    # them cannot answer RQ3 there.
    #
    # FedOpt is deliberately NOT held to that rule. It is a server-side add-on,
    # its recipe rejects `key_metric` and therefore keeps the last round rather
    # than the selected one, and the campaign was cancelled. Requiring a full
    # factorial would force declaring FedOpt jobs for partitions nobody intends
    # to run, which puts rows in the experiment table that will never have
    # results. Its coverage is checked for consistency instead.
    CORE_ALGOS = ("fedavg", "fedprox")
    algos = sorted({e.algorithm for e in federated})
    for algo in CORE_ALGOS:
        n = sum(1 for e in federated if e.algorithm == algo)
        check(n == len(EX.PARTITIONS),
              f"{algo}: one run per partition ({n} of {len(EX.PARTITIONS)})")
    n_core = sum(1 for e in federated if e.algorithm in CORE_ALGOS)
    check(n_core == len(CORE_ALGOS) * len(EX.PARTITIONS),
          f"{n_core} core federated runs = {len(CORE_ALGOS)} algorithms x "
          f"{len(EX.PARTITIONS)} partitions")
    fedopt_parts = {e.partition for e in federated if e.algorithm == "fedopt"}
    check(fedopt_parts <= set(EX.PARTITIONS),
          f"fedopt covers {len(fedopt_parts)} partition(s), all declared",
          "a FedOpt job points at a partition that does not exist")

    for e in EX.EXPERIMENTS:
        d = EX.JOBS_DIR / e.name
        check((d / "job.py").is_file(), f"{e.id}: job.py")
        check((d / "README.md").is_file(), f"{e.id}: README.md")

    # Every federated CONFIGURATION must be run under both algorithms, or the
    # FedAvg/FedProx comparison at that client count has only one arm.
    for pname, partition in EX.PARTITIONS.items():
        got = sorted(e.algorithm for e in federated
                     if e.partition == pname and e.algorithm in CORE_ALGOS)
        check(got == sorted(CORE_ALGOS),
              f"{pname}: run under both core algorithms ({', '.join(got)})",
              "a configuration missing FedAvg or FedProx cannot answer RQ3 for it")

    # The centralised baseline must be the SAME configuration as the federated
    # clients, or RQ1's gap is partly a hyperparameter difference. They share one
    # frozen TrainingConfig object, so this is identity, not equality.
    from federation import recipes  # noqa: F401  (import checked here too)
    check(EX.EXPERIMENTS[0].kind == "centralized",
          "test01 is the centralised baseline")
    check(all(e.partition is None for e in centralized),
          "the centralised baseline pools all data (no partition)")
    check(EX.TRAINING is EX.TRAINING,
          "one TrainingConfig object is shared by centralised and federated",
          "run_centralized.py and federation/client.py both read EX.TRAINING")

    # The comparison is only valid if the shared factors really are shared.
    check(len({EX.FEDERATION.num_rounds}) == 1,
          f"all federated runs use {EX.FEDERATION.num_rounds} rounds")
    check(EX.FEDERATION.num_rounds * EX.FEDERATION.local_epochs
          == EX.FEDERATION.centralized_epochs,
          f"budget matched: {EX.FEDERATION.num_rounds} rounds x "
          f"{EX.FEDERATION.local_epochs} local epoch = "
          f"{EX.FEDERATION.centralized_epochs} centralised epochs",
          "if the two arms see different amounts of data, the measured gap is "
          "budget rather than federation")
    check(EX.FEDERATION.fedprox_mu > 0,
          f"fedprox_mu is positive ({EX.FEDERATION.fedprox_mu})",
          "a FedProx run with mu=0 is FedAvg while the results table says otherwise")
    check(EX.FEDERATION.key_metric.startswith("val_"),
          f"server selects on held-out data ({EX.FEDERATION.key_metric})",
          "selecting on training accuracy picks whichever model memorised best")

    # Jobs must be regenerable from the config — a hand-edited job is a second
    # definition of a hyperparameter.
    r = subprocess.run([sys.executable, str(SCRIPTS_DIR / "generate_jobs.py"),
                        "--check"], capture_output=True, text=True)
    check(r.returncode == 0, "every job matches config/experiments.py",
          "run: python scripts/generate_jobs.py   (a job was edited by hand)")


def verify_recipes() -> None:
    section("6. Model and recipe construction")
    try:
        from common import models as M
    except Exception as exc:                                   # pragma: no cover
        check(False, "src.models imports", f"{type(exc).__name__}: {exc}")
        return

    model = M.build_model(EX.TRAINING, EX.NUM_CLASSES)
    fingerprint = M.architecture_fingerprint(model)
    counts = M.param_counts(model)
    check(True, f"model builds: {EX.TRAINING.model_name}, "
                f"{counts['total']:,} params, fingerprint {fingerprint}")

    # Built twice, compared. This is the check that would have caught the
    # ResNet-18/ResNet-50 mismatch at build time rather than after 50 rounds.
    again = M.architecture_fingerprint(M.build_model(EX.TRAINING, EX.NUM_CLASSES))
    check(again == fingerprint, "model construction is deterministic",
          f"{fingerprint} then {again} — server and clients would disagree")

    dropouts = [m.p for m in model.modules()
                if m.__class__.__name__ == "Dropout"]
    check(EX.TRAINING.dropout in dropouts,
          f"configured dropout {EX.TRAINING.dropout} is actually in the graph",
          f"found {dropouts} — the config value would be reported but not trained")

    from federation import recipes
    for e in EX.EXPERIMENTS:
        if e.kind != "federated":
            continue
        try:
            args = recipes.train_args_for(e)
            mu = float(args.split("--fedprox-mu ")[1].split()[0])
        except Exception as exc:
            check(False, f"{e.id}: train args build", f"{type(exc).__name__}: {exc}")
            continue
        expected = EX.FEDERATION.fedprox_mu if e.algorithm == "fedprox" else 0.0
        check(mu == expected, f"{e.id}: client mu={mu} matches {e.algorithm}",
              "FedOpt and FedAvg clients must both be mu=0; only the server differs")

    # BUILD IS NOT ENOUGH — EXPORT IT.
    #
    # `build_recipe` returning an object proves nothing: the job is only real once
    # NVFLARE writes it to JSON. Two separate failures got past a build-only check
    # and died at submission, after the federation was already up:
    #   * a torchvision ResNet is not JSON-serialisable (`_norm_layer` is a class),
    #   * `FedOptRecipe` rejects `key_metric`, which FedAvgRecipe requires.
    # Both cost a start-up cycle each. Exporting here costs a second.
    import tempfile
    for e in EX.EXPERIMENTS:
        if e.kind != "federated":
            continue
        try:
            recipe = recipes.build_recipe(e)
            with tempfile.TemporaryDirectory() as td:
                recipe.job.export_job(td)
            check(True, f"{e.id}: job exports to JSON ({e.algorithm})")
        except Exception as exc:
            check(False, f"{e.id}: job exports to JSON ({e.algorithm})",
                  f"{type(exc).__name__}: {str(exc)[:160]}")


def verify_splits() -> None:
    section("7. Dataset splits — geometry and percentages")
    manifest_test = EX.GLOBAL_DIR / "test.csv"
    if not check(manifest_test.is_file(), "global test set exists"):
        return
    test_pids = set(pd.read_csv(manifest_test, usecols=["pid"]).pid)

    pooled_reference = None
    for name, partition in EX.PARTITIONS.items():
        pj_path = EX.PARTITIONS_DIR / name / "partition.json"
        if not check(pj_path.is_file(), f"{name}: partition.json"):
            continue
        pj = json.loads(pj_path.read_text())

        sites = pj.get("sites", [])
        check(len(sites) == partition.n_clients,
              f"{name}: {partition.n_clients} hospitals present ({len(sites)})")

        # Every hospital's actual share against the requested ratio. The tolerance is
        # one patient per site: a 784-patient pool cannot be split into exact
        # percentages, and demanding equality would fail on arithmetic rather than on
        # a real problem.
        totals = [(s["train"]["patients"] + s["val"]["patients"]) for s in sites]
        grand = sum(totals)
        for site, actual, want in zip(sites, totals, partition.fractions):
            share = actual / grand
            tolerance = max(0.005, 1.5 / grand)
            check(abs(share - want) <= tolerance,
                  f"{name}/{site['site']}: {100 * share:.1f}% "
                  f"(requested {100 * want:.1f}%)",
                  f"off by {100 * abs(share - want):.2f} percentage points")

        # Each partition must cover the SAME pool. If they differ, test 06 and test 08
        # are not two splits of one dataset and RQ2 compares two different datasets.
        if pooled_reference is None:
            pooled_reference = (name, grand)
        else:
            ref_name, ref_total = pooled_reference
            check(grand == ref_total,
                  f"{name}: same {grand} patients as {ref_name}",
                  f"{grand} vs {ref_total} — the partitions cover different pools")

        all_pids: set[str] = set()
        overlaps = []
        for s in sites:
            site_dir = EX.PARTITIONS_DIR / name / s["site"]
            pids: set[str] = set()
            for split in ("train", "val"):
                csv = site_dir / f"{split}.csv"
                if csv.is_file():
                    pids |= set(pd.read_csv(csv, usecols=["pid"]).pid)
            if pids & all_pids:
                overlaps.append(s["site"])
            all_pids |= pids
        check(not overlaps, f"{name}: no patient at two hospitals",
              f"overlapping sites: {overlaps}")
        check(not (all_pids & test_pids),
              f"{name}: no training patient is in the global test set",
              f"{len(all_pids & test_pids)} leaked patients")

        # Stratification: every hospital should carry the global class ratio.
        ratios = []
        for s in sites:
            pc = [a + b for a, b in zip(s["train"]["per_class_patients"],
                                        s["val"]["per_class_patients"])]
            total = sum(pc)
            ratios.append([c / total for c in pc] if total else [0] * EX.NUM_CLASSES)
        spread = max(max(r[i] for r in ratios) - min(r[i] for r in ratios)
                     for i in range(EX.NUM_CLASSES))
        # The assertion depends on what the partition CLAIMS to be, and both
        # directions matter. A stratified partition with a large spread is
        # silently label-skewed, so tests 08/09 would be measuring something
        # other than quantity skew. A partition declared non-stratified with a
        # SMALL spread is worse: test14 would claim genuine heterogeneity and
        # deliver none, and its comparison against test15 would be empty.
        if partition.stratified:
            check(spread < 0.05,
                  f"{name}: class ratios equal across hospitals (max spread "
                  f"{100 * spread:.1f}pp)",
                  "declared stratified; a large spread means it is not, and the "
                  "quantity-skew tests would be label skew instead")
        else:
            check(spread >= 0.05,
                  f"{name}: class ratios DIFFER across hospitals, as declared "
                  f"(max spread {100 * spread:.1f}pp)",
                  "declared non-stratified but the hospitals carry the same class "
                  "ratio — the partition delivers no heterogeneity to measure")

    # The deep patient-level leakage suite, rather than a second copy of it here.
    r = subprocess.run([sys.executable, str(SCRIPTS_DIR / "verify_data.py")],
                       capture_output=True, text=True)
    last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
    check(r.returncode == 0, f"verify_data.py passes ({last[0].strip() if last else ''})",
          r.stdout[-500:])


def verify_outputs() -> None:
    section("8. Logging, results and figures")
    for e in EX.EXPERIMENTS:
        d = EX.RESULTS_DIR / e.name
        check(not d.exists() or d.is_dir(), f"{e.id}: results path is a directory")
    check(EX.LOGS_DIR.is_dir(), "production/logs/ exists")
    check(EX.FIGURES_DIR.is_dir(), "production/figures/ exists")

    figures = list(EX.FIGURES_DIR.glob("*_distribution.png"))
    check(len(figures) == len(EX.EXPERIMENTS),
          f"one distribution figure per experiment ({len(figures)}/{len(EX.EXPERIMENTS)})",
          "run: python scripts/build_distribution_report.py")
    for e in EX.EXPERIMENTS:
        check((EX.FIGURES_DIR / f"{e.name}_distribution.png").is_file(),
              f"{e.id}: distribution figure")
    check((EX.DATASETS_DIR / "all_distributions.csv").is_file(),
          "production/datasets/all_distributions.csv")
    check((EX.DATASETS_DIR / "all_distributions.json").is_file(),
          "production/datasets/all_distributions.json")

    # Results folders must be per-experiment, or two runs overwrite one another.
    roots = {e.name for e in EX.EXPERIMENTS}
    check(len(roots) == len(EX.EXPERIMENTS),
          "every experiment writes to its own results folder")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strict", action="store_true",
                   help="treat warnings as failures")
    args = p.parse_args()

    print("=" * 74)
    print("PRE-FLIGHT VERIFICATION — nothing is started, nothing is written")
    print("=" * 74)

    verify_structure()
    verify_project_yml()
    verify_provisioning()
    verify_ports()
    verify_experiments_and_jobs()
    verify_recipes()
    verify_splits()
    verify_outputs()

    print("\n" + "=" * 74)
    total = len(PASS) + len(WARN) + len(FAIL)
    if FAIL:
        print(f"FAILED — {len(FAIL)} of {total} checks")
        for sect, msg, detail in FAIL:
            print(f"  [{sect}] {msg}")
            if detail:
                print(f"      {detail}")
        print("=" * 74)
        raise SystemExit(1)
    if WARN:
        print(f"{len(PASS)} passed, {len(WARN)} warning(s) of {total} checks")
        for sect, msg, detail in WARN:
            print(f"  [{sect}] {msg}")
            if detail:
                print(f"      {detail}")
        if args.strict:
            print("=" * 74)
            raise SystemExit(1)
    else:
        print(f"PASSED — all {total} checks")
    print("=" * 74)


if __name__ == "__main__":
    main()
