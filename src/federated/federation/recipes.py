"""One row of `config/experiments.py` in, one NVFLARE recipe out.

WHY FEDPROX DOES NOT HAVE ITS OWN RECIPE
----------------------------------------
It surprises people, so it is stated here rather than discovered later: **FedProx and
FedAvg use the same recipe and the same server-side aggregation.** FedProx does not
change how updates are combined. It changes what each client optimises, by adding

    mu/2 * || w_local - w_global ||^2

to the local loss, which keeps a site from drifting too far from the model it was
given. That term lives entirely on the client, so the only difference between test06
and test07 in this file is one number passed to the training script.

The practical consequence is a failure mode worth guarding: a client that receives the
coefficient and ignores it is running FedAvg while the results table says FedProx, and
nothing anywhere will warn you. `build_recipe` therefore refuses to build a FedProx
experiment with `mu <= 0`, and `federation/client.py` refuses to apply a proximal term
without the global weights to anchor to.

WHY THE MODEL IS PASSED AS AN INSTANCE
--------------------------------------
`FedAvgRecipe(model=...)` takes the actual `nn.Module`. It also accepts a dotted path
plus keyword arguments, and that is how this project once shipped a server that built
a ResNet-18 while every client built a ResNet-50: the path resolved, the defaults
differed, the run completed, and the numbers were meaningless. Passing a built object
makes the server's copy the same object the clients construct from the same config.

This requires the **PyTorch** recipe, `nvflare.app_opt.pt.recipes.fedavg`. The generic
`nvflare.recipe.FedAvgRecipe` accepts `model` only as a dict and rejects an
`nn.Module` outright.

The architecture fingerprint is passed to the clients on top of that, so the check is
made at both ends rather than assumed at one.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import experiments as EX          # noqa: E402
from config import federation as FED          # noqa: E402
from common import models as M                   # noqa: E402

CLIENT_SCRIPT = PROJECT_ROOT / "federation" / "client.py"


def train_args_for(experiment) -> str:
    """The command line the client is started with.

    Everything the client needs to know that is NOT in `config/experiments.py` goes
    here — which partition to read, and whether a proximal term applies. Everything
    else the client reads from the shared config, so a hyperparameter cannot differ
    between the recipe and the trainer.
    """
    if experiment.kind != "federated":
        raise ValueError(f"{experiment.id} is not a federated experiment")

    partition = EX.PARTITIONS[experiment.partition]
    mu = EX.FEDERATION.fedprox_mu if experiment.algorithm == "fedprox" else 0.0

    if experiment.algorithm == "fedprox" and mu <= 0:
        raise ValueError(
            f"{experiment.id} is declared FedProx but fedprox_mu is {mu}. "
            "A client with mu=0 runs plain FedAvg while the results table claims "
            "otherwise — refusing to build this job.")
    if experiment.algorithm in ("fedavg", "fedopt") and mu != 0.0:
        raise ValueError(
            f"{experiment.id} is {experiment.algorithm} but mu={mu} would be sent. "
            "FedOpt is a SERVER-side optimiser; its clients must be plain FedAvg "
            "clients, or the run measures two changes at once.")

    model = M.build_model(EX.TRAINING, EX.NUM_CLASSES)
    return " ".join([
        f"--partition {partition.name}",
        f"--local-epochs {EX.FEDERATION.local_epochs}",
        f"--num-rounds {EX.FEDERATION.num_rounds}",
        f"--fedprox-mu {mu}",
        f"--seed {EX.TRAINING.seed}",
        f"--architecture {M.architecture_fingerprint(model)}",
        f"--results-dir {EX.RESULTS_DIR / experiment.name / 'sites'}",
    ])


def build_recipe(experiment):
    """Build the NVFLARE recipe for one federated experiment."""
    if experiment.kind != "federated":
        raise ValueError(
            f"{experiment.id} is centralised — it is not an NVFLARE job. "
            "Run it with scripts/run_centralized.py.")

    # The PyTorch recipe, NOT `nvflare.recipe.FedAvgRecipe`. The base recipe accepts
    # `model` only as a dict or None and raises
    #     "model must be a dict or None for the base recipe. Got ResNet."
    # The framework-specific one takes the built `nn.Module` and brings the PyTorch
    # persistor with it, which is what serialises the exact object the clients build.
    partition = EX.PARTITIONS[experiment.partition]
    # The WRAPPER, not the bare network. NVFLARE rebuilds the server's copy from
    # the recorded constructor arguments; a torchvision ResNet cannot be recorded
    # (it stores `_norm_layer` as a class, which is not JSON-serialisable) and
    # would rebuild as a default 1000-class model even if it could. See
    # src/models.py::FederatedClassifier.
    model = M.federated_model(EX.TRAINING, EX.NUM_CLASSES)

    # Identical for every algorithm. Only the recipe CLASS and, for FedOpt, the
    # server-side optimiser differ — so a difference between two of these runs
    # cannot be a difference in client count, rounds, script or selection metric.
    common = dict(
        name=experiment.name,
        model=model,
        min_clients=partition.n_clients,
        num_rounds=EX.FEDERATION.num_rounds,
        train_script=str(CLIENT_SCRIPT),
        train_args=train_args_for(experiment),
        # The server selects the global model on this metric, reported by clients
        # from HELD-OUT patients. Training accuracy here would make the server pick
        # whichever model let clients memorise their own shard best.
        key_metric=EX.FEDERATION.key_metric,
    )

    if experiment.algorithm == "fedopt":
        # FedOptRecipe does NOT accept `key_metric` (nor `best_model_filename`,
        # `stop_cond`, `patience` ...). Passing it raises TypeError at build time.
        #
        # AND THAT IS A REAL ASYMMETRY, NOT JUST AN ARGUMENT LIST.
        # FedAvg and FedProx select the global model by `val_balanced_accuracy` and
        # write `best_FL_global_model.pt`. FedOpt in NVFLARE 2.8 has no such
        # mechanism: it keeps the LAST round's model. So a FedOpt number is the
        # round-30 model while a FedAvg number is the best of 30, and the two are
        # not measured the same way.
        #
        # It is not hidden: `collect_results.py::find_global_model` reports which
        # file it scored, `selected` or `last_round`, and that column travels into
        # the summary. The dissertation must state it — a FedOpt run that looks
        # slightly worse may only be unselected.
        common.pop("key_metric", None)
        # FedOpt keeps the FedAvg CLIENT and changes only what the server does with
        # the aggregated update: instead of adopting the mean, it treats the mean as
        # a gradient and steps an optimiser with it. SGD at lr=1.0, momentum=0 is
        # FedAvg exactly, so the momentum is the entire difference.
        #
        # `device="cpu"` on purpose: the server holds one model and applies one
        # optimiser step per round. Putting it on an accelerator buys nothing and
        # adds a device the aggregation could fail on.
        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe
        return FedOptRecipe(
            optimizer_args={
                "path": "torch.optim.SGD",
                "args": {"lr": EX.FEDERATION.fedopt_lr,
                         "momentum": EX.FEDERATION.fedopt_momentum},
                "config_type": "dict",
            },
            device="cpu",
            **common,
        )

    # FedAvg and FedProx share this recipe and share the server-side aggregation.
    # FedProx differs only in the mu the client receives — see the module docstring.
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    return FedAvgRecipe(**common)


def build_env(n_clients: int):
    """The production environment: real PKI startup kits, real ports, admin API.

    `ProdEnv` and not `SimEnv`. The simulator runs clients as threads inside one
    process, which is fast and useless as evidence — there is no PKI, no network and
    no separate processes. Every number this dissertation reports comes from here.
    """
    from nvflare.recipe import ProdEnv

    # Resolved through config/federation.py so every script picks the SAME prod_NN.
    # Two scripts disagreeing about which provisioning run to use produces a TLS
    # handshake failure whose error message never mentions provisioning.
    admin_dir = FED.startup_kit(FED.ADMIN_USER)
    return ProdEnv(startup_kit_location=str(admin_dir), username=FED.ADMIN_USER)


def describe(experiment) -> str:
    lines = [
        f"{experiment.id} — {experiment.name}",
        f"  objective : {experiment.objective}",
        f"  question  : {experiment.research_question}",
        f"  clients   : {experiment.n_clients}",
        f"  split     : {experiment.split_label}",
    ]
    if experiment.kind == "federated":
        mu = EX.FEDERATION.fedprox_mu if experiment.algorithm == "fedprox" else 0.0
        lines += [
            (f"  algorithm : fedopt (server SGD lr={EX.FEDERATION.fedopt_lr}, "
             f"momentum={EX.FEDERATION.fedopt_momentum}; client mu=0)"
             if experiment.algorithm == "fedopt"
             else f"  algorithm : {experiment.algorithm} (mu={mu})"),
            f"  rounds    : {EX.FEDERATION.num_rounds} x "
            f"{EX.FEDERATION.local_epochs} local epoch",
            f"  selection : {EX.FEDERATION.key_metric}",
        ]
    else:
        lines.append(f"  epochs    : {EX.FEDERATION.centralized_epochs} (budget-matched)")
    if experiment.notes:
        lines.append(f"  notes     : {experiment.notes}")
    return "\n".join(lines)
