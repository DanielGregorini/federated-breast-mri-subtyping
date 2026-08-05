"""Single source of truth for the nine dissertation experiments.

Every other file in this project reads this one. To change the protocol, change it
here and rebuild — never edit a number inside a script, a job, or a notebook.

WHY A DECLARATIVE TABLE
-----------------------
Nine experiments differing along three axes (client count, split shape, aggregation
algorithm) is exactly the shape of problem where hand-written per-experiment configs
drift apart. In the previous iteration of this project three separate bugs came from
that drift: the server built a ResNet-18 while the clients built a ResNet-50, the
evaluation script had an architecture name hard-coded, and the federated clients used
different regularisation from the centralised baseline they were compared against.
All three were invisible until results looked wrong.

Here every experiment is a row. The model, the hyperparameters and the partition
geometry are defined once and shared, so a mismatch between server and client, or
between centralised and federated, is not expressible.

THE DESIGN
----------
One CENTRALISED baseline against eight FEDERATED runs. The federated runs vary the
number of hospitals (2, 3, 4) and how the data is divided between them (balanced vs
skewed), and each configuration runs once with FedAvg and once with FedProx, with
everything else held fixed.

    Test 01 - Centralised (all data on one machine)
    Test 02 - 2 hospitals, 50/50            , FedAvg
    Test 03 - 2 hospitals, 50/50            , FedProx
    Test 04 - 3 hospitals, 33/33/33         , FedAvg
    Test 05 - 3 hospitals, 33/33/33         , FedProx
    Test 06 - 4 hospitals, 25/25/25/25      , FedAvg
    Test 07 - 4 hospitals, 25/25/25/25      , FedProx
    Test 08 - 4 hospitals, 50/20/10/10      , FedAvg
    Test 09 - 4 hospitals, 50/20/10/10      , FedProx

Tests 01-09 vary only how MUCH data each site holds: all four partitions are
stratified, so the class-share spread between hospitals never exceeds 0.6
percentage points. That is quantity skew, the weakest form of non-IID data, and
it is why tests 10-13 were added.

    Test 10 - 3 hospitals, one cohort each   , FedAvg   <- DUKE | I-SPY1 | I-SPY2
    Test 11 - 3 hospitals, one cohort each   , FedProx
    Test 12 - 3 hospitals, cohorts mixed     , FedAvg   <- same sizes as Test 10
    Test 13 - 3 hospitals, cohorts mixed     , FedProx

Numbered so that each partition owns a consecutive FedAvg/FedProx pair, exactly
as tests 02-09 are numbered: 10/11 is the cohort split, 12/13 its control.

10 vs 12 and 11 vs 13 are the MATCHED PAIRS that answer RQ2. All four hold 642,
101 and 784 patients; the only difference is whether a site draws from one cohort
or from all three. The class spread is 27.5 percentage points against 0.3, so
anything measured between them is attributable to cohort identity rather than to
site size.

RESEARCH QUESTION MAPPING
-------------------------
    RQ1  can federated match centralised?      Test 01 vs Tests 02-09
    RQ2  impact of non-IID heterogeneity?      Test 10 vs Test 12, Test 11 vs 13
                                               (Tests 06 vs 08 = quantity skew only)
    RQ3  FedAvg vs FedProx trade-offs?         every even/odd pair; the pair that
                                               matters is 10 vs 11, where FedProx
                                               has real heterogeneity to correct
    RQ4  what mitigates FL limitations?        Test 09, Test 11, plus the security
                                               measures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# PATHS                                                                        #
# --------------------------------------------------------------------------- #
# config/ -> federated/ -> src/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

# The prepared 2D dataset produced by `core/dataset_builder.py`. Images are PNG, one
# per slice, three DCE phases fused as RGB, 224x224 at a constant 0.357 mm/px.
#
# `multi_subtype_80mm` — the combined dataset, all three cohorts:
# I-SPY2 + I-SPY1 + DUKE. 2,063 patients, 16,378 images.
#
# THE IMAGES ARE NEVER REGENERATED HERE
# -------------------------------------
# This project consumes that dataset; it does not build it. `prepare_data.py` and
# `partition_data.py` both HARDLINK the existing PNGs into the federated layout, so
# the bytes under `data/global/` and `data/partitions/` are the same inodes as the
# bytes under `dataset/multi_subtype_80mm/`. Nothing is resampled,
# re-cropped, re-normalised or re-encoded.
#
# The pipeline that produced it is `core/dataset_builder.py`,
# driven by `notebooks/03_build_dataset_mine.ipynb`, and its exact parameters are
# recorded in that dataset's own `config.json`. Both are kept in the active tree
# precisely so the preprocessing stays reproducible and unchanged. See
# `docs/DATASET_SPEC.md` for the full specification.
#
# THE CONFOUND THIS CARRIES, STATED ONCE AND NOT AGAIN
# ----------------------------------------------------
# A source-signature probe on pooled cohorts reached macro-AUC 0.9978 predicting
# WHICH COHORT a slice came from, against 0.6078 for the actual subtype task. The
# cohorts differ systematically: DUKE is 64.6% HRposHER2neg against I-SPY2's 38.8%,
# and its tumours are ~5x smaller by volume. "Small tumour -> DUKE ->
# HRposHER2neg" is therefore available to the model as a shortcut that has nothing
# to do with biology.
#
# That is a real effect and it must be reported beside any result measured here —
# `SONDA_r18` in the classifier phase is the probe, and it is the evidence. It is
# NOT a reason to avoid this dataset: pooled data is what makes the hospitals
# genuinely non-IID, which is what RQ2 is actually about. Quantity skew across
# stratified sites (what the single-cohort split gave) is a much weaker test.
#
# With `--by-cohort`, `partition_data.py` puts one real cohort per hospital, and
# the sites then differ in class distribution and tumour size the way real
# hospitals do. That is the strongest available version of RQ2.
SOURCE_DATASET = REPO_ROOT / "dataset" / "multi_subtype_80mm"

DATA_DIR = REPO_ROOT / "deployment" / "data"   # per-hospital datasets live here
GLOBAL_DIR = DATA_DIR / "global"           # the held-out global test set
PARTITIONS_DIR = DATA_DIR / "partitions"   # one physical split per scenario

# --------------------------------------------------------------------------- #
# THE PRODUCTION DEPLOYMENT                                                    #
# --------------------------------------------------------------------------- #
# `production/` is the deployment and output layer: the provisioning file, the PKI
# workspace, the nine jobs, and everything the nine runs emit. It holds no second
# definition of any hyperparameter — the jobs under `production/jobs/` are GENERATED
# from this file by `scripts/generate_jobs.py`, and the scripts under
# `production/scripts/` delegate to `scripts/`.
#
# That separation is the point. This project has already shipped three bugs whose
# single cause was two copies of one setting drifting apart: a server that built a
# ResNet-18 while the clients built a ResNet-50, an evaluation script with the
# architecture hard-coded, and federated clients regularised differently from the
# baseline they were compared against. All three were invisible until the numbers
# looked wrong. So configuration lives here, once, and `production/` is where the
# deployment and its output live.
PRODUCTION_DIR = REPO_ROOT / "deployment"

PROJECT_YML = PRODUCTION_DIR / "project.yml"   # the NVFLARE provisioning file
WORKSPACE_DIR = PRODUCTION_DIR / "workspace"   # `nvflare provision -w workspace`
JOBS_DIR = PRODUCTION_DIR / "jobs"             # one folder per experiment
RESULTS_DIR = REPO_ROOT / "results" / "federated"  # metrics, models, predictions
LOGS_DIR = PRODUCTION_DIR / "logs"             # one folder per test, one file per site
FIGURES_DIR = PRODUCTION_DIR / "figures"       # distribution and result figures
DATASETS_DIR = PRODUCTION_DIR / "datasets"     # split manifests and provenance

# --------------------------------------------------------------------------- #
# CLASSIFICATION TASK                                                          #
# --------------------------------------------------------------------------- #
# Three molecular subtypes. Everything else in BreastDCEDL is ignored.
CLASS_NAMES = ["HRposHER2neg", "TripleNeg", "HER2pos"]
NUM_CLASSES = len(CLASS_NAMES)

# The trivial baseline (always predict the majority class) is NOT a constant. It
# depends on the split, and quoting accuracy without it is meaningless. It is
# computed from the test CSV at evaluation time — see src/evaluation/.

# --------------------------------------------------------------------------- #
# MODEL AND HYPERPARAMETERS — shared by every experiment                       #
# --------------------------------------------------------------------------- #
# These are used identically by the centralised baseline and by every federated
# client. That is what makes RQ1 a fair comparison: if the federated runs used
# different regularisation, the measured gap would be the regularisation, not the
# federation.


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters. Identical for centralised and federated.

    THESE ARE NOT DEFAULTS, THEY ARE A MEASURED CONFIGURATION.

    Every value below is read out of the winning checkpoint of the classifier phase,
    `results/classifier/checkpoints/FREEZE_R18_*.pt`. Reproducing the
    federated arm against a *different* configuration from the centralised one it is
    compared to is the single easiest way to make RQ1 meaningless, so the two share
    this object and nothing overrides it.

    WHICH RUN IS "THE BEST MODEL", AND WHY IT IS NOT THE HIGHEST NUMBER
    ------------------------------------------------------------------
        run             test macro-AUC     seeds
        R18_s42         0.6263             1
        FREEZE_R18      0.6159 ± 0.003     2
        R18             0.6078 ± 0.026     2

    `R18_s42` has the highest number. Its sibling seed scored 0.5894 — the same
    configuration, 0.037 apart. `FREEZE_R18` scored 0.6140 and 0.6178, 0.003 apart.
    Against a measured noise floor of 0.067 the two configurations are
    indistinguishable in AUC, but one of them is reproducible and the other is a
    lottery ticket, and a federated experiment that has to attribute a 0.07 gap to
    federation cannot afford a baseline that moves by 0.04 on its own.

    So the baseline is the FROZEN configuration, and `R18_s42` is not used.
    """

    # resnet18 and not resnet50: measured winner, and 6x faster (12 min vs 74.5).
    # Across 1.5M to 87.6M parameters, CNNs, transformers and 3D networks, every
    # architecture lands in the same 0.55-0.63 band — the ceiling is signal, not
    # capacity, so the cheapest adequate backbone is the right one.
    model_name: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    num_classes: int = NUM_CLASSES

    batch_size: int = 24
    learning_rate: float = 1e-4          # 3e-4 measured worst, 3e-5 no better
    weight_decay: float = 5e-4
    dropout: float = 0.5
    label_smoothing: float = 0.1
    optimizer: str = "adamw"
    scheduler: str = "cosine"            # evaluated per round; see src/training.py

    # Freeze conv1 + bn1 + layer1 + layer2. This is what makes the baseline
    # reproducible: seed spread fell from 0.026 to 0.003, nearly ten-fold, for
    # +0.008 AUC that means nothing on its own.
    #
    # Honest caveat, and it belongs here rather than in a footnote: on ResNet-18 this
    # removes only 683,072 of 11,178,051 parameters (6.1%), because the weight is
    # almost all in layer4. Whether it "reduces overfitting" is NOT supported — the
    # two seeds moved the train/test gap in opposite directions. "Stabilises the
    # result" is what was measured, and that is the reason it is used.
    freeze_until: str = "layer3"
    freeze_bn: bool = False

    # Inverse-frequency weights counted per PATIENT, not per slice. Counting slices
    # conflates how many patients carry a class with how large their tumours are.
    class_weighted_loss: bool = True

    # WHOSE class frequencies? "local" = each hospital's own rows; "global" = the
    # pooled training split, identical everywhere. Under the stratified partitions
    # of tests 02-09 the two agree to three decimals and the choice is cosmetic.
    # Under a cohort-based partition it is not, and it becomes RQ4 material: local
    # weights mean the sites optimise different objectives and FedAvg averages
    # models trained on different losses; global weights mean one objective and one
    # leaked vector of class counts. See src/data.py::class_weights.
    class_weight_scope: str = "local"

    # At most one slice per patient per batch. Neighbouring slices of one tumour are
    # near-duplicates and produce almost the same gradient, which removes the
    # stochastic noise that makes SGD generalise.
    max_slices_per_patient_per_batch: int = 1

    # Slice probabilities averaged into one prediction per patient before any
    # metric. Every number this project reports is per patient.
    aggregation: str = "mean"
    monitor_metric: str = "auc"          # selects the centralised checkpoint

    seed: int = 42
    num_workers: int = 8
    mixed_precision: bool = True


@dataclass(frozen=True)
class FederationConfig:
    """Federated protocol. Budget-matched to the centralised baseline on purpose."""

    # 30 rounds x 1 local epoch means the model sees 30 epochs of data — the same
    # budget as the centralised baseline. Matching this is what makes the comparison
    # fair: if the federated side saw less data, the measured difference would be
    # budget, not federation.
    num_rounds: int = 30
    local_epochs: int = 1
    centralized_epochs: int = 30

    # FedProx proximal coefficient. 0.01 is the value used throughout the FL
    # literature for cross-silo medical imaging, and the one NVFLARE defaults to.
    fedprox_mu: float = 0.01

    # FedOpt — a SERVER-side optimiser, where FedProx is a CLIENT-side penalty.
    #
    # FedAvg replaces the global model with the weighted mean of the client models.
    # FedOpt instead treats that mean as a pseudo-gradient and takes an optimiser
    # step with it:
    #
    #     delta    = mean(w_client - w_global)
    #     w_global = ServerOpt(w_global, delta)
    #
    # The clients are untouched: they run exactly the FedAvg client, with mu = 0.
    # That makes FedOpt orthogonal to FedProx rather than an alternative — the two
    # could be combined, and are not here.
    #
    # SGD with lr 1.0 and momentum 0 IS FedAvg exactly, so the momentum below is the
    # whole of the difference: it gives the server memory of the previous rounds'
    # direction. 1.0/0.6 is NVFLARE's own default pairing and what the earlier
    # iteration of this project used.
    fedopt_lr: float = 1.0
    fedopt_momentum: float = 0.6

    # The metric the SERVER uses to select the global model. It must come from
    # held-out client data, never from training accuracy: a previous iteration of
    # this project selected on training accuracy and therefore picked whichever
    # global model let clients memorise their shard best (99%+).
    key_metric: str = "val_balanced_accuracy"

    # Fraction of each hospital's patients held out locally. Serves the per-round
    # convergence curve and the per-client evaluation. The OFFICIAL number always
    # comes from the global test set, which is identical across all nine tests.
    local_val_fraction: float = 0.2


TRAINING = TrainingConfig()
FEDERATION = FederationConfig()

# --------------------------------------------------------------------------- #
# PARTITIONS                                                                   #
# --------------------------------------------------------------------------- #
# One physical data split per scenario. Patients are never shared between
# hospitals: every slice of a patient goes to exactly one client. Splitting by
# slice instead would let a model recognise the patient rather than the disease.


@dataclass(frozen=True)
class Partition:
    """How the training patients are divided between hospitals.

    `ratio` is the SHAPE of the split, not a set of fractions. The dissertation
    describes the skewed case as "50/20/10/10", which sums to 90 rather than 100 —
    it is a 5:2:1:1 ratio, and normalising it gives 55.6/22.2/11.1/11.1. Storing the
    ratio and normalising here keeps the dissertation's wording and the code's
    behaviour in agreement instead of silently disagreeing by ten percent.
    """

    name: str
    n_clients: int
    ratio: tuple[float, ...]
    label: str
    # Stratified keeps each hospital's class ratio equal to the global one, so the
    # only thing that varies between hospitals is QUANTITY. This is a deliberate
    # limitation and must be stated in the dissertation: it is quantity skew, not
    # genuine non-IID heterogeneity. See docs/EXPERIMENTS.md.
    stratified: bool = True

    def __post_init__(self) -> None:
        if len(self.ratio) != self.n_clients:
            raise ValueError(f"{self.name}: {self.n_clients} clients but {len(self.ratio)} ratios")
        if any(r <= 0 for r in self.ratio):
            raise ValueError(f"{self.name}: every share must be positive, got {self.ratio}")

    @property
    def fractions(self) -> tuple[float, ...]:
        """The ratio normalised to sum to exactly 1. This is what the splitter uses."""
        total = sum(self.ratio)
        return tuple(r / total for r in self.ratio)

    @property
    def client_names(self) -> list[str]:
        return [f"hospital_{i + 1}" for i in range(self.n_clients)]

    def describe(self) -> str:
        pct = " / ".join(f"{100 * f:.1f}%" for f in self.fractions)
        return f"{self.label}  ->  {pct}"


PARTITIONS: dict[str, Partition] = {
    "2_clients_balanced": Partition(
        name="2_clients_balanced", n_clients=2, ratio=(1, 1),
        label="2 hospitals, balanced (50/50)"),
    "3_clients_balanced": Partition(
        name="3_clients_balanced", n_clients=3, ratio=(1, 1, 1),
        label="3 hospitals, balanced (33.3 each)"),
    "4_clients_balanced": Partition(
        name="4_clients_balanced", n_clients=4, ratio=(1, 1, 1, 1),
        label="4 hospitals, balanced (25 each)"),
    "4_clients_skewed": Partition(
        name="4_clients_skewed", n_clients=4, ratio=(5, 2, 1, 1),
        label="4 hospitals, skewed 5:2:1:1 (dissertation: 50/20/10/10)"),

    # ----------------------------------------------------------------------- #
    # THE RQ2 PAIR — genuine heterogeneity, and its size-matched control       #
    # ----------------------------------------------------------------------- #
    # Partitions 1-4 above are stratified: the class-share spread between sites
    # never exceeds 0.6 percentage points, so they vary QUANTITY and nothing
    # else. That is quantity skew, the weakest entry in the non-IID taxonomy of
    # Kairouz et al., and it is why RQ2 currently has no real answer.
    #
    # These two partitions are built to be compared against each other and only
    # against each other. They hold the SAME three site sizes; the only thing
    # that differs is whether a site's patients come from one cohort or from all
    # three. Anything measured between them is therefore attributable to cohort
    # identity rather than to how much data a site holds.
    #
    # Sizes come from the cohorts themselves, and the ORDER matters:
    # `cohort_shares` assigns `sorted(cohorts)` to hospital_1..n, and
    # sorted(["duke", "spy1", "spy2"]) puts DUKE first, I-SPY1 second, I-SPY2
    # third. The control below must repeat that order or the two partitions stop
    # being size-matched.
    #
    #   hospital_1  DUKE     642 patients   66.4 / 15.9 / 17.8 %
    #   hospital_2  I-SPY1   101 patients   41.6 / 26.7 / 31.7 %
    #   hospital_3  I-SPY2   784 patients   38.9 / 35.8 / 25.3 %
    #
    # Class-share spread: 27.5 pp, against 0.6 pp for the skewed partition above.
    "3_clients_cohort": Partition(
        name="3_clients_cohort", n_clients=3, ratio=(642, 101, 784),
        label="3 hospitals, one cohort each (DUKE | I-SPY1 | I-SPY2)",
        stratified=False),
    "3_clients_sizematched": Partition(
        name="3_clients_sizematched", n_clients=3, ratio=(642, 101, 784),
        label="3 hospitals, cohorts mixed, sizes matched to 3_clients_cohort"),
}

# --------------------------------------------------------------------------- #
# THE NINE EXPERIMENTS                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Experiment:
    """One row of the dissertation's experiment table."""

    id: str                     # test01 ... test09
    name: str                   # folder name under jobs/
    kind: str                   # "centralized" | "federated"
    objective: str              # one line, goes into the job README
    research_question: str      # which RQ this row answers
    partition: str | None = None
    algorithm: str | None = None   # "fedavg" | "fedprox"
    notes: str = ""

    @property
    def job_dir(self) -> Path:
        return JOBS_DIR / self.name

    @property
    def n_clients(self) -> int:
        return 1 if self.partition is None else PARTITIONS[self.partition].n_clients

    @property
    def split_label(self) -> str:
        return "all data pooled" if self.partition is None else PARTITIONS[self.partition].describe()


EXPERIMENTS: list[Experiment] = [
    Experiment(
        id="test01", name="test01_centralized", kind="centralized",
        objective="Upper-bound baseline: one machine, all training data pooled.",
        research_question="RQ1 — the reference every federated run is measured against.",
        notes="Not an NVFLARE job. Runs the same trainer the clients run, on the "
              "union of every hospital's data, for the same number of epochs."),
    Experiment(
        id="test02", name="test02_fedavg_2h", kind="federated",
        partition="2_clients_balanced", algorithm="fedavg",
        objective="Smallest federation: does splitting across two sites cost anything?",
        research_question="RQ1 — federated versus centralised."),
    Experiment(
        id="test03", name="test03_fedprox_2h", kind="federated",
        partition="2_clients_balanced", algorithm="fedprox",
        objective="Same two-site split under FedProx.",
        research_question="RQ3 — FedAvg versus FedProx at two clients."),
    Experiment(
        id="test04", name="test04_fedavg_3h", kind="federated",
        partition="3_clients_balanced", algorithm="fedavg",
        objective="Three sites: does further fragmentation compound the cost?",
        research_question="RQ1 — the shape of degradation as sites increase."),
    Experiment(
        id="test05", name="test05_fedprox_3h", kind="federated",
        partition="3_clients_balanced", algorithm="fedprox",
        objective="Same three-site split under FedProx.",
        research_question="RQ3 — FedAvg versus FedProx at three clients."),
    Experiment(
        id="test06", name="test06_fedavg_4h", kind="federated",
        partition="4_clients_balanced", algorithm="fedavg",
        objective="Four balanced sites. The primary federated configuration.",
        research_question="RQ1 — the headline comparison against Test 01. "
                          "Also the IID control for RQ2."),
    Experiment(
        id="test07", name="test07_fedprox_4h", kind="federated",
        partition="4_clients_balanced", algorithm="fedprox",
        objective="Same four-site split under FedProx.",
        research_question="RQ3 — FedAvg versus FedProx at four clients."),
    Experiment(
        id="test08", name="test08_fedavg_skewed", kind="federated",
        partition="4_clients_skewed", algorithm="fedavg",
        objective="Four sites of very unequal size, one holding half the data.",
        research_question="RQ2 — the impact of heterogeneity, against Test 06.",
        notes="Quantity skew only: each hospital keeps the global class ratio. "
              "A genuinely non-IID partition would also skew the label "
              "distribution. Stated as a limitation in docs/EXPERIMENTS.md."),
    Experiment(
        id="test09", name="test09_fedprox_skewed", kind="federated",
        partition="4_clients_skewed", algorithm="fedprox",
        objective="The skewed four-site split under FedProx, which exists precisely "
                  "to stabilise training under heterogeneity.",
        research_question="RQ4 — does FedProx mitigate the skew that Test 08 exposes?"),

    # FedOpt ONCE OCCUPIED test10-test13 AND WAS REMOVED FROM THIS TABLE.
    # ------------------------------------------------------------------------
    # Four FedOpt runs were declared against the same four partitions. test12 and
    # test13 failed at launch — `FedOptRecipe` rejects `key_metric` and raises
    # TypeError — and test10 was cancelled at round 19 of 30 on CPU. Nothing
    # completed, so nothing can be reported.
    #
    # They are removed rather than left as empty rows because a results table
    # carrying four permanently blank experiments invites the reader to ask what
    # happened to them in every chapter. The episode is recorded in
    # docs/PROJECT_HISTORY.md.
    #
    # The asymmetry that made FedOpt unusable here is worth keeping in mind if it
    # is ever revived: with `key_metric` refused, FedOpt keeps the LAST round
    # while FedAvg and FedProx keep the best of thirty, so the three are not
    # measured the same way.
    #
    # test10-test13 below are the RQ2 experiments that DID complete.

    # ----------------------------------------------------------------------- #
    # THE RQ2 PAIR                                                             #
    # ----------------------------------------------------------------------- #
    # test10 and test12 are a matched pair and are meaningless apart. Both hold
    # three sites of 642, 101 and 784 patients. In test10 each site is one real
    # cohort; in test15 the same three sites are filled with a stratified mix of
    # all three. Sizes, client count, rounds, seed and algorithm are identical,
    # so the difference between them isolates COHORT IDENTITY — a 27.5 pp spread
    # in class prior, tumours five times smaller at DUKE, and a different scanner
    # population — from the quantity skew that tests 08/09 already vary.
    #
    # The source probe (macro-AUC 0.9978 predicting cohort) MUST be reported
    # beside test10: with one cohort per site, "identify the cohort, then use
    # that cohort's prior" becomes available to the aggregated global model.
    #
    # Worth watching, and the reason this pair may say something new: within a
    # single client of test10 the cohort is CONSTANT, so it carries no
    # discriminative information locally and cannot be learned as a shortcut
    # there. It can only re-emerge after aggregation.
    Experiment(
        id="test10", name="test10_fedavg_cohort", kind="federated",
        partition="3_clients_cohort", algorithm="fedavg",
        objective="One real cohort per hospital — genuine label and feature "
                  "heterogeneity, not quantity skew.",
        research_question="RQ2 — the primary test, against test12.",
        notes="Report the cohort probe beside this result."),
    Experiment(
        id="test12", name="test12_fedavg_sizematched", kind="federated",
        partition="3_clients_sizematched", algorithm="fedavg",
        objective="The control for test10: identical site sizes, cohorts mixed.",
        research_question="RQ2 — the control that isolates cohort identity."),

    # The FedProx arms of the same pair. FedProx exists to mitigate exactly the
    # heterogeneity test14 introduces, so this is where RQ3 has its strongest
    # available test: on the stratified partitions its proximal term had almost
    # nothing to correct.
    Experiment(
        id="test11", name="test11_fedprox_cohort", kind="federated",
        partition="3_clients_cohort", algorithm="fedprox",
        objective="FedProx under genuine cohort heterogeneity.",
        research_question="RQ3 — against test10, under real heterogeneity."),
    Experiment(
        id="test13", name="test13_fedprox_sizematched", kind="federated",
        partition="3_clients_sizematched", algorithm="fedprox",
        objective="FedProx on the size-matched control.",
        research_question="RQ3 — against test12."),
]

EXPERIMENTS_BY_ID = {e.id: e for e in EXPERIMENTS}
EXPERIMENTS_BY_NAME = {e.name: e for e in EXPERIMENTS}


def get(identifier: str) -> Experiment:
    """Look an experiment up by id (test03) or by folder name."""
    if identifier in EXPERIMENTS_BY_ID:
        return EXPERIMENTS_BY_ID[identifier]
    if identifier in EXPERIMENTS_BY_NAME:
        return EXPERIMENTS_BY_NAME[identifier]
    raise KeyError(f"unknown experiment: {identifier}. "
                   f"Known: {', '.join(EXPERIMENTS_BY_ID)}")


def summary() -> str:
    """Human-readable protocol summary. Printed by every entry-point script."""
    lines = [
        "FEDERATED PROTOCOL — breast-cancer molecular subtype",
        f"  dataset  : {SOURCE_DATASET.name}",
        f"  classes  : {NUM_CLASSES} — {', '.join(CLASS_NAMES)}",
        f"  model    : {TRAINING.model_name} (ImageNet pretrained={TRAINING.pretrained})",
        f"  federated: {FEDERATION.num_rounds} rounds x {FEDERATION.local_epochs} local epoch",
        f"  central  : {FEDERATION.centralized_epochs} epochs (same data budget)",
        f"  lr {TRAINING.learning_rate} · batch {TRAINING.batch_size} · "
        f"dropout {TRAINING.dropout} · seed {TRAINING.seed}",
        f"  FedProx mu: {FEDERATION.fedprox_mu}",
        f"  model selection: {FEDERATION.key_metric} (held-out client data)",
        "",
        f"  {len(EXPERIMENTS)} experiments, {len(PARTITIONS)} partitions",
    ]
    for e in EXPERIMENTS:
        algo = e.algorithm or "—"
        lines.append(f"    {e.id}  {e.name:<36} {e.n_clients}c  {algo:<8} {e.split_label}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
