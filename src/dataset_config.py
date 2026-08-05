"""Single configuration file. Change experiments here and nowhere else.

Everything an experiment needs is one of three things: WHICH pipeline, WHICH task,
and WHICH model. Those three lines are at the top. Everything below them is a
default you can override without touching a line of training code.

    from config import Config
    cfg = Config()                                  # the defaults below
    cfg = Config(model="convnext_tiny", epochs=50)   # override anything

The two pipelines are deliberately isolated. `authors` reproduces the published
BreastDCEDL implementation; `mine` carries this thesis's proposals. Nothing is
shared between their preprocessing, so a comparison between them is a comparison
of methods rather than of leftovers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# =========================================================================== #
#  THE THREE LINES THAT DEFINE AN EXPERIMENT                                  #
# =========================================================================== #

# --- 1. PIPELINE ----------------------------------------------------------- #
# PIPELINE = "reference"   # faithful reproduction of the BreastDCEDL repository
PIPELINE = "thesis"        # this thesis's proposals

# --- 2. TASK --------------------------------------------------------------- #
# Authors' pipeline (binary):
#   TASK = "hr"        hormone-receptor status
#   TASK = "her2"      HER2 status
#   TASK = "pcr"       pathological complete response
# My pipeline (3 classes):
#   TASK = "subtype"   HRposHER2neg / TripleNeg / HER2pos
TASK = "subtype"

# --- 3. MODEL -------------------------------------------------------------- #
# Uncomment exactly one. Every entry below is implemented and tested.
#
# MODEL = "resnet18"            11.2M   the measured winner on the subtype task
# MODEL = "resnet34"            21.3M
# MODEL = "resnet50"            23.5M   the reference in the comparable literature
# MODEL = "efficientnet_b0"      4.0M
# MODEL = "efficientnet_b3"     10.7M
# MODEL = "convnext_tiny"       27.8M
# MODEL = "convnext_small"      49.5M
# MODEL = "mobilenet_v3_large"   4.2M
# MODEL = "densenet121"          7.0M
# MODEL = "vit_b_16"            85.8M   torchvision ViT, ImageNet-pretrained
# MODEL = "vit_mae_base"        85.8M   facebook/vit-mae-base — the AUTHORS' model
# MODEL = "swin_t"              27.5M
# MODEL = "thda_resnet34"       21.3M   dual-attention ResNet, arXiv:2510.13897
MODEL = "resnet18"

# =========================================================================== #
#  PATHS                                                                      #
# =========================================================================== #
# src/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# Raw Zenodo download — NEVER written to. See raw_dataset_BreastDCEDL/README.md
# for how to obtain it; it is too large to keep under version control.
#
# This path was wrong for a long time. It pointed at the authors' cloned code
# repository, which ships the metadata CSV but none of the imaging, so the
# builder could not have been re-run even though the figures still regenerated
# through a fallback resolver. Both the CSV and the three cohort folders now live
# under one directory, and the fallback is gone.
RAW_DIR = REPO_ROOT / "raw_dataset_BreastDCEDL"
METADATA_CSV = RAW_DIR / "BreastDCEDL_metadata_min_crop.csv"

# One folder per cohort of the MinCrop release.
COHORT_DIRS = {
    "spy2": RAW_DIR / "BreastDCEDL_ISPY2_min_crop",
    "spy1": RAW_DIR / "BreastDCEDL_ISPY1_min_crop",
    "duke": RAW_DIR / "BreastDCEDL_DUKE_min_crop",
}

# Data and results live at the repository root, not beside the code that
# writes them, so they are found without knowing where the code sits.
DATA_DIR = REPO_ROOT / "dataset"                    # generated 2D datasets
RESULTS_DIR = REPO_ROOT / "results" / "classifier"

# =========================================================================== #
#  TASK DEFINITIONS                                                           #
# =========================================================================== #


@dataclass(frozen=True)
class Task:
    """One classification target.

    `column` is read straight from the authors' harmonised metadata — no label is
    ever derived in code, so a label can never disagree with the published one.
    """

    name: str
    column: str
    classes: tuple[str, ...]
    pipeline: str                  # which pipeline this task belongs to
    description: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2


TASKS: dict[str, Task] = {
    # ---- authors' pipeline: three binary targets ------------------------- #
    "hr": Task(
        name="hr", column="HR", classes=("HRneg", "HRpos"), pipeline="reference",
        description="Hormone-receptor status. Positive in ~55% of I-SPY2."),
    "her2": Task(
        name="her2", column="HER2", classes=("HER2neg", "HER2pos"), pipeline="reference",
        description="HER2 status. The authors report AUC 0.744 with THDA-ResNet "
                    "(arXiv:2510.13897)."),
    "pcr": Task(
        name="pcr", column="pCR", classes=("no_pCR", "pCR"), pipeline="reference",
        description="Pathological complete response. The authors report AUC 0.72 "
                    "overall — 0.78 I-SPY2, 0.68 I-SPY1, 0.54 DUKE. Note this is a "
                    "PRE-treatment scan predicting a post-chemotherapy outcome."),
    # ---- my pipeline: the thesis target ---------------------------------- #
    "subtype": Task(
        name="subtype", column="HR_HER2_STATUS",
        classes=("HRposHER2neg", "TripleNeg", "HER2pos"), pipeline="thesis",
        description="Molecular subtype, three classes. The authors never attempted "
                    "this; it is this thesis's target."),
}

# =========================================================================== #
#  CONFIGURATION                                                              #
# =========================================================================== #


@dataclass
class Config:
    """Everything one experiment needs. Override any field at construction."""

    # ---- identity -------------------------------------------------------- #
    pipeline: Literal["reference", "thesis"] = PIPELINE
    task: str = TASK
    model: str = MODEL

    # ---- data ------------------------------------------------------------ #
    # Which cohorts to use. The authors pool all three; this thesis defaults to
    # I-SPY2 alone, because the source probe reaches macro-AUC 0.9978 predicting
    # the cohort on pooled data — the images identify the scanner almost perfectly.
    cohorts: tuple[str, ...] = ("spy2",)
    image_size: int = 224
    batch_size: int = 24
    num_workers: int = 8

    # At most one slice per patient per batch. Neighbouring slices of one tumour
    # are near-duplicates and give almost the same gradient, which removes the
    # stochastic noise that makes SGD generalise.
    max_slices_per_patient_per_batch: int = 1

    # ---- optimisation ---------------------------------------------------- #
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-4          # 3e-5 for transformers — see lr_for_model()
    weight_decay: float = 5e-4
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    epochs: int = 100
    early_stopping_patience: int = 30
    label_smoothing: float = 0.1

    # Inverse-frequency weights counted per PATIENT, not per slice: counting
    # slices conflates how many patients carry a class with how large their
    # tumours are.
    class_weighted_loss: bool = True

    # ---- regularisation -------------------------------------------------- #
    dropout: float = 0.5
    # "none" | "layer3" (freeze conv1+bn1+layer1+layer2) | "layer4"
    freeze_until: str = "none"
    freeze_bn: bool = False
    mixup_alpha: float = 0.0

    # ---- augmentation ---------------------------------------------------- #
    # "default" | "half" (same amplitudes, applied 50% of the time) | "none"
    # Measured: "half" tripled the train/test gap (0.135 -> 0.512) and lost 0.040
    # macro-AUC. The default is what holds this model back from memorising.
    augmentation: str = "default"

    # ---- reproducibility ------------------------------------------------- #
    seed: int = 42
    mixed_precision: bool = True

    # ---- evaluation ------------------------------------------------------ #
    # Slice probabilities are aggregated per patient before any metric. The
    # authors use median for HER2; mean elsewhere.
    aggregation: Literal["mean", "median"] = "mean"
    monitor_metric: str = "auc"          # what selects the best checkpoint

    # ---- filled in automatically ----------------------------------------- #
    dataset_name: str = ""               # resolved from pipeline + task
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}. Known: {list(TASKS)}")
        spec = TASKS[self.task]
        if spec.pipeline != self.pipeline:
            raise ValueError(
                f"task {self.task!r} belongs to the {spec.pipeline!r} pipeline, "
                f"not {self.pipeline!r}. Mixing them would make the comparison "
                f"between pipelines meaningless.")
        if not self.dataset_name:
            self.dataset_name = f"{self.pipeline}_{self.task}"

    # ---- derived --------------------------------------------------------- #
    @property
    def task_spec(self) -> Task:
        return TASKS[self.task]

    @property
    def num_classes(self) -> int:
        return self.task_spec.num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.task_spec.classes

    @property
    def dataset_dir(self) -> Path:
        return DATA_DIR / self.dataset_name

    def as_dict(self) -> dict:
        d = asdict(self)
        d["num_classes"] = self.num_classes
        d["class_names"] = list(self.class_names)
        return d

    def summary(self) -> str:
        return "\n".join([
            f"pipeline   : {self.pipeline}",
            f"task       : {self.task} — {self.task_spec.description}",
            f"classes    : {self.num_classes} — {', '.join(self.class_names)}",
            f"model      : {self.model}",
            f"cohorts    : {', '.join(self.cohorts)}",
            f"dataset    : {self.dataset_dir}",
            f"optimiser  : {self.optimizer} lr={self.learning_rate} wd={self.weight_decay}",
            f"schedule   : {self.scheduler}, {self.epochs} epochs, patience "
            f"{self.early_stopping_patience}",
            f"batch      : {self.batch_size} @ {self.image_size}px",
            f"regularise : dropout {self.dropout}, freeze {self.freeze_until}, "
            f"aug {self.augmentation}",
            f"seed       : {self.seed}",
        ])


def lr_for_model(model: str, default: float = 1e-4) -> float:
    """Learning rate appropriate to the architecture family.

    A transformer diverges at the rate swept for a CNN. This project watched a ViT
    sit at balanced accuracy exactly 0.5000 for sixteen epochs at 1e-4 before the
    cause was found. Transformers get 3e-5.
    """
    return 3e-5 if model.startswith(("vit", "swin")) else default
