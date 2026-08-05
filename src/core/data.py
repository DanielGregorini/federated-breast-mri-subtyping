"""Datasets, samplers and augmentation. Shared by both pipelines.

What is NOT here is as important as what is: this module reads PNGs that a
pipeline already produced. Everything that differs between the authors' pipeline
and mine — the crop, the normalisation of the volume, the slice selection — is
baked into those PNGs by `pipelines/*/preprocessing.py`. By the time an image
reaches this file the two pipelines are indistinguishable, which is exactly what
makes them comparable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

# ImageNet statistics — the backbones were pretrained with these.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# Augmentation                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AugmentConfig:
    """MRI-appropriate augmentation, each transform independently switchable.

    Deliberately absent, and why:
      * MixUp/CutMix at image level — they blend lesions from different patients;
        the diagnostic signal in MRI is small, local and low-contrast.
      * Aggressive vertical flip — a cranio-caudal flip produces anatomy that does
        not exist. Left/right is fine: it looks like the contralateral breast.
      * Strong elastic deformation — it alters tumour morphology, which is
        precisely the feature being classified.
    """

    enabled: bool = True

    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0

    # Rotation, zoom and translation are drawn independently but composed into ONE
    # affine transform. Three successive `grid_sample` calls would blur the image
    # three times over.
    rotation_degrees: float = 15.0
    rotation_probability: float = 1.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    scale_probability: float = 1.0
    translate_fraction: float = 0.08
    translate_probability: float = 1.0

    brightness_range: tuple[float, float] = (0.8, 1.2)
    brightness_probability: float = 1.0

    gaussian_noise_std: tuple[float, float] = (0.005, 0.03)
    noise_probability: float = 0.25

    cutout_probability: float = 0.0


# Same amplitudes, applied half the time. Measured on the 3-class subtype task:
# training accuracy rose from 0.57 to 0.99 and the train/test gap tripled
# (0.135 -> 0.512). Kept for the record; not recommended.
AUGMENT_HALF = replace(
    AugmentConfig(), rotation_probability=0.5, scale_probability=0.5,
    translate_probability=0.5, brightness_probability=0.5)

AUGMENT_NONE = replace(AugmentConfig(), enabled=False)

PROFILES: dict[str, AugmentConfig] = {
    "default": AugmentConfig(),
    "half": AUGMENT_HALF,
    "none": AUGMENT_NONE,
}


def augment_for(profile: str) -> AugmentConfig:
    if profile not in PROFILES:
        raise ValueError(f"unknown augmentation profile {profile!r}. "
                         f"Known: {list(PROFILES)}")
    return PROFILES[profile]


def apply_augment(img: torch.Tensor, aug: AugmentConfig) -> torch.Tensor:
    """Augment one image, shape (3, H, W), values in [0, 1]."""
    import torch.nn.functional as F

    if random.random() < aug.horizontal_flip:
        img = torch.flip(img, dims=[2])
    if random.random() < aug.vertical_flip:
        img = torch.flip(img, dims=[1])

    do_rot = aug.rotation_degrees > 0 and random.random() < aug.rotation_probability
    do_scale = aug.scale_range != (1.0, 1.0) and random.random() < aug.scale_probability
    do_trans = aug.translate_fraction > 0 and random.random() < aug.translate_probability

    if do_rot or do_scale or do_trans:
        angle = np.deg2rad(random.uniform(-aug.rotation_degrees,
                                          aug.rotation_degrees)) if do_rot else 0.0
        scale = random.uniform(*aug.scale_range) if do_scale else 1.0
        tx = random.uniform(-aug.translate_fraction, aug.translate_fraction) if do_trans else 0.0
        ty = random.uniform(-aug.translate_fraction, aug.translate_fraction) if do_trans else 0.0
        cos_a, sin_a = np.cos(angle) / scale, np.sin(angle) / scale
        theta = img.new_tensor([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]).unsqueeze(0)
        grid = F.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False).squeeze(0)

    if aug.brightness_range != (1.0, 1.0) and random.random() < aug.brightness_probability:
        img = torch.clamp(img * random.uniform(*aug.brightness_range), 0, 1)
    if aug.gaussian_noise_std and random.random() < aug.noise_probability:
        sigma = random.uniform(*aug.gaussian_noise_std)
        img = torch.clamp(img + torch.randn_like(img) * sigma, 0, 1)

    if aug.cutout_probability > 0 and random.random() < aug.cutout_probability:
        _, h, w = img.shape
        side = int(min(h, w) * random.uniform(0.1, 0.3))
        if side > 0:
            top, left = random.randint(0, h - side), random.randint(0, w - side)
            img = img.clone()
            img[:, top:top + side, left:left + side] = 0.0
    return img


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class SliceDataset(Dataset):
    """One PNG per item. The patient id is kept for patient-level aggregation."""

    def __init__(self, rows: pd.DataFrame, images_dir: Path, image_size: int,
                 augment: AugmentConfig | None = None) -> None:
        self.rows = rows.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        r = self.rows.iloc[i]
        img = Image.open(self.images_dir / r.filename).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.augment is not None and self.augment.enabled:
            x = apply_augment(x, self.augment)
        # ImageNet normalisation LAST, after augmentation: brightness and noise
        # operate in [0, 1] where clamping to [0, 1] is meaningful.
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(r.label), i


class PatientBatchSampler(Sampler):
    """Batches holding at most `max_per_patient` slices of any one patient.

    Plain shuffling draws 24 indices from thousands of slices belonging to
    hundreds of patients, so a batch routinely contains several near-duplicate
    slices of the same tumour. Those produce almost the same gradient, which
    quietly removes the stochastic noise that makes SGD generalise.

    Every slice is still seen exactly once per epoch — this is not subsampling.
    Only batch composition changes.
    """

    def __init__(self, pids: list[str], batch_size: int, max_per_patient: int = 1,
                 seed: int = 0) -> None:
        self.batch_size = batch_size
        self.max_per_patient = max(1, max_per_patient)
        self.seed = seed
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for i, pid in enumerate(pids):
            self.groups.setdefault(pid, []).append(i)
        self.n_samples = len(pids)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle differently each epoch; identical batch composition every
        epoch would itself be a form of overfitting."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        queues = {p: rng.sample(idx, len(idx)) for p, idx in self.groups.items()}
        batch: list[int] = []
        used: dict[str, int] = {}
        while queues:
            available = list(queues)
            rng.shuffle(available)
            added = 0
            for pid in available:
                if len(batch) >= self.batch_size:
                    break
                if used.get(pid, 0) >= self.max_per_patient:
                    continue
                batch.append(queues[pid].pop())
                used[pid] = used.get(pid, 0) + 1
                added += 1
                if not queues[pid]:
                    del queues[pid]
            # `used` resets when the batch is EMITTED, not on every pass. Resetting
            # per pass let a patient enter the same batch twice once few patients
            # remained — precisely what this sampler exists to prevent.
            if len(batch) >= self.batch_size or added == 0 or not queues:
                if batch:
                    yield batch
                batch, used = [], {}
        if batch:
            yield batch

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def effective_num_workers(requested: int) -> int:
    """Zero on macOS: DataLoader workers use `spawn` there and reliably hang when
    the parent holds an MPS context. Costs little — data loading is ~10% of a
    step. Throughput only; cannot change a result."""
    import platform
    return 0 if platform.system() == "Darwin" else requested


def make_loaders(dataset_dir: Path, cfg) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    """train / val / test loaders and the frames behind them.

    Returns the frames too, because every downstream step — class weights,
    patient-level aggregation, per-cohort reporting — must score exactly the rows
    the loader fed, not a re-read of the CSV.
    """
    images = Path(dataset_dir) / "images"
    workers = effective_num_workers(cfg.num_workers)
    aug = augment_for(cfg.augmentation)

    loaders: dict[str, DataLoader] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        csv = Path(dataset_dir) / f"{split}.csv"
        if not csv.is_file():
            continue
        rows = pd.read_csv(csv)
        frames[split] = rows
        ds = SliceDataset(rows, images, cfg.image_size,
                          aug if split == "train" else None)
        common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)
        if split == "train":
            sampler = PatientBatchSampler(rows.pid.tolist(), cfg.batch_size,
                                          cfg.max_slices_per_patient_per_batch,
                                          seed=cfg.seed)
            loaders[split] = DataLoader(ds, batch_sampler=sampler, **common)
        else:
            loaders[split] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                        drop_last=False, **common)
    return loaders, frames


def class_weights(rows: pd.DataFrame, num_classes: int, device) -> torch.Tensor:
    """Inverse-frequency weights counted per PATIENT, not per slice.

    Slice counts conflate two different things: how many patients carry a class,
    and how large their tumours are. A subtype whose tumours happen to be large
    would get a small weight simply for contributing more slices.
    """
    per_patient = rows.groupby("pid").label.first()
    counts = np.bincount(per_patient.values, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(per_patient) / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def trivial_baseline(rows: pd.DataFrame) -> float:
    """Majority-class rate among PATIENTS. Accuracy is meaningless without it,
    and it changes with the cohort — 0.404 on I-SPY2 alone, 0.511 pooled."""
    per_patient = rows.drop_duplicates("pid").label
    return float(per_patient.value_counts().max() / len(per_patient)) if len(per_patient) else float("nan")


def set_seed(seed: int) -> None:
    """Pin what can be pinned.

    Note what this does NOT pin: cuDNN kernel selection, AMP, and DataLoader
    worker ordering. Two runs of a byte-identical configuration differing only in
    seed were measured 0.067 macro-AUC apart on this task. One seed is not a
    result.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
