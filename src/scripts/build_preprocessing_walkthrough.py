#!/usr/bin/env python3
"""The preprocessing pipeline, step by step, on ONE patient — plus a flowchart.

    python3 scripts/build_preprocessing_walkthrough.py
    python3 scripts/build_preprocessing_walkthrough.py --pid ISPY2-111881

Writes to `docs/preprocessing_figures/`. Companion to
`scripts/build_dataset_report_figures.py`: that script characterises the dataset
as a whole, this one follows a SINGLE slice of a SINGLE patient through every
transformation.

FIGURES CARRY STATE, NOT ARGUMENT
---------------------------------
Each panel is labelled with what the step DID and the resulting state — shape,
dtype, range, mm/px — and nothing else. The reasoning lives in
`docs/PREPROCESSING.md`, which the figures accompany. A figure that argues with
itself in marginal notes cannot be dropped into a dissertation unedited.

NOTHING HERE IS ILLUSTRATIVE
----------------------------
Every panel is produced by calling the same functions `core/dataset_builder.py`
calls, in the same order, with the parameters read from the dataset's own
`config.json`. The final panel is the PNG read back from disk — the actual file
the network trains on — not a recomputation of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
REPO_ROOT = SRC.parent

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import dataset_config as CFG                                           # noqa: E402
from core import dataset_builder as DB                         # noqa: E402
from pipelines.thesis import preprocessing as P                  # noqa: E402

DATASET = REPO_ROOT / "dataset" / "multi_subtype_80mm"
OUT = REPO_ROOT / "docs" / "images" / "preprocessing_figures"

COHORT_LABEL = {"spy2": "I-SPY2", "duke": "Duke", "spy1": "I-SPY1"}
CLASS_LABEL = {"HRposHER2neg": "HR+/HER2−", "TripleNeg": "Triple Negative",
               "HER2pos": "HER2+"}

# One patient with a voxel mask, so the lesion contour can be drawn rather than
# asserted. Overridable with --pid; any patient in the dataset works.
DEFAULT_PID = "ISPY2-111881"

# Image panels are 30% smaller (linear) than the first draft and every label is
# larger, matching the sizing used by `build_dataset_report_figures.py`.
IMG = 0.70

C_MASK = "#FFD400"      # voxel mask contour
C_ROI = "#00D5FF"       # tumour bounding box
C_WIN = "#FF2D2D"       # the 80 mm window
C_STEP = "#0072B2"      # build-time accents
C_TRAIN = "#D55E00"     # load-time (per-epoch) accents

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 13.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def cohort_root(cohort: str) -> Path:
    """Where this cohort's NIfTI volumes are.

    A single configured location, with no fallback. There used to be one: for a
    long time `COHORT_DIRS` pointed at the authors' cloned code repository, which
    ships the metadata CSV but none of the imaging, and these scripts silently
    resolved around it while the dataset builder could not have been re-run at
    all. Both now point at `raw_dataset_BreastDCEDL/`, so a missing directory is
    a real error and says so.
    """
    root = CFG.COHORT_DIRS[cohort]
    if (root / DB.DCE_SUBDIR[cohort]).is_dir():
        return root
    raise FileNotFoundError(
        f"no DCE directory for {cohort} at {root / DB.DCE_SUBDIR[cohort]}. "
        f"Download the imaging first: python raw_dataset_BreastDCEDL/download_dataset.py")


def blank(a):
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    return a


def state(a, text):
    """The resulting state under a panel. Facts only — no justification."""
    a.set_xlabel(text, fontsize=11.0, labelpad=4, color="0.2", linespacing=1.5)


def arrow(fig, a_from, a_to):
    """A plain arrow between two axes, in figure coordinates."""
    b0, b1 = a_from.get_position(), a_to.get_position()
    y = (b0.y0 + b0.y1) / 2
    fig.patches.append(FancyArrowPatch(
        (b0.x1 + 0.003, y), (b1.x0 - 0.003, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, lw=1.8, color=C_STEP, zorder=5))


# --------------------------------------------------------------------------- #
# Load one patient exactly as the builder does                                 #
# --------------------------------------------------------------------------- #
def load_patient(pid: str, cfg: dict, meta: pd.DataFrame) -> dict:
    """Every intermediate state of the real pipeline, for one patient."""
    import nibabel as nib

    src = pd.read_csv(CFG.METADATA_CSV, low_memory=False)
    row = src[src.pid == pid].iloc[0]
    cohort = row["dataset"]
    root = cohort_root(cohort)

    # ---- 1. read the three DCE phases (NaN/inf -> 0, as the builder does) ----
    paths, phases = DB._phase_paths(
        root / DB.DCE_SUBDIR[cohort], pid, cohort,
        [row["pre"], row["post_early"], row["post_late"]])
    raw_vols = [np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)
                for p in paths]
    vols = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in raw_vols]
    vol_raw = np.stack(vols)                                   # (3, Z, Y, X)

    # ---- 2. locate the lesion ----
    if cohort in DB.MASK_SUBDIR:
        mf = next((root / DB.MASK_SUBDIR[cohort]).glob(f"{pid}_*_mask.nii.gz"))
        mask = np.asarray(nib.load(str(mf)).dataobj) > 0
        roi = DB._roi_from_mask(mask, cfg["min_tumor_px"])
    else:
        mask = None
        roi = DB._roi_from_box(row, vol_raw.shape[1])

    # ---- 3. normalise the WHOLE volume, then select slices ----
    vol_norm = P.normalize_volume(vol_raw)
    chosen = P.select_slices(roi.zs, cfg["n_slices"], cfg["trim_frac"])

    # ---- 4. the 80 mm window ----
    spacing = float(row["xy_spacing"])
    box, side = P.crop_window(roi.row0, roi.row1, roi.col0, roi.col1,
                              spacing, cfg["crop_mm"])

    # The slice shown throughout: the middle one of the eight kept, so it is
    # representative rather than chosen for looks.
    z = int(chosen[len(chosen) // 2])

    saved = meta[(meta.pid == pid) & (meta.slice_index == z)]
    return dict(
        pid=pid, cohort=cohort, row=row, phases=phases, mask=mask, roi=roi,
        vol_raw=vol_raw, vol_norm=vol_norm, chosen=chosen, z=z,
        box=box, side=side, spacing=spacing, saved=saved,
        label_name=str(meta[meta.pid == pid].iloc[0].label_name),
    )


# --------------------------------------------------------------------------- #
# FIGURE 1 — the walkthrough: the same slice at every step                     #
# --------------------------------------------------------------------------- #
def fig_walkthrough(D: dict, cfg: dict) -> None:
    """One slice, five states, left to right. Labels state what was done.

    Steps 1 and 2 differ in the NUMBERS, not in the rendering: min–max is an
    affine rescale, and any float array must be display-windowed to be drawn at
    all. The state line under each panel is therefore what carries the change,
    and it is why the ranges are printed.
    """
    from PIL import Image

    z, box, side, sp = D["z"], D["box"], D["side"], D["spacing"]
    vr, vn, mask, roi = D["vol_raw"], D["vol_norm"], D["mask"], D["roi"]
    H, W = vr.shape[2], vr.shape[3]

    fig = plt.figure(figsize=(15.0 * IMG * 1.30, 5.0 * IMG * 1.30))
    gs = fig.add_gridspec(1, 5, wspace=0.16, left=0.02, right=0.985,
                          top=0.735, bottom=0.185)
    ax = [blank(fig.add_subplot(gs[0, i])) for i in range(5)]

    # (1) the source slice, as released
    disp = vr[:, z].transpose(1, 2, 0)
    ax[0].imshow(np.clip(disp / max(float(disp.max()), 1e-6), 0, 1))
    ax[0].set_title("1 · Source slice", fontsize=13.5, pad=6, color=C_STEP)
    state(ax[0], f"{W}×{H} px · {sp:.3f} mm/px\nfloat32 · range [0, {vr.max():.0f}]")

    # (2) min-max over the whole 4-D volume
    ax[1].imshow(np.clip(vn[:, z], 0, 1).transpose(1, 2, 0))
    ax[1].set_title("2 · Volume min–max", fontsize=13.5, pad=6, color=C_STEP)
    state(ax[1], f"{W}×{H} px · {sp:.3f} mm/px\nfloat32 · range [0, 1]")

    # (3) lesion located, 80 mm window placed
    ax[2].imshow(np.clip(vn[:, z], 0, 1).transpose(1, 2, 0))
    if mask is not None:
        ax[2].contour(mask[z], levels=[0.5], colors=C_MASK, linewidths=1.5)
    ax[2].add_patch(Rectangle((roi.col0, roi.row0), roi.col1 - roi.col0,
                              roi.row1 - roi.row0, fill=False, ec=C_ROI,
                              lw=1.3, ls=":"))
    ax[2].add_patch(Rectangle((box[2], box[0]), side, side, fill=False,
                              ec=C_WIN, lw=2.0))
    ax[2].set_title("3 · Lesion + 80 mm window", fontsize=13.5, pad=6,
                    color=C_STEP)
    state(ax[2], f"window {side}×{side} px\n= {side * sp:.0f} mm")

    # (4) the crop, zero-padded, before resizing
    crop = DB._crop_pad(vn[:, z], box)
    ax[3].imshow(np.clip(crop, 0, 1).transpose(1, 2, 0))
    ax[3].set_title("4 · Cropped", fontsize=13.5, pad=6, color=C_STEP)
    state(ax[3], f"{side}×{side} px · {sp:.3f} mm/px\nzero-padded")

    # (5) the final saved PNG, read back from disk
    if len(D["saved"]):
        ax[4].imshow(Image.open(DATASET / "images" / D["saved"].iloc[0].filename))
    else:
        img = (np.clip(crop, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        ax[4].imshow(np.asarray(Image.fromarray(img).resize((224, 224),
                                                            Image.LANCZOS)))
    ax[4].set_title("5 · Resized + 8-bit", fontsize=13.5, pad=6, color=C_STEP)
    state(ax[4], f"224×224 px · {sp * side / 224:.3f} mm/px\nuint8 RGB — saved")

    fig.canvas.draw()
    for i in range(4):
        arrow(fig, ax[i], ax[i + 1])

    fig.suptitle(
        f"Preprocessing — the same slice at every step\n"
        f"{D['pid']} · {COHORT_LABEL[D['cohort']]} · "
        f"{CLASS_LABEL[D['label_name']]} · z = {z}",
        fontsize=16.0, y=0.985)
    fig.text(0.5, 0.055,
             "yellow = voxel mask · cyan dotted = tumour extent · "
             "red = the 80 mm window",
             ha="center", fontsize=11.0, color="0.35")
    save(fig, "fig_p1_walkthrough")


# --------------------------------------------------------------------------- #
# FIGURE 2 — normalisation scope: ours against the authors'                    #
# --------------------------------------------------------------------------- #
def fig_normalisation(D: dict) -> None:
    """Volume min–max (ours) against per-slice min–max (the authors').

    WHY THIS IS THE COMPARISON AND NOT "RAW vs NORMALISED"
    -----------------------------------------------------
    Min–max is an affine rescale, so a raw-versus-normalised pair would show one
    curve with a relabelled axis and two identical-looking images. What changes
    the pixels is the SCOPE of the statistic: one (min, max) for the whole 4-D
    volume, or one per slice.

    Both arms call the real functions: `pipelines.thesis.normalize_volume` and
    `pipelines.reference.normalize`.
    """
    from pipelines.reference import preprocessing as AP

    vr, vn, chosen, box = D["vol_raw"], D["vol_norm"], D["chosen"], D["box"]
    roi = D["roi"]

    fig = plt.figure(figsize=(14.0 * IMG * 1.30, 8.0 * IMG * 1.30))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.0, 1.0, 1.30], hspace=0.22,
                          wspace=0.06, left=0.085, right=0.985, top=0.845,
                          bottom=0.085)

    for i, z in enumerate(chosen[:8]):
        a = blank(fig.add_subplot(gs[0, i]))
        a.imshow(np.clip(DB._crop_pad(vn[:, int(z)], box), 0, 1).transpose(1, 2, 0),
                 vmin=0, vmax=1)
        a.set_title(f"z = {int(z)}", fontsize=11.0, pad=3, color="0.25")
        if i == 0:
            a.set_ylabel("OURS\nwhole volume", fontsize=12.0, color=C_STEP)

    for i, z in enumerate(chosen[:8]):
        a = blank(fig.add_subplot(gs[1, i]))
        # the authors normalise the SLICE, jointly over its three channels
        a.imshow(np.clip(DB._crop_pad(AP.normalize(vr[:, int(z)]), box), 0, 1)
                 .transpose(1, 2, 0), vmin=0, vmax=1)
        if i == 0:
            a.set_ylabel("AUTHORS'\nper slice", fontsize=12.0, color=C_TRAIN)

    a = fig.add_subplot(gs[2, :])
    zs = roi.zs
    ours = [float(vn[:, int(z)].max()) for z in zs]
    theirs = [float(AP.normalize(vr[:, int(z)]).max()) for z in zs]

    a.plot(zs, ours, color=C_STEP, lw=2.2, marker="o", ms=3.4,
           label="ours — each slice keeps its true brightness")
    a.plot(zs, theirs, color=C_TRAIN, lw=2.2, ls="--", marker="s", ms=3.4,
           label="authors' — every slice forced to 1.0")
    for z in chosen:
        a.axvline(int(z), color="0.78", lw=1.0, ls=":", zorder=0)

    a.set_title("Maximum intensity of each slice", fontsize=13.0)
    a.set_xlabel("slice index z", fontsize=12.0)
    a.set_ylabel("max intensity", fontsize=12.0)
    a.set_ylim(0, 1.16)
    a.legend(fontsize=11.0, loc="lower center", ncol=2)

    fig.suptitle("Normalisation scope — whole volume or per slice\n"
                 f"{D['pid']} · the same eight slices under each rule",
                 fontsize=15.0, y=0.985)
    save(fig, "fig_p2_normalisation")


# --------------------------------------------------------------------------- #
# FIGURE 3 — slice selection, the step invisible in a single-slice view        #
# --------------------------------------------------------------------------- #
def fig_slice_selection(D: dict, cfg: dict) -> None:
    """Which slices are kept and which are discarded. The step acts along z."""
    roi, mask, chosen = D["roi"], D["mask"], D["chosen"]
    vn, box = D["vol_norm"], D["box"]
    zs = roi.zs

    fig = plt.figure(figsize=(14.0 * IMG * 1.30, 6.6 * IMG * 1.30))
    gs = fig.add_gridspec(2, 8, height_ratios=[1.25, 1.0], hspace=0.40,
                          wspace=0.06, left=0.065, right=0.985, top=0.845,
                          bottom=0.085)

    a = fig.add_subplot(gs[0, :])
    if mask is not None:
        areas = mask.reshape(mask.shape[0], -1).sum(1)
    else:
        areas = np.zeros(vn.shape[1]); areas[zs] = 1
    a.fill_between(np.arange(len(areas)), areas, color="0.82",
                   label="tumour pixels per slice")
    a.plot(np.arange(len(areas)), areas, color="0.55", lw=1.2)

    # headroom, so the legend and the trim labels never reach the curve
    a.set_ylim(0, max(float(areas.max()), 1.0) * 1.48)

    cut = int(np.floor(len(zs) * cfg["trim_frac"]))
    if cut and len(zs) - 2 * cut >= max(cfg["n_slices"], 3):
        for lo, hi, mid in [(zs[0] - 0.5, zs[cut] - 0.5, zs[cut // 2]),
                            (zs[len(zs) - cut] - 0.5, zs[-1] + 0.5,
                             zs[len(zs) - cut // 2 - 1])]:
            a.axvspan(lo, hi, color=C_WIN, alpha=0.13)
            a.text(mid, a.get_ylim()[1] * 0.87,
                   f"trimmed\n{cfg['trim_frac']:.0%}", ha="center",
                   fontsize=11.0, color=C_WIN)

    for i, z in enumerate(chosen):
        a.axvline(z, color=C_STEP, lw=1.8, ls="--", alpha=0.9,
                  label="kept (8, evenly spaced)" if i == 0 else None)
    a.axvline(D["z"], color=C_TRAIN, lw=2.6, alpha=0.9,
              label="the slice in Figure 1")

    a.set_title(f"{len(zs)} slices contain tumour · {len(chosen)} are kept "
                f"({100 * len(chosen) / len(zs):.0f}%)", fontsize=13.5)
    a.set_xlabel("slice index z", fontsize=12.0)
    a.set_ylabel("tumour pixels", fontsize=12.0)
    a.legend(fontsize=10.8, ncol=3, loc="upper center")
    a.set_xlim(max(zs[0] - 3, 0), min(zs[-1] + 3, len(areas) - 1))

    for i, z in enumerate(chosen[:8]):
        b = blank(fig.add_subplot(gs[1, i]))
        z = int(z)
        b.imshow(np.clip(DB._crop_pad(vn[:, z], box), 0, 1).transpose(1, 2, 0))
        b.set_title(f"z = {z}", fontsize=11.0, pad=3,
                    color=C_TRAIN if z == D["z"] else "0.25")
        if z == D["z"]:
            for s in b.spines.values():
                s.set_visible(True); s.set_color(C_TRAIN); s.set_linewidth(2.2)

    fig.suptitle("Slice selection — trim 15% off each end, then 8 evenly spaced",
                 fontsize=15.0, y=0.985)
    save(fig, "fig_p3_slice_selection")


# --------------------------------------------------------------------------- #
# FIGURE 4 — what happens every epoch, at load time                            #
# --------------------------------------------------------------------------- #
def fig_training_time(D: dict) -> None:
    """The PNG on disk is not what reaches the network."""
    import random

    import torch
    from PIL import Image

    from core.data import IMAGENET_MEAN, IMAGENET_STD, apply_augment, augment_for

    if not len(D["saved"]):
        print("  load-time figure skipped: shown slice is not on disk")
        return

    png = Image.open(DATASET / "images" / D["saved"].iloc[0].filename)
    base = torch.from_numpy(np.asarray(png).astype(np.float32) / 255.0)
    base = base.permute(2, 0, 1)                                # (3, 224, 224)

    fig = plt.figure(figsize=(14.0 * IMG * 1.30, 4.4 * IMG * 1.30))
    gs = fig.add_gridspec(1, 6, wspace=0.13, left=0.02, right=0.985,
                          top=0.700, bottom=0.145)

    a = blank(fig.add_subplot(gs[0, 0]))
    a.imshow(png)
    a.set_title("On disk", fontsize=13.0, pad=6)
    state(a, "uint8 RGB\n224×224")

    random.seed(7); torch.manual_seed(7)
    aug = augment_for("default")
    for i in range(4):
        a = blank(fig.add_subplot(gs[0, 1 + i]))
        a.imshow(np.clip(apply_augment(base.clone(), aug)
                         .permute(1, 2, 0).numpy(), 0, 1))
        a.set_title(f"Augmented · draw {i + 1}", fontsize=13.0, pad=6,
                    color=C_TRAIN)
        state(a, "a new draw\nevery epoch")

    a = blank(fig.add_subplot(gs[0, 5]))
    norm = (base - IMAGENET_MEAN) / IMAGENET_STD
    shown = (norm - norm.min()) / (norm.max() - norm.min())
    a.imshow(shown.permute(1, 2, 0).numpy())
    a.set_title("ImageNet normalised", fontsize=13.0, pad=6, color=C_TRAIN)
    state(a, f"mean {norm.mean():.2f} · std {norm.std():.2f}\n"
             "rescaled to draw")

    fig.suptitle("Load time — applied every epoch, training split only\n"
                 "flip · ±15° · zoom 0.9–1.1 · shift ±8% · "
                 "brightness ×0.8–1.2 · noise (p = 0.25)",
                 fontsize=15.0, y=0.985)
    save(fig, "fig_p4_load_time")


# --------------------------------------------------------------------------- #
# FIGURE 5 — the flowchart                                                     #
# --------------------------------------------------------------------------- #
def fig_flowchart() -> None:
    """The pipeline as a diagram: the steps, in order, and nothing else.

    Deliberately free of marginal justification — every "why" is a numbered
    section of `docs/PREPROCESSING.md`, and a figure that carries its own essay
    cannot be placed in a dissertation without being cut down first.
    """
    fig, ax = plt.subplots(figsize=(8.6, 12.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 116)
    ax.axis("off")

    def box(y, text, h=9.0, w=70, x=15, fc="#EAF2FA", ec=C_STEP, fs=14.0,
            bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
            fc=fc, ec=ec, lw=1.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.45,
                fontweight="bold" if bold else "normal")

    def down(y_from, y_to):
        ax.add_patch(FancyArrowPatch(
            (50, y_from), (50, y_to), arrowstyle="-|>", mutation_scale=18,
            lw=2.0, color="0.45", zorder=1))

    # ---------------- build time ----------------
    ax.add_patch(FancyBboxPatch(
        (3, 23.5), 94, 86.5, boxstyle="round,pad=0.8,rounding_size=2",
        fc="#FBFCFD", ec="#C3D2E0", lw=1.5, ls="--", zorder=0))
    ax.text(6.0, 108.6, "BUILD TIME — runs once", fontsize=14.5,
            fontweight="bold", color=C_STEP, va="top")

    steps = [
        (96.0, "DCE-MRI volumes\n3 phases per patient", "#F2F2F2", "0.5", True),
        (84.0, "Sanitise\nNaN and ±inf → 0", "#EAF2FA", C_STEP, False),
        (72.0, "Locate the lesion\nmask (I-SPY) or box (Duke)", "#EAF2FA",
         C_STEP, False),
        (60.0, "Min–max over the whole volume", "#EAF2FA", C_STEP, False),
        (48.0, "Keep 8 slices\ntrim 15% off each end", "#EAF2FA", C_STEP, False),
        (36.0, "Crop an 80 mm physical window", "#EAF2FA", C_STEP, False),
        (25.0, "8-bit → resize 224×224 → save PNG", "#DCEBF7", C_STEP, True),
    ]
    for i, (y, text, fc, ec, bold) in enumerate(steps):
        h = 7.2 if i == len(steps) - 1 else 9.0
        box(y, text, h=h, fc=fc, ec=ec, bold=bold)
        if i:
            # from the bottom edge of the previous box to the top edge of this one
            down(steps[i - 1][0], y + h)

    # ---------------- the artefact ----------------
    down(25.0, 20.5)
    ax.add_patch(FancyBboxPatch(
        (15, 13.5), 70, 7.0, boxstyle="round,pad=0.6,rounding_size=1.6",
        fc="#FFF4E0", ec=C_TRAIN, lw=2.0, zorder=2))
    ax.text(50, 17.0, "2,063 patients · 16,378 images\n224×224 uint8 RGB",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            zorder=3, linespacing=1.45)

    # ---------------- load time ----------------
    down(13.5, 8.0)
    ax.add_patch(FancyBboxPatch(
        (15, 1.0), 70, 7.0, boxstyle="round,pad=0.6,rounding_size=1.6",
        fc="#FDEBD8", ec=C_TRAIN, lw=1.8, zorder=2))
    ax.text(50, 4.5, "LOAD TIME, every epoch\naugment → ImageNet normalise",
            ha="center", va="center", fontsize=13.5, zorder=3, linespacing=1.45)

    fig.suptitle("Preprocessing pipeline", fontsize=18.0, y=0.955)
    fig.tight_layout()
    save(fig, "fig_p5_flowchart")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=DEFAULT_PID,
                    help="patient to walk through (must be in the dataset)")
    args = ap.parse_args()

    meta = pd.read_csv(DATASET / "metadata.csv", low_memory=False)
    raw = json.loads((DATASET / "config.json").read_text())
    cfg = {"crop_mm": raw["crop_mm"], "n_slices": raw["n_slices"],
           "trim_frac": raw["trim_frac"], "min_tumor_px": raw["min_tumor_px"]}

    if args.pid not in set(meta.pid):
        raise SystemExit(f"{args.pid} is not in the dataset")

    print(f"dataset: {meta.pid.nunique():,} patients, {len(meta):,} images")
    print(f"config : {cfg}")
    print(f"patient: {args.pid}")

    D = load_patient(args.pid, cfg, meta)
    print(f"  cohort {D['cohort']} · {D['label_name']} · "
          f"{len(D['roi'].zs)} tumour slices · kept {len(D['chosen'])} · "
          f"window {D['side']} px = {D['side'] * D['spacing']:.0f} mm")

    fig_walkthrough(D, cfg)
    fig_normalisation(D)
    fig_slice_selection(D, cfg)
    fig_training_time(D)
    fig_flowchart()
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
