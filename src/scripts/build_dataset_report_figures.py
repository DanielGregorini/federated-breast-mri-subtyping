#!/usr/bin/env python3
"""Every figure in `docs/DATASET_REPORT.md`, regenerated from the data.

    python3 scripts/build_dataset_report_figures.py

Writes to `docs/report_figures/`. Nothing here is hand-drawn or illustrative: each
panel is produced from the source NIfTI volumes and the built dataset, so a figure
and the report's tables cannot disagree.

THE PREPROCESSING PANELS RE-RUN THE REAL PIPELINE
-------------------------------------------------
`stage_panels` does not approximate the four stages — it calls the same functions
`core/dataset_builder.py` calls, in the same order, with the same parameters read
from the dataset's own `config.json`. A figure claiming to show "what the network
sees" has to BE what the network sees, or it is decoration.
"""

from __future__ import annotations

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
from matplotlib.patches import Rectangle                       # noqa: E402

import dataset_config as CFG                                           # noqa: E402
from core import dataset_builder as DB                         # noqa: E402
from pipelines.thesis import preprocessing as P                  # noqa: E402

DATASET = REPO_ROOT / "dataset" / "multi_subtype_80mm"


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

OUT = REPO_ROOT / "docs" / "images" / "report_figures"
CLASSES = ["HRposHER2neg", "TripleNeg", "HER2pos"]
CLASS_LABEL = {"HRposHER2neg": "HR+/HER2−", "TripleNeg": "Triple Negative",
               "HER2pos": "HER2+"}
COHORTS = ["spy2", "duke", "spy1"]
COHORT_LABEL = {"spy2": "I-SPY2", "duke": "Duke", "spy1": "I-SPY1"}
CLASS_COLOUR = ["#0072B2", "#D55E00", "#009E73"]
COHORT_COLOUR = {"spy2": "#0072B2", "duke": "#D55E00", "spy1": "#009E73"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 12.6,   # 9 * 1.4
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# --------------------------------------------------------------------------- #
# 1. Composition                                                               #
# --------------------------------------------------------------------------- #
# Each panel is a function of one axis, so the SAME code draws it inside the
# six-panel overview and inside its own standalone file. Two drawing paths for one
# panel is how a combined figure and a standalone figure drift apart.

def panel_patients_per_cohort(a, pat, meta):
    v = [int((pat.cohort == c).sum()) for c in COHORTS]
    b = a.bar([COHORT_LABEL[c] for c in COHORTS], v,
              color=[COHORT_COLOUR[c] for c in COHORTS])
    for r, n in zip(b, v):
        a.text(r.get_x() + r.get_width() / 2, n + max(v) * 0.015,
               f"{n:,}\n{100 * n / sum(v):.1f}%", ha="center", fontsize=10.5)
    a.set_title("Patients per cohort", fontsize=13.3)
    a.set_ylabel("patients"); a.set_ylim(0, max(v) * 1.28)


def panel_images_per_cohort(a, pat, meta):
    v = [int((meta.cohort == c).sum()) for c in COHORTS]
    b = a.bar([COHORT_LABEL[c] for c in COHORTS], v,
              color=[COHORT_COLOUR[c] for c in COHORTS])
    for r, n in zip(b, v):
        a.text(r.get_x() + r.get_width() / 2, n + max(v) * 0.015,
               f"{n:,}\n{100 * n / sum(v):.1f}%", ha="center", fontsize=10.5)
    a.set_title("Images per cohort", fontsize=13.3)
    a.set_ylabel("images"); a.set_ylim(0, max(v) * 1.28)


def panel_per_class(a, pat, meta):
    pv = [int((pat.label_name == c).sum()) for c in CLASSES]
    iv = [int((meta.label_name == c).sum()) for c in CLASSES]
    x = np.arange(3)
    a.bar(x - 0.2, pv, 0.4, color=CLASS_COLOUR)
    a2 = a.twinx(); a2.grid(False)
    a2.bar(x + 0.2, iv, 0.4, color=CLASS_COLOUR, alpha=0.45)
    for i, (p_, i_) in enumerate(zip(pv, iv)):
        a.text(i - 0.2, p_ + max(pv) * 0.015, f"{p_:,}", ha="center", fontsize=9.8)
        a2.text(i + 0.2, i_ + max(iv) * 0.015, f"{i_:,}", ha="center", fontsize=9.8,
                alpha=0.8)
    a.set_xticks(x, [CLASS_LABEL[c] for c in CLASSES], fontsize=11.2)
    a.set_title("Patients and images per class", fontsize=13.3)
    a.set_ylabel("patients"); a2.set_ylabel("images")
    a.set_ylim(0, max(pv) * 1.25); a2.set_ylim(0, max(iv) * 1.25)


def panel_class_within_cohort(a, pat, meta):
    """Stacked to 100%, so there is NO free space inside the axes for a legend.

    The legend therefore goes BELOW the axes. Placing it inside covered the
    percentage labels on the first bar, which are the point of the panel.
    """
    bottom = np.zeros(len(COHORTS))
    for i, c in enumerate(CLASSES):
        share = np.array([100 * ((pat.cohort == co) & (pat.label_name == c)).sum()
                          / max((pat.cohort == co).sum(), 1) for co in COHORTS])
        bars = a.bar([COHORT_LABEL[co] for co in COHORTS], share, bottom=bottom,
                     color=CLASS_COLOUR[i], label=CLASS_LABEL[c])
        for r, sh in zip(bars, share):
            if sh > 6:
                a.text(r.get_x() + r.get_width() / 2, r.get_y() + sh / 2,
                       f"{sh:.1f}%", ha="center", va="center", fontsize=10.5,
                       color="white")
        bottom += share
    a.set_title("Class composition within each cohort", fontsize=12.6)
    a.set_ylabel("% of the cohort"); a.set_ylim(0, 100)
    a.legend(fontsize=10.5, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.12), frameon=False)


def panel_splits(a, pat, meta):
    order = ["train", "val", "test"]
    bottom = np.zeros(len(order))
    for i, c in enumerate(CLASSES):
        v = np.array([int(((pat.split == s) & (pat.label_name == c)).sum())
                      for s in order])
        a.bar(order, v, bottom=bottom, color=CLASS_COLOUR[i], label=CLASS_LABEL[c])
        bottom += v
    for i, s in enumerate(order):
        a.text(i, bottom[i] + bottom.max() * 0.015, f"{int(bottom[i]):,}",
               ha="center", fontsize=10.5)
    a.set_title("Patients per split", fontsize=13.3)
    a.set_ylabel("patients"); a.set_ylim(0, bottom.max() * 1.3)
    a.legend(fontsize=10.5, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.12), frameon=False)


def panel_spacing(a, pat, meta):
    for co in COHORTS:
        sp = pat[pat.cohort == co].xy_spacing.dropna()
        a.hist(sp, bins=28, alpha=0.55, color=COHORT_COLOUR[co],
               label=f"{COHORT_LABEL[co]} (median {sp.median():.3f})")
    a.set_title("In-plane pixel spacing per cohort", fontsize=12.6)
    a.set_xlabel("mm / pixel"); a.set_ylabel("patients"); a.legend(fontsize=9.8)


# Every panel is still written as its own file — nothing is lost.
PANELS = [
    ("p1_patients_per_cohort", panel_patients_per_cohort),
    ("p2_images_per_cohort", panel_images_per_cohort),
    ("p3_patients_images_per_class", panel_per_class),
    ("p4_class_within_cohort", panel_class_within_cohort),
    ("p5_train_val_test", panel_splits),
    ("p6_pixel_spacing", panel_spacing),
]

# ...but the OVERVIEW carries four, not six, so each is large enough to read in a
# printed thesis. The two dropped ones are the two that add least:
#
#   * "images per cohort" is ~redundant with "patients per cohort" — every patient
#     contributes at most 8 images, so the two share the same proportions to within
#     0.3 pp (47.6/44.3/8.1 vs 47.8/44.0/8.1). The per-class panel already carries
#     an image count.
#   * "train / validation / test" is three numbers (1,527 / 268 / 268). A table
#     states them better than a bar chart, and §3.3 of the report has that table.
#
# Both remain available as standalone files for anyone who wants them.
OVERVIEW_PANELS = [
    ("p1_patients_per_cohort", panel_patients_per_cohort),
    ("p3_patients_images_per_class", panel_per_class),
    ("p4_class_within_cohort", panel_class_within_cohort),
    ("p6_pixel_spacing", panel_spacing),
]


def composition(meta: pd.DataFrame) -> None:
    pat = meta.drop_duplicates("pid")

    # the four-panel overview — 2x2 so each panel is roughly twice the area it had
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.2))
    for (name, fn), a in zip(OVERVIEW_PANELS, ax.ravel()):
        fn(a, pat, meta)
    fig.suptitle(f"Dataset composition — {pat.pid.nunique():,} patients, "
                 f"{len(meta):,} images", fontsize=15.4, y=1.01)
    fig.tight_layout()
    save(fig, "fig1_dataset_composition")

    # and each panel on its own, from the SAME functions
    for name, fn in PANELS:
        fig, a = plt.subplots(figsize=(5.4, 4.2))
        fn(a, pat, meta)
        fig.tight_layout()
        save(fig, f"fig1_{name}")


def tumour_size(meta: pd.DataFrame) -> None:
    """Tumour size by cohort — the measurable difference the crop must preserve."""
    pat = meta.drop_duplicates("pid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

    a = ax[0]
    data, labels = [], []
    for co in COHORTS:
        v = pat[(pat.cohort == co) & (pat.tum_vol > 0)].tum_vol.dropna()
        if len(v):
            data.append(v); labels.append(f"{COHORT_LABEL[co]}\nn={len(v)}")
    bp = a.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, co in zip(bp["boxes"], COHORTS):
        patch.set_facecolor(COHORT_COLOUR[co]); patch.set_alpha(0.6)
    for i, v in enumerate(data):
        a.text(i + 1, v.median(), f" {v.median():.1f}", va="center", fontsize=10.5)
    a.set_title("Tumour volume by cohort (metadata `tum_vol`)", fontsize=13.3)
    a.set_ylabel("volume (dataset units)")

    a = ax[1]
    for co in COHORTS:
        v = pat[(pat.cohort == co) & (pat.tumor_area_mm2 > 0)].tumor_area_mm2.dropna()
        if len(v):
            a.hist(v, bins=40, alpha=0.55, color=COHORT_COLOUR[co],
                   label=f"{COHORT_LABEL[co]} (median {v.median():.0f} mm²)")
    a.set_title("Tumour area on the saved slice", fontsize=12.6)
    a.set_xlabel("mm²"); a.set_ylabel("images"); a.legend(fontsize=9.8)

    fig.tight_layout()
    save(fig, "fig2_tumour_size_by_cohort")


# --------------------------------------------------------------------------- #
# 2. Example final images, one per cohort x class                              #
# --------------------------------------------------------------------------- #
def examples(meta: pd.DataFrame, rng: np.random.Generator) -> None:
    """One final training image per cohort and class.

    LAYOUT NOTE
    -----------
    Each panel carries a title above and a caption below, and in a tight 3x3 grid
    the caption of one row lands on the title of the next. `constrained_layout`
    with an explicit `hspace` reserves that room instead of letting the two
    collide, which no amount of shrinking the images would have fixed on its own.
    """
    from PIL import Image

    # Panels 30% smaller (linear); text 10% larger than the surrounding figures.
    fig, axes = plt.subplots(3, 3, figsize=(5.9, 6.9), layout="constrained")
    fig.get_layout_engine().set(hspace=0.14, wspace=0.04)
    chosen: dict[tuple[str, str], tuple[str, int]] = {}

    for r, co in enumerate(COHORTS):
        for c, cl in enumerate(CLASSES):
            a = axes[r][c]
            sub = meta[(meta.cohort == co) & (meta.label_name == cl)]
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            if sub.empty:
                a.text(0.5, 0.5, "none", ha="center", va="center"); continue
            # the middle slice of a random patient — not the prettiest one
            pid = rng.choice(sub.pid.unique())
            rows = sub[sub.pid == pid].sort_values("slice_order")
            row = rows.iloc[len(rows) // 2]
            chosen[(co, cl)] = (pid, int(row.slice_index))
            a.imshow(Image.open(DATASET / "images" / row.filename))
            a.set_title(f"{COHORT_LABEL[co]} — {CLASS_LABEL[cl]}", fontsize=9.4,
                        pad=3)
            # patient id and source spacing on one line, small and unobtrusive
            a.set_xlabel(f"{pid} · {row.xy_spacing:.3f} mm/px", fontsize=7.6,
                         labelpad=2)

    fig.suptitle("Final 224x224 training images\n"
                 "RGB = (pre-contrast, early post-contrast, late post-contrast)",
                 fontsize=11.0)
    save(fig, "fig3_examples_cohort_class")
    return chosen


def examples_raw(meta: pd.DataFrame, chosen: dict) -> None:
    """The SAME slices as Figure 3, before any of our preprocessing.

    WHAT "RAW" MEANS HERE, PRECISELY
    --------------------------------
    These are the source NIfTI volumes as released. They have already been through
    the dataset authors' pipeline — DICOM to NIfTI conversion and the MinCrop
    tumour-centred crop — and that cannot be undone from what the repository holds.
    The original DICOM series are not present.

    So this figure shows what is raw RELATIVE TO US: no volume min-max
    normalisation, no 80 mm window, no resize to 224, no 8-bit conversion, and no
    RGB fusion. A single phase is shown in greyscale, at the volume's native
    in-plane resolution.

    The display uses per-slice windowing (vmin/vmax of that slice). That is a
    DISPLAY choice, not a data change — a float array cannot be rendered without
    one — and it is stated on each panel so it is not mistaken for normalisation.
    """
    import nibabel as nib

    src = pd.read_csv(CFG.METADATA_CSV, low_memory=False)
    fig, axes = plt.subplots(3, 3, figsize=(5.9, 6.9), layout="constrained")
    fig.get_layout_engine().set(hspace=0.14, wspace=0.04)

    for r, co in enumerate(COHORTS):
        for c, cl in enumerate(CLASSES):
            a = axes[r][c]
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            key = (co, cl)
            if key not in chosen:
                a.text(0.5, 0.5, "none", ha="center", va="center"); continue
            pid, z = chosen[key]
            row = src[src.pid == pid].iloc[0]
            root = cohort_root(co)
            paths, phases = DB._phase_paths(
                root / DB.DCE_SUBDIR[co], pid, co,
                [row["pre"], row["post_early"], row["post_late"]])
            # the EARLY POST-CONTRAST phase alone, unfused and unnormalised
            vol = np.asarray(nib.load(str(paths[1])).dataobj, dtype=np.float32)
            plane = np.nan_to_num(vol[z], nan=0.0, posinf=0.0, neginf=0.0)

            a.imshow(plane, cmap="gray", vmin=float(plane.min()),
                     vmax=float(plane.max()))
            a.set_title(f"{COHORT_LABEL[co]} — {CLASS_LABEL[cl]}", fontsize=9.4,
                        pad=3)
            a.set_xlabel(f"{plane.shape[1]}x{plane.shape[0]} px · "
                         f"{row['xy_spacing']:.3f} mm/px", fontsize=7.6,
                         labelpad=2)

    # Self-contained: no cross-reference to another figure, and it states only
    # what is being shown.
    fig.suptitle("Source slices before preprocessing\n"
                 "early post-contrast phase, greyscale, native resolution",
                 fontsize=11.0)
    save(fig, "fig3b_examples_raw")


# --------------------------------------------------------------------------- #
# 3. The four preprocessing stages, re-running the real pipeline               #
# --------------------------------------------------------------------------- #
def stage_panels(meta: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> None:
    """Original slice -> ROI -> physical crop -> final image, per class."""
    import nibabel as nib
    from PIL import Image

    src = pd.read_csv(CFG.METADATA_CSV, low_memory=False)

    # Three masked rows (one per class) plus one Duke row. Duke earns its own row
    # because its ROI is a BOUNDING BOX, not a voxel mask — the single largest
    # methodological difference between the cohorts, and one a reader should see
    # rather than take on trust.
    rows = [(cl, "mask") for cl in CLASSES] + [("HRposHER2neg", "bbox")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13.5, 13.4))

    for r, (cl, want_roi) in enumerate(rows):
        pool = meta[(meta.label_name == cl) & (meta.roi_source == want_roi)]
        pid = rng.choice(pool.pid.unique())
        row = src[src.pid == pid].iloc[0]
        cohort = row["dataset"]
        root = cohort_root(cohort)

        paths, phases = DB._phase_paths(root / DB.DCE_SUBDIR[cohort], pid, cohort,
                                        [row["pre"], row["post_early"], row["post_late"]])
        vols = [np.nan_to_num(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0) for p in paths]
        vol = np.stack(vols)                                   # (3, Z, Y, X)

        if want_roi == "mask":
            mf = next((root / DB.MASK_SUBDIR[cohort]).glob(f"{pid}_*_mask.nii.gz"))
            mask = np.asarray(nib.load(str(mf)).dataobj) > 0
            roi = DB._roi_from_mask(mask, cfg["min_tumor_px"])
        else:
            mask = None
            roi = DB._roi_from_box(row, vol.shape[1])

        # exactly the builder's order: normalise the whole volume, then select
        norm = P.normalize_volume(vol)
        chosen = P.select_slices(roi.zs, cfg["n_slices"], cfg["trim_frac"])
        z = int(chosen[len(chosen) // 2])
        spacing = float(row["xy_spacing"])
        box, side = P.crop_window(roi.row0, roi.row1, roi.col0, roi.col1,
                                  spacing, cfg["crop_mm"])

        # (a) original slice, RGB-fused, before cropping
        a = axes[r][0]
        a.imshow(np.clip(norm[:, z], 0, 1).transpose(1, 2, 0))
        a.set_title("1. Fused RGB slice (full field)" if r == 0 else "", fontsize=12.6)
        a.set_ylabel(f"{CLASS_LABEL[cl]}\n{COHORT_LABEL[cohort]} · {pid}",
                     fontsize=11.2)
        a.set_xlabel(f"{norm.shape[3]}x{norm.shape[2]} px · {spacing:.3f} mm/px",
                     fontsize=9.1)

        # (b) ROI: mask contour + the 80 mm window that will be taken
        a = axes[r][1]
        a.imshow(np.clip(norm[:, z], 0, 1).transpose(1, 2, 0))
        if mask is not None:
            a.contour(mask[z], levels=[0.5], colors="yellow", linewidths=1.2)
        a.add_patch(Rectangle((roi.col0, roi.row0), roi.col1 - roi.col0,
                              roi.row1 - roi.row0, fill=False, ec="cyan", lw=1.0,
                              ls=":"))
        a.add_patch(Rectangle((box[2], box[0]), side, side, fill=False,
                              ec="red", lw=1.4))
        a.set_title("2. Tumour ROI + 80 mm window" if r == 0 else "", fontsize=12.6)
        if want_roi == "bbox":
            a.text(0.02, 0.98, "bounding box only\n(no voxel mask in Duke)",
                   transform=a.transAxes, va="top", fontsize=9.1, color="cyan")
        a.set_xlabel(f"window {side} px = {side * spacing:.0f} mm", fontsize=9.1)

        # (c) the crop, before resizing
        a = axes[r][2]
        crop = DB._crop_pad(norm[:, z], box)
        a.imshow(np.clip(crop, 0, 1).transpose(1, 2, 0))
        a.set_title("3. Cropped, before resize" if r == 0 else "", fontsize=12.6)
        a.set_xlabel(f"{side}x{side} px", fontsize=9.1)

        # (d) the final saved PNG — read from disk, not recomputed
        a = axes[r][3]
        saved = meta[(meta.pid == pid) & (meta.slice_index == z)]
        if len(saved):
            a.imshow(Image.open(DATASET / "images" / saved.iloc[0].filename))
            lab = "read from disk"
        else:
            img = (np.clip(crop, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            a.imshow(np.asarray(Image.fromarray(img).resize((224, 224),
                                                            Image.LANCZOS)))
            lab = "recomputed (slice not among the 8 kept)"
        a.set_title("4. Final training image" if r == 0 else "", fontsize=12.6)
        a.set_xlabel(f"224x224 · {spacing * side / 224:.3f} mm/px · {lab}",
                     fontsize=9.1)

        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)

    fig.suptitle("Preprocessing, stage by stage — produced by the same functions "
                 "the dataset builder calls\nR = pre-contrast, G = early "
                 "post-contrast, B = late post-contrast.  Yellow = voxel mask, "
                 "cyan dotted = tumour extent, red = 80 mm window",
                 fontsize=14.7, y=1.003)
    fig.tight_layout()
    save(fig, "fig4_preprocessing_stages")


def window_rationale(meta: pd.DataFrame) -> None:
    """Why a fixed physical window and not a proportional crop."""
    pat = meta.drop_duplicates("pid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

    a = ax[0]
    for co in COHORTS:
        s = pat[pat.cohort == co]
        # 224 output pixels drawn from an 80 mm window of `side` source pixels.
        # Values > 1 mean UPSAMPLING. Reported as 1.96x / 1.91x / 2.20x in
        # DATASET_REPORT.md; the reciprocal was plotted here by mistake.
        f = 224.0 / (80.0 / s.xy_spacing)
        a.hist(f.dropna(), bins=30, alpha=0.55, color=COHORT_COLOUR[co],
               label=f"{COHORT_LABEL[co]} (median {f.median():.2f}x)")
    a.set_title("Resampling factor applied per patient", fontsize=12.6)
    a.set_xlabel("resampling factor (output px per source px; >1 = upsampling)")
    a.set_ylabel("patients")
    a.legend(fontsize=9.8)

    a = ax[1]
    frac = pat[pat.tumor_fraction > 0].copy()
    for co in COHORTS:
        v = frac[frac.cohort == co].tumor_fraction * 100
        if len(v):
            a.hist(v, bins=35, alpha=0.55, color=COHORT_COLOUR[co],
                   label=f"{COHORT_LABEL[co]} (median {v.median():.1f}%)")
    a.set_title("Tumour share of the saved frame", fontsize=12.6)
    a.set_xlabel("% of frame occupied by tumour"); a.set_ylabel("images")
    a.legend(fontsize=9.8)

    fig.tight_layout()
    save(fig, "fig5_why_physical_window")


def main() -> None:
    meta = pd.read_csv(DATASET / "metadata.csv", low_memory=False)
    raw = json.loads((DATASET / "config.json").read_text())
    cfg = {"crop_mm": raw["crop_mm"], "n_slices": raw["n_slices"],
           "trim_frac": raw["trim_frac"], "min_tumor_px": raw["min_tumor_px"]}
    print(f"dataset: {meta.pid.nunique():,} patients, {len(meta):,} images")
    print(f"config : {cfg}")
    rng = np.random.default_rng(42)

    composition(meta)
    tumour_size(meta)
    chosen = examples(meta, rng)
    try:
        examples_raw(meta, chosen)
    except Exception as exc:
        print(f"  raw examples FAILED: {type(exc).__name__}: {exc}")
    window_rationale(meta)
    try:
        stage_panels(meta, cfg, rng)
    except Exception as exc:
        print(f"  stage panels FAILED: {type(exc).__name__}: {exc}")
        print("  (needs the source NIfTI volumes to be present)")
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
