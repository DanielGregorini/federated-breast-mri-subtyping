"""Turns MinCrop NIfTI volumes into a 2D PNG dataset for one pipeline and task.

One builder, two pipelines. The pipeline supplies three functions — slice
selection, crop window, normalisation — and this module does everything that is
common: reading volumes, locating the lesion, splitting by patient, writing PNGs
and a CSV, and verifying there is no leakage.

    from core.dataset_builder import build
    build(Config(pipeline="reference", task="pcr"))

WHY THE COHORTS ARE HANDLED DIFFERENTLY, AND WHY THAT IS NOT A DIFFERENCE
------------------------------------------------------------------------
I-SPY1 and I-SPY2 ship a voxel-level 3D mask. DUKE does not — it ships an
expert-drawn bounding box, which the dataset paper states explicitly (§2.3:
"Expert radiologists annotated bounding boxes on planes containing the largest
tumor area"). The `ROI` abstraction below carries whichever exists.

The box was VERIFIED against the real masks on I-SPY2, where both are available:
it matches exactly for 767 of 767 patients whose original scan was already
256x256 — which covers 914 of the 916 DUKE patients. The other two are dropped.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dataset_config as CFG

SPLIT_MAP = {0: "train", 1: "test", 2: "val"}
DCE_SUBDIR = {"spy2": "dce", "spy1": "dce", "duke": "crop_min_dce"}
MASK_SUBDIR = {"spy2": "mask", "spy1": "mask"}          # DUKE has none
Z_MARGIN = 2      # the authors' `slice_padding`, verified against released data


@dataclass
class Record:
    """One row of the output CSV — one per generated image."""
    filename: str
    pid: str
    cohort: str
    split: str
    label: int
    label_name: str
    slice_index: int
    slice_order: int
    z_rel: float
    n_slices_tumor: int
    phase_pre: int
    phase_early: int
    phase_late: int
    roi_source: str          # "mask" (I-SPY) or "bbox" (DUKE)
    tumor_pixels: int        # -1 where only a box exists — never imputed
    tumor_area_mm2: float
    tumor_fraction: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    xy_spacing: float
    slice_thick: float
    crop_px: int
    img_size: int
    mm_per_px: float
    tum_vol: float


@dataclass
class ROI:
    """Where the lesion is, from a mask or from a box."""
    zs: np.ndarray
    row0: int
    row1: int
    col0: int
    col1: int
    source: str
    mask: np.ndarray | None = None


def _roi_from_mask(mask: np.ndarray, min_px: int) -> ROI | None:
    """Exact ROI from the 3D binary mask, centred on the slice with most tumour.

    Centring on the largest-area slice rather than the 3D union matches what the
    DUKE annotation actually is — a box drawn on the plane with the largest
    tumour area — so the two cohorts get the same treatment. The two centres
    differ by 2.3 mm at the median and up to 25.6 mm in the tail.
    """
    areas = mask.reshape(mask.shape[0], -1).sum(1)
    zs = np.where(areas >= min_px)[0]
    if zs.size == 0:
        return None
    plane = mask[int(np.argmax(areas))]
    ys, xs = np.nonzero(plane)
    return ROI(zs=zs, row0=int(ys.min()), row1=int(ys.max()) + 1,
               col0=int(xs.min()), col1=int(xs.max()) + 1, source="mask", mask=mask)


def _roi_from_box(row, n_z: int) -> ROI | None:
    """ROI from the metadata box (DUKE).

    The z extent does NOT come from mask_start/mask_end — those index the ORIGINAL
    volume, and MinCrop is already cropped in z. It comes from the construction
    rule, verified: the MinCrop volume is exactly the tumour range plus two slices
    of margin each side, so the tumour occupies z in [2, nz-3].
    """
    if int(row["n_xy"]) != 256:
        return None                     # box does not map — see module docstring
    z0, z1 = Z_MARGIN, n_z - Z_MARGIN
    if z1 <= z0:
        z0, z1 = 0, n_z
    r0, r1, c0, c1 = (int(row["sraw"]), int(row["eraw"]),
                      int(row["scol"]), int(row["ecol"]))
    if not (0 <= r0 < r1 <= 256 and 0 <= c0 < c1 <= 256):
        return None
    return ROI(zs=np.arange(z0, z1), row0=r0, row1=r1, col0=c0, col1=c1,
               source="bbox")


def _phase_paths(dce_dir: Path, pid: str, cohort: str,
                 wanted: list[int]) -> tuple[list[Path], list[int]]:
    """Files for the three phases, and the indices actually used.

    When a requested index is missing, fall back to the LAST available
    acquisition rather than an arbitrary file — that is what the authors specify
    for DUKE ("timepoints 0, 1, and the final acquisition"). The index actually
    used goes into the CSV so no substitution is invisible.
    """
    available: dict[int, Path] = {}
    for f in dce_dir.glob(f"{pid}_{cohort}*_dce_aqc_*.nii.gz"):
        available[int(f.name.split("aqc_")[1].split(".")[0])] = f
    if not available:
        raise FileNotFoundError(f"{pid}: no DCE files")
    paths, used = [], []
    for i in wanted:
        i = int(i) if int(i) in available else max(available)
        paths.append(available[i])
        used.append(i)
    return paths, used


def _crop_pad(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop, zero-padding whatever falls outside."""
    y0, y1, x0, x1 = box
    h, w = img.shape[-2:]
    out = np.zeros(img.shape[:-2] + (y1 - y0, x1 - x0), dtype=img.dtype)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y1, h), min(x1, w)
    if sy1 > sy0 and sx1 > sx0:
        out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[..., sy0:sy1, sx0:sx1]
    return out


def _process(args) -> tuple[list[dict], str | None]:
    """One patient. Returns (records, error)."""
    import nibabel as nib
    from PIL import Image

    row, opts, out_dir = args
    pid, cohort = row["pid"], row["dataset"]
    pipeline = opts["pipeline"]
    root = CFG.COHORT_DIRS[cohort]

    if pipeline == "reference":
        from pipelines.reference import preprocessing as P
    else:
        from pipelines.thesis import preprocessing as P

    try:
        paths, phases = _phase_paths(root / DCE_SUBDIR[cohort], pid, cohort,
                                     [row["pre"], row["post_early"], row["post_late"]])
        vols = [np.nan_to_num(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0) for p in paths]
    except Exception as exc:
        return [], f"read: {exc}"

    if len({v.shape for v in vols}) != 1:
        return [], "phases have different shapes"
    vol = np.stack(vols)                                   # (3, Z, Y, X)
    n_z = vol.shape[1]

    # ---- locate the lesion ---- #
    if cohort in MASK_SUBDIR:
        mf = next((root / MASK_SUBDIR[cohort]).glob(f"{pid}_*_mask.nii.gz"), None)
        if mf is None:
            return [], "no mask"
        mask = np.asarray(nib.load(str(mf)).dataobj) > 0
        if mask.shape != vol.shape[1:]:
            return [], f"mask {mask.shape} != dce {vol.shape[1:]}"
        roi = _roi_from_mask(mask, opts["min_tumor_px"])
    else:
        roi = _roi_from_box(row, n_z)
    if roi is None:
        return [], "no usable ROI"

    # ---- normalise (volume-level only; the authors normalise per slice) ---- #
    if pipeline == "thesis":
        vol = (P.normalize_channel_clip(vol) if opts["normalization"] == "chanclip"
               else P.normalize_volume(vol))

    # ---- select slices ---- #
    chosen = (P.select_slices(roi.zs) if pipeline == "reference"
              else P.select_slices(roi.zs, opts["n_slices"], opts["trim_fraction"]))
    if len(chosen) == 0:
        return [], "no slice survived selection"

    # ---- crop window ---- #
    spacing = float(row["xy_spacing"])
    if not np.isfinite(spacing) or spacing <= 0:
        return [], "invalid xy_spacing"
    if pipeline == "reference":
        box = P.crop_window(roi.row0, roi.row1, roi.col0, roi.col1, opts["crop_px"])
        side = opts["crop_px"]
    else:
        box, side = P.crop_window(roi.row0, roi.row1, roi.col0, roi.col1,
                                  spacing, opts["crop_mm"])

    pdir = out_dir / "images" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    S = opts["save_size"]
    scale = S / side
    z0, z1 = int(roi.zs[0]), int(roi.zs[-1])
    span = max(z1 - z0, 1)
    records: list[dict] = []

    for order, z in enumerate(chosen):
        z = int(z)
        plane = vol[:, z]
        if pipeline == "reference":
            plane = P.normalize(plane)          # per slice, joint over channels
        plane = _crop_pad(plane, box)
        img = (np.clip(plane, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        if img.shape[0] != S:
            img = np.asarray(Image.fromarray(img).resize((S, S), Image.LANCZOS))
        name = f"slice_{z:03d}.png"
        Image.fromarray(img, "RGB").save(pdir / name, optimize=True)

        # Tumour measurements inside the SAVED frame. Where there is no mask they
        # stay at -1 — nothing is ever imputed.
        if roi.mask is not None:
            mz = _crop_pad(roi.mask[z], box)
            px = int(mz.sum())
            ys, xs = np.nonzero(mz)
            if ys.size:
                bx, by = int(xs.min() * scale), int(ys.min() * scale)
                bw = int((xs.max() - xs.min() + 1) * scale)
                bh = int((ys.max() - ys.min() + 1) * scale)
            else:
                bx = by = bw = bh = 0
            area, frac = px * spacing * spacing, float(mz.mean())
        else:
            px, area, frac = -1, -1.0, -1.0
            bx = int((roi.col0 - box[2]) * scale)
            by = int((roi.row0 - box[0]) * scale)
            bw = int((roi.col1 - roi.col0) * scale)
            bh = int((roi.row1 - roi.row0) * scale)

        records.append(asdict(Record(
            filename=f"{pid}/{name}", pid=pid, cohort=cohort, split=row["split"],
            label=int(row["label"]), label_name=str(row["label_name"]),
            slice_index=z, slice_order=order, z_rel=round((z - z0) / span, 4),
            n_slices_tumor=len(roi.zs),
            phase_pre=phases[0], phase_early=phases[1], phase_late=phases[2],
            roi_source=roi.source, tumor_pixels=px,
            tumor_area_mm2=round(float(area), 2), tumor_fraction=round(frac, 6),
            bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
            xy_spacing=round(spacing, 4),
            slice_thick=round(float(row["slice_thick"]), 3),
            crop_px=side, img_size=S, mm_per_px=round(spacing * side / S, 5),
            tum_vol=float(row["tum_vol"]) if np.isfinite(row["tum_vol"]) else -1.0)))
    return records, None


def _run_tasks(tasks: list, workers: int) -> tuple[list[dict], dict[str, str]]:
    """Process every patient, in parallel where that works and serially where not.

    macOS starts worker processes with `spawn`, which re-imports the parent
    module. From a notebook or a heredoc there is no importable parent, and the
    pool dies with BrokenProcessPool. Falling back to serial keeps the notebooks
    usable on a laptop; on Linux the pool works and is used.
    """
    records: list[dict] = []
    errors: dict[str, str] = {}

    def _absorb(i: int, result) -> None:
        recs, err = result
        if err:
            errors[tasks[i - 1][0]["pid"]] = err
        records.extend(recs)
        if i % 200 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} patients · {len(records)} images · "
                  f"{len(errors)} errors", flush=True)

    if workers and workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, res in enumerate(ex.map(_process, tasks, chunksize=4), 1):
                    _absorb(i, res)
            return records, errors
        except Exception as exc:
            print(f"  [note] parallel pool unavailable ({type(exc).__name__}); "
                  f"falling back to serial. Same output, slower.")
            records, errors = [], {}

    for i, t in enumerate(tasks, 1):
        _absorb(i, _process(t))
    return records, errors


def patient_table(cfg) -> pd.DataFrame:
    """Patients with a valid label, carrying the authors' official split."""
    df = pd.read_csv(CFG.METADATA_CSV)
    df = df[df.dataset.isin(cfg.cohorts)].copy()
    spec = cfg.task_spec

    if spec.num_classes > 2:
        df = df[df[spec.column].isin(spec.classes)].copy()
        df["label"] = df[spec.column].map({c: i for i, c in enumerate(spec.classes)})
        df["label_name"] = df[spec.column]
    else:
        before = len(df)
        df = df[df[spec.column].notna()].copy()
        if before - len(df):
            print(f"[note] {before - len(df)} patients dropped: no {spec.column}")
        df["label"] = df[spec.column].astype(int)
        df["label_name"] = df.label.map(dict(enumerate(spec.classes)))

    df = df[df.test.isin(SPLIT_MAP)].copy()
    df["split"] = df.test.map(SPLIT_MAP)

    before = len(df)
    df = df[~((df.dataset == "duke") & (df.n_xy != 256))].copy()
    if before - len(df):
        print(f"[note] {before - len(df)} DUKE patients dropped: n_xy != 256, "
              f"the metadata box does not map onto the MinCrop volume")
    return df.reset_index(drop=True)


def verify(meta: pd.DataFrame, out_dir: Path) -> None:
    """Checks that must pass before the dataset can be trusted. Raises if not."""
    print("\nVERIFICATION")
    failures = []

    by_split = meta.groupby("split").pid.unique()
    for a in by_split.index:
        for b in by_split.index:
            if a < b:
                shared = set(by_split[a]) & set(by_split[b])
                ok = not shared
                print(f"  [{'ok' if ok else 'FAIL'}] no patient in both {a} and {b}")
                if not ok:
                    failures.append(f"leak {a}/{b}")

    checks = [
        ("one label per patient", (meta.groupby("pid").label.nunique() == 1).all()),
        ("no duplicate filenames", not meta.filename.duplicated().any()),
        ("all files on disk",
         not [f for f in meta.filename if not (out_dir / "images" / f).is_file()]),
        ("single image size", meta.img_size.nunique() == 1),
        ("labels in range", set(meta.label.unique()) <= set(range(meta.label.max() + 1))),
    ]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    if failures:
        raise SystemExit(f"\nVERIFICATION FAILED: {failures}")
    print("  all checks passed")


def build(cfg, n_slices: int = 8, trim_fraction: float = 0.15,
          crop_mm: float = 80.0, crop_px: int = 224, save_size: int = 224,
          normalization: str = "minmax", min_tumor_px: int = 10,
          workers: int = 8, dry_run: bool = False) -> Path | None:
    """Build the dataset for `cfg`. Returns the output folder."""
    patients = patient_table(cfg)
    out_dir = cfg.dataset_dir
    opts = dict(pipeline=cfg.pipeline, n_slices=n_slices,
                trim_fraction=trim_fraction, crop_mm=crop_mm, crop_px=crop_px,
                save_size=save_size, normalization=normalization,
                min_tumor_px=min_tumor_px)

    print(f"BUILD  {cfg.pipeline} / {cfg.task}  ->  {out_dir}")
    print(f"  cohorts : {', '.join(cfg.cohorts)}")
    print(f"  patients: {len(patients)}")
    if cfg.pipeline == "reference":
        print(f"  slices  : 4, range(idx-2, idx+2) — the authors' rule")
        print(f"  crop    : {crop_px} px fixed, centred on the ROI")
        print(f"  norm    : per slice, joint over channels")
    else:
        print(f"  slices  : {n_slices} spread, trimming {trim_fraction:.0%} each end")
        print(f"  crop    : {crop_mm:.0f} mm physical -> {save_size} px")
        print(f"  norm    : {normalization}, whole volume")
    print()
    print(pd.crosstab(patients.dataset, [patients.split, patients.label_name]).to_string())
    if dry_run:
        print("\n--dry-run: nothing written")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(r, opts, out_dir) for _, r in patients.iterrows()]
    records, errors = _run_tasks(tasks, workers)

    if errors:
        from collections import Counter
        print(f"\n{len(errors)} patients failed:")
        for reason, n in Counter(errors.values()).most_common():
            print(f"  {n:4d}  {reason}")

    meta = pd.DataFrame(records)
    fields = list(Record.__dataclass_fields__)
    meta[fields].to_csv(out_dir / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    for split, sub in meta.groupby("split"):
        sub[fields].to_csv(out_dir / f"{split}.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {**opts, "task": cfg.task, "cohorts": list(cfg.cohorts),
         "n_patients": int(meta.pid.nunique()), "n_images": len(meta),
         "fromarray_fix": cfg.pipeline == "reference", "errors": errors}, indent=2))

    verify(meta, out_dir)
    print(f"\n{len(meta):,} images from {meta.pid.nunique():,} patients -> {out_dir}")
    print("\nPATIENTS — cohort x class")
    print(pd.crosstab(meta.drop_duplicates('pid').cohort,
                      meta.drop_duplicates('pid').label_name, margins=True).to_string())
    return out_dir
