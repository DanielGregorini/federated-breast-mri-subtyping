#!/usr/bin/env python3
"""Check that the notebooks reproduce the recorded results.

    python notebooks/verify_notebooks.py

The notebooks carry their own logic rather than importing `src/`, so this runs
their cells and asserts the outcome against numbers that were recorded before the
notebooks existed: the reported centralised macro AUC, the parameter counts, the
dataset split sizes and the built federated partitions.

It stops before anything expensive. No training happens here.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "notebooks")          # notebooks resolve REPO_ROOT as Path.cwd().parent

FAILS = []


def cells(nb_name):
    nb = json.loads((REPO / "notebooks" / nb_name).read_text())
    return [("".join(c["source"]) if c["cell_type"] == "code" else None)
            for c in nb["cells"]]


def run(nb_name, upto_marker, ns=None):
    """Run code cells in order until the cell containing `upto_marker` has run."""
    ns = ns if ns is not None else {}
    for src in cells(nb_name):
        if src is None:
            continue
        exec(compile(src, nb_name, "exec"), ns)
        if upto_marker in src:
            return ns
    raise RuntimeError(f"{nb_name}: marker {upto_marker!r} never reached")


def check(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------- notebook 03
print("03_train_centralized.ipynb")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = run("03_train_centralized.ipynb", "print(\"metrics defined")

import numpy as np
import pandas as pd

# 1. The metric stack must reproduce the reported centralised macro AUC exactly.
pred = pd.read_csv(REPO / "results/federated/test01_centralized/seed_42/predictions_test.csv")
P = pred[["prob_HRposHER2neg", "prob_TripleNeg", "prob_HER2pos"]].to_numpy()
m = ns["compute_metrics"](pred.label.to_numpy(), P, ns["CLASS_NAMES"])
check("test01 macro AUC from the notebook's compute_metrics", m["auc"],
      0.6067918080145278, tol=1e-12)
check("test01 accuracy", round(m["accuracy"], 4), 0.5299)
check("test01 balanced accuracy", round(m["balanced_accuracy"], 4), 0.4503)
check("test01 macro F1", round(m["macro_f1"], 4), 0.4523)
check("test01 trivial baseline", round(m["trivial_baseline_accuracy"], 4), 0.5112)
check("test01 confusion", m["confusion"], [[96, 20, 21], [30, 36, 12], [33, 10, 10]])
check("test01 per-class AUC", m["per_class_auc"], [0.6238, 0.6886, 0.5079])
check("test01 per-class recall", m["per_class_recall"], [0.7007, 0.4615, 0.1887])

# 2. Balanced accuracy must equal macro recall, as recorded for all 13 experiments.
check("balanced accuracy == macro recall", round(m["balanced_accuracy"] - m["macro_recall"], 12), 0)

# 3. Patient aggregation: the config the campaign used.
check("aggregation", ns["AGGREGATION"], "mean")
check("monitor metric", ns["MONITOR_METRIC"], "auc")
check("epochs", ns["EPOCHS"], 30)
check("batch size", ns["BATCH_SIZE"], 24)
check("learning rate", ns["LEARNING_RATE"], 1e-4)
check("weight decay", ns["WEIGHT_DECAY"], 5e-4)
check("dropout", ns["DROPOUT"], 0.5)
check("label smoothing", ns["LABEL_SMOOTHING"], 0.1)
check("freeze_until", ns["FREEZE_UNTIL"], "layer3")
check("seed", ns["SEED"], 42)
check("early stopping disabled", ns["EARLY_STOPPING_PATIENCE"], 0)

# 4. Model parameter counts, as recorded.
net = ns["build_model"]("resnet18", 3, pretrained=False, dropout=0.5)
ns["freeze_until"](net, "layer3")
pc = ns["param_counts"](net)
check("resnet18 total parameters", pc["total"], 11178051)
check("resnet18 trainable", pc["trainable"], 10494979)
check("resnet18 frozen", pc["frozen"], 683072)

# 5. The head layout the saved checkpoints require: fc.1.weight / fc.1.bias.
keys = [k for k in net.state_dict() if k.startswith("fc.")]
check("head keys", sorted(keys), ["fc.1.bias", "fc.1.weight"])

# 6. The dataset splits.
check("train patients", int(ns["frames"]["train"].pid.nunique()), 1527)
check("val patients", int(ns["frames"]["val"].pid.nunique()), 268)
check("test patients", int(ns["frames"]["test"].pid.nunique()), 268)
check("total images", int(sum(len(f) for f in ns["frames"].values())), 16378)
check("test trivial baseline", round(ns["trivial_baseline"](ns["frames"]["test"]), 4), 0.5112)

# 7. Class weights, patient level.
import torch
w = ns["class_weights"](ns["frames"]["train"], 3, torch.device("cpu"))
per_patient = ns["frames"]["train"].groupby("pid").label.first()
counts = np.bincount(per_patient.values, minlength=3)
# weight_c * count_c is n/num_classes for every class, so the total is n.
# float32 tensors, so compare with a tolerance rather than for equality.
check("class weights reweight every class to n/3",
      float((w.numpy() * counts).sum()), float(len(per_patient)), tol=1e-3)
check("weight per class equals n/(3*count)",
      [round(float(x), 4) for x in w.numpy()],
      [round(len(per_patient) / (3 * c), 4) for c in counts])

# ---------------------------------------------------------------- notebook 02
print("\n02_build_dataset.ipynb")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns2 = run("02_build_dataset.ipynb", "NORMALIZERS[NORMALIZATION].__name__")

check("N_SLICES", ns2["N_SLICES"], 8)
check("TRIM_FRACTION", ns2["TRIM_FRACTION"], 0.15)
check("CROP_MM", ns2["CROP_MM"], 80.0)
check("SAVE_SIZE", ns2["SAVE_SIZE"], 224)
check("normalisation", ns2["NORMALIZATION"], "minmax")
check("class order", tuple(ns2["CLASS_NAMES"]), ("HRposHER2neg", "TripleNeg", "HER2pos"))

# Against the config.json the real dataset was built with.
built = json.loads((REPO / "dataset/multi_subtype_80mm/config.json").read_text())
for key, nb_key in [("n_slices", "N_SLICES"), ("trim_fraction", "TRIM_FRACTION"),
                    ("trim_frac", "TRIM_FRACTION"),
                    ("crop_mm", "CROP_MM"), ("save_size", "SAVE_SIZE"),
                    ("normalization", "NORMALIZATION"), ("min_tumor_px", "MIN_TUMOR_PX")]:
    if key in built:
        check(f"matches the built dataset: {key}", ns2[nb_key], built[key])
check("matches the built dataset: cohorts", list(ns2["COHORTS"]), built["cohorts"])
check("built dataset patients", built["n_patients"], 2063)
check("built dataset images", built["n_images"], 16378)

# The patient table the notebook builds must be the one the dataset was built from.
pt = ns2["patient_table"]()
check("patient_table finds the same patients", int(len(pt)), 2063)
check("patient_table split sizes",
      pt.split.value_counts().reindex(["train", "val", "test"]).tolist(),
      [1527, 268, 268])

# The slice rule, on the shapes it actually meets.
sel = ns2["select_slices"]
check("a 60-slice lesion keeps 8", len(sel(np.arange(60))), 8)
check("a 6-slice lesion keeps all 6", len(sel(np.arange(6))), 6)
check("an 8-slice lesion keeps 8", len(sel(np.arange(8))), 8)

# The crop rule.
_, side = ns2["crop_window"](100, 140, 100, 150, 0.357)
check("80 mm at 0.357 mm/px is 224 px", side, 224)

# ---------------------------------------------------------------- notebook 06
print("\n06_federated_setup.ipynb")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns6 = run("06_federated_setup.ipynb", "local validation carving defined")

check("seed", ns6["SEED"], 42)
check("local val fraction", ns6["LOCAL_VAL_FRACTION"], 0.2)
check("partitions defined", len(ns6["PARTITIONS"]), 6)
check("skewed ratio", ns6["PARTITIONS"]["4_clients_skewed"].ratio, (5, 2, 1, 1))
check("skewed normalises to 55.6%",
      round(100 * ns6["PARTITIONS"]["4_clients_skewed"].fractions[0], 1), 55.6)
check("cohort sizes", ns6["PARTITIONS"]["3_clients_cohort"].ratio, (642, 101, 784))
check("size-matched repeats them", ns6["PARTITIONS"]["3_clients_sizematched"].ratio,
      (642, 101, 784))

# Reproduce a real partition and compare against the one on disk.
src = REPO / "dataset/multi_subtype_80mm"
train_rows = pd.read_csv(src / "train.csv")
patients = train_rows.drop_duplicates("pid")[["pid", "label", "cohort"]].reset_index(drop=True)

for name in ("3_clients_cohort", "4_clients_skewed"):
    disk = REPO / f"deployment/data/partitions/{name}/partition.json"
    if not disk.is_file():
        print(f"  [skip] {name}: not built on disk")
        continue
    meta = json.loads(disk.read_text())
    rng = np.random.default_rng(ns6["SEED"])
    part = ns6["PARTITIONS"][name]
    if not part.stratified:
        shares = ns6["cohort_shares"](patients, part.n_clients)
    else:
        shares = ns6["stratified_shares"](patients, part.fractions, rng)
    got = [len(s) for s in shares]
    want = [s["train"]["patients"] + s["val"]["patients"] for s in meta["sites"]]
    check(f"{name}: site sizes reproduce the built partition", got, want)

    # And the local validation carve, on the same rng stream as the real build.
    rng2 = np.random.default_rng(ns6["SEED"])
    if not part.stratified:
        shares2 = ns6["cohort_shares"](patients, part.n_clients)
    else:
        shares2 = ns6["stratified_shares"](patients, part.fractions, rng2)
    got_train = []
    for pids in shares2:
        tr, va = ns6["carve_local_val"](pids, patients, ns6["LOCAL_VAL_FRACTION"], rng2)
        got_train.append(len(tr))
    check(f"{name}: training patients per site reproduce",
          got_train, [s["train"]["patients"] for s in meta["sites"]])

print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} CHECKS FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("every check passed")
