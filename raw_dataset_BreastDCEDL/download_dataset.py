#!/usr/bin/env python3
"""Download the BreastDCEDL imaging release from Zenodo into this folder.

    python raw_dataset_BreastDCEDL/download_dataset.py           # MinCrop only
    python raw_dataset_BreastDCEDL/download_dataset.py --all     # everything
    python raw_dataset_BreastDCEDL/download_dataset.py --list    # show, download nothing

Roughly 22 GB to download, about 35 GB once extracted. Downloads resume where they stopped, so an
interrupted run can simply be repeated.

WHY THE FILE LIST IS NOT HARD-CODED
-----------------------------------
It is read from the Zenodo API at run time. A published record can gain a file or
change an archive name, and a hard-coded list would then fail with a 404 that says
nothing useful. Asking the record what it contains means this script keeps working,
and `--list` shows exactly what is there.

EVERY FAILURE PATH ENDS IN MANUAL INSTRUCTIONS
----------------------------------------------
No network, a proxy, a Zenodo outage, a corrupted transfer: whatever goes wrong, the
script prints the record URL and what to do by hand rather than a traceback. The
dataset is a prerequisite for everything else in the repository, so failing to fetch
it must not also leave you guessing.

Standard library only. This runs before any project dependency is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

RECORD_ID = "18114231"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
HERE = Path(__file__).resolve().parent

# The MinCrop release is what this project builds on: tumour-centred 256x256 crops.
# The Full release is 206 GB of whole volumes and is not used anywhere.
MINCROP_KEYWORDS = ("min_crop", "metadata")

CHUNK = 1 << 20  # 1 MiB


def manual_instructions(reason: str) -> None:
    """Print what to do by hand. Called from every failure path."""
    print(f"\n  Automatic download failed: {reason}\n")
    print("  DOWNLOAD IT MANUALLY")
    print("  --------------------")
    print(f"  1. Open  {RECORD_URL}")
    print("  2. Download the MinCrop files:")
    print("       BreastDCEDL_ISPY2_min_crop.tar.gz")
    print("       BreastDCEDL_ISPY1_min_crop.tar.gz")
    print("       BreastDCEDL_DUKE_min_crop.tar.gz")
    print("       BreastDCEDL_metadata_min_crop.csv")
    print(f"  3. Put them in  {HERE}")
    print("  4. Extract each archive in place:")
    print("       tar xzf BreastDCEDL_ISPY2_min_crop.tar.gz")
    print("\n  The result must look like this, or the builder will not find the")
    print("  imaging:")
    print("       raw_dataset_BreastDCEDL/BreastDCEDL_metadata_min_crop.csv")
    print("       raw_dataset_BreastDCEDL/BreastDCEDL_ISPY2_min_crop/dce/")
    print("       raw_dataset_BreastDCEDL/BreastDCEDL_ISPY2_min_crop/mask/")
    print("       raw_dataset_BreastDCEDL/BreastDCEDL_ISPY1_min_crop/dce/")
    print("       raw_dataset_BreastDCEDL/BreastDCEDL_DUKE_min_crop/crop_min_dce/")
    print("\n  See raw_dataset_BreastDCEDL/README.md for what each file contains.\n")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def fetch_record() -> list[dict]:
    """Ask Zenodo what the record contains. Raises on any failure."""
    request = urllib.request.Request(
        API_URL, headers={"User-Agent": "breastdcedl-thesis-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.loads(response.read().decode())
    files = record.get("files", [])
    if not files:
        raise RuntimeError("the record reports no files")
    return files


def wanted(files: list[dict], want_all: bool) -> list[dict]:
    if want_all:
        return files
    chosen = [f for f in files
              if any(k in f["key"].lower() for k in MINCROP_KEYWORDS)]
    return chosen or files


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict) -> Path:
    """Download one file, resuming if a partial copy is already here.

    Zenodo publishes an md5 for every file, so a transfer that was truncated by a
    dropped connection is detected here rather than three steps later as an
    unreadable archive.
    """
    name = entry["key"]
    url = entry["links"]["self"]
    total = int(entry.get("size", 0))
    expected = str(entry.get("checksum", "")).replace("md5:", "")
    target = HERE / name

    if target.exists() and total and target.stat().st_size == total:
        if expected and md5_of(target) == expected:
            print(f"  {name}  already present and verified")
            return target
        if not expected:
            print(f"  {name}  already present ({human(total)}), no checksum published")
            return target
        print(f"  {name}  present but the checksum does not match — downloading again")
        target.unlink()

    start = target.stat().st_size if target.exists() else 0
    headers = {"User-Agent": "breastdcedl-thesis-downloader"}
    if start:
        headers["Range"] = f"bytes={start}-"
        print(f"  {name}  resuming at {human(start)}")
    else:
        print(f"  {name}  {human(total)}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        # A server that ignores the Range header sends 200 and the whole file; in
        # that case appending would corrupt it, so start over.
        mode = "ab" if response.status == 206 else "wb"
        if mode == "wb":
            start = 0
        done = start
        with target.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {pct:5.1f}%  {human(done)} / {human(total)}",
                          end="", flush=True)
        print()

    if expected:
        print("    verifying checksum", end="", flush=True)
        actual = md5_of(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"{name}: checksum mismatch (expected {expected}, got {actual}). "
                "The partial file was removed; run the script again.")
        print(" ok")
    return target


def extract(archive: Path) -> None:
    """Extract a .tar.gz in place, skipping if it already looks extracted."""
    if archive.suffixes[-2:] != [".tar", ".gz"]:
        return
    stem = archive.name[: -len(".tar.gz")]
    if (HERE / stem).is_dir():
        print(f"  {stem}/  already extracted")
        return
    print(f"  extracting {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse members that would write outside this folder. A malicious or
        # malformed archive is not expected from Zenodo, but extracting one
        # blindly is how a download turns into arbitrary file writes.
        for member in tar.getmembers():
            destination = (HERE / member.name).resolve()
            if not str(destination).startswith(str(HERE)):
                raise RuntimeError(
                    f"{archive.name} contains a path outside the target "
                    f"directory: {member.name}")
        tar.extractall(HERE)


def report_layout() -> None:
    """Say whether the result is what the dataset builder expects."""
    expected = {
        "metadata CSV": HERE / "BreastDCEDL_metadata_min_crop.csv",
        "I-SPY2 volumes": HERE / "BreastDCEDL_ISPY2_min_crop" / "dce",
        "I-SPY2 masks": HERE / "BreastDCEDL_ISPY2_min_crop" / "mask",
        "I-SPY1 volumes": HERE / "BreastDCEDL_ISPY1_min_crop" / "dce",
        "I-SPY1 masks": HERE / "BreastDCEDL_ISPY1_min_crop" / "mask",
        "Duke volumes": HERE / "BreastDCEDL_DUKE_min_crop" / "crop_min_dce",
    }
    print("\n  LAYOUT")
    missing = []
    for label, path in expected.items():
        ok = path.exists()
        print(f"    [{'ok' if ok else '--'}] {label}")
        if not ok:
            missing.append(label)
    if missing:
        print(f"\n  {len(missing)} item(s) missing. Duke ships no mask folder by "
              "design; anything else missing means the download is incomplete.")
    else:
        print("\n  Complete. Build the dataset with build_dataset.ipynb.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="download every file in the record, not just MinCrop")
    parser.add_argument("--list", action="store_true",
                        help="list the files and exit without downloading")
    parser.add_argument("--no-extract", action="store_true",
                        help="download the archives but leave them packed")
    args = parser.parse_args()

    print(f"BreastDCEDL — Zenodo record {RECORD_ID}")
    print(f"  {RECORD_URL}")
    print(f"  target: {HERE}\n")

    try:
        files = fetch_record()
    except urllib.error.HTTPError as exc:
        manual_instructions(f"Zenodo returned HTTP {exc.code}")
        return 1
    except urllib.error.URLError as exc:
        manual_instructions(f"could not reach Zenodo ({exc.reason})")
        return 1
    except Exception as exc:                                  # noqa: BLE001
        manual_instructions(f"{type(exc).__name__}: {exc}")
        return 1

    selected = wanted(files, args.all)
    total = sum(int(f.get("size", 0)) for f in selected)
    print(f"  {len(selected)} file(s), {human(total)} total\n")
    for entry in selected:
        print(f"    {entry['key']:<45} {human(int(entry.get('size', 0)))}")

    if args.list:
        print(f"\n  Listing only. Run without --list to download.")
        return 0

    print()
    archives = []
    for entry in selected:
        try:
            path = download(entry)
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Run the script again to resume.")
            return 130
        except Exception as exc:                              # noqa: BLE001
            manual_instructions(f"{entry['key']}: {type(exc).__name__}: {exc}")
            return 1
        if path.name.endswith(".tar.gz"):
            archives.append(path)

    if not args.no_extract:
        print()
        for archive in archives:
            try:
                extract(archive)
            except Exception as exc:                          # noqa: BLE001
                manual_instructions(
                    f"could not extract {archive.name}: {type(exc).__name__}: {exc}")
                return 1

    report_layout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
