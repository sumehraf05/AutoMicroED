#!/usr/bin/env python3

"""
xds_pipeline.py  —  MIDAS: Micro-Electron Diffraction Integrated Data Automation System
========================================================================================
Automated batch processing pipeline for MicroED crystallographic datasets.
Handles everything from raw diffraction images to a publication-ready merged
dataset with minimal user intervention.

What it does:
    1.  Renames image files starting at 1 (rerun-safe, backs up originals)
    2.  Reads detector parameters from image headers (distance, wavelength,
        starting angle) and generates a per-dataset XDS.INP automatically
    3.  Adaptive spot finding: tests MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT = 3, 4, 6
        and picks the value that gives the highest indexed fraction
    4.  Runs XDS Phase 1 (XYCORR INIT COLSPOT IDXREF) and auto-retries with
        relaxed parameters if indexing falls below 15%
    5.  Runs XDS Phase 2 (DEFPIX INTEGRATE CORRECT)
    6.  Detects and excludes bad frames based on per-frame I/sigma and
        fraction-observed statistics from INTEGRATE.LP
    7.  Detects ice rings from CORRECT.LP and adds EXCLUDE_RESOLUTION_RANGE
    8.  Applies automated resolution cutoff where I/sigma drops below threshold
    9.  Extracts full quality statistics per dataset:
          - Space group, unit cell (a, b, c, alpha, beta, gamma)
          - Completeness, Rmeas, I/sigma, CC½  (overall AND hi-res shell)
          - Resolution limit (Angstroms)
   10.  CNN quality check: ResNet-18 network independently predicts the unit
        cell and a quality score (0-1) from raw diffraction images
   11.  Computes consensus unit cell across all datasets and re-indexes any
        dataset that deviates, forcing it toward the true lattice
   12.  Identifies the dominant crystal form using MAD-based outlier detection
        and filters incompatible datasets automatically
   13.  Ranks compatible datasets by quality score
   14.  Merges all compatible datasets with XSCALE (baseline)
   15.  Greedy forward selection: finds the optimal subset of datasets that
        maximises completeness while keeping Rmeas and CC½ high
   16.  Pushes the resolution limit by extending 0.1 A beyond the best
        individual dataset and testing whether merged signal is present
   17.  Converts the final merged XSCALE.HKL to SHELX format (shelx.hkl)
        using XDSCONV for direct use with SHELXS/SHELXL
   18.  Saves a timestamped log file named after the parent folder
   19.  Supports parallel processing (--workers N), watch mode (--watch),
        and direct folder specification (--folder PATH)

Usage:
  python xds_pipeline.py                          # interactive prompt
  python xds_pipeline.py --folder /path/to/data  # direct path
  python xds_pipeline.py --workers 4             # parallel (4 crystals at once)
  python xds_pipeline.py --watch                 # continuous monitoring mode

Requirements:
  xds_par and xscale_par must be on your PATH.
  xdsconv must be on PATH for SHELX format conversion.
  See README.md for conda environment setup instructions.

"""

import os
import time
import logging
import subprocess
import shutil
import csv
import re
import argparse
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import fabio

from microed_cnn import (
    load_cnn_model,
    predict_unit_cell,
    compare_xds_and_cnn,
    CNNPrediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default I/sigma cutoff for automated resolution trimming (Week 7 task 6).
# Processing will rerun CORRECT with a high-resolution limit set to the shell
# where mean I/sigma first drops below this value.
# ---------------------------------------------------------------------------
ISIGI_CUTOFF = 2.0


# ===========================================================================
# Utility helpers
# ===========================================================================

def safe_float(value, default=None):
    """Convert value to float, return default on any failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def already_renamed(filename: str, subdir_name: str) -> bool:
    """Return True if filename already matches <subdir>_NNNNN.img."""
    return re.fullmatch(
        rf"{re.escape(subdir_name)}_\d{{5}}\.img", filename
    ) is not None


def prompt_parent_dir() -> Path:
    """Prompt for parent directory with drag-and-drop support."""
    while True:
        raw = input("Drag the PARENT folder here, then press Enter:\n> ").strip()
        if (raw.startswith('"') and raw.endswith('"')) or \
           (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1]
        raw = raw.replace("\\ ", " ")
        p = Path(raw).expanduser()
        if p.exists():
            p = p.resolve()
        if p.is_dir():
            return p
        print(f"  Not a valid directory: {p} -- please try again.\n")


# ===========================================================================
# Week 6 Task 1 -- Image file renaming (1-indexed, rerun-safe)
# ===========================================================================

def rename_and_backup(root_path: Path, subdir_name: str) -> list:
    """
    Rename .img files to <subdir_name>_NNNNN.img starting at 00001.

    XDS requires frames numbered from 1 (not 0). Originals are copied to a
    backup subdirectory before renaming so raw data is never lost.
    Rerun-safe: if all files already match the pattern, nothing is touched.
    """
    img_files = sorted(f for f in os.listdir(root_path) if f.endswith(".img"))

    if all(already_renamed(f, subdir_name) for f in img_files):
        log.info("  Images already renamed -- skipping.")
        return img_files

    backup_dir = root_path / f"{subdir_name}_backup"
    backup_dir.mkdir(exist_ok=True)
    to_rename = sorted(f for f in img_files if not already_renamed(f, subdir_name))

    for idx, img in enumerate(to_rename, start=1):
        old_path = root_path / img
        shutil.copy2(old_path, backup_dir / f"{subdir_name}_backup_{idx-1:05d}.img")
        new_path = root_path / f"{subdir_name}_{idx:05d}.img"
        if new_path.exists():
            raise FileExistsError(
                f"Cannot rename '{img}' -> '{new_path.name}': target exists."
            )
        old_path.rename(new_path)

    return sorted(f for f in os.listdir(root_path) if f.endswith(".img"))


# ===========================================================================
# Week 6 Task 2 -- XDS.INP generation and patching
# ===========================================================================

_XDS_INP_TEMPLATE = """! Auto-generated XDS.INP for dataset: {name}
! Instrument: UCSF cryo-EM (ADSC 2048x2048 CCD, microED geometry)
!=============================================================================
JOB= XYCORR INIT COLSPOT IDXREF
!
! Beam centre: hardcoded for this instrument.
! ORGX (horizontal) and ORGY (vertical) are intentionally different values.
ORGX= {orgx:.1f}  ORGY= {orgy:.1f}
!
DETECTOR_DISTANCE= {distance:.3f}
OSCILLATION_RANGE= 1.0
STARTING_ANGLE= {starting_angle:.3f}
X-RAY_WAVELENGTH= {wavelength:.6f}
NAME_TEMPLATE_OF_DATA_FRAMES= {template}
DATA_RANGE= {data_start} {n_images}
SPOT_RANGE= {spot_start} {spot_end}
!
! Resolution range: 20A low, 0.97A high.
! After CORRECT, insert actual high-res cutoff and re-run CORRECT.
INCLUDE_RESOLUTION_RANGE= 20 0.97
!
SPACE_GROUP_NUMBER= 0                    ! 0 = let XDS determine space group
UNIT_CELL_CONSTANTS= 0 0 0 0 0 0        ! replace with known values if available
!
! --- Indexing thresholds ---
MINIMUM_FRACTION_OF_INDEXED_SPOTS= 0.30
INDEX_QUALITY= 0.70
MAXIMUM_ERROR_OF_SPOT_POSITION= 15.0
MAXIMUM_ERROR_OF_SPINDLE_POSITION= 7.5
!
! --- Spot finding ---
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= 6
SEPMIN= 7.0  CLUSTER_RADIUS= 3.5
!
! --- Data collection geometry ---
OFFSET= 100
DELPHI= 20
GAIN= 15
!
! --- Trusted region and dynamic range ---
TRUSTED_REGION= 0.0 1.2
VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS= 6000. 60000.
!
! --- Untrusted detector regions ---
UNTRUSTED_ELLIPSE= 966 1115 980 1120
UNTRUSTED_QUADRILATERAL= 992 1024 1 1056 2 1113 990 1072
!
! --- Detector hardware ---
NX= 2048  NY= 2048  QX= 0.028  QY= 0.028
DETECTOR= ADSC  MINIMUM_VALID_PIXEL_VALUE= 1  OVERLOAD= 65000
SENSOR_THICKNESS= 0.01
!
! --- Refinement strategy for electron diffraction ---
REFINE(IDXREF)=   CELL BEAM ORIENTATION AXIS
REFINE(INTEGRATE)= POSITION BEAM ORIENTATION
REFINE(CORRECT)=   ORIENTATION CELL AXIS BEAM
!
! --- Geometry ---
ROTATION_AXIS= 0 1 0
DIRECTION_OF_DETECTOR_X-AXIS= 1 0 0
DIRECTION_OF_DETECTOR_Y-AXIS= 0 1 0
INCIDENT_BEAM_DIRECTION= 0 0 1
FRACTION_OF_POLARIZATION= 0.98
POLARIZATION_PLANE_NORMAL= 0 1 0
FRIEDEL'S_LAW= FALSE
"""


def generate_xds_inp(root: Path, subdir_name: str, n_images: int,
                     orgx: float, orgy: float,
                     distance: float, wavelength: float,
                     starting_angle: float = 0.0) -> None:
    """
    Write a fresh XDS.INP from the template.

    Beam centre (orgx, orgy) is hardcoded for this instrument:
      ORGX ~ 1043  (horizontal, X direction)
      ORGY ~ 1046  (vertical,   Y direction)
    These are intentionally different -- do not set both to 1024.

    DATA_RANGE starts at 1.
    SPOT_RANGE uses the middle third of the dataset for spot finding,
    which avoids radiation-damaged frames at the start and end.
    """
    # Use the middle third of frames for spot finding -- more reliable
    # than the first N frames which may have higher radiation damage
    spot_start = max(1, n_images // 3)
    spot_end   = min(n_images, 2 * n_images // 3)

    file_content = _XDS_INP_TEMPLATE.format(
        name=subdir_name,
        orgx=orgx,
        orgy=orgy,
        distance=distance,
        wavelength=wavelength,
        starting_angle=starting_angle,
        template=f"{subdir_name}_?????.img",
        data_start=1,
        n_images=n_images,
        spot_start=spot_start,
        spot_end=spot_end,
    )
    (root / "XDS.INP").write_text(file_content)
    log.info("  XDS.INP generated.")


def patch_xds_inp(root: Path, subdir_name: str, n_images: int,
                  orgx: float, orgy: float) -> None:
    """
    Patch an existing XDS.INP in-place.

    Updates: beam centre, data/spot ranges, template name, and all
    instrument-specific parameters. Leaves any lines we do not control
    exactly as they are.
    """
    xds_inp  = root / "XDS.INP"
    lines    = xds_inp.read_text(errors="replace").splitlines()

    spot_start = max(1, n_images // 3)
    spot_end   = min(n_images, 2 * n_images // 3)

    desired = {
        # Beam centre -- hardcoded for this instrument, X != Y
        "ORGX=":                               f"ORGX= {orgx:.1f}  ORGY= {orgy:.1f}",
        "ORGY=":                               None,   # consumed with ORGX=
        # Data ranges
        "DATA_RANGE=":                         f"DATA_RANGE= 1 {n_images}",
        "SPOT_RANGE=":                         f"SPOT_RANGE= {spot_start} {spot_end}",
        "NAME_TEMPLATE_OF_DATA_FRAMES=":       f"NAME_TEMPLATE_OF_DATA_FRAMES= {subdir_name}_?????.img",
        # Resolution
        "INCLUDE_RESOLUTION_RANGE=":           "INCLUDE_RESOLUTION_RANGE= 20 0.97",
        # Indexing thresholds
        "INDEX_QUALITY=":                      "INDEX_QUALITY= 0.70",
        "MINIMUM_FRACTION_OF_INDEXED_SPOTS=":  "MINIMUM_FRACTION_OF_INDEXED_SPOTS= 0.30",
        "MAXIMUM_ERROR_OF_SPOT_POSITION=":     "MAXIMUM_ERROR_OF_SPOT_POSITION= 15.0",
        "MAXIMUM_ERROR_OF_SPINDLE_POSITION=":  "MAXIMUM_ERROR_OF_SPINDLE_POSITION= 7.5",
        # Spot finding
        "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=": "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= 6",
        "SEPMIN=":                             "SEPMIN= 7.0  CLUSTER_RADIUS= 3.5",
        "CLUSTER_RADIUS=":                     None,   # consumed with SEPMIN=
        # Geometry
        "OFFSET=":                             "OFFSET= 100",
        "DELPHI=":                             "DELPHI= 20",
        "GAIN=":                               "GAIN= 15",
        # Trusted region
        "TRUSTED_REGION=":                     "TRUSTED_REGION= 0.0 1.2",
        "VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS=":
                                               "VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS= 6000. 60000.",
        # Refinement strategy for electron diffraction
        "REFINE(IDXREF)=":                     "REFINE(IDXREF)=   CELL BEAM ORIENTATION AXIS",
        "REFINE(INTEGRATE)=":                  "REFINE(INTEGRATE)= POSITION BEAM ORIENTATION",
        "REFINE(CORRECT)=":                    "REFINE(CORRECT)=   ORIENTATION CELL AXIS BEAM",
    }

    out, seen = [], set()
    for line in lines:
        stripped = line.strip()
        key = next((k for k in desired if stripped.startswith(k)), None)
        if key is None:
            out.append(line)
        elif key not in seen:
            seen.add(key)
            if desired[key] is not None:
                out.append(desired[key])
        # else: duplicate or consumed key -- drop it

    # Append any parameters that were not in the original file
    for key, value in desired.items():
        if key not in seen and value is not None:
            out.append(value)

    xds_inp.write_text("\n".join(out) + "\n")
    log.info("  XDS.INP patched.")


def set_job_line(root: Path, job_value: str) -> None:
    """Replace the JOB= line in XDS.INP."""
    xds_inp = root / "XDS.INP"
    if not xds_inp.exists():
        return
    lines = xds_inp.read_text(errors="replace").splitlines()
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith("JOB=") and not replaced:
            out.append(f"JOB= {job_value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, f"JOB= {job_value}")
    xds_inp.write_text("\n".join(out) + "\n")


def set_resolution_limit(root: Path, high_res: float) -> None:
    """
    Set INCLUDE_RESOLUTION_RANGE in XDS.INP to 'low high_res'.
    Used for the automated resolution cutoff rerun (Week 7 task 6).
    """
    xds_inp = root / "XDS.INP"
    if not xds_inp.exists():
        return
    lines = xds_inp.read_text(errors="replace").splitlines()
    out, found = [], False
    for line in lines:
        if line.strip().startswith("INCLUDE_RESOLUTION_RANGE=") and not found:
            # Preserve the low-resolution limit, replace the high-res limit
            parts = line.split("=", 1)[1].split()
            low = parts[0] if parts else "30.0"
            out.append(f"INCLUDE_RESOLUTION_RANGE= {low} {high_res:.2f}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"INCLUDE_RESOLUTION_RANGE= 30.0 {high_res:.2f}")
    xds_inp.write_text("\n".join(out) + "\n")


# ===========================================================================
# Week 6 Task 3 / Week 7 Task 4 -- XDS execution
# ===========================================================================

def run_xds(root_path: Path, job: str, log_path: Path) -> int:
    """
    Set JOB=, run xds_par, capture all output to log_path.

    Returns exit code (0 = success).
    Returns -1 with a clear error message when xds_par is not found.
    Never raises -- all failures are caught and reported.
    """
    set_job_line(root_path, job)
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["xds_par"],
                cwd=str(root_path),
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        return proc.returncode
    except FileNotFoundError:
        log.error(
            "  xds_par not found on PATH. "
            "Make sure XDS is installed and 'xds_par' is on your PATH. "
            "Run: which xds_par"
        )
        return -1
    except Exception as exc:
        log.error("  Unexpected error running xds_par: %s", exc)
        return -1


def tail_log(log_path: Path, n: int = 25) -> str:
    """Return the last n lines of a log file as an indented string."""
    if not log_path.exists():
        return "    (log file not found)"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        tail  = lines[-n:] if len(lines) > n else lines
        return "\n".join(f"    {ln}" for ln in tail)
    except OSError:
        return "    (could not read log file)"



def adaptive_spot_size(root_path: Path, subdir_name: str,
                        n_images: int, log_dir: Path) -> int:
    """
    Find the best MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT value by trying
    multiple values and picking the one that gives the best indexing.

    For microED on a CETA detector, spots are typically 3-6 pixels wide.
    - Too small (2-3): picks up noise along with real spots
    - Too large (8+): misses weak high-resolution spots

    We try [3, 4, 6] and return the value giving the best indexed fraction.
    The best value is also patched into XDS.INP so subsequent steps use it.

    Returns the best MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT value found.
    """
    candidates = [3, 4, 6]
    best_val   = 6       # conservative default
    best_frac  = 0.0

    log.info("  Adaptive spot size: testing MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT = %s",
             candidates)

    for val in candidates:
        # Patch the value into XDS.INP
        xds_inp = root_path / "XDS.INP"
        if not xds_inp.exists():
            break
        lines = xds_inp.read_text(errors="replace").splitlines()
        out = []
        found = False
        for line in lines:
            if line.strip().startswith("MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT="):
                out.append(f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {val}")
                found = True
            else:
                out.append(line)
        if not found:
            out.append(f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {val}")
        xds_inp.write_text("\n".join(out) + "\n")

        # Run just COLSPOT + IDXREF to test
        log_test = log_dir / f"{subdir_name}_spottest_px{val}.log"
        rc = run_xds(root_path, "XYCORR INIT COLSPOT IDXREF", log_test)

        idx   = parse_idxref(root_path)
        frac  = idx["indexed_fraction"] or 0.0
        n_idx = idx["indexed_spots"]    or 0

        log.info("    PIXELS=%d -> %d/%s indexed (%.1f%%)",
                 val, n_idx, idx["total_spots"], frac * 100)

        if frac > best_frac:
            best_frac = frac
            best_val  = val

    log.info("  Best MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT = %d  (%.1f%% indexed)",
             best_val, best_frac * 100)

    # Patch the winning value permanently
    xds_inp = root_path / "XDS.INP"
    if xds_inp.exists():
        lines = xds_inp.read_text(errors="replace").splitlines()
        out = []
        for line in lines:
            if line.strip().startswith("MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT="):
                out.append(f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {best_val}")
            else:
                out.append(line)
        xds_inp.write_text("\n".join(out) + "\n")

    return best_val


def reindex_with_reference_cell(root_path: Path, subdir_name: str,
                                  ref_cell: dict, log_dir: Path) -> dict:
    """
    Re-run IDXREF with the consensus cell as a reference to correct
    datasets that indexed to a slightly wrong alternative lattice.

    When multiple indexing solutions are possible (e.g. different
    orientations of the same lattice), XDS may pick the wrong one.
    Providing the known correct cell as a starting point forces XDS
    to find the solution closest to the reference.

    This is the key to getting consistent indexing across all datasets:
    every dataset ends up describing the same lattice in the same setting.

    ref_cell: dict with keys a,b,c,alpha,beta,gamma (floats)
    Returns updated parse_idxref() result.
    """
    xds_inp = root_path / "XDS.INP"
    if not xds_inp.exists():
        return parse_idxref(root_path)

    a   = ref_cell.get("a",     0)
    b   = ref_cell.get("b",     0)
    c   = ref_cell.get("c",     0)
    alp = ref_cell.get("alpha", 90)
    bet = ref_cell.get("beta",  90)
    gam = ref_cell.get("gamma", 90)

    log.info("  Re-indexing with reference cell: %.2f %.2f %.2f  %.2f %.2f %.2f",
             a, b, c, alp, bet, gam)

    # Patch UNIT_CELL_CONSTANTS and SPACE_GROUP_NUMBER into XDS.INP
    lines = xds_inp.read_text(errors="replace").splitlines()
    out = []
    seen_cell = seen_sg = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("UNIT_CELL_CONSTANTS="):
            out.append(f"UNIT_CELL_CONSTANTS= {a:.3f} {b:.3f} {c:.3f} "
                       f"{alp:.3f} {bet:.3f} {gam:.3f}")
            seen_cell = True
        elif stripped.startswith("SPACE_GROUP_NUMBER="):
            out.append("SPACE_GROUP_NUMBER= 1")
            seen_sg = True
        else:
            out.append(line)
    if not seen_cell:
        out.append(f"UNIT_CELL_CONSTANTS= {a:.3f} {b:.3f} {c:.3f} "
                   f"{alp:.3f} {bet:.3f} {gam:.3f}")
    if not seen_sg:
        out.append("SPACE_GROUP_NUMBER= 1")
    xds_inp.write_text("\n".join(out) + "\n")

    # Re-run IDXREF only (no need for COLSPOT again -- spots are already found)
    log_reindex = log_dir / f"{subdir_name}_XDS_reindex.log"
    t0  = time.time()
    rc  = run_xds(root_path, "IDXREF", log_reindex)
    idx = parse_idxref(root_path)
    frac  = idx["indexed_fraction"] or 0.0
    n_idx = idx["indexed_spots"]    or 0
    log.info("  Re-index result: %d/%s indexed (%.1f%%)  cell=%s  (%.1f s)",
             n_idx, idx["total_spots"], frac * 100,
             idx["unit_cell"], time.time() - t0)
    return idx


def compute_consensus_cell(csv_rows: list) -> dict:
    """
    Compute the consensus unit cell from all successfully processed datasets.

    Uses the median of each cell parameter across datasets that have valid
    cells and produced HKL files. Returns a dict with keys a,b,c,alpha,beta,gamma.
    """
    import statistics

    params = {"a": [], "b": [], "c": [], "alpha": [], "beta": [], "gamma": []}
    for row in csv_rows:
        if str(row.get("has_hkl", "NO")).upper() != "YES":
            continue
        try:
            for p in params:
                v = float(row.get(p) or "n/a")
                params[p].append(v)
        except (ValueError, TypeError):
            pass

    if not params["a"]:
        return {}

    return {p: round(statistics.median(vals), 3)
            for p, vals in params.items() if vals}

def check_xds_available() -> bool:
    """Check xds_par is on PATH. Logs a clear error if not. Call at startup."""
    path = shutil.which("xds_par")
    if path:
        log.info("xds_par found: %s", path)
        return True
    log.error("=" * 60)
    log.error("ERROR: xds_par not found on PATH.")
    log.error("XDS must be installed and accessible before running this pipeline.")
    log.error("To check: which xds_par")
    log.error("Typical fix on a cluster: module load xds")
    log.error("=" * 60)
    return False


def check_images_accessible(root_path: Path, img_files: list) -> bool:
    """
    Verify the first image file is readable and non-empty.
    Logs clear warnings if something looks wrong so the user knows
    before XDS fails silently.
    """
    if not img_files:
        return False
    first = root_path / img_files[0]
    if not first.exists():
        log.error("  First image not found: %s", first)
        return False
    size_bytes = first.stat().st_size
    if size_bytes == 0:
        log.error("  First image is empty (0 bytes): %s", first)
        return False
    n_found = sum(1 for f in img_files if (root_path / f).exists())
    if n_found < len(img_files):
        log.warning("  Only %d of %d .img files found in %s",
                    n_found, len(img_files), root_path)
    log.info("  Images: %d files  first=%s  (%.1f MB)",
             len(img_files), img_files[0], size_bytes / 1e6)
    return True


# ===========================================================================
# Week 6 Task 3 -- Log-file parsers
# ===========================================================================

def parse_idxref(root: Path) -> dict:
    """Parse IDXREF.LP for indexing statistics."""
    result = {
        "indexed_spots":    None,
        "total_spots":      None,
        "indexed_fraction": None,
        "unit_cell":        None,
        "failure_hint":     None,
    }
    lp = root / "IDXREF.LP"
    if not lp.exists():
        result["failure_hint"] = "IDXREF.LP missing"
        return result

    text = lp.read_text(errors="replace")

    m = re.search(r"UNIT CELL PARAMETERS\s+([^\n]+)", text)
    if m:
        result["unit_cell"] = m.group(1).strip()

    m = re.search(r"(\d+)\s+OUT OF\s+(\d+)\s+SPOTS INDEXED", text)
    if m:
        indexed, total = int(m.group(1)), int(m.group(2))
        result["indexed_spots"]    = indexed
        result["total_spots"]      = total
        result["indexed_fraction"] = indexed / total if total else 0.0

    if "ERROR IN REFINE" in text:
        result["failure_hint"] = "ERROR IN REFINE"
    elif "CANNOT INDEX REFLECTIONS" in text:
        result["failure_hint"] = "CANNOT INDEX REFLECTIONS"
    elif "INSUFFICIENT PERCENTAGE" in text:
        result["failure_hint"] = "INSUFFICIENT % INDEXED"
    elif result["indexed_fraction"] is not None and result["indexed_fraction"] < 0.10:
        result["failure_hint"] = "Very low indexed fraction (<10%)"
    else:
        result["failure_hint"] = "IDXREF completed"

    return result


# ===========================================================================
# Week 7 Task 5 -- Full summary statistics from CORRECT.LP
# (space group, unit cell, overall + highest-res shell stats)
# ===========================================================================

def parse_correct_lp(root: Path) -> dict:
    """
    Parse CORRECT.LP for the complete set of quality statistics.

    Handles two CORRECT.LP formats:
      - Standard format: numeric values without % signs
      - Newer XDS format: values include % signs, -99.9 means not available

    Returns a flat dict with space group, cell parameters, and statistics
    for both overall and highest-resolution shell.
    All values are None when not found or not available (-99.9).
    """
    result = {
        "space_group_number":   None,
        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,
        "completeness_overall": None,
        "rmeas_overall":        None,
        "isigi_overall":        None,
        "cc_half_overall":      None,
        "completeness_hi":      None,
        "rmeas_hi":             None,
        "isigi_hi":             None,
        "cc_half_hi":           None,
        "resolution_high":      None,
        "resolution_low":       None,
    }

    lp = root / "CORRECT.LP"
    if not lp.exists():
        return result

    text = lp.read_text(errors="replace")

    # --- Space group ---
    m = re.search(r"SPACE_GROUP_NUMBER\s*=?\s*(\d+)", text)
    if m:
        result["space_group_number"] = int(m.group(1))

    # --- Unit cell ---
    m = re.search(
        r"UNIT CELL PARAMETERS\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text,
    )
    if m:
        result["a"], result["b"], result["c"] = m.group(1), m.group(2), m.group(3)
        result["alpha"], result["beta"], result["gamma"] = m.group(4), m.group(5), m.group(6)

    def clean_val(s):
        """Strip % sign and return float, or None if -99.9 or unparseable."""
        if s is None:
            return None
        s = str(s).strip().rstrip("%").strip()
        try:
            v = float(s)
            return None if v <= -99.0 else v
        except (ValueError, TypeError):
            return None

    # --- Statistics table ---
    # Parse every 'total' line -- XDS CORRECT writes multiple intermediate
    # totals; we take the LAST one which is the final result.
    #
    # Supported formats:
    #   total  n_obs  n_uniq  n_poss  comp%  rsym%  rmeas%  ?  isigi  ?  cc  ...
    #   total  n_obs  n_uniq  mult    comp   isigi  rsym    rmeas  ranom  cc
    #
    # We try multiple column mappings and take the one that gives valid values.

    total_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*total\s+\d", line, re.IGNORECASE)
    ]

    if total_lines:
        last_total = total_lines[-1]
        # Tokenise, stripping % signs from each token
        tokens = [t.rstrip("%") for t in last_total.split()]
        # tokens[0] = "total"
        # Try to extract n_uniq, completeness, isigi, rmeas, cc_half
        # by finding which tokens are plausible values for each field.
        try:
            n_uniq = int(tokens[2]) if len(tokens) > 2 else None

            # Completeness: first token after n fields that looks like a %
            # In the new format: tokens[4] = completeness%
            # In the old format: tokens[4] = completeness (no %)
            comp = None
            for i in [4, 5, 6]:
                if i < len(tokens):
                    v = clean_val(tokens[i])
                    if v is not None and 0 <= v <= 100:
                        comp = v
                        break

            # Rmeas: look for a reasonable Rmeas value (usually 5-200%)
            rmeas = None
            # In new format: tokens[6] or tokens[5] is Rmeas%
            for i in [6, 5, 7]:
                if i < len(tokens):
                    v = clean_val(tokens[i])
                    if v is not None and 0 < v < 500:
                        rmeas = v / 100.0  # convert % to fraction
                        break

            # I/sigma: usually a small positive number (0-100)
            isigi = None
            for i in [10, 9, 8, 11]:
                if i < len(tokens):
                    v = clean_val(tokens[i])
                    if v is not None and -5 <= v <= 200:
                        isigi = v
                        break

            # CC1/2: usually 0-100 or 0-1
            cc_half = None
            for i in [12, 13, 11, 10]:
                if i < len(tokens):
                    v = clean_val(tokens[i])
                    if v is not None and 0 <= v <= 100:
                        cc_half = v / 100.0 if v > 1.0 else v
                        break

            result["completeness_overall"] = comp
            result["rmeas_overall"]        = rmeas
            result["isigi_overall"]        = isigi
            result["cc_half_overall"]      = cc_half

        except (IndexError, ValueError):
            pass

    # --- Shell rows for hi-resolution stats ---
    # Match lines starting with a resolution value (d in Angstroms)
    shell_lines = []
    for line in text.splitlines():
        m = re.match(r"^\s*([\d.]+)\s+([\d]+)\s+([\d]+)\s+", line)
        if m:
            d = safe_float(m.group(1))
            if d is not None and 0.5 <= d <= 50.0:
                tokens = [t.rstrip("%") for t in line.split()]
                shell_lines.append((d, tokens))

    if shell_lines:
        # Sort by d-spacing; smallest d = highest resolution
        shell_lines.sort(key=lambda x: x[0])
        result["resolution_high"] = shell_lines[0][0]
        result["resolution_low"]  = shell_lines[-1][0]

        # Use the highest-resolution shell for hi stats
        hi_d, hi_tokens = shell_lines[0]
        try:
            comp_hi = None
            for i in [4, 5, 6]:
                if i < len(hi_tokens):
                    v = clean_val(hi_tokens[i])
                    if v is not None and 0 <= v <= 100:
                        comp_hi = v
                        break
            rmeas_hi = None
            for i in [6, 5, 7]:
                if i < len(hi_tokens):
                    v = clean_val(hi_tokens[i])
                    if v is not None and 0 < v < 500:
                        rmeas_hi = v / 100.0
                        break
            isigi_hi = None
            for i in [10, 9, 8, 11]:
                if i < len(hi_tokens):
                    v = clean_val(hi_tokens[i])
                    if v is not None and -5 <= v <= 200:
                        isigi_hi = v
                        break
            cc_hi = None
            for i in [12, 13, 11, 10]:
                if i < len(hi_tokens):
                    v = clean_val(hi_tokens[i])
                    if v is not None and 0 <= v <= 100:
                        cc_hi = v / 100.0 if v > 1.0 else v
                        break
            result["completeness_hi"] = comp_hi
            result["rmeas_hi"]        = rmeas_hi
            result["isigi_hi"]        = isigi_hi
            result["cc_half_hi"]      = cc_hi
        except (IndexError, ValueError):
            pass

    return result

    return result


def find_resolution_cutoff(root: Path, isigi_threshold: float = ISIGI_CUTOFF) -> float:
    """
    Scan the resolution-shell table in CORRECT.LP and return the d-spacing of
    the highest-resolution shell where <I/sigma> is still >= isigi_threshold.

    This gives the automated resolution cutoff for the data (Week 7 task 6).
    Returns None if CORRECT.LP is missing or no suitable shell is found.
    """
    lp = root / "CORRECT.LP"
    if not lp.exists():
        return None

    text = lp.read_text(errors="replace")

    row_pat = re.compile(
        r"^\s*([\d.]+)\s+"    # d_limit
        r"\d+\s+\d+\s+"       # n_obs, n_uniq
        r"[\d.]+\s+"          # mult
        r"[\d.]+\s+"          # comp
        r"([\d.]+)\s+",       # <I/sigI>
        re.MULTILINE,
    )

    # Collect (d_limit, isigi) pairs, smallest d first = highest resolution first
    shells = []
    for m in row_pat.finditer(text):
        d     = safe_float(m.group(1))
        isigi = safe_float(m.group(2))
        if d is not None and isigi is not None:
            shells.append((d, isigi))

    if not shells:
        return None

    # Sort highest-resolution (smallest d) first
    shells.sort(key=lambda x: x[0])

    # Walk from low resolution to high resolution; the cutoff is the last shell
    # where I/sigma is still above the threshold
    cutoff = None
    for d, isigi in reversed(shells):   # reversed = low res to high res
        if isigi >= isigi_threshold:
            cutoff = d
            break

    return cutoff


# ===========================================================================
# Cell parameter extraction (fallback when CORRECT.LP is absent)
# ===========================================================================

def extract_cell_params(root: Path):
    """
    Extract unit-cell parameters from CORRECT.LP (preferred) or IDXREF.LP.
    Returns a 6-tuple of strings, or ('n/a', ...) if nothing found.
    """
    def _try(text):
        m = re.search(
            r"UNIT CELL PARAMETERS\s+"
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            text,
        )
        if m:
            return m.groups()
        m = re.search(
            r"PARAMETERS OF THE REDUCED CELL.*?\n\s*"
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            text, re.DOTALL,
        )
        if m:
            return m.groups()
        return None

    for src in [root / "CORRECT.LP", root / "IDXREF.LP"]:
        if src.exists():
            try:
                p = _try(src.read_text(errors="replace"))
                if p:
                    return p
            except OSError:
                pass
    return ("n/a",) * 6


# ===========================================================================
# XSCALE helpers (Week 7 Tasks 7 & 8)
# ===========================================================================

def write_xscale_inp(output_path: Path, hkl_entries: list,
                     resolution_high: float = 2.0,
                     space_group_number: int = 0,
                     unit_cell: str = "") -> None:
    """
    Write a valid XSCALE.INP file.

    hkl_entries is a list of dicts, each with:
        'hkl_path'  -- absolute Path to XDS_ASCII.HKL
        'name'      -- short dataset label (used in the INPUT_FILE block)

    Space group and unit cell constants are written at the global level so
    XSCALE merges all datasets in the same symmetry.  Pass space_group_number=0
    to let XSCALE determine symmetry automatically.
    """
    lines = [
        "! XSCALE.INP -- auto-generated by xds_pipeline.py",
        "!",
        f"RESOLUTION_SHELLS= 50.0 {resolution_high:.2f}",
        "SAVE_CORRECTION_IMAGES= FALSE",
        "",
        "OUTPUT_FILE= XSCALE.HKL",
        "",
    ]
    if space_group_number and space_group_number != 0:
        lines.append(f"SPACE_GROUP_NUMBER= {space_group_number}")
    if unit_cell:
        lines.append(f"UNIT_CELL_CONSTANTS= {unit_cell}")
    lines.append("")

    for entry in hkl_entries:
        lines.append(f"INPUT_FILE= {entry['hkl_path']}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def run_xscale(run_dir: Path) -> int:
    """Run xscale_par in run_dir, write output to xscale.log. Returns exit code."""
    log_path = run_dir / "xscale.log"
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            ["xscale_par"],
            cwd=str(run_dir),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def parse_xscale_lp(run_dir: Path) -> dict:
    """
    Parse XSCALE.LP for merged-dataset quality statistics.

    XSCALE.LP contains multiple "total" lines:
      1. The main statistics table total  -- has 13+ tokens including n_obs,
         n_uniq, completeness%, Rsym%, Rmeas%, I/sig, CC1/2 etc.
      2. Per-dataset R-factor totals       -- short lines with only 3-4 tokens

    We specifically look for the main statistics total line (>= 10 tokens).

    XSCALE.LP column layout (new XDS format with % signs):
      total  n_obs  n_uniq  n_poss  comp%  rsym%  rmeas%  ?  isigi  ranom%  cc_half*  ...

    Values with * (e.g. 96.6*) mean the CC1/2 is statistically significant.
    Values of -99.9 mean the statistic could not be calculated.
    """
    result = {
        "completeness_overall": None,
        "rmerge_overall":       None,
        "rmeas_overall":        None,
        "isigi_overall":        None,
        "cc_half_overall":      None,
        "n_unique_overall":     None,
        "completeness_hi":      None,
        "rmerge_hi":            None,
        "rmeas_hi":             None,
        "isigi_hi":             None,
        "cc_half_hi":           None,
        "resolution_high":      None,
    }

    lp = run_dir / "XSCALE.LP"
    if not lp.exists():
        return result

    text = lp.read_text(errors="replace")

    def clean_val(s):
        """Strip %, * and whitespace. Return float or None if -99.9/invalid."""
        if s is None:
            return None
        s = str(s).strip().rstrip("%").rstrip("*").strip()
        try:
            v = float(s)
            return None if v <= -99.0 else v
        except (ValueError, TypeError):
            return None

    # ---------------------------------------------------------------
    # Find the MAIN statistics total line.
    # It has >= 10 tokens and the 4th token (index 4) is completeness%.
    # Short "total" lines from per-dataset R-factor tables have only 3-4 tokens.
    # ---------------------------------------------------------------
    main_total_tokens = None
    for line in text.splitlines():
        if not re.match(r"^\s*total\s+\d", line, re.IGNORECASE):
            continue
        tokens = [t.rstrip("%").rstrip("*") for t in line.split()]
        if len(tokens) >= 10:
            # Verify this looks like the main statistics line:
            # token[4] should be completeness (0-100)
            v = clean_val(tokens[4]) if len(tokens) > 4 else None
            if v is not None and 0 <= v <= 100:
                main_total_tokens = tokens
                # Don't break -- take the LAST matching total line
    
    if main_total_tokens:
        t = main_total_tokens
        # Column mapping confirmed from XSCALE.LP inspection:
        # [0]=total [1]=n_obs [2]=n_uniq [3]=n_poss [4]=comp%
        # [5]=rsym% [6]=rmeas% [7]=? [8]=isigi [9]=ranom% [10]=cc_half*
        try:
            result["n_unique_overall"]     = int(t[2]) if len(t) > 2 else None
            result["completeness_overall"] = clean_val(t[4]) if len(t) > 4 else None
            result["rmerge_overall"]       = (clean_val(t[5]) / 100.0
                                              if len(t) > 5 and clean_val(t[5]) else None)
            result["rmeas_overall"]        = (clean_val(t[6]) / 100.0
                                              if len(t) > 6 and clean_val(t[6]) else None)
            result["isigi_overall"]        = clean_val(t[8])  if len(t) > 8  else None
            # CC1/2: token[10] is e.g. "96.6*" -- already stripped above
            cc_val = clean_val(t[10]) if len(t) > 10 else None
            if cc_val is not None:
                result["cc_half_overall"] = cc_val / 100.0 if cc_val > 1.0 else cc_val
        except (IndexError, ValueError, TypeError):
            pass

    # ---------------------------------------------------------------
    # Shell rows for highest-resolution statistics
    # Lines that start with a numeric d-spacing (resolution in Angstroms)
    # ---------------------------------------------------------------
    shells = []
    for line in text.splitlines():
        m = re.match(r"^\s*([\d.]+)\s+\d+\s+\d+\s+\d+\s+", line)
        if m:
            d = safe_float(m.group(1))
            if d and 0.5 <= d <= 50.0:
                tokens = [t.rstrip("%").rstrip("*") for t in line.split()]
                shells.append((d, tokens))

    if shells:
        shells.sort(key=lambda x: x[0])
        hi_d, hi_tok = shells[0]
        result["resolution_high"] = hi_d
        try:
            result["completeness_hi"] = clean_val(hi_tok[4]) if len(hi_tok) > 4 else None
            result["rmerge_hi"]  = (clean_val(hi_tok[5]) / 100.0
                                    if len(hi_tok) > 5 and clean_val(hi_tok[5]) else None)
            result["rmeas_hi"]   = (clean_val(hi_tok[6]) / 100.0
                                    if len(hi_tok) > 6 and clean_val(hi_tok[6]) else None)
            result["isigi_hi"]   = clean_val(hi_tok[8]) if len(hi_tok) > 8 else None
            cc_hi = clean_val(hi_tok[10]) if len(hi_tok) > 10 else None
            if cc_hi is not None:
                result["cc_half_hi"] = cc_hi / 100.0 if cc_hi > 1.0 else cc_hi
        except (IndexError, ValueError, TypeError):
            pass

    return result



def merge_quality_score(stats: dict) -> float:
    """
    Composite quality score for a merged dataset in [0, 1].

    Weights (tuned for crystallographic relevance):
      completeness_overall  0.40  -- most important: complete data is essential
      cc_half_overall       0.30  -- statistical reliability of measurements
      isigi_overall         0.20  -- signal strength (saturates at I/sig = 20)
      rmeas_overall         0.10  -- internal consistency (inverted, lower = better)
    """
    score = 0.0
    if stats.get("completeness_overall") is not None:
        score += 0.40 * min(stats["completeness_overall"] / 100.0, 1.0)
    if stats.get("cc_half_overall") is not None:
        score += 0.30 * max(0.0, min(stats["cc_half_overall"], 1.0))
    if stats.get("isigi_overall") is not None:
        score += 0.20 * min(stats["isigi_overall"] / 20.0, 1.0)
    if stats.get("rmeas_overall") is not None:
        score += 0.10 * max(0.0, 1.0 - stats["rmeas_overall"] / 0.5)
    return round(score, 4)


def score_individual_dataset(row: dict) -> float:
    """
    Pre-merge quality score for one dataset (0-1).
    Used only to rank candidates before the greedy search starts.
    """
    score = 0.0
    try:
        score += 0.35 * min(float(row.get("idxref_fraction") or 0), 1.0)
    except (TypeError, ValueError):
        pass
    if str(row.get("has_hkl", "NO")).strip().upper() == "YES":
        score += 0.25
    try:
        score += 0.25 * min(float(row.get("cnn_quality_score") or 0), 1.0)
    except (TypeError, ValueError):
        pass
    if str(row.get("cnn_disagreement", "NO")).strip().upper() != "YES":
        score += 0.15
    return round(score, 4)


def run_merge_trial(trial_dir: Path, hkl_entries: list,
                    resolution_high: float = 2.0,
                    space_group_number: int = 0,
                    unit_cell: str = "") -> dict:
    """
    Write XSCALE.INP, run XSCALE, parse XSCALE.LP -- all in one call.
    Returns parsed stats dict plus 'merge_score' and 'xscale_rc'.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_xscale_inp(
        trial_dir / "XSCALE.INP",
        hkl_entries,
        resolution_high=resolution_high,
        space_group_number=0,   # let XSCALE determine symmetry
        unit_cell="",           # let XSCALE determine cell
    )
    rc = run_xscale(trial_dir)
    stats = parse_xscale_lp(trial_dir)
    stats["xscale_rc"]   = rc
    stats["merge_score"] = merge_quality_score(stats) if rc == 0 else 0.0
    return stats


def check_space_group_consistency(candidates: list) -> tuple:
    """
    Check that all candidate datasets share the same space group.

    Returns (space_group_number, unit_cell_str, filtered_candidates) where
    filtered_candidates excludes any dataset with a different space group.
    The majority space group is used as the reference.
    """
    from collections import Counter

    sg_counts = Counter(c.get("space_group_number") for c in candidates
                        if c.get("space_group_number") is not None)
    if not sg_counts:
        return 0, "", candidates

    majority_sg, _ = sg_counts.most_common(1)[0]

    # Use the unit cell from the first dataset that has the majority space group
    unit_cell_str = ""
    for c in candidates:
        if c.get("space_group_number") == majority_sg:
            a = c.get("a", "")
            b = c.get("b", "")
            cv = c.get("c", "")
            al = c.get("alpha", "")
            be = c.get("beta", "")
            ga = c.get("gamma", "")
            if all(v not in ("n/a", "", None) for v in (a, b, cv, al, be, ga)):
                unit_cell_str = f"{a} {b} {cv} {al} {be} {ga}"
            break

    # Filter out datasets with a different space group
    consistent = [c for c in candidates
                  if c.get("space_group_number") in (majority_sg, None)]
    excluded = [c for c in candidates
                if c.get("space_group_number") not in (majority_sg, None)]

    if excluded:
        log.warning("  Excluding %d dataset(s) with non-matching space group:",
                    len(excluded))
        for c in excluded:
            log.warning("    %s  (SG %s)", c["name"], c.get("space_group_number"))

    return majority_sg, unit_cell_str, consistent


def greedy_subset_search(candidates: list, trials_dir: Path,
                         resolution_high: float = 2.0,
                         space_group_number: int = 0,
                         unit_cell: str = "") -> tuple:
    """
    Greedy forward selection to find the dataset subset that maximises
    the composite merge quality score (completeness, CC½, I/sig, Rmeas).

    Algorithm:
      1. Sort candidates by pre-merge individual quality score (best first).
      2. Seed with the single best dataset.
      3. For each remaining dataset, run a trial XSCALE merge including it.
      4. Accept if the merge score improves; reject otherwise.
      5. Repeat until all candidates have been evaluated.

    Runs O(N) XSCALE calls for N datasets -- efficient for typical batch sizes.

    Returns (best_subset_list, best_stats_dict).
    """
    if not candidates:
        return [], {}

    ranked = sorted(candidates, key=lambda c: c["ind_score"], reverse=True)
    log.info("  Greedy search over %d candidate datasets...", len(ranked))

    # -- Seed --
    current_subset = [ranked[0]]
    seed_stats = run_merge_trial(
        trials_dir / "trial_seed",
        [{"hkl_path": c["hkl_path"], "name": c["name"]} for c in current_subset],
        resolution_high=resolution_high,
        space_group_number=space_group_number,
        unit_cell=unit_cell,
    )
    best_score = seed_stats["merge_score"]
    best_stats = seed_stats

    log.info(
        "  Seed: %-30s  score=%.4f  comp=%.1f%%  Rmeas=%s  CC½=%s  I/sig=%s",
        ranked[0]["name"], best_score,
        seed_stats.get("completeness_overall") or 0,
        f'{seed_stats["rmeas_overall"]:.3f}'  if seed_stats.get("rmeas_overall")  else "n/a",
        f'{seed_stats["cc_half_overall"]:.3f}' if seed_stats.get("cc_half_overall") else "n/a",
        f'{seed_stats["isigi_overall"]:.1f}'  if seed_stats.get("isigi_overall")  else "n/a",
    )

    # -- Forward selection --
    for i, candidate in enumerate(ranked[1:], start=1):
        trial_subset = current_subset + [candidate]
        trial_entries = [{"hkl_path": c["hkl_path"], "name": c["name"]}
                         for c in trial_subset]
        trial_dir = trials_dir / f"trial_{i:02d}_{candidate['name']}"

        trial_stats = run_merge_trial(
            trial_dir, trial_entries,
            resolution_high=resolution_high,
            space_group_number=space_group_number,
            unit_cell=unit_cell,
        )
        trial_score = trial_stats["merge_score"]

        if trial_score > best_score:
            current_subset = trial_subset
            best_score = trial_score
            best_stats = trial_stats
            log.info(
                "  + ACCEPTED %-28s  score=%.4f  comp=%.1f%%  Rmeas=%s  CC½=%s",
                candidate["name"], trial_score,
                trial_stats.get("completeness_overall") or 0,
                f'{trial_stats["rmeas_overall"]:.3f}'   if trial_stats.get("rmeas_overall")  else "n/a",
                f'{trial_stats["cc_half_overall"]:.3f}'  if trial_stats.get("cc_half_overall") else "n/a",
            )
        else:
            log.info(
                "  - REJECTED %-28s  trial=%.4f  best=%.4f",
                candidate["name"], trial_score, best_score,
            )
            # If trial score is 0, show why XSCALE failed
            if trial_score == 0.0 and trial_stats.get("xscale_rc") == 0:
                lp = trial_dir / "XSCALE.LP"
                if lp.exists():
                    # Show the total line so we can see what XSCALE produced
                    for line in lp.read_text(errors="replace").splitlines():
                        if "total" in line.lower() or "error" in line.lower():
                            log.info("    XSCALE.LP: %s", line.strip())
                            break

    return current_subset, best_stats



# ===========================================================================
# IDXREF retry with relaxed parameters
# ===========================================================================

def retry_idxref_relaxed(root_path: Path, subdir_name: str,
                          log_dir: Path) -> dict:
    """
    If IDXREF fails with strict settings, retry once with relaxed thresholds.

    Relaxed settings:
      MINIMUM_FRACTION_OF_INDEXED_SPOTS = 0.10  (was 0.30)
      MAXIMUM_ERROR_OF_SPOT_POSITION    = 25.0  (was 15.0)
      MAXIMUM_ERROR_OF_SPINDLE_POSITION = 10.0  (was 7.5)
      INDEX_QUALITY                     = 0.50  (was 0.70)

    This often recovers datasets where the crystal was slightly off-centre,
    had higher mosaicity, or had fewer spots than average.
    Returns parse_idxref() dict after the retry.
    """
    log.info("  Retrying IDXREF with relaxed parameters...")

    xds_inp = root_path / "XDS.INP"
    if not xds_inp.exists():
        return parse_idxref(root_path)

    # Patch in relaxed values temporarily
    lines = xds_inp.read_text(errors="replace").splitlines()
    relaxed = {
        "MINIMUM_FRACTION_OF_INDEXED_SPOTS=": "MINIMUM_FRACTION_OF_INDEXED_SPOTS= 0.10",
        "MAXIMUM_ERROR_OF_SPOT_POSITION=":    "MAXIMUM_ERROR_OF_SPOT_POSITION= 25.0",
        "MAXIMUM_ERROR_OF_SPINDLE_POSITION=": "MAXIMUM_ERROR_OF_SPINDLE_POSITION= 10.0",
        "INDEX_QUALITY=":                     "INDEX_QUALITY= 0.50",
    }
    out, seen = [], set()
    for line in lines:
        key = next((k for k in relaxed if line.strip().startswith(k)), None)
        if key is None:
            out.append(line)
        elif key not in seen:
            seen.add(key)
            out.append(relaxed[key])
    for key, value in relaxed.items():
        if key not in seen:
            out.append(value)
    xds_inp.write_text("\n".join(out) + "\n")

    # Run IDXREF only (not the full Phase 1 chain)
    log_retry = log_dir / f"{subdir_name}_XDS_idxref_retry.log"
    t0 = time.time()
    rc = run_xds(root_path, "IDXREF", log_retry)
    log.info("  Retry finished in %.1f s  (exit code %d)", time.time() - t0, rc)

    result = parse_idxref(root_path)
    frac  = result["indexed_fraction"] or 0.0
    n_idx = result["indexed_spots"]    or 0
    log.info("  Retry IDXREF: %s/%s indexed (%.1f%%)  hint=%s",
             n_idx, result["total_spots"], frac * 100, result["failure_hint"])
    return result


# ===========================================================================
# Bad frame exclusion from INTEGRATE.LP
# ===========================================================================

def parse_integrate_lp_frames(root: Path) -> list:
    """
    Parse INTEGRATE.LP for per-frame statistics.

    Returns a list of dicts, one per frame, with keys:
      frame, n_obs, fraction_observed, isigi_mean

    Used to identify frames with unusually low completeness or I/sigma
    that should be excluded before running CORRECT.
    """
    lp = root / "INTEGRATE.LP"
    if not lp.exists():
        return []

    text  = lp.read_text(errors="replace")
    frames = []

    # INTEGRATE.LP frame table format (typical):
    # IMAGE   IER  SCALE   NBKG NOUT NADD  NFULL ... FRACTION ...
    # Each data line starts with the frame number
    frame_pat = re.compile(
        r"^\s*(\d+)\s+"       # frame number
        r"[\d.]+\s+"          # IER
        r"[\d.]+\s+"          # SCALE
        r"\d+\s+\d+\s+\d+\s+" # NBKG NOUT NADD
        r"\d+\s+"             # NFULL
        r"([\d.]+)\s+"        # FRACTION_OBSERVED
        r"[\d.]+\s+"          # CORR
        r"([\d.]+)",          # MNSIG (I/sigma estimate)
        re.MULTILINE,
    )

    for m in frame_pat.finditer(text):
        frame_num    = int(m.group(1))
        frac_obs     = safe_float(m.group(2))
        isigi_mean   = safe_float(m.group(3))
        frames.append({
            "frame":              frame_num,
            "fraction_observed":  frac_obs,
            "isigi_mean":         isigi_mean,
        })

    return frames


def find_bad_frames(frames: list,
                    isigi_threshold: float = 0.1,
                    frac_threshold:  float = 0.05,
                    max_bad_fraction: float = 0.30) -> list:
    """
    Identify frames that should be excluded from CORRECT.

    Very conservative thresholds are used -- removing too many frames from
    a microED dataset (which has few frames to begin with) is worse than
    keeping some weak ones. Only frames that are clearly broken are excluded.

    Rules:
      - Frame number must be > 0 (frame 0 is invalid -- parser matched wrong row)
      - Both I/sigma AND fraction_observed must be below threshold simultaneously
      - Never exclude more than max_bad_fraction (30%) of total frames
    """
    if not frames:
        return []

    total_frames = len(frames)
    bad = []

    for f in frames:
        frame_num = f.get("frame", 0)
        if frame_num <= 0:
            continue   # invalid frame number -- skip

        isigi = f.get("isigi_mean")
        frac  = f.get("fraction_observed")

        # Only exclude when BOTH metrics are clearly bad simultaneously
        if (isigi is not None and isigi < isigi_threshold and
                frac is not None and frac < frac_threshold):
            bad.append(frame_num)

    # Safety cap: never exclude more than 30% of frames
    max_to_exclude = max(1, int(total_frames * max_bad_fraction))
    if len(bad) > max_to_exclude:
        log.info("  Frame exclusion capped at %d/%d (30%% safety limit).",
                 max_to_exclude, total_frames)
        bad = sorted(bad)[:max_to_exclude]

    return sorted(bad)

def add_exclude_frames(root: Path, bad_frames: list) -> None:
    """
    Add EXCLUDE_FRAMES= lines to XDS.INP for each bad frame identified.

    XDS accepts individual frame numbers or ranges:
      EXCLUDE_FRAMES= 5 5    ! exclude frame 5
      EXCLUDE_FRAMES= 10 15  ! exclude frames 10 through 15

    We write one line per contiguous range for compactness.
    """
    if not bad_frames:
        return

    xds_inp = root / "XDS.INP"
    if not xds_inp.exists():
        return

    # Build contiguous ranges from the list of bad frame numbers
    ranges = []
    start = bad_frames[0]
    end   = bad_frames[0]
    for fn in bad_frames[1:]:
        if fn == end + 1:
            end = fn
        else:
            ranges.append((start, end))
            start = end = fn
    ranges.append((start, end))

    # Remove any existing EXCLUDE_FRAMES lines then append fresh ones
    lines = xds_inp.read_text(errors="replace").splitlines()
    lines = [l for l in lines if not l.strip().startswith("EXCLUDE_FRAMES=")]
    for s, e in ranges:
        lines.append(f"EXCLUDE_FRAMES= {s} {e}  ! auto-excluded (low quality)")

    xds_inp.write_text("\n".join(lines) + "\n")
    log.info("  Excluded %d bad frame(s) in %d range(s): %s",
             len(bad_frames), len(ranges),
             ", ".join(f"{s}-{e}" if s != e else str(s) for s, e in ranges))


# ===========================================================================
# Ice ring detection and exclusion
# ===========================================================================

# Common ice ring d-spacings in Angstroms (from Thorn et al. 2017)
_ICE_RINGS = [
    (3.897, 0.03), (3.669, 0.03), (3.441, 0.03),
    (2.671, 0.03), (2.249, 0.03), (2.072, 0.03),
    (1.948, 0.03), (1.918, 0.03), (1.883, 0.03),
    (1.721, 0.03),
]


def detect_ice_rings(root: Path) -> list:
    """
    Scan CORRECT.LP shell statistics for anomalously high completeness or
    Rmeas in shells that correspond to known ice ring d-spacings.

    Returns a list of (d_low, d_high) exclusion ranges for detected ice rings.
    These can be added to XDS.INP as EXCLUDE_RESOLUTION_RANGE lines.
    """
    lp = root / "CORRECT.LP"
    if not lp.exists():
        return []

    text = lp.read_text(errors="replace")

    # Parse shell rows: d_limit and Rmeas
    row_pat = re.compile(
        r"^\s*([\d.]+)\s+"   # d_limit
        r"\d+\s+\d+\s+"      # n_obs n_uniq
        r"[\d.]+\s+"         # mult
        r"([\d.]+)\s+"       # comp
        r"[\d.]+\s+"         # isigi
        r"[\d.]+\s+"         # Rsym
        r"([\d.]+)",         # Rmeas
        re.MULTILINE,
    )

    shells = []
    for m in row_pat.finditer(text):
        d     = safe_float(m.group(1))
        rmeas = safe_float(m.group(3))
        if d is not None and rmeas is not None:
            shells.append((d, rmeas))

    if not shells:
        return []

    # Compute mean Rmeas across all shells as a baseline
    mean_rmeas = sum(r for _, r in shells) / len(shells)
    detected   = []

    for ice_d, half_width in _ICE_RINGS:
        d_lo = ice_d - half_width
        d_hi = ice_d + half_width
        # Find shells that fall within this ice ring range
        ring_shells = [(d, r) for d, r in shells if d_lo <= d <= d_hi]
        if ring_shells:
            ring_rmeas = sum(r for _, r in ring_shells) / len(ring_shells)
            # Flag if Rmeas in this shell is more than 2x the mean
            if ring_rmeas > 2.0 * mean_rmeas:
                detected.append((round(d_hi, 3), round(d_lo, 3)))
                log.info("  Ice ring detected at %.3f A  (Rmeas=%.3f vs mean=%.3f)",
                         ice_d, ring_rmeas, mean_rmeas)

    return detected


def add_ice_ring_exclusions(root: Path, exclusions: list) -> None:
    """
    Add EXCLUDE_RESOLUTION_RANGE lines to XDS.INP for detected ice rings.
    Removes any existing EXCLUDE_RESOLUTION_RANGE lines first to avoid
    duplicates on rerun.
    """
    if not exclusions:
        return

    xds_inp = root / "XDS.INP"
    if not xds_inp.exists():
        return

    lines = xds_inp.read_text(errors="replace").splitlines()
    lines = [l for l in lines
             if not (l.strip().startswith("EXCLUDE_RESOLUTION_RANGE=")
                     and "ice" in l.lower())]

    for d_hi, d_lo in exclusions:
        lines.append(
            f"EXCLUDE_RESOLUTION_RANGE= {d_hi:.3f} {d_lo:.3f}  ! ice ring"
        )

    xds_inp.write_text("\n".join(lines) + "\n")


# ===========================================================================
# Unit cell clustering (smarter than space group check alone)
# ===========================================================================

def cell_distance(cell_a: tuple, cell_b: tuple,
                  len_tol: float = 5.0, ang_tol: float = 5.0) -> float:
    """
    Compute a normalised distance between two unit cells.

    Length parameters (a, b, c) are compared in Angstroms.
    Angle parameters (alpha, beta, gamma) are compared in degrees.
    Returns 0.0 for identical cells, larger values for more different cells.
    Returns infinity if either cell contains non-numeric values.
    """
    try:
        dists = []
        for i in range(3):   # a, b, c
            dists.append(abs(float(cell_a[i]) - float(cell_b[i])) / len_tol)
        for i in range(3, 6):  # alpha, beta, gamma
            dists.append(abs(float(cell_a[i]) - float(cell_b[i])) / ang_tol)
        return sum(dists) / 6.0
    except (TypeError, ValueError):
        return float("inf")


def cluster_by_unit_cell(candidates: list,
                          distance_threshold: float = 1.0) -> list:
    """
    Group datasets by unit cell similarity and return only those in the
    largest cluster (most common unit cell).

    This catches cases where XDS indexed some crystals in a different
    setting or orientation of the same unit cell -- those datasets would
    pass the space group check but produce poor merges with XSCALE.

    Parameters:
        candidates          : list of candidate dicts (must have a,b,c,alpha,beta,gamma)
        distance_threshold  : max normalised distance to be in the same cluster (default 1.0)

    Returns the filtered list containing only the largest cluster.
    """
    if len(candidates) <= 1:
        return candidates

    # Build a cell tuple for each candidate
    def get_cell(c):
        return (c.get("a"), c.get("b"), c.get("c"),
                c.get("alpha"), c.get("beta"), c.get("gamma"))

    # Simple greedy clustering
    clusters = []
    assigned = [False] * len(candidates)

    for i, c in enumerate(candidates):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        cell_i = get_cell(c)
        for j, d in enumerate(candidates):
            if assigned[j]:
                continue
            cell_j = get_cell(d)
            if cell_distance(cell_i, cell_j) <= distance_threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Pick the largest cluster
    largest = max(clusters, key=len)

    if len(largest) < len(candidates):
        excluded_count = len(candidates) - len(largest)
        log.info("  Unit cell clustering: keeping %d datasets, "
                 "excluding %d with dissimilar cell.",
                 len(largest), excluded_count)
        for idx in range(len(candidates)):
            if idx not in largest:
                c = candidates[idx]
                log.info("    Excluded by cell clustering: %s  cell=%s %s %s %s %s %s",
                         c["name"],
                         c.get("a"), c.get("b"), c.get("c"),
                         c.get("alpha"), c.get("beta"), c.get("gamma"))

    return [candidates[i] for i in largest]


# ===========================================================================
# Formatted summary table
# ===========================================================================

def print_summary_table(csv_rows: list) -> None:
    """
    Print a formatted ASCII table of all processed datasets to the terminal,
    ranked by overall completeness (best first).

    Shows the most important statistics at a glance without opening the CSV.
    """
    if not csv_rows:
        return

    # Sort by completeness descending, datasets with no HKL at the bottom
    def sort_key(row):
        try:
            comp = float(row.get("completeness_overall") or -1)
        except (TypeError, ValueError):
            comp = -1
        return comp

    sorted_rows = sorted(csv_rows, key=sort_key, reverse=True)

    header = (
        f"{'Dataset':<25} {'SG':>4} {'HKL':>4} "
        f"{'Comp%':>6} {'Rmeas':>6} {'I/sig':>6} {'CC½':>6} "
        f"{'Hi-comp':>7} {'Hi-Isig':>7}"
    )
    sep = "-" * len(header)

    log.info("\n%s", sep)
    log.info("DATASET SUMMARY (ranked by completeness)")
    log.info("%s", sep)
    log.info(header)
    log.info("%s", sep)

    for row in sorted_rows:
        def fmt(v, fmt_str=".1f", na="  n/a"):
            try:
                return format(float(v), fmt_str) if v not in (None, "n/a", "") else na
            except (TypeError, ValueError):
                return na

        name = str(row.get("subdirectory", "?"))[:24]
        sg   = str(row.get("space_group") or "?")[:4]
        hkl  = "YES" if str(row.get("has_hkl", "NO")).upper() == "YES" else "NO"

        log.info(
            "%-25s %4s %4s %6s %6s %6s %6s %7s %7s",
            name, sg, hkl,
            fmt(row.get("completeness_overall"), ".1f"),
            fmt(row.get("rmeas_overall"),        ".3f"),
            fmt(row.get("isigi_overall"),         ".1f"),
            fmt(row.get("cc_half_overall"),       ".3f"),
            fmt(row.get("completeness_hi"),       ".1f"),
            fmt(row.get("isigi_hi"),              ".1f"),
        )

    log.info("%s\n", sep)


# ===========================================================================
# Watch mode -- keep processing new datasets as they appear
# ===========================================================================

def get_processed_subdirs(parent_dir: Path) -> set:
    """Return the set of subdirectory names that already have CORRECT.LP."""
    done = set()
    for d in parent_dir.iterdir():
        if d.is_dir() and (d / "CORRECT.LP").exists():
            done.add(d.name)
    return done



# ===========================================================================
# Smart pre-filtering: find the dominant crystal form automatically
# ===========================================================================

def find_dominant_crystal_form(candidates: list,
                                sigma_cutoff: float = 2.0) -> tuple:
    """
    Identify the dominant crystal form by statistical outlier detection.

    For each of the six unit cell parameters, compute the median and
    median absolute deviation (MAD) across all candidates that have valid
    cell values. Any dataset whose cell differs from the median by more than
    sigma_cutoff * MAD is flagged as an outlier and excluded.

    This approach is more robust than a fixed tolerance because it adapts
    to the actual spread of your data rather than requiring you to know
    the expected cell in advance.

    Parameters
    ----------
    candidates   : list of candidate dicts with keys a,b,c,alpha,beta,gamma
    sigma_cutoff : datasets more than this many MADs from median are excluded
                   (default 2.0 -- catches clear outliers without over-filtering)

    Returns
    -------
    (clean, rejected, consensus_cell) where:
        clean         : list of candidates that passed all filters
        rejected      : list of (candidate, reason) tuples
        consensus_cell: dict with median a,b,c,alpha,beta,gamma values
    """
    import statistics

    param_names = ("a", "b", "c", "alpha", "beta", "gamma")

    # Collect numeric cell values for each parameter
    param_values = {p: [] for p in param_names}
    valid_indices = []

    for i, c in enumerate(candidates):
        try:
            vals = {p: float(c.get(p) or "n/a") for p in param_names}
            for p in param_names:
                param_values[p].append(vals[p])
            valid_indices.append(i)
        except (TypeError, ValueError):
            pass  # cell has n/a values -- will be caught later

    if len(valid_indices) < 2:
        # Not enough data to do statistics -- keep everything
        return candidates, [], {}

    # Compute median and MAD for each parameter
    medians = {}
    mads    = {}
    for p in param_names:
        vals   = param_values[p]
        median = statistics.median(vals)
        mad    = statistics.median([abs(v - median) for v in vals])
        # Use a minimum MAD to avoid division by zero for very tight clusters
        mads[p]    = max(mad, 0.1)
        medians[p] = median

    consensus_cell = {p: round(medians[p], 3) for p in param_names}

    # Score each candidate against the consensus
    clean    = []
    rejected = []

    for c in candidates:
        try:
            vals = {p: float(c.get(p) or "n/a") for p in param_names}
        except (TypeError, ValueError):
            rejected.append((c, "no valid unit cell parameters"))
            continue

        # Find worst-offending parameter
        deviations = {
            p: abs(vals[p] - medians[p]) / mads[p]
            for p in param_names
        }
        worst_param = max(deviations, key=deviations.get)
        worst_dev   = deviations[worst_param]

        if worst_dev > sigma_cutoff:
            reason = (
                f"{worst_param}: value={vals[worst_param]:.2f}  "
                f"median={medians[worst_param]:.2f}  "
                f"deviation={worst_dev:.1f} MADs (cutoff={sigma_cutoff})"
            )
            rejected.append((c, reason))
        else:
            clean.append(c)

    return clean, rejected, consensus_cell


def quality_score_from_row(row: dict) -> float:
    """
    Compute a quality score from actual CORRECT.LP statistics when available,
    falling back to indexing metrics when they are not.

    Priority order:
      1. Real merged statistics (completeness, CC1/2, I/sigma, Rmeas)
         -- these are the most meaningful quality indicators
      2. Indexing fraction and HKL presence
         -- used when CORRECT.LP stats are missing or zero

    Returns a score in [0, 1], higher is better.
    """
    score = 0.0
    has_real_stats = False

    # --- Real CORRECT.LP statistics ---
    comp = safe_float(row.get("completeness_overall"), None)
    cc   = safe_float(row.get("cc_half_overall"),      None)
    isig = safe_float(row.get("isigi_overall"),        None)
    rmeas = safe_float(row.get("rmeas_overall"),       None)

    # Only use real stats if they look valid (not zero or -99.9)
    if comp and comp > 0:
        score += 0.40 * min(comp / 100.0, 1.0)
        has_real_stats = True
    if cc and cc > 0:
        score += 0.25 * max(0.0, min(cc, 1.0))
        has_real_stats = True
    if isig and isig > 0:
        score += 0.20 * min(isig / 20.0, 1.0)
        has_real_stats = True
    if rmeas and 0 < rmeas < 0.5:
        score += 0.10 * max(0.0, 1.0 - rmeas / 0.5)
        has_real_stats = True

    # --- Indexing fallback ---
    if not has_real_stats:
        try:
            frac = float(row.get("idxref_fraction") or 0)
            score += 0.50 * min(frac, 1.0)
        except (TypeError, ValueError):
            pass
        if str(row.get("has_hkl", "NO")).upper() == "YES":
            score += 0.35
        try:
            cnn_q = float(row.get("cnn_quality_score") or 0)
            score += 0.15 * min(cnn_q, 1.0)
        except (TypeError, ValueError):
            pass

    return round(min(score, 1.0), 4)

# ===========================================================================
# Main pipeline
# ===========================================================================


def _write_csv(csv_path: Path, csv_rows: list) -> None:
    """Write the full summary CSV from csv_rows."""
    fieldnames = [
        "subdirectory", "space_group",
        "a", "b", "c", "alpha", "beta", "gamma",
        "has_hkl",
        "idxref_indexed", "idxref_total", "idxref_fraction", "idxref_hint",
        "completeness_overall", "rmeas_overall", "isigi_overall", "cc_half_overall",
        "completeness_hi", "rmeas_hi", "isigi_hi", "cc_half_hi",
        "resolution_high", "resolution_low",
        "cnn_a", "cnn_b", "cnn_c", "cnn_alpha", "cnn_beta", "cnn_gamma",
        "cnn_quality_score", "cnn_disagreement", "cnn_flag_reason",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)


def clean_xds_output(root_path: Path) -> None:
    """
    Delete all XDS output files from a previous run, keeping only the
    raw .img files and the backup folder.

    This ensures the pipeline always starts fresh rather than picking up
    stale output from a previous run that used different parameters.

    Files removed: all XDS output (.LP, .HKL, .cbf, .XDS, SPOT.XDS etc.)
    Files kept: *.img files, *_backup/ folder
    """
    # XDS output file patterns to remove
    xds_patterns = [
        "*.LP", "*.cbf", "*.XDS", "*.HKL",
        "SPOT.XDS", "XDS.INP", "*.xml",
        "dials", "arcimboldo",
    ]
    removed = 0
    for pattern in xds_patterns:
        for f in root_path.glob(pattern):
            try:
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()
                removed += 1
            except OSError:
                pass
    # Also remove the log subfolder (will be recreated)
    log_dir = root_path / "log"
    if log_dir.exists():
        try:
            shutil.rmtree(log_dir)
            removed += 1
        except OSError:
            pass
    if removed:
        log.info("  Cleaned %d old XDS file(s) from previous run.", removed)

def process_one_dataset(args_tuple):
    """
    Process a single dataset through all XDS steps.
    Called by the parallel executor and also directly in serial mode.
    Returns a CSV row dict, or None if the dataset should be skipped.
    """
    root_path, cnn_model, known_cell, known_sg = args_tuple
    subdir_name = root_path.name
    t_start = time.time()

    log.info("\n=== Processing: %s ===", root_path)

    # Step 1 -- Clean old XDS output then rename images
    clean_xds_output(root_path)
    try:
        img_files = rename_and_backup(root_path, subdir_name)
    except FileExistsError as exc:
        log.error("  Rename error: %s -- skipping.", exc)
        return None
    n_images = len(img_files)
    if n_images == 0:
        log.warning("  No .img files after rename -- skipping.")
        return None

    # Step 2 -- Header
    first_img = root_path / img_files[0]
    try:
        header = fabio.open(str(first_img)).header
    except Exception as exc:
        log.error("  Cannot read header: %s -- skipping.", exc)
        return None

    # Detector distance: try multiple header key names (instrument-dependent)
    # UCSC: DETECTOR_DISTANCE   Johnstone lab: DISTANCE
    distance = (
        safe_float(header.get("DETECTOR_DISTANCE"), None) or
        safe_float(header.get("DISTANCE"), None) or
        787.953834
    )

    wavelength  = safe_float(header.get("WAVELENGTH", 0.025082), 0.025082)
    # Beam centre: hardcoded to instrument values, header is ignored.
    # ORGX=1043, ORGY=1046 are the calibrated values for this microscope.
    # These are intentionally asymmetric and must not be changed to 1024/1024.
    orgx, orgy = 1043.0, 1046.0

    # Starting angle: try both key names
    starting_angle = (
        safe_float(header.get("OSC_START"), None) or
        safe_float(header.get("STARTING_ANGLE"), None) or
        0.0
    )
    log.info("  dist=%.2fmm  wl=%.6fA  ORGX=%.1f  ORGY=%.1f  angle=%.3f",
              distance, wavelength, orgx, orgy, starting_angle)

    # Step 3 -- Clean up any existing XDS output files so we always start fresh.
    # This ensures old results from a previous run don't interfere.
    XDS_OUTPUT_FILES = [
        "XDS.INP", "XPARM.XDS", "GXPARM.XDS", "IDXREF.LP", "COLSPOT.LP",
        "XYCORR.LP", "INIT.LP", "DEFPIX.LP", "INTEGRATE.LP", "INTEGRATE.HKL",
        "CORRECT.LP", "XDS_ASCII.HKL", "SPOT.XDS",
        "BKGINIT.cbf", "BKGPIX.cbf", "BLANK.cbf", "DECAY.cbf",
        "MODPIX.cbf", "ABS.cbf", "ABSORP.cbf", "GAIN.cbf",
        "X-CORRECTIONS.cbf", "Y-CORRECTIONS.cbf",
        "DX-CORRECTIONS.cbf", "DY-CORRECTIONS.cbf",
        "GX-CORRECTIONS.cbf", "GY-CORRECTIONS.cbf",
        "SHOW_BKG.cbf", "SHOW_HKL.cbf",
    ]
    cleaned = []
    for fname in XDS_OUTPUT_FILES:
        fpath = root_path / fname
        if fpath.exists():
            fpath.unlink()
            cleaned.append(fname)
    # Also remove the log directory from previous runs
    log_dir_old = root_path / "log"
    if log_dir_old.exists():
        shutil.rmtree(log_dir_old)
    if cleaned:
        log.info("  Cleaned %d existing XDS file(s) for fresh run.", len(cleaned))

    # Now generate a fresh XDS.INP
    log_dir = root_path / "log"
    log_dir.mkdir(exist_ok=True)
    generate_xds_inp(root_path, subdir_name, n_images,
                     orgx, orgy, distance, wavelength,
                     starting_angle=starting_angle)

    # If a known unit cell was provided, seed XDS.INP with it.
    # This is the single most effective way to improve indexing on
    # difficult datasets -- XDS searches around the provided cell
    # rather than trying to determine it from scratch.
    if known_cell:
        xds_inp = root_path / "XDS.INP"
        if xds_inp.exists():
            lines = xds_inp.read_text(errors="replace").splitlines()
            out = []
            for line in lines:
                s = line.strip()
                if s.startswith("UNIT_CELL_CONSTANTS="):
                    out.append(f"UNIT_CELL_CONSTANTS= {known_cell}")
                elif s.startswith("SPACE_GROUP_NUMBER="):
                    out.append(f"SPACE_GROUP_NUMBER= {known_sg or 1}")
                else:
                    out.append(line)
            xds_inp.write_text("\n".join(out) + "\n")
            log.info("  Seeded XDS.INP with known cell: %s  SG=%s",
                     known_cell, known_sg or 1)

    # Step 4 -- Phase 1
    if not check_images_accessible(root_path, img_files):
        log.error("  Image check failed -- skipping.")
        return None
    log_idxref = log_dir / f"{subdir_name}_XDS_idxref.log"
    # Step 4a: Adaptive spot size test
    # Try MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT = 3, 4, 6 and keep the best.
    # This runs XYCORR INIT COLSPOT IDXREF three times but only takes ~30s
    # extra and can significantly improve the number of indexed spots.
    log.info("  Finding optimal MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT...")
    best_px = adaptive_spot_size(root_path, subdir_name, n_images, log_dir)

    log.info("  Running Phase 1 (XYCORR INIT COLSPOT IDXREF) with PIXELS=%d...",
             best_px)
    t0 = time.time()
    rc1 = run_xds(root_path, "XYCORR INIT COLSPOT IDXREF", log_idxref)
    log.info("  Phase 1 done in %.1f s  (exit %d)", time.time() - t0, rc1)

    if not (root_path / "IDXREF.LP").exists():
        log.error("  IDXREF.LP not produced.")
        log.error("%s", tail_log(log_idxref))
        if log_idxref.exists():
            txt = log_idxref.read_text(errors="replace").lower()
            if "expired" in txt:
                log.error("  -> XDS LICENSE EXPIRED. Get new XDS from xds.mr.mpg.de")
            elif "illegal" in txt or "obsolete" in txt:
                log.error("  -> XDS rejected an obsolete keyword in XDS.INP.")
            elif "cannot open" in txt or "no such file" in txt:
                log.error("  -> XDS cannot find image files.")

    idx   = parse_idxref(root_path)
    frac  = idx["indexed_fraction"] or 0.0
    n_idx = idx["indexed_spots"]    or 0
    log.info("  IDXREF: %s/%s indexed (%.1f%%)  cell=%s  hint=%s",
              n_idx, idx["total_spots"], frac * 100,
              idx["unit_cell"], idx["failure_hint"])

    # Auto-retry with relaxed parameters if indexing very low
    if n_idx == 0 or frac < 0.15:
        idx   = retry_idxref_relaxed(root_path, subdir_name, log_dir)
        frac  = idx["indexed_fraction"] or 0.0
        n_idx = idx["indexed_spots"]    or 0

    # Step 5 -- Phase 2
    log_integrate = log_dir / f"{subdir_name}_XDS_integrate.log"
    if n_idx > 0:
        log.info("  Running Phase 2 (DEFPIX INTEGRATE CORRECT)...")
        t2 = time.time()
        rc2 = run_xds(root_path, "DEFPIX INTEGRATE CORRECT", log_integrate)
        log.info("  Phase 2 done in %.1f s  (exit %d)", time.time() - t2, rc2)
        if rc2 != 0:
            log.warning("%s", tail_log(log_integrate))
    else:
        log.info("  Skipping Phase 2 -- no spots indexed.")

    # Step 5b -- Bad frame exclusion
    if (root_path / "INTEGRATE.LP").exists() and n_idx > 0:
        frames = parse_integrate_lp_frames(root_path)
        bad    = find_bad_frames(frames)
        if bad:
            log.info("  %d bad frame(s) detected -- excluding and rerunning CORRECT.", len(bad))
            add_exclude_frames(root_path, bad)
            run_xds(root_path, "CORRECT",
                    log_dir / f"{subdir_name}_XDS_correct_frameexclude.log")

    # Step 5c -- Ice ring detection
    if (root_path / "CORRECT.LP").exists():
        ice = detect_ice_rings(root_path)
        if ice:
            add_ice_ring_exclusions(root_path, ice)
            log.info("  Ice rings detected -- rerunning CORRECT with exclusions.")
            run_xds(root_path, "CORRECT",
                    log_dir / f"{subdir_name}_XDS_correct_ice.log")

    # Step 6 -- Statistics
    stats = parse_correct_lp(root_path)
    if stats["a"] and stats["a"] != "n/a":
        a, b, c = stats["a"], stats["b"], stats["c"]
        alpha, beta, gamma = stats["alpha"], stats["beta"], stats["gamma"]
    else:
        a, b, c, alpha, beta, gamma = extract_cell_params(root_path)

    sg = stats.get("space_group_number")
    log.info("  SG=%s  Cell: %s %s %s %s %s %s", sg, a, b, c, alpha, beta, gamma)
    log.info("  Overall: comp=%.1f%%  Rmeas=%.3f  I/sig=%.1f  CC½=%.3f",
              stats["completeness_overall"] or 0, stats["rmeas_overall"] or 0,
              stats["isigi_overall"] or 0, stats["cc_half_overall"] or 0)
    log.info("  Hi-shell: comp=%.1f%%  Rmeas=%.3f  I/sig=%.1f  CC½=%.3f  d=%.2fA",
              stats["completeness_hi"] or 0, stats["rmeas_hi"] or 0,
              stats["isigi_hi"] or 0, stats["cc_half_hi"] or 0,
              stats["resolution_high"] or 0)

    # Step 7 -- Auto resolution cutoff
    hkl_path = root_path / "XDS_ASCII.HKL"
    if hkl_path.exists() and n_idx > 0:
        cutoff       = find_resolution_cutoff(root_path)
        nominal_high = stats.get("resolution_high")
        if cutoff and nominal_high and cutoff > nominal_high + 0.05:
            log.info("  I/sig drops at %.2fA -- rerunning CORRECT with cutoff.", cutoff)
            set_resolution_limit(root_path, cutoff)
            if run_xds(root_path, "CORRECT",
                       log_dir / f"{subdir_name}_XDS_correct_recut.log") == 0:
                stats = parse_correct_lp(root_path)

    # Step 8 -- CNN
    cnn_pred   = predict_unit_cell(cnn_model, root_path, img_files)
    xds_params = {"a": a, "b": b, "c": c,
                  "alpha": alpha, "beta": beta, "gamma": gamma}
    cnn_pred   = compare_xds_and_cnn(xds_params, cnn_pred)
    if cnn_pred.is_valid():
        log.info("  CNN quality=%.4f  disagree=%s", cnn_pred.quality_score or 0, cnn_pred.disagreement)
    else:
        log.info("  CNN: %s", cnn_pred.flag_reason)

    log.info("  Dataset done in %.1f s  HKL=%s",
              time.time() - t_start, "YES" if hkl_path.exists() else "NO")

    return {
        "subdirectory":         subdir_name,
        "space_group":          sg,
        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma,
        "has_hkl":              "YES" if hkl_path.exists() else "NO",
        "idxref_indexed":       idx.get("indexed_spots"),
        "idxref_total":         idx.get("total_spots"),
        "idxref_fraction":      f"{frac:.3f}" if idx["indexed_fraction"] is not None else "n/a",
        "idxref_hint":          idx.get("failure_hint"),
        "completeness_overall": stats.get("completeness_overall"),
        "rmeas_overall":        stats.get("rmeas_overall"),
        "isigi_overall":        stats.get("isigi_overall"),
        "cc_half_overall":      stats.get("cc_half_overall"),
        "completeness_hi":      stats.get("completeness_hi"),
        "rmeas_hi":             stats.get("rmeas_hi"),
        "isigi_hi":             stats.get("isigi_hi"),
        "cc_half_hi":           stats.get("cc_half_hi"),
        "resolution_high":      stats.get("resolution_high"),
        "resolution_low":       stats.get("resolution_low"),
        "cnn_a":                cnn_pred.params.get("a",     "n/a"),
        "cnn_b":                cnn_pred.params.get("b",     "n/a"),
        "cnn_c":                cnn_pred.params.get("c",     "n/a"),
        "cnn_alpha":            cnn_pred.params.get("alpha", "n/a"),
        "cnn_beta":             cnn_pred.params.get("beta",  "n/a"),
        "cnn_gamma":            cnn_pred.params.get("gamma", "n/a"),
        "cnn_quality_score":    cnn_pred.quality_score if cnn_pred.quality_score is not None else "n/a",
        "cnn_disagreement":     "YES" if cnn_pred.disagreement else "NO",
        "cnn_flag_reason":      cnn_pred.flag_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Automated MicroED XDS processing pipeline"
    )
    parser.add_argument("--watch", action="store_true",
        help="Keep running and process new datasets as they appear (stop with Ctrl+C).")
    parser.add_argument("--workers", type=int, default=1,
        help="Number of datasets to process in parallel (default: 1). Use 4-8 on a server.")
    parser.add_argument("--folder", type=str, default=None,
        help="Parent folder path (skips the interactive prompt).")
    parser.add_argument("--unit-cell", type=str, default=None,
        dest="unit_cell",
        help=('Known unit cell as "a b c alpha beta gamma" (Angstroms/degrees). '
              'Example for acetaminophen: "9.44 9.44 15.23 89.7 89.8 60.2". '
              'When provided, XDS is seeded with this cell and the consensus '
              'step uses it as the reference, dramatically improving indexing '
              'on difficult datasets.'))
    parser.add_argument("--space-group", type=int, default=1,
        dest="space_group",
        help="Known space group number (default: 1 = P1). Example: --space-group 2 for P-1.")
    args = parser.parse_args()

    if args.folder:
        parent_dir = Path(args.folder).expanduser().resolve()
        if not parent_dir.is_dir():
            log.error("Not a valid directory: %s", args.folder)
            return
    else:
        parent_dir = prompt_parent_dir()

    log.info("Parent directory : %s", parent_dir)
    log.info("Parallel workers : %d", args.workers)
    log.info("Watch mode       : %s", args.watch)
    if args.unit_cell:
        log.info("Known unit cell  : %s  (SG %d)", args.unit_cell, args.space_group)
    else:
        log.info("Known unit cell  : not provided (pipeline will determine automatically)")

    # Save all terminal output to a timestamped log file in the parent folder.
    # A new file is created every run so previous logs are never overwritten.
    # e.g. test16_20260510_143022.log
    import datetime
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = parent_dir / f"{parent_dir.name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    log.info("Log file: %s", log_filename)

    xds_on_path = check_xds_available()
    if not xds_on_path:
        log.warning("xds_par not found -- XDS.INP files will be written but not run.")

    cnn_model = load_cnn_model("microed_cnn_weights.pt")
    log.info("CNN: %s", "loaded" if cnn_model else "not loaded")

    csv_rows = []

    # -----------------------------------------------------------------------
    # Discover all dataset subdirectories
    # -----------------------------------------------------------------------
    def collect_datasets(parent: Path) -> list:
        """Walk parent_dir and return list of Paths that contain .img files."""
        datasets = []
        for root, dirs, files in os.walk(parent):
            dirs[:] = sorted(
                d for d in dirs
                if not d.endswith("_backup")
                and d not in ("log", "xscale", "trials")
            )
            img_count = sum(1 for f in files if f.endswith(".img"))
            if img_count > 0:
                datasets.append(Path(root))
                log.info("  Found dataset: %s  (%d .img files)", Path(root).name, img_count)
        if not datasets:
            log.warning("No directories with .img files found under %s", parent)
            log.warning("Checking what IS in that folder:")
            try:
                top_contents = sorted(os.listdir(parent))[:20]
                for item in top_contents:
                    full = parent / item
                    if full.is_dir():
                        sub_files = os.listdir(full)
                        img_n = sum(1 for f in sub_files if f.endswith(".img"))
                        log.warning("  DIR  %s  (%d .img files inside)", item, img_n)
                    else:
                        log.warning("  FILE %s", item)
            except Exception as exc:
                log.error("  Cannot list folder: %s", exc)
        return datasets

    def run_batch(dataset_paths: list, workers: int, csv_rows: list) -> list:
        """
        Process a list of dataset paths, serially or in parallel.
        Appends results to csv_rows. Returns updated csv_rows.
        """
        known_cell = args.unit_cell
        known_sg   = args.space_group
        work = [(p, cnn_model, known_cell, known_sg) for p in dataset_paths]

        if workers <= 1:
            for item in work:
                row = process_one_dataset(item)
                if row is not None:
                    csv_rows.append(row)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_one_dataset, item): item for item in work}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row = fut.result()
                        if row is not None:
                            csv_rows.append(row)
                    except Exception as exc:
                        path = futures[fut][0]
                        log.error("  Worker error on %s: %s", path, exc)
        return csv_rows

    if args.watch:
        # ------------------------------------------------------------------
        # Watch mode: poll for new datasets every 60 seconds
        # ------------------------------------------------------------------
        log.info("Watch mode active. Ctrl+C to stop.")
        processed = get_processed_subdirs(parent_dir)
        log.info("Already processed: %d dataset(s).", len(processed))

        try:
            while True:
                all_datasets = collect_datasets(parent_dir)
                new_datasets = [p for p in all_datasets
                                if p.name not in processed]

                if new_datasets:
                    log.info("Found %d new dataset(s) -- processing...",
                             len(new_datasets))
                    csv_rows = run_batch(new_datasets, args.workers, csv_rows)
                    for p in new_datasets:
                        processed.add(p.name)

                    # Write CSV after every batch so results are not lost
                    csv_path = parent_dir / "cell_parameters_summary.csv"
                    _write_csv(csv_path, csv_rows)
                    log.info("CSV updated (%d datasets total).", len(csv_rows))
                else:
                    log.info("No new datasets. Waiting 60 seconds... (Ctrl+C to stop)")
                    time.sleep(60)

        except KeyboardInterrupt:
            log.info("Watch mode stopped by user.")

    else:
        # ------------------------------------------------------------------
        # Normal mode: discover all datasets and process them once
        # ------------------------------------------------------------------
        all_datasets = collect_datasets(parent_dir)
        log.info("Found %d dataset(s) with .img files to process.", len(all_datasets))
        if not all_datasets:
            log.error("No datasets found. Make sure your folder contains "
                      "subdirectories with .img files inside them.")
        else:
            csv_rows = run_batch(all_datasets, args.workers, csv_rows)

    # -----------------------------------------------------------------------
    # Second pass: re-index datasets using the consensus cell
    # to correct any that XDS indexed to a wrong alternative lattice
    # -----------------------------------------------------------------------
    # Use the known unit cell as the reference if provided,
    # otherwise compute it from the processed datasets.
    if args.unit_cell:
        parts = args.unit_cell.split()
        if len(parts) == 6:
            consensus_cell = {
                "a":     float(parts[0]), "b":     float(parts[1]),
                "c":     float(parts[2]), "alpha": float(parts[3]),
                "beta":  float(parts[4]), "gamma": float(parts[5]),
            }
            log.info("\nUsing provided reference cell: a=%(a)s b=%(b)s c=%(c)s  "
                     "alpha=%(alpha)s beta=%(beta)s gamma=%(gamma)s", consensus_cell)
        else:
            consensus_cell = compute_consensus_cell(csv_rows)
    else:
        consensus_cell = compute_consensus_cell(csv_rows)

    if consensus_cell and len(csv_rows) >= 3:
        log.info("\nConsensus cell: a=%(a)s b=%(b)s c=%(c)s  alpha=%(alpha)s beta=%(beta)s gamma=%(gamma)s",
                 consensus_cell)

        tol_len = 1.5   # Angstroms
        tol_ang = 3.0   # degrees
        reindex_candidates = []
        for row in csv_rows:
            if str(row.get("has_hkl","NO")).upper() != "YES":
                continue
            try:
                dl = max(abs(float(row[p]) - consensus_cell[p]) for p in ("a","b","c"))
                da = max(abs(float(row[p]) - consensus_cell[p]) for p in ("alpha","beta","gamma"))
                if dl > tol_len or da > tol_ang:
                    reindex_candidates.append((row["subdirectory"], dl, da))
            except (ValueError, TypeError, KeyError):
                pass

        if reindex_candidates:
            log.info("  %d dataset(s) deviate from consensus -- attempting re-indexing:",
                     len(reindex_candidates))
            for name, dl, da in reindex_candidates:
                log.info("    %s  Dlength=%.2fA  Dangle=%.1f deg", name, dl, da)

            updated = []
            for row in csv_rows:
                if row["subdirectory"] not in [n for n,_,_ in reindex_candidates]:
                    updated.append(row)
                    continue
                root_path = parent_dir / row["subdirectory"]
                log_dir   = root_path / "log"
                log_dir.mkdir(exist_ok=True)
                old_frac  = float(row.get("idxref_fraction") or 0)
                idx = reindex_with_reference_cell(root_path, row["subdirectory"],
                                                   consensus_cell, log_dir)
                new_frac = idx["indexed_fraction"] or 0.0
                if new_frac > old_frac and (idx["indexed_spots"] or 0) > 0:
                    log.info("  %s improved: %.1f%% -> %.1f%% indexed",
                             row["subdirectory"], old_frac*100, new_frac*100)
                    log_int = log_dir / f"{row['subdirectory']}_XDS_reindex_integrate.log"
                    if run_xds(root_path, "DEFPIX INTEGRATE CORRECT", log_int) == 0:
                        stats = parse_correct_lp(root_path)
                        for k in ("a","b","c","alpha","beta","gamma",
                                  "completeness_overall","rmeas_overall",
                                  "isigi_overall","cc_half_overall"):
                            if stats.get(k):
                                row[k] = stats[k]
                        row["idxref_fraction"] = f"{new_frac:.3f}"
                        row["has_hkl"] = "YES" if (root_path/"XDS_ASCII.HKL").exists() else "NO"
                        log.info("  Updated stats for %s: comp=%.1f%%",
                                 row["subdirectory"],
                                 stats.get("completeness_overall") or 0)
                updated.append(row)
            csv_rows = updated
        else:
            log.info("  All datasets consistent with consensus cell. No re-indexing needed.")

    # Write final CSV and summary
    csv_path = parent_dir / "cell_parameters_summary.csv"
    _write_csv(csv_path, csv_rows)
    log.info("\nCSV summary saved at %s", csv_path)
    log.info("Pipeline complete. Processed %d dataset(s).", len(csv_rows))

    print_summary_table(csv_rows)

    # =======================================================================
    # XSCALE: automated intelligent merging
    # =======================================================================
    # Step A: build candidate list from all datasets that have HKL files
    # Step B: pre-filter to the dominant crystal form (statistical outliers out)
    # Step C: rank clean datasets by actual quality metrics
    # Step D: merge all clean datasets (baseline)
    # Step E: greedy subset search to find the optimal combination
    # Step F: print a final comparison table
    # =======================================================================

    # --- Step A: collect candidates ---
    candidates = []
    for row in csv_rows:
        if str(row.get("has_hkl", "NO")).strip().upper() != "YES":
            continue
        hkl_path = parent_dir / row["subdirectory"] / "XDS_ASCII.HKL"
        if not hkl_path.exists():
            continue
        quality = quality_score_from_row(row)
        candidates.append({
            "name":               row["subdirectory"],
            "hkl_path":           hkl_path,
            "ind_score":          quality,
            "space_group_number": row.get("space_group"),
            "a":     row.get("a"),
            "b":     row.get("b"),
            "c":     row.get("c"),
            "alpha": row.get("alpha"),
            "beta":  row.get("beta"),
            "gamma": row.get("gamma"),
            "completeness": safe_float(row.get("completeness_overall"), 0),
            "rmeas":         safe_float(row.get("rmeas_overall"),        None),
            "isigi":         safe_float(row.get("isigi_overall"),        0),
            "cc_half":       safe_float(row.get("cc_half_overall"),      0),
            "idxref_fraction": safe_float(row.get("idxref_fraction"),    0),
        })

    if not candidates:
        log.info("\nNo datasets with XDS_ASCII.HKL found -- skipping XSCALE.")
        return

    W = 72  # width for separator lines
    sep  = "=" * W
    sep2 = "-" * W

    # ===================================================================
    # DATASET QUALITY REPORT
    # Show every dataset, its quality, and what happens to it
    # ===================================================================
    log.info("\n%s", sep)
    log.info("  DATASET QUALITY REPORT  (%d datasets with HKL files)", len(candidates))
    log.info("%s", sep)
    log.info("  %-20s  %5s  %6s  %6s  %6s  %6s  %5s",
             "Dataset", "Idx%", "Comp%", "I/sig", "a", "c", "Res(A)")
    log.info("%s", sep2)
    for c in sorted(candidates, key=lambda x: x["ind_score"], reverse=True):
        # Get resolution from CORRECT.LP for this dataset
        res_val = "n/a"
        try:
            lp = parent_dir / c["name"] / "CORRECT.LP"
            if lp.exists():
                s = parse_correct_lp(lp.parent)
                r = s.get("resolution_high")
                if r and float(r) > 0:
                    res_val = f"{float(r):.2f}"
        except Exception:
            pass
        log.info("  %-20s  %5.1f  %6.1f  %6.1f  %6s  %6s  %5s",
                 c["name"],
                 (c["idxref_fraction"] or 0) * 100,
                 c["completeness"] or 0,
                 c["isigi"] or 0,
                 c.get("a") or "n/a",
                 c.get("c") or "n/a",
                 res_val)
    log.info("%s", sep)

    # ===================================================================
    # FILTER 1: Find the dominant crystal form
    # Datasets with very different unit cells are wrong indexing solutions
    # and will never merge correctly with the good ones
    # ===================================================================
    log.info("\n  FILTER 1: Removing datasets with incompatible unit cells")
    log.info("%s", sep2)

    # When a known cell is provided, use it as the consensus directly
    # and tighten the tolerance (sigma_cutoff=2.0 instead of 3.0)
    # so only datasets genuinely consistent with the target compound pass.
    if args.unit_cell and consensus_cell:
        # Inject the known cell as the consensus and filter against it
        clean    = []
        rejected = []
        ABS_KNOWN_LEN = 3.0   # Angstroms -- relaxed for difficult data
        ABS_KNOWN_ANG = 15.0  # degrees  -- relaxed for difficult data
        for c in candidates:
            try:
                dl = max(abs(float(c.get(p, 0)) - consensus_cell[p])
                         for p in ("a", "b", "c"))
                da = max(abs(float(c.get(p, 90)) - consensus_cell[p])
                         for p in ("alpha", "beta", "gamma"))
                if dl > ABS_KNOWN_LEN or da > ABS_KNOWN_ANG:
                    reason = (f"deviates from known cell "
                              f"(Δlength={dl:.2f}A tol={ABS_KNOWN_LEN}A, "
                              f"Δangle={da:.1f}° tol={ABS_KNOWN_ANG}°)")
                    rejected.append((c, reason))
                else:
                    clean.append(c)
            except (ValueError, TypeError):
                rejected.append((c, "no valid unit cell"))
        consensus = consensus_cell
    else:
        clean, rejected, consensus = find_dominant_crystal_form(candidates, sigma_cutoff=3.0)

    # Second tightening pass (only when NOT using a known reference cell,
    # since in that case we already applied tight tolerances above).
    ABS_LEN_TOL = 2.0   # Angstroms -- same compound = cells within 2A
    ABS_ANG_TOL = 5.0   # degrees
    if consensus and not args.unit_cell:
        tight_clean    = []
        tight_rejected = list(rejected)
        for c in clean:
            try:
                dl = max(abs(float(c.get(p,0)) - consensus[p])
                         for p in ("a","b","c"))
                da = max(abs(float(c.get(p,90)) - consensus[p])
                         for p in ("alpha","beta","gamma"))
                if dl > ABS_LEN_TOL or da > ABS_ANG_TOL:
                    reason = (f"absolute deviation too large "
                              f"(Δlength={dl:.2f}A tol={ABS_LEN_TOL}A, "
                              f"Δangle={da:.1f}° tol={ABS_ANG_TOL}°)")
                    tight_rejected.append((c, reason))
                else:
                    tight_clean.append(c)
            except (ValueError, TypeError):
                tight_clean.append(c)
        if len(tight_clean) < len(clean):
            log.info("  Absolute tolerance check removed %d more dataset(s).",
                     len(clean) - len(tight_clean))
        clean    = tight_clean
        rejected = tight_rejected

    if consensus:
        log.info("  Target cell:  a=%(a)s  b=%(b)s  c=%(c)s  "
                 "alpha=%(alpha)s  beta=%(beta)s  gamma=%(gamma)s", consensus)
    log.info("%s", sep2)

    for c, reason in rejected:
        log.info("  EXCLUDED  %-20s  Wrong cell -- %s", c["name"], reason)
    for c in clean:
        log.info("  KEPT      %-20s  Cell matches consensus", c["name"])
    log.info("%s", sep2)
    log.info("  Result: %d kept, %d excluded", len(clean), len(rejected))

    if not clean:
        # Fallback: when all datasets fail the cell filter, use every dataset
        # that produced an HKL file. This ensures XSCALE always gets to run
        # and produces output. The merged statistics will honestly reflect
        # the data quality. The user can review which crystals were compatible.
        log.warning("  All datasets excluded by cell filter.")
        log.warning("  Falling back to all %d datasets with HKL files to attempt XSCALE merge.",
                    len(candidates))
        log.warning("  Statistics will reflect actual data consistency.")
        clean = candidates[:]
        rejected = []

    # ===================================================================
    # FILTER 2: Rank by quality within the compatible set
    # Datasets that indexed well and have higher completeness go first
    # ===================================================================
    log.info("\n  FILTER 2: Ranking compatible datasets by quality")
    log.info("%s", sep2)
    clean_sorted = sorted(clean, key=lambda c: c["ind_score"], reverse=True)
    sg, unit_cell_str, clean_sorted = check_space_group_consistency(clean_sorted)

    for rank, c in enumerate(clean_sorted, 1):
        log.info("  #%d  score=%.3f  idx=%4.1f%%  comp=%4.1f%%  %s",
                 rank,
                 c["ind_score"],
                 (c["idxref_fraction"] or 0) * 100,
                 c["completeness"] or 0,
                 c["name"])
    log.info("%s", sep2)

    # Resolution limit
    hi_res_vals = []
    for c in clean_sorted:
        lp = c["hkl_path"].parent / "CORRECT.LP"
        if lp.exists():
            s = parse_correct_lp(lp.parent)
            v = safe_float(s.get("resolution_high"), None)
            if v and v > 0.5:
                hi_res_vals.append(v)
    if hi_res_vals:
        hi_res_vals.sort()
        # Use the best (lowest = highest resolution) individual limit
        # rather than the median -- this pushes the merged dataset as
        # far as the best crystal reaches.
        # Then try extending by 0.1 A to see if there is more signal.
        best_individual = hi_res_vals[0]
        extended         = round(best_individual - 0.10, 2)
        resolution_high  = max(extended, 0.80)   # never go below 0.80 A
        log.info("  Best individual resolution: %.2f A  "
                 "-> Attempting extended limit: %.2f A",
                 best_individual, resolution_high)
    else:
        resolution_high = 2.0

    log.info("  Space group: %s   Resolution: %.2f A   Unit cell: %s",
             sg or "auto", resolution_high, unit_cell_str or "auto")

    xscale_dir = parent_dir / "xscale"
    xscale_dir.mkdir(exist_ok=True)
    xscale_available = bool(shutil.which("xscale_par"))
    if not xscale_available:
        log.warning("  xscale_par not on PATH -- INP files written but not run.")

    # ===================================================================
    # MERGE A: All compatible datasets
    # Baseline reference -- includes every dataset that passed Filter 1
    # ===================================================================
    log.info("\n%s", sep)
    log.info("  MERGE A: All %d compatible datasets (baseline)", len(clean_sorted))
    log.info("%s", sep2)
    all_dir = xscale_dir / "all_compatible"
    all_dir.mkdir(exist_ok=True)
    all_entries = [{"hkl_path": c["hkl_path"], "name": c["name"]}
                   for c in clean_sorted]
    write_xscale_inp(all_dir / "XSCALE.INP", all_entries,
                     resolution_high=resolution_high,
                     space_group_number=0,
                     unit_cell="")  # let XSCALE determine cell from input HKLs

    all_stats = {}
    if xscale_available:
        rc = run_xscale(all_dir)
        if rc == 0:
            all_stats = parse_xscale_lp(all_dir)
            log.info("  Datasets : %s",
                     ", ".join(c["name"] for c in clean_sorted))
            log.info("  comp=%.1f%%  Rmeas=%.3f  CC1/2=%.3f  I/sig=%.1f  n_unique=%s",
                     all_stats.get("completeness_overall") or 0,
                     all_stats.get("rmeas_overall")        or 0,
                     all_stats.get("cc_half_overall")      or 0,
                     all_stats.get("isigi_overall")        or 0,
                     all_stats.get("n_unique_overall") or "n/a")
            # Diagnose if stats still show zero
            if not all_stats.get("completeness_overall"):
                log.warning("  WARNING: XSCALE ran but stats are 0 -- showing XSCALE.LP tail:")
                lp_tail = tail_log(all_dir / "XSCALE.LP", n=30)
                log.warning("%s", lp_tail)
        else:
            log.warning("  XSCALE exit code %d -- see %s", rc, all_dir / "xscale.log")
            log.warning("%s", tail_log(all_dir / "xscale.log"))

    # ===================================================================
    # MERGE B: Optimal subset (greedy forward selection)
    # Tests adding each dataset one at a time and only keeps it if the
    # merged statistics actually improve. Bad datasets are rejected here
    # even if their unit cell looked compatible.
    # ===================================================================
    opt_dir    = xscale_dir / "optimal"
    opt_dir.mkdir(exist_ok=True)
    opt_stats  = {}
    opt_subset = clean_sorted

    if xscale_available and len(clean_sorted) > 1:
        log.info("\n%s", sep)
        log.info("  MERGE B: Finding optimal subset (greedy search)")
        log.info("  Testing each dataset -- only kept if it improves the merge")
        log.info("%s", sep2)

        trials_dir = xscale_dir / "trials"
        trials_dir.mkdir(exist_ok=True)
        opt_subset, opt_stats = greedy_subset_search(
            clean_sorted, trials_dir,
            resolution_high=resolution_high,
            space_group_number=sg,
            unit_cell=unit_cell_str,
        )

        kept     = [c["name"] for c in opt_subset]
        dropped  = [c["name"] for c in clean_sorted if c not in opt_subset]

        log.info("%s", sep2)
        for c in opt_subset:
            log.info("  ACCEPTED  %-20s  Improved the merge", c["name"])
        for name in dropped:
            log.info("  DROPPED   %-20s  Did not improve the merge", name)
        log.info("%s", sep2)

        # Final clean run with accepted datasets only
        opt_entries = [{"hkl_path": c["hkl_path"], "name": c["name"]}
                       for c in opt_subset]
        write_xscale_inp(opt_dir / "XSCALE.INP", opt_entries,
                         resolution_high=resolution_high,
                         space_group_number=0,
                         unit_cell="")  # let XSCALE determine cell
        rc_opt = run_xscale(opt_dir)
        if rc_opt == 0:
            opt_stats = parse_xscale_lp(opt_dir)
            log.info("  Datasets : %s", ", ".join(kept))
            log.info("  comp=%.1f%%  Rmeas=%.3f  CC1/2=%.3f  I/sig=%.1f  n_unique=%s",
                     opt_stats.get("completeness_overall") or 0,
                     opt_stats.get("rmeas_overall")        or 0,
                     opt_stats.get("cc_half_overall")      or 0,
                     opt_stats.get("isigi_overall")        or 0,
                     opt_stats.get("n_unique_overall") or "n/a")
        else:
            log.warning("  XSCALE exit code %d -- see %s",
                        rc_opt, opt_dir / "xscale.log")
    elif len(clean_sorted) == 1:
        log.info("\n  Only one compatible dataset -- no subset search needed.")
        opt_entries = [{"hkl_path": clean_sorted[0]["hkl_path"],
                        "name": clean_sorted[0]["name"]}]
        write_xscale_inp(opt_dir / "XSCALE.INP", opt_entries,
                         resolution_high=resolution_high,
                         space_group_number=sg,
                         unit_cell=unit_cell_str)
        if xscale_available:
            run_xscale(opt_dir)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    log.info("\n%s", sep)
    log.info("  FINAL SUMMARY")
    log.info("%s", sep)

    all_names  = [c["name"] for c in candidates]
    bad_names  = [c["name"] for c, _ in rejected]
    good_names = [c["name"] for c in clean_sorted]
    opt_names  = [c["name"] for c in opt_subset]

    log.info("  Total datasets processed : %d", len(all_names))
    log.info("  Excluded (wrong cell)    : %d  -- %s",
             len(bad_names), ", ".join(bad_names) if bad_names else "none")
    log.info("  Compatible datasets      : %d  -- %s",
             len(good_names), ", ".join(good_names))
    log.info("  Optimal subset           : %d  -- %s",
             len(opt_names), ", ".join(opt_names))
    log.info("%s", sep)

    if all_stats and opt_stats:
        log.info("  %-22s  %6s  %6s  %6s  %6s  %8s  %6s",
                 "Merge", "Comp%", "Rmeas", "CC1/2", "I/sig", "N_uniq", "Res(A)")
        log.info("%s", sep2)
        def _fmt(label, s, res=None):
            log.info("  %-22s  %6.1f  %6.3f  %6.3f  %6.1f  %8s  %6s",
                     label,
                     s.get("completeness_overall") or 0,
                     s.get("rmeas_overall")        or 0,
                     s.get("cc_half_overall")      or 0,
                     s.get("isigi_overall")        or 0,
                     str(s.get("n_unique_overall") or "n/a"),
                     f"{res:.2f}" if res else "n/a")
        _fmt(f"All compatible ({len(good_names)})", all_stats, resolution_high)
        _fmt(f"Optimal ({len(opt_names)})",          opt_stats, resolution_high)
        log.info("%s", sep2)

    # ===================================================================
    # SHELX FORMAT CONVERSION
    # Convert the optimal XSCALE.HKL to SHELX format using XDSCONV.
    # This produces a shelx.hkl file ready for direct use with SHELXS/SHELXL.
    # ===================================================================
    xscale_hkl = opt_dir / "XSCALE.HKL"
    if xscale_hkl.exists() and shutil.which("xdsconv"):
        log.info("\n%s", sep)
        log.info("  SHELX FORMAT CONVERSION (XDSCONV)")
        log.info("%s", sep2)
        shelx_out = opt_dir / "shelx.hkl"

        # Write XDSCONV.INP
        xdsconv_inp = opt_dir / "XDSCONV.INP"
        xdsconv_inp.write_text(
            f"INPUT_FILE= {xscale_hkl}\n"
            f"OUTPUT_FILE= shelx.hkl SHELX\n"
            f"FRIEDEL'S_LAW= FALSE\n"
            f"INCLUDE_RESOLUTION_RANGE= 50.0 {resolution_high:.2f}\n"
        )

        xdsconv_log = opt_dir / "XDSCONV.LP"
        with open(xdsconv_log, "w") as lf:
            subprocess.run(
                ["xdsconv"],
                cwd=str(opt_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        if shelx_out.exists() and shelx_out.stat().st_size > 100:
            n_ref = sum(1 for _ in open(shelx_out))
            log.info("  SHELX HKL written: %s", shelx_out)
            log.info("  Reflections      : %d", n_ref)
            log.info("  Resolution limit : %.2f Angstroms", resolution_high)
            log.info("  Ready for SHELXS/SHELXL structure solution")
        else:
            log.warning("  XDSCONV did not produce shelx.hkl -- see %s", xdsconv_log)
    elif not shutil.which("xdsconv"):
        log.info("  (xdsconv not on PATH -- skipping SHELX conversion)")

    log.info("%s", sep)
    log.info("  DATA SUMMARY")
    log.info("%s", sep2)
    log.info("  Resolution limit : %.2f Angstroms", resolution_high)
    log.info("  Space group      : %s", sg or "1 (P1)")
    if unit_cell_str:
        parts = unit_cell_str.split()
        if len(parts) == 6:
            log.info("  Unit cell        : a=%-7s b=%-7s c=%-7s",
                     parts[0], parts[1], parts[2])
            log.info("                     alpha=%-5s beta=%-5s gamma=%-5s",
                     parts[3], parts[4], parts[5])
    log.info("%s", sep2)
    log.info("  File to use for structure determination:")
    log.info("  --> %s", opt_dir / "XSCALE.HKL")
    log.info("%s", sep)


if __name__ == "__main__":
    main()
