#!/usr/bin/env python3

"""
xds_pipeline.py
===============
Automated batch processing pipeline for MicroED crystallographic datasets.
    1. Renumber image files starting at 1 (rerun-safe, backs up originals)
    2. Auto-generate or patch XDS.INP with per-dataset parameters from the
       image header (distance, wavelength, beam centre, data range, etc.)
    3. Loop XDS processing through all required steps automatically:
       Phase 1 -> XYCORR INIT COLSPOT IDXREF
       Phase 2 -> DEFPIX INTEGRATE CORRECT
    4. Recursively detect and process all dataset subdirectories in one run
    5. Extract and report full summary statistics per dataset:
         - Space group number
         - Unit cell parameters (a, b, c, alpha, beta, gamma)
         - Completeness, Rmeas, I/sig, CC½  (overall AND highest-res shell)
    6. Automated resolution cutoff: detect the shell where <I/sig> drops
       below a threshold (default 2.0) and rerun CORRECT with that limit
    7. Write XSCALE.INP (with space group + unit cell) and run XSCALE
    8. Greedy subset search: find the combination of datasets that maximises
       completeness while minimising Rmerge / CC½ degradation

Usage:
  python3 xds_pipeline.py

You will be prompted to drag-and-drop the parent folder containing all dataset
subdirectories. XDS (xds_par) and XSCALE (xscale_par) must be on your PATH.

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

    Extracts:
      - space_group_number
      - unit cell: a, b, c, alpha, beta, gamma
      - For BOTH the overall dataset and the highest-resolution shell:
            completeness, Rmeas, I/sigma, CC½

    These are the exact statistics listed in the Week 7 assignment.

    Returns a flat dict.  All values are None if CORRECT.LP is missing or
    a field could not be parsed.
    """
    result = {
        # Space group and cell
        "space_group_number":   None,
        "a": None, "b": None, "c": None,
        "alpha": None, "beta": None, "gamma": None,

        # Overall statistics
        "completeness_overall": None,
        "rmeas_overall":        None,
        "isigi_overall":        None,
        "cc_half_overall":      None,

        # Highest-resolution shell statistics
        "completeness_hi":      None,
        "rmeas_hi":             None,
        "isigi_hi":             None,
        "cc_half_hi":           None,

        # Resolution limits actually used
        "resolution_high":      None,
        "resolution_low":       None,
    }

    lp = root / "CORRECT.LP"
    if not lp.exists():
        return result

    text = lp.read_text(errors="replace")

    # -- Space group --
    m = re.search(r"SPACE_GROUP_NUMBER\s*=?\s*(\d+)", text)
    if m:
        result["space_group_number"] = int(m.group(1))

    # -- Unit cell (from CORRECT.LP "UNIT CELL PARAMETERS" line) --
    m = re.search(
        r"UNIT CELL PARAMETERS\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text,
    )
    if m:
        result["a"], result["b"], result["c"] = m.group(1), m.group(2), m.group(3)
        result["alpha"], result["beta"], result["gamma"] = m.group(4), m.group(5), m.group(6)

    # -- Statistics table --
    # CORRECT.LP contains a table like:
    #
    # SUBSET OF REFLECTIONS...
    # ...
    #  d(A)   #obs  #uniq  mult  %comp  <I/sI>  Rsym   Rmeas  Ranom  CC(1/2) CCano
    #  30.00   ...
    #   2.00   ...   <- highest-res shell (last data line before 'total')
    #  total   ...
    #
    # We parse each data line (non-header, non-total, starts with a number)
    # and also the 'total' line.

    # Regex for one statistics row:
    # d_limit  n_obs  n_uniq  mult  comp  isigi  rsym  rmeas  ranom  cc_half  [ccano]
    row_pat = re.compile(
        r"^\s*([\d.]+)\s+"     # 1: d_limit (resolution in A)
        r"\d+\s+"              # n_obs
        r"\d+\s+"              # n_uniq
        r"[\d.]+\s+"           # mult
        r"([\d.]+)\s+"         # 2: % completeness
        r"([\d.]+)\s+"         # 3: <I/sigI>
        r"[\d.]+\s+"           # Rsym (skip -- use Rmeas)
        r"([\d.]+)\s+"         # 4: Rmeas
        r"[\d.]+\s+"           # Ranom (skip)
        r"([\d.*]+)",          # 5: CC(1/2)  (may have *)
        re.MULTILINE,
    )
    total_pat = re.compile(
        r"^\s*total\s+"
        r"\d+\s+"
        r"\d+\s+"
        r"[\d.]+\s+"
        r"([\d.]+)\s+"         # 1: % completeness
        r"([\d.]+)\s+"         # 2: <I/sigI>
        r"[\d.]+\s+"
        r"([\d.]+)\s+"         # 3: Rmeas
        r"[\d.]+\s+"
        r"([\d.*]+)",          # 4: CC(1/2)
        re.MULTILINE | re.IGNORECASE,
    )

    # Collect all shell rows so we can pick the highest-resolution one
    shell_rows = []
    for m in row_pat.finditer(text):
        d_lim       = safe_float(m.group(1))
        comp        = safe_float(m.group(2))
        isigi       = safe_float(m.group(3))
        rmeas_pct   = safe_float(m.group(4))
        cc_str      = m.group(5).replace("*", "")
        cc_half     = safe_float(cc_str)
        if d_lim is not None:
            shell_rows.append({
                "d":      d_lim,
                "comp":   comp,
                "isigi":  isigi,
                "rmeas":  rmeas_pct / 100.0 if rmeas_pct is not None else None,
                "cc_half": cc_half / 100.0  if cc_half  is not None else None,
            })

    # Highest-resolution shell = row with smallest d value
    if shell_rows:
        hi_shell = min(shell_rows, key=lambda r: r["d"])
        result["completeness_hi"] = hi_shell["comp"]
        result["rmeas_hi"]        = hi_shell["rmeas"]
        result["isigi_hi"]        = hi_shell["isigi"]
        result["cc_half_hi"]      = hi_shell["cc_half"]
        result["resolution_high"] = hi_shell["d"]
        result["resolution_low"]  = max(r["d"] for r in shell_rows)

    # Overall ('total') row
    m = total_pat.search(text)
    if m:
        result["completeness_overall"] = safe_float(m.group(1))
        result["isigi_overall"]        = safe_float(m.group(2))
        rmeas_pct = safe_float(m.group(3))
        result["rmeas_overall"]        = rmeas_pct / 100.0 if rmeas_pct is not None else None
        cc_str = m.group(4).replace("*", "")
        cc_val = safe_float(cc_str)
        result["cc_half_overall"]      = cc_val / 100.0 if cc_val is not None else None

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

    Returns overall and highest-resolution shell values for:
    completeness, Rmerge, Rmeas, I/sigma, CC½, n_unique.
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

    # Shell rows
    row_pat = re.compile(
        r"^\s*([\d.]+)\s+"      # d_limit
        r"(\d+)\s+"             # n_obs
        r"(\d+)\s+"             # n_uniq
        r"[\d.]+\s+"            # mult
        r"([\d.]+)\s+"          # comp
        r"([\d.]+)\s+"          # isigi
        r"([\d.]+)\s+"          # Rsym/Rmerge
        r"([\d.]+)\s+"          # Rmeas
        r"[\d.]+\s+"            # Ranom
        r"([\d.*]+)",           # CC(1/2)
        re.MULTILINE,
    )

    shells = []
    for m in row_pat.finditer(text):
        d         = safe_float(m.group(1))
        comp      = safe_float(m.group(4))
        isigi     = safe_float(m.group(5))
        rmerge_p  = safe_float(m.group(6))
        rmeas_p   = safe_float(m.group(7))
        cc_str    = m.group(8).replace("*", "")
        cc_half   = safe_float(cc_str)
        if d is not None:
            shells.append({
                "d":       d,
                "comp":    comp,
                "isigi":   isigi,
                "rmerge":  rmerge_p / 100.0 if rmerge_p is not None else None,
                "rmeas":   rmeas_p  / 100.0 if rmeas_p  is not None else None,
                "cc_half": cc_half  / 100.0 if cc_half  is not None else None,
            })

    if shells:
        hi = min(shells, key=lambda r: r["d"])
        result["completeness_hi"] = hi["comp"]
        result["rmerge_hi"]       = hi["rmerge"]
        result["rmeas_hi"]        = hi["rmeas"]
        result["isigi_hi"]        = hi["isigi"]
        result["cc_half_hi"]      = hi["cc_half"]
        result["resolution_high"] = hi["d"]

    # Total row
    total_pat = re.compile(
        r"^\s*total\s+"
        r"(\d+)\s+"             # n_obs
        r"(\d+)\s+"             # n_uniq
        r"[\d.]+\s+"            # mult
        r"([\d.]+)\s+"          # comp
        r"([\d.]+)\s+"          # isigi
        r"([\d.]+)\s+"          # Rmerge
        r"([\d.]+)\s+"          # Rmeas
        r"[\d.]+\s+"
        r"([\d.*]+)",           # CC(1/2)
        re.MULTILINE | re.IGNORECASE,
    )
    m = total_pat.search(text)
    if m:
        result["n_unique_overall"]     = int(m.group(2))
        result["completeness_overall"] = safe_float(m.group(3))
        result["isigi_overall"]        = safe_float(m.group(4))
        rmerge_p = safe_float(m.group(5))
        rmeas_p  = safe_float(m.group(6))
        result["rmerge_overall"] = rmerge_p / 100.0 if rmerge_p is not None else None
        result["rmeas_overall"]  = rmeas_p  / 100.0 if rmeas_p  is not None else None
        cc_str = m.group(7).replace("*", "")
        cc_val = safe_float(cc_str)
        result["cc_half_overall"] = cc_val / 100.0 if cc_val is not None else None

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
        space_group_number=space_group_number,
        unit_cell=unit_cell,
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
                    isigi_threshold: float = 1.0,
                    frac_threshold:  float = 0.3) -> list:
    """
    Identify frames that should be excluded from CORRECT.

    A frame is considered bad if:
      - its mean I/sigma is below isigi_threshold (default 1.0), OR
      - its fraction of observed reflections is below frac_threshold (default 0.3)

    Returns a sorted list of bad frame numbers.
    """
    if not frames:
        return []

    bad = []
    for f in frames:
        isigi = f.get("isigi_mean")
        frac  = f.get("fraction_observed")
        if (isigi is not None and isigi < isigi_threshold) or            (frac  is not None and frac  < frac_threshold):
            bad.append(f["frame"])

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

def process_one_dataset(args_tuple):
    """
    Process a single dataset through all XDS steps.
    Called by the parallel executor and also directly in serial mode.
    Returns a CSV row dict, or None if the dataset should be skipped.
    """
    root_path, cnn_model = args_tuple
    subdir_name = root_path.name
    t_start = time.time()

    log.info("\n=== Processing: %s ===", root_path)

    # Step 1 -- Rename
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

    distance       = safe_float(header.get("DETECTOR_DISTANCE", 1304.067765), 1304.067765)
    wavelength     = safe_float(header.get("WAVELENGTH", 0.025082), 0.025082)
    _ORGX_DEFAULT  = 1043.0
    _ORGY_DEFAULT  = 1046.0
    orgx_h = safe_float(header.get("ORGX", _ORGX_DEFAULT), _ORGX_DEFAULT)
    orgy_h = safe_float(header.get("ORGY", _ORGY_DEFAULT), _ORGY_DEFAULT)
    if abs(orgx_h - _ORGX_DEFAULT) < 100 and abs(orgy_h - _ORGY_DEFAULT) < 100:
        orgx, orgy = orgx_h, orgy_h
    else:
        orgx, orgy = _ORGX_DEFAULT, _ORGY_DEFAULT
        log.info("  Header beam centre suspicious -- using instrument default (%.1f, %.1f).", orgx, orgy)
    starting_angle = safe_float(header.get("STARTING_ANGLE", 0.0), 0.0)
    log.info("  dist=%.2fmm  wl=%.6fA  ORGX=%.1f  ORGY=%.1f  angle=%.3f",
              distance, wavelength, orgx, orgy, starting_angle)

    # Step 3 -- XDS.INP
    log_dir = root_path / "log"
    log_dir.mkdir(exist_ok=True)
    if not (root_path / "XDS.INP").exists():
        generate_xds_inp(root_path, subdir_name, n_images,
                         orgx, orgy, distance, wavelength,
                         starting_angle=starting_angle)
    else:
        patch_xds_inp(root_path, subdir_name, n_images, orgx, orgy)

    # Step 4 -- Phase 1
    if not check_images_accessible(root_path, img_files):
        log.error("  Image check failed -- skipping.")
        return None
    log_idxref = log_dir / f"{subdir_name}_XDS_idxref.log"
    log.info("  Running Phase 1 (XYCORR INIT COLSPOT IDXREF)...")
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
        work = [(p, cnn_model) for p in dataset_paths]

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

    # Write final CSV and summary
    csv_path = parent_dir / "cell_parameters_summary.csv"
    _write_csv(csv_path, csv_rows)
    log.info("\nCSV summary saved at %s", csv_path)
    log.info("Pipeline complete. Processed %d dataset(s).", len(csv_rows))

    # -----------------------------------------------------------------------
    # XSCALE: intelligent merging (Week 7 Tasks 7 & 8)
    # -----------------------------------------------------------------------
    # Build candidate list from csv_rows
    csv_header = [
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
    candidates = []
    for row in csv_rows:
        if str(row.get("has_hkl", "NO")).strip().upper() != "YES":
            continue
        hkl_path = parent_dir / row["subdirectory"] / "XDS_ASCII.HKL"
        if not hkl_path.exists():
            continue
        ind_score = score_individual_dataset(row)
        candidates.append({
            "name":               row["subdirectory"],
            "hkl_path":           hkl_path,
            "ind_score":          ind_score,
            "space_group_number": row.get("space_group"),
            "a": row.get("a"), "b": row.get("b"), "c": row.get("c"),
            "alpha": row.get("alpha"), "beta": row.get("beta"),
            "gamma": row.get("gamma"),
        })

    if not candidates:
        log.info("\nNo datasets with XDS_ASCII.HKL -- skipping XSCALE.")
        return

    log.info("\n%d dataset(s) eligible for merging:", len(candidates))
    for c in sorted(candidates, key=lambda x: x["ind_score"], reverse=True):
        log.info("  pre_score=%.4f  SG=%s  %s",
                 c["ind_score"], c.get("space_group_number"), c["name"])

    # Check space-group consistency before merging
    sg, unit_cell_str, candidates = check_space_group_consistency(candidates)

    # Unit cell clustering: exclude datasets with significantly different cells
    # even if they share the same space group number
    candidates = cluster_by_unit_cell(candidates)
    log.info("  Reference space group: %s  unit cell: %s", sg, unit_cell_str or "auto")

    if not candidates:
        log.warning("No consistent datasets remain after space-group check.")
        return

    # Determine resolution high limit: use median of individual dataset limits
    res_limits = [safe_float(c.get("resolution_high"), None)
                  for c in candidates
                  if safe_float(c.get("resolution_high"), None) is not None]
    if res_limits:
        res_limits.sort()
        resolution_high = res_limits[len(res_limits) // 2]   # median
    else:
        resolution_high = 2.0
    log.info("  Merging to resolution: %.2f A", resolution_high)

    xscale_dir = parent_dir / "xscale"
    xscale_dir.mkdir(exist_ok=True)
    xscale_available = bool(shutil.which("xscale_par"))

    if not xscale_available:
        log.warning("xscale_par not found on PATH.")
        log.warning("XSCALE.INP files will be written but not executed.")

    # -- Merge 1: all datasets (baseline reference) --
    all_dir = xscale_dir / "all_datasets"
    all_dir.mkdir(exist_ok=True)
    all_entries = [{"hkl_path": c["hkl_path"], "name": c["name"]} for c in candidates]
    write_xscale_inp(all_dir / "XSCALE.INP", all_entries,
                     resolution_high=resolution_high,
                     space_group_number=sg,
                     unit_cell=unit_cell_str)
    log.info("\nXSCALE all_datasets: XSCALE.INP written -> %s", all_dir)

    if xscale_available:
        rc_all = run_xscale(all_dir)
        all_stats = parse_xscale_lp(all_dir)
        if rc_all == 0:
            log.info(
                "all_datasets: comp=%.1f%%  Rmeas=%.3f  CC½=%.3f  "
                "I/sig=%.1f  n_unique=%s",
                all_stats.get("completeness_overall") or 0,
                all_stats.get("rmeas_overall")         or 0,
                all_stats.get("cc_half_overall")       or 0,
                all_stats.get("isigi_overall")         or 0,
                all_stats.get("n_unique_overall"),
            )
            log.info(
                "all_datasets hi-shell: comp=%.1f%%  Rmeas=%.3f  "
                "CC½=%.3f  I/sig=%.1f",
                all_stats.get("completeness_hi") or 0,
                all_stats.get("rmeas_hi")        or 0,
                all_stats.get("cc_half_hi")      or 0,
                all_stats.get("isigi_hi")        or 0,
            )
        else:
            log.warning("all_datasets XSCALE exit code %d", rc_all)

    # -- Merge 2: optimal subset via greedy forward selection --
    opt_dir = xscale_dir / "optimal"
    opt_dir.mkdir(exist_ok=True)

    if xscale_available and len(candidates) > 1:
        log.info("\nRunning greedy subset search...")
        trials_dir = xscale_dir / "trials"
        trials_dir.mkdir(exist_ok=True)

        optimal_subset, opt_stats = greedy_subset_search(
            candidates, trials_dir,
            resolution_high=resolution_high,
            space_group_number=sg,
            unit_cell=unit_cell_str,
        )

        log.info("\nOptimal subset (%d / %d datasets):", len(optimal_subset), len(candidates))
        for c in optimal_subset:
            log.info("  pre_score=%.4f  %s", c["ind_score"], c["name"])
        log.info(
            "Optimal merge: comp=%.1f%%  Rmeas=%.3f  CC½=%.3f  "
            "I/sig=%.1f  n_unique=%s",
            opt_stats.get("completeness_overall") or 0,
            opt_stats.get("rmeas_overall")        or 0,
            opt_stats.get("cc_half_overall")      or 0,
            opt_stats.get("isigi_overall")        or 0,
            opt_stats.get("n_unique_overall"),
        )
        log.info(
            "Optimal hi-shell: comp=%.1f%%  Rmeas=%.3f  CC½=%.3f  I/sig=%.1f",
            opt_stats.get("completeness_hi") or 0,
            opt_stats.get("rmeas_hi")        or 0,
            opt_stats.get("cc_half_hi")      or 0,
            opt_stats.get("isigi_hi")        or 0,
        )

        # Final clean run in opt_dir
        opt_entries = [{"hkl_path": c["hkl_path"], "name": c["name"]}
                       for c in optimal_subset]
        write_xscale_inp(opt_dir / "XSCALE.INP", opt_entries,
                         resolution_high=resolution_high,
                         space_group_number=sg,
                         unit_cell=unit_cell_str)
        rc_opt = run_xscale(opt_dir)
        if rc_opt == 0:
            log.info("Optimal XSCALE run complete -> %s", opt_dir / "XSCALE.HKL")
        else:
            log.warning("Optimal XSCALE exit code %d -- see %s",
                        rc_opt, opt_dir / "xscale.log")

    elif len(candidates) == 1:
        log.info("\nOnly one dataset -- writing single-dataset XSCALE.INP.")
        write_xscale_inp(opt_dir / "XSCALE.INP",
                         [{"hkl_path": candidates[0]["hkl_path"],
                           "name": candidates[0]["name"]}],
                         resolution_high=resolution_high,
                         space_group_number=sg,
                         unit_cell=unit_cell_str)
        if xscale_available:
            run_xscale(opt_dir)

    else:
        # XSCALE not available -- write best-guess INP ranked by individual score
        log.info("\nWriting best-guess XSCALE.INP (greedy search needs xscale_par).")
        ranked = sorted(candidates, key=lambda c: c["ind_score"], reverse=True)
        entries = [{"hkl_path": c["hkl_path"], "name": c["name"]} for c in ranked]
        write_xscale_inp(opt_dir / "XSCALE.INP", entries,
                         resolution_high=resolution_high,
                         space_group_number=sg,
                         unit_cell=unit_cell_str)
        log.info("Run manually: cd %s && xscale_par", opt_dir)

    log.info("\nAll XSCALE output is in: %s", xscale_dir)


if __name__ == "__main__":
    main()
