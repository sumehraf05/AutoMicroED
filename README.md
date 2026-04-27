# MicroED Automation Pipeline

An automated pipeline for processing Micro-Electron Diffraction (microED)
crystallographic datasets. The pipeline handles everything from raw image
files through to an optimally merged final dataset, without requiring
manual intervention at any step.

---

## What This Does

MicroED is a technique that uses electrons to determine the atomic structure
of molecules too small to study with traditional X-ray crystallography. Each
crystal can only be measured briefly before being destroyed, so data from
many crystals must be combined. This pipeline automates the entire process:

1. Renames and backs up raw diffraction image files
2. Generates XDS configuration files tailored to each dataset
3. Runs XDS to find the crystal lattice and measure reflection intensities
4. Retries indexing automatically with relaxed parameters if the first attempt fails
5. Detects and excludes bad frames and ice rings automatically
6. Extracts quality statistics (completeness, Rmeas, I/sigma, CC1/2)
7. Applies automated resolution cutoffs based on signal quality
8. Uses a neural network to independently check each dataset's quality
9. Merges all datasets with XSCALE
10. Finds the optimal combination of datasets to maximise completeness
    while minimising degradation of data quality

---

## Files

| File | Description |
|------|-------------|
| `xds_pipeline.py` | Main pipeline — runs everything end to end |
| `microed_cnn.py` | Neural network module — model, preprocessing, inference |
| `train_cnn.py` | Training script — teaches the CNN using XDS results |
| `microed_cnn_weights.pt` | Trained model weights (created by `train_cnn.py`) |
| `cell_parameters_summary.csv` | Output spreadsheet generated after each run |
| `LICENSE` | MIT License |

---

## Requirements

### Python packages

```bash
pip install fabio torch torchvision numpy
```

### XDS

XDS must be installed and on your PATH. Download from https://xds.mr.mpg.de

```bash
cd ~
wget https://xds.mr.mpg.de/XDS-gfortran_Linux_x86_64.tar.gz
tar -xzf XDS-gfortran_Linux_x86_64.tar.gz

# Add to PATH (add to ~/.bashrc to make permanent)
export PATH=~/XDS-gfortran_Linux_x86_64:$PATH

# Verify installation
which xds_par
xds_par | head -4
```

---

## Usage

### Basic run

```bash
python xds_pipeline.py
```

When prompted, drag your parent folder (the one containing all your crystal
subdirectories) into the terminal and press Enter.

Your folder structure should look like this:

```
parent_folder/
    crystal-1/
        crystal-1_0001.img
        crystal-1_0002.img
        ...
    crystal-2/
        crystal-2_0001.img
        ...
```

### Command line options

```bash
# Pass the folder path directly (skips the interactive prompt)
python xds_pipeline.py --folder /path/to/your/data

# Process multiple crystals in parallel (recommended: 4-8 on a server)
python xds_pipeline.py --folder /path/to/your/data --workers 4

# Watch mode: keep running and process new crystals as they appear
# Stop with Ctrl+C when done
python xds_pipeline.py --folder /path/to/your/data --watch

# Run in the background so the job continues after you close the terminal
nohup python xds_pipeline.py --folder /path/to/your/data --workers 4 > pipeline.log 2>&1 &

# Monitor a background run
tail -f pipeline.log
```

### Full recommended workflow

```bash
# Step 1: Process all crystals through XDS
python xds_pipeline.py --folder /path/to/your/data

# Step 2: Train the CNN on your results
python train_cnn.py

# Step 3: Re-run with CNN quality scores feeding into dataset selection
python xds_pipeline.py --folder /path/to/your/data
```

On Step 3, the pipeline loads the trained CNN weights automatically and adds
quality scores to every dataset. These scores then feed into the XSCALE
subset selection to help choose the best datasets to merge.

---

## Output

After a successful run, your data folder will contain:

```
parent_folder/
    cell_parameters_summary.csv        # Quality metrics for every crystal
    crystal-1/
        XDS.INP                        # XDS configuration (auto-generated)
        XDS_ASCII.HKL                  # Processed reflections
        CORRECT.LP                     # XDS statistics log
        IDXREF.LP                      # Indexing log
        log/
            crystal-1_XDS_idxref.log
            crystal-1_XDS_integrate.log
    xscale/
        all_datasets/
            XSCALE.INP
            XSCALE.HKL                 # Merged file using all crystals (baseline)
            XSCALE.LP
        optimal/
            XSCALE.INP
            XSCALE.HKL                 # Merged file using best subset (use this)
            XSCALE.LP
        trials/                        # One folder per greedy search step
```

The file to use for structure determination is `xscale/optimal/XSCALE.HKL`.

---

## CSV Column Reference

The `cell_parameters_summary.csv` file contains one row per crystal.

| Column | Source | Description |
|--------|--------|-------------|
| `subdirectory` | pipeline | Crystal folder name |
| `space_group` | CORRECT.LP | Space group number determined by XDS |
| `a` `b` `c` | CORRECT.LP | Unit cell lengths in Angstroms |
| `alpha` `beta` `gamma` | CORRECT.LP | Unit cell angles in degrees |
| `has_hkl` | pipeline | YES if XDS produced a usable HKL file |
| `idxref_indexed` | IDXREF.LP | Number of spots successfully indexed |
| `idxref_total` | IDXREF.LP | Total spots found |
| `idxref_fraction` | IDXREF.LP | Fraction of spots indexed (0 to 1) |
| `idxref_hint` | IDXREF.LP | Plain-English description of indexing outcome |
| `completeness_overall` | CORRECT.LP | Overall data completeness (%) |
| `rmeas_overall` | CORRECT.LP | Overall Rmeas — internal consistency (lower is better) |
| `isigi_overall` | CORRECT.LP | Overall mean I/sigma — signal strength (higher is better) |
| `cc_half_overall` | CORRECT.LP | Overall CC1/2 — statistical reliability (closer to 1 is better) |
| `completeness_hi` | CORRECT.LP | Completeness in the highest-resolution shell |
| `rmeas_hi` | CORRECT.LP | Rmeas in the highest-resolution shell |
| `isigi_hi` | CORRECT.LP | I/sigma in the highest-resolution shell |
| `cc_half_hi` | CORRECT.LP | CC1/2 in the highest-resolution shell |
| `resolution_high` | CORRECT.LP | High-resolution limit achieved (Angstroms) |
| `resolution_low` | CORRECT.LP | Low-resolution limit (Angstroms) |
| `cnn_a` to `cnn_gamma` | CNN | CNN-predicted unit cell parameters |
| `cnn_quality_score` | CNN | CNN quality score (0 = poor, 1 = excellent) |
| `cnn_disagreement` | pipeline | YES if CNN and XDS disagree on the unit cell |
| `cnn_flag_reason` | pipeline | Details of any disagreement |

---

## How Dataset Selection Works

Rather than merging all crystals together (which lets bad datasets degrade
the result), the pipeline uses a greedy forward selection algorithm:

1. Each crystal is scored individually based on indexed fraction, whether
   it produced an HKL file, CNN quality score, and CNN/XDS agreement
2. Crystals are sorted best-first by individual score
3. Starting with the single best crystal, each remaining crystal is tested
   by actually running XSCALE with it included
4. A crystal is accepted only if the merged statistics improve
   (higher completeness, better CC1/2 and Rmeas). Otherwise it is rejected.
5. This continues until all crystals have been evaluated

Two output merges are always produced for comparison:

- `xscale/all_datasets/XSCALE.HKL` — every crystal merged together (baseline)
- `xscale/optimal/XSCALE.HKL` — best subset only (recommended)

---

## How the Neural Network Works

`microed_cnn.py` contains a convolutional neural network based on ResNet-18,
adapted for single-channel grayscale diffraction images. It has two outputs:

**Unit cell prediction** — independently predicts the six unit cell parameters
from the raw diffraction images. This is compared against what XDS measured.
If they disagree by more than 5 Angstroms or 5 degrees on any parameter,
the dataset is flagged for review.

**Quality scoring** — predicts a score from 0 to 1 based on image features
such as spot sharpness and signal-to-background ratio. This score feeds into
the dataset ranking for XSCALE merging.

The CNN needs to be trained on your own data before its predictions become
meaningful. Run `train_cnn.py` after the pipeline has processed at least a
few crystals. It uses the cell parameters and indexed fractions from the CSV
as training labels. The pipeline runs fully without trained weights — CNN
columns will just show `n/a`.

---

## Instrument Configuration

The XDS.INP template is configured for the UCSF cryo-EM instrument with an
ADSC CCD detector. Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DETECTOR` | ADSC | Detector type |
| `NX` / `NY` | 2048 / 2048 | Detector dimensions in pixels |
| `QX` / `QY` | 0.028 mm | Pixel size |
| `ORGX` / `ORGY` | 1043 / 1046 | Beam centre (pixels) |
| `DETECTOR_DISTANCE` | 1304.07 mm | Sample to detector distance |
| `X-RAY_WAVELENGTH` | 0.025082 A | Electron wavelength at 200 kV |
| `GAIN` | 15 | Detector gain |
| `OVERLOAD` | 65000 | Pixel saturation value |
| `OSCILLATION_RANGE` | 1.0 deg | Rotation per frame |

To use this pipeline with a different instrument, update `_XDS_INP_TEMPLATE`
and the beam centre defaults `_ORGX_DEFAULT` and `_ORGY_DEFAULT` in
`xds_pipeline.py`.

---

## Troubleshooting

**XDS not found on PATH:**
```bash
export PATH=~/XDS-gfortran_Linux_x86_64:$PATH
which xds_par
```

**XDS license expired:**
Download the latest XDS from https://xds.mr.mpg.de — the license is renewed
annually at no cost.

**ILLEGAL KEYWORD error in XDS:**
An outdated parameter is in a generated XDS.INP. Delete existing XDS.INP
files and re-run to regenerate clean ones:
```bash
find /path/to/your/data -name "XDS.INP" -delete
python xds_pipeline.py --folder /path/to/your/data
```

**0 datasets processed:**
Check that your crystal folders actually contain `.img` files:
```bash
find /path/to/your/data -name "*.img" | head -5
```

**Very low completeness or -99.9% Rmeas:**
This means each reflection was only measured once so internal consistency
cannot be calculated. This is normal for individual microED datasets — the
XSCALE merge step is what combines many crystals to build up completeness
and redundancy. If most datasets show this, consider collecting more frames
per crystal (wider oscillation range).

**CNN weights not loading:**
`microed_cnn_weights.pt` has not been created yet. Run `train_cnn.py` first.
The pipeline will still run without it.

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
