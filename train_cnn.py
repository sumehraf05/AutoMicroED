#!/usr/bin/env python3
"""
train_cnn.py
============
Training script for MicroEdCNN.

How to use:
  1. Run xds_pipeline.py first on your data folder. This generates
     cell_parameters_summary.csv which contains the training labels.
  2. Run this script and point it at the same data folder.
  3. The best model weights are saved to microed_cnn_weights.pt in
     that folder. xds_pipeline.py will find and load them automatically
     the next time you run it.

What gets trained:
  - Cell parameter prediction: the network learns to predict a, b, c,
    alpha, beta, gamma from averaged diffraction images. Labels come
    from XDS (CORRECT.LP) via the CSV.
  - Quality score prediction: the network learns to predict a score
    between 0 and 1 representing how good the dataset is. The label
    is the indexed fraction from XDS -- a crystal where 90% of spots
    indexed scores close to 1.0, a crystal where only 10% indexed
    scores close to 0.0.

Usage:
  python train_cnn.py

You will be prompted for the parent folder (same one you gave xds_pipeline.py).
"""

import os
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Import from microed_cnn.py -- must be in the same folder as this script
from microed_cnn import (
    MicroEdCNN,
    load_and_average_frames,
    save_checkpoint,
    _N_FRAMES_DEFAULT,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Dataset
# ===========================================================================

class MicroEdDataset(Dataset):
    """
    Reads cell_parameters_summary.csv to build a labelled training set.

    Each sample returns three things:
      image_tensor   : (1, 256, 256) float32  -- averaged diffraction frames
      cell_label     : (6,)          float32  -- [a, b, c, alpha, beta, gamma]
      quality_label  : scalar        float32  -- indexed_fraction in [0, 1]

    A row is skipped when:
      - any cell parameter is 'n/a' (XDS did not produce a valid cell)
      - indexed_fraction is 0 or missing (dataset completely failed)
      - the subdirectory has no .img files
    """

    def __init__(self, parent_dir: Path, n_frames: int = _N_FRAMES_DEFAULT):
        self.n_frames = n_frames
        self.samples  = []

        csv_path = parent_dir / "cell_parameters_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found at {csv_path}\n"
                "Run xds_pipeline.py first to generate it."
            )

        skipped = 0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subdir = row.get("subdirectory", "").strip()

                # --- Cell labels ---
                try:
                    cell_label = [
                        float(row["a"]),
                        float(row["b"]),
                        float(row["c"]),
                        float(row["alpha"]),
                        float(row["beta"]),
                        float(row["gamma"]),
                    ]
                except (ValueError, KeyError):
                    log.debug("Skipping %s -- invalid cell parameters.", subdir)
                    skipped += 1
                    continue

                # --- Quality label (indexed fraction) ---
                try:
                    quality_label = float(row.get("idxref_fraction") or 0)
                except (ValueError, TypeError):
                    quality_label = 0.0

                if quality_label <= 0.0:
                    log.debug("Skipping %s -- indexed fraction is 0.", subdir)
                    skipped += 1
                    continue

                # --- Image files ---
                dataset_dir = parent_dir / subdir
                if not dataset_dir.is_dir():
                    log.debug("Skipping %s -- directory not found.", subdir)
                    skipped += 1
                    continue

                img_files = sorted(
                    f for f in os.listdir(dataset_dir) if f.endswith(".img")
                )
                if not img_files:
                    log.debug("Skipping %s -- no .img files found.", subdir)
                    skipped += 1
                    continue

                self.samples.append((dataset_dir, img_files, cell_label, quality_label))

        log.info(
            "Dataset: %d valid samples found, %d skipped.",
            len(self.samples), skipped,
        )
        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid training samples found.\n"
                "Make sure xds_pipeline.py has been run and produced "
                "cell_parameters_summary.csv with valid cell parameters."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        dataset_dir, img_files, cell_label, quality_label = self.samples[idx]

        # Load and average diffraction frames
        averaged = load_and_average_frames(dataset_dir, img_files, self.n_frames)
        if averaged is None:
            # If frames cannot be read, return a blank image.
            # The high loss for this sample signals the loading problem.
            log.warning("Could not load frames from %s -- using blank image.", dataset_dir)
            averaged = np.zeros((1, 256, 256), dtype=np.float32)

        return (
            torch.from_numpy(averaged).float(),               # (1, 256, 256)
            torch.tensor(cell_label,    dtype=torch.float32), # (6,)
            torch.tensor(quality_label, dtype=torch.float32), # scalar
        )


# ===========================================================================
# Loss function
# ===========================================================================

class CombinedLoss(nn.Module):
    """
    Weighted sum of two losses:
      - Huber loss  for cell parameters (robust to occasional bad XDS labels)
      - BCE loss    for quality score   (binary cross-entropy for [0,1] output)

    The two losses are kept separate rather than combined into one output
    because they have different scales and gradients -- mixing them into
    one linear layer would cause them to interfere with each other.
    """

    def __init__(
        self,
        w_cell:      float = 1.0,
        w_quality:   float = 0.5,
        huber_delta: float = 10.0,
    ):
        super().__init__()
        self.w_cell       = w_cell
        self.w_quality    = w_quality
        self.cell_loss_fn = nn.HuberLoss(delta=huber_delta)
        self.qual_loss_fn = nn.BCELoss()

    def forward(
        self,
        cell_pred:    torch.Tensor,  # (B, 6)
        cell_gt:      torch.Tensor,  # (B, 6)
        quality_pred: torch.Tensor,  # (B,)
        quality_gt:   torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        lc = self.cell_loss_fn(cell_pred, cell_gt)
        lq = self.qual_loss_fn(quality_pred, quality_gt)
        return self.w_cell * lc + self.w_quality * lq


# ===========================================================================
# Training
# ===========================================================================

def train(
    parent_dir:  str | Path,
    epochs:      int   = 50,
    lr:          float = 1e-3,
    batch_size:  int   = 4,
    val_split:   float = 0.2,
    n_frames:    int   = _N_FRAMES_DEFAULT,
    w_cell:      float = 1.0,
    w_quality:   float = 0.5,
):
    """
    Full training run.

    Parameters
    ----------
    parent_dir  : the same folder you gave xds_pipeline.py
    epochs      : number of training epochs (default 50)
    lr          : Adam learning rate (default 0.001)
    batch_size  : samples per mini-batch (default 4, keep low for few datasets)
    val_split   : fraction of data held out for validation (default 0.2 = 20%)
    n_frames    : diffraction frames averaged per sample (default 15)
    w_cell      : weight for cell regression loss (default 1.0)
    w_quality   : weight for quality score loss (default 0.5)
    """
    parent_dir   = Path(parent_dir)
    weights_path = parent_dir / "microed_cnn_weights.pt"

    # --- Build dataset ---
    dataset  = MicroEdDataset(parent_dir, n_frames=n_frames)
    n_total  = len(dataset)
    n_val    = max(1, int(n_total * val_split))
    n_train  = n_total - n_val

    if n_train < 1:
        raise RuntimeError(
            f"Not enough samples to train ({n_total} total, need at least 2)."
        )

    train_set, val_set = random_split(dataset, [n_train, n_val])
    log.info("Training on %d samples, validating on %d samples.", n_train, n_val)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False, num_workers=0
    )

    # --- Model, optimiser, scheduler, loss ---
    model     = MicroEdCNN(pretrained=True)
    criterion = CombinedLoss(w_cell=w_cell, w_quality=w_quality)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing: LR smoothly decays to near-zero over the training run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val_loss = float("inf")

    log.info("Starting training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):

        # ---- Training phase ----
        model.train()
        train_loss = 0.0

        for images, cell_gt, quality_gt in train_loader:
            optimizer.zero_grad()

            cell_pred, quality_pred = model(images)
            loss = criterion(cell_pred, cell_gt, quality_pred, quality_gt)

            loss.backward()

            # Gradient clipping prevents large unstable updates early in training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, cell_gt, quality_gt in val_loader:
                cell_pred, quality_pred = model(images)
                loss = criterion(cell_pred, cell_gt, quality_pred, quality_gt)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader),   1)
        scheduler.step()

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, epochs, avg_train, avg_val,
        )

        # Save only the best checkpoint (lowest validation loss)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(
                model,
                weights_path,
                optimizer=optimizer,
                epoch=epoch,
                metadata={"val_loss": best_val_loss},
            )
            log.info("  --> Best model saved (val_loss=%.4f)", best_val_loss)

    log.info("Training complete.")
    log.info("Best validation loss : %.4f", best_val_loss)
    log.info("Weights saved to     : %s",   weights_path)
    log.info(
        "Run xds_pipeline.py again to use these weights for CNN inference."
    )


# ===========================================================================
# Entry point
# ===========================================================================

def prompt_parent_dir() -> Path:
    """Prompt for parent directory with drag-and-drop support."""
    while True:
        raw = input("\nDrag the PARENT folder here, then press Enter:\n> ").strip()
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


if __name__ == "__main__":
    parent = prompt_parent_dir()
    train(parent)
