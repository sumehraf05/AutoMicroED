#!/usr/bin/env python3
"""
microed_cnn.py
==============
CNN module for MicroED unit-cell prediction and data-quality scoring.

This file is imported automatically by xds_pipeline.py and train_cnn.py.
You never run this file directly.

What it provides:
  - MicroEdCNN         : the neural network model
  - load_cnn_model     : load trained weights from disk
  - predict_unit_cell  : run the CNN on one dataset's diffraction images
  - compare_xds_and_cnn: check whether CNN and XDS agree on the unit cell
  - save_checkpoint    : save model weights during training
  - load_and_average_frames : load and preprocess diffraction images

Requirements:
  pip install torch torchvision numpy fabio
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# fabio is needed to read .img diffraction image files.
# If it is not installed the CNN will still load but frame reading will fail.
# ---------------------------------------------------------------------------
try:
    import fabio
    _HAS_FABIO = True
except ImportError:
    _HAS_FABIO = False
    log.warning(
        "fabio not installed -- diffraction frame loading will not work. "
        "Install it with: pip install fabio"
    )


# ===========================================================================
# Constants
# ===========================================================================

# Every diffraction frame is resized to this square size before feeding to the CNN.
# 256x256 is a good balance between detail and memory usage.
_IMG_SIZE = 256

# Number of frames sampled evenly from each dataset and averaged together.
# Using 15 frames gives a representative average without loading everything.
_N_FRAMES_DEFAULT = 15

# Physical bounds for unit cell parameters.
# Cell predictions are clamped to these ranges at inference time so the
# network cannot output physically impossible values.
_CELL_MIN = torch.tensor([1.0,   1.0,   1.0,   60.0,  60.0,  60.0])
_CELL_MAX = torch.tensor([500.0, 500.0, 500.0, 120.0, 120.0, 120.0])

# The six unit cell parameter names in order
PARAM_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class CNNPrediction:
    """
    The result returned by predict_unit_cell().

    Fields
    ------
    params        : dict with keys a, b, c, alpha, beta, gamma.
                    Values are floats when inference succeeded,
                    or the string 'n/a' when it did not.
    quality_score : float between 0 and 1 (higher = better quality dataset).
                    None when inference was not run.
    disagreement  : True when CNN and XDS disagree on the unit cell by more
                    than the tolerance threshold.
    flag_reason   : plain-English explanation of any disagreement or failure.
    """
    params: dict = field(
        default_factory=lambda: {k: "n/a" for k in PARAM_NAMES}
    )
    quality_score: Optional[float] = None
    disagreement:  bool = False
    flag_reason:   str  = "CNN inference not run"

    def is_valid(self) -> bool:
        """Return True when real predictions are present (not 'n/a')."""
        return self.params.get("a") != "n/a"


# This is returned whenever the model is None or frame loading fails.
# is_valid() returns False so callers know no real prediction was made.
_NULL_CNN_RESULT = CNNPrediction()


# ===========================================================================
# Image preprocessing
# ===========================================================================

def _block_downsample(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Shrink a 2D float32 array to (out_h, out_w) by block-averaging.

    Block averaging (taking the mean of rectangular blocks of pixels) is
    better than simple subsampling for diffraction images because it
    preserves the integrated intensity of weak spots.

    Falls back to nearest-neighbour selection if the input is already
    smaller than the target size.
    """
    in_h, in_w = arr.shape

    # If the image is already smaller than the target, use nearest-neighbour
    if in_h < out_h or in_w < out_w:
        row_idx = np.linspace(0, in_h - 1, out_h, dtype=int)
        col_idx = np.linspace(0, in_w - 1, out_w, dtype=int)
        return arr[np.ix_(row_idx, col_idx)].astype(np.float32)

    # Crop to the largest multiple of the block size, then reshape and average
    crop_h = (in_h // out_h) * out_h
    crop_w = (in_w // out_w) * out_w
    bh = crop_h // out_h
    bw = crop_w // out_w
    return (
        arr[:crop_h, :crop_w]
        .reshape(out_h, bh, out_w, bw)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def _preprocess_frame(raw: np.ndarray) -> np.ndarray:
    """
    Prepare one diffraction frame for input to the CNN.

    Steps:
      1. Clip negative pixels to zero.
         Negative values come from background subtraction and are
         not physically meaningful -- they would confuse the network.

      2. Log-compress the pixel values.
         Diffraction images have a huge dynamic range: the direct beam
         and strong Bragg peaks can be thousands of times brighter than
         weak spots. Log compression brings them into a range where the
         network can learn from both strong and weak features.

      3. Block-downsample to _IMG_SIZE x _IMG_SIZE pixels.
         Standardises all images to the same size regardless of detector.

      4. Min-max normalise to [0, 1].
         Neural networks train better when inputs are in a standard range.

    Returns an array of shape (1, H, W) -- the leading 1 is the channel
    dimension that PyTorch expects for grayscale images.
    """
    raw        = np.clip(raw, 0, None).astype(np.float32)
    compressed = np.log1p(raw)
    resized    = _block_downsample(compressed, _IMG_SIZE, _IMG_SIZE)
    mn, mx     = resized.min(), resized.max()
    if mx > mn:
        resized = (resized - mn) / (mx - mn)
    else:
        resized = np.zeros_like(resized)
    return resized[np.newaxis]    # shape: (1, H, W)


def load_and_average_frames(
    dataset_dir: Path,
    img_files:   Sequence[str],
    n_frames:    int = _N_FRAMES_DEFAULT,
) -> Optional[np.ndarray]:
    """
    Load diffraction frames from a dataset directory, preprocess them,
    and return their average as a single (1, H, W) float32 array.

    Rather than loading every frame (which could be hundreds), we sample
    n_frames evenly spaced across the dataset. This captures how the
    crystal quality changes through the rotation while keeping memory use low.

    Returns None if fabio is not installed or no frames could be read.
    """
    if not _HAS_FABIO:
        log.error("fabio must be installed to load diffraction frames.")
        return None

    total = len(img_files)
    if total == 0:
        return None

    # Pick n_frames indices evenly spread from first to last frame
    indices     = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    accumulated = None
    loaded      = 0

    for i in indices:
        fpath = dataset_dir / img_files[i]
        try:
            raw   = fabio.open(str(fpath)).data.astype(np.float32)
            frame = _preprocess_frame(raw)
            accumulated = frame if accumulated is None else accumulated + frame
            loaded += 1
        except Exception as exc:
            log.debug("Skipping frame %s: %s", fpath.name, exc)

    if loaded == 0 or accumulated is None:
        log.warning("No frames could be loaded from %s", dataset_dir)
        return None

    return accumulated / loaded    # shape: (1, H, W)


# Backward-compatible alias used by older versions of train_cnn.py
_load_and_average_frames = load_and_average_frames


# ===========================================================================
# Neural network architecture
# ===========================================================================

class _CellHead(nn.Module):
    """
    Regression head that predicts the 6 unit cell parameters.

    Takes the feature vector from the ResNet backbone and maps it to
    [a, b, c, alpha, beta, gamma].

    Uses Dropout to reduce overfitting when training on small datasets.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _QualityHead(nn.Module):
    """
    Classification head that predicts a data quality score in [0, 1].

    Takes the same backbone feature vector as _CellHead but has its own
    separate weights. Keeping it separate prevents the quality score
    gradient (from BCE loss) from interfering with the cell parameter
    gradients (from Huber loss) during training.

    The Sigmoid at the end ensures the output is always between 0 and 1.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1) removes the trailing dimension so output is (B,) not (B, 1)
        return self.fc(x).squeeze(-1)


class MicroEdCNN(nn.Module):
    """
    Convolutional neural network for MicroED diffraction image analysis.

    Architecture:
      ResNet-18 backbone (pretrained on ImageNet, adapted for grayscale)
          |
          +--> _CellHead    --> 6 unit cell parameters  (a, b, c, alpha, beta, gamma)
          |
          +--> _QualityHead --> 1 quality score in [0, 1]

    Why ResNet-18?
      ResNet-18 is a well-tested architecture that works well for feature
      extraction from images. It was originally trained on millions of
      colour photographs (ImageNet). We adapt it for diffraction images by:
        1. Replacing the first convolutional layer to accept 1 channel
           (grayscale) instead of 3 channels (RGB).
        2. Initialising the new layer by averaging the 3 RGB channel weights
           so we keep the useful features ResNet learned, just adapted for
           single-channel input.
        3. Replacing the final classification layer with our two output heads.

    Parameters
    ----------
    pretrained : bool
        If True, start from ImageNet weights (recommended -- gives better
        results with less training data). If False, use random weights.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet-18, suppressing the deprecation warning about weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights  = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)

        # Replace the first conv layer: 3-channel RGB -> 1-channel grayscale
        old_conv        = backbone.conv1
        backbone.conv1  = nn.Conv2d(
            in_channels  = 1,
            out_channels = old_conv.out_channels,
            kernel_size  = old_conv.kernel_size,
            stride       = old_conv.stride,
            padding      = old_conv.padding,
            bias         = False,
        )

        with torch.no_grad():
            if pretrained:
                # Average the 3 RGB channel weights into 1 channel.
                # This preserves the ImageNet features instead of discarding them.
                backbone.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
            else:
                nn.init.kaiming_normal_(
                    backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

        # Remove ResNet's original 1000-class ImageNet classifier
        in_features    = backbone.fc.in_features
        backbone.fc    = nn.Identity()
        self.backbone  = backbone

        # Add our two output heads
        self.cell_head    = _CellHead(in_features)
        self.quality_head = _QualityHead(in_features)

    def forward(self, x: torch.Tensor):
        """
        Forward pass -- use this during training.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, H, W)
            Batch of preprocessed diffraction images.
            B = batch size, 1 = grayscale channel, H = W = 256.

        Returns
        -------
        cell_params   : (B, 6)  -- raw cell parameter predictions
        quality_score : (B,)    -- quality scores in [0, 1]
        """
        features      = self.backbone(x)
        cell_params   = self.cell_head(features)
        quality_score = self.quality_head(features)
        return cell_params, quality_score

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Inference-only forward pass with physical clamping.

        Same as forward() but additionally clamps the cell parameter
        predictions to physically valid ranges:
          lengths : 1 to 500 Angstroms
          angles  : 60 to 120 degrees

        Use this when running the pipeline. Use forward() during training
        so that gradients can flow freely through the network.
        """
        cell_raw, quality = self.forward(x)
        cell_clamped = torch.clamp(
            cell_raw,
            _CELL_MIN.to(x.device),
            _CELL_MAX.to(x.device),
        )
        return cell_clamped, quality


# ===========================================================================
# Checkpoint saving and loading
# ===========================================================================

def save_checkpoint(
    model:     MicroEdCNN,
    path:      Path,
    optimizer: Optional[optim.Optimizer] = None,
    epoch:     int  = 0,
    metadata:  Optional[dict] = None,
) -> None:
    """
    Save the model weights and optional training metadata to a .pt file.

    The saved file contains:
      - model_state     : the network weights
      - epoch           : which training epoch this checkpoint is from
      - metadata        : any extra info (e.g. validation loss)
      - optimizer_state : optimizer state (optional, useful for resuming)

    Called by train_cnn.py whenever a new best validation loss is found.
    """
    payload: dict = {
        "model_state": model.state_dict(),
        "epoch":       epoch,
        "metadata":    metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, path)
    log.info("Checkpoint saved -> %s  (epoch %d)", path, epoch)


def _remap_legacy_state(raw: dict) -> dict:
    """
    Remap weight keys from the old single-head model to the new two-head model.

    The original model used key names like 'model.conv1.weight'.
    The current model uses 'backbone.conv1.weight'.
    This function remaps the old names so old weight files still load.
    """
    return {k.replace("model.", "backbone.", 1): v for k, v in raw.items()}


def load_cnn_model(weights_path) -> Optional[MicroEdCNN]:
    """
    Load a trained MicroEdCNN from a weights file.

    Accepts two file formats:
      1. New format  : dict containing 'model_state' key (from save_checkpoint)
      2. Legacy format: bare state dict (from older torch.save calls)

    Returns None (with a warning) if the file does not exist or cannot be
    loaded. The pipeline will continue without CNN inference in that case.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        log.warning(
            "CNN weights not found at '%s'. "
            "Run train_cnn.py first to generate them. "
            "CNN inference will be skipped for this run.",
            weights_path,
        )
        return None

    try:
        model = MicroEdCNN(pretrained=False)
        raw   = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(raw, dict) and "model_state" in raw:
            # New format -- extract the model state and report which epoch
            state = raw["model_state"]
            log.info(
                "CNN model loaded from '%s' (epoch %s)",
                weights_path, raw.get("epoch", "?"),
            )
        else:
            # Legacy format -- remap key names for the new architecture
            state = _remap_legacy_state(raw)
            log.info(
                "CNN model loaded from '%s' (legacy format, keys remapped)",
                weights_path,
            )

        # strict=False means missing or extra keys don't crash the load
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Checkpoint is missing keys: %s", missing)
        if unexpected:
            log.warning("Checkpoint has unexpected keys: %s", unexpected)

        model.eval()
        return model

    except Exception as exc:
        log.error("Failed to load CNN model from '%s': %s", weights_path, exc)
        return None


# ===========================================================================
# Inference
# ===========================================================================

def predict_unit_cell(
    model:       Optional[MicroEdCNN],
    dataset_dir: Path,
    img_files:   Sequence[str],
    n_frames:    int = _N_FRAMES_DEFAULT,
) -> CNNPrediction:
    """
    Run the CNN on one dataset and return a unit cell prediction.

    This function ALWAYS returns a CNNPrediction object -- it never raises
    an exception or returns None. If something goes wrong (model not loaded,
    frames cannot be read, inference error), it returns a CNNPrediction with
    is_valid() == False and a flag_reason explaining what happened.

    This design means xds_pipeline.py never needs to check for None.

    Parameters
    ----------
    model       : loaded MicroEdCNN, or None to skip inference
    dataset_dir : folder containing the .img diffraction frames
    img_files   : sorted list of .img filenames in dataset_dir
    n_frames    : how many frames to sample and average (default 15)

    Returns
    -------
    CNNPrediction with predicted cell parameters and quality score.
    """
    # No model loaded -- return the null sentinel
    if model is None:
        return _NULL_CNN_RESULT

    # Load and average the diffraction frames
    averaged = load_and_average_frames(dataset_dir, img_files, n_frames)
    if averaged is None:
        return CNNPrediction(flag_reason="Frame loading failed")

    # Add batch dimension: (1, H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(averaged).unsqueeze(0).float()

    # Run inference
    try:
        cell_tensor, quality_tensor = model.predict(tensor)
    except Exception as exc:
        log.error("CNN inference error on %s: %s", dataset_dir, exc)
        return CNNPrediction(flag_reason=f"Inference error: {exc}")

    # Convert output tensors to Python values
    cell_np = cell_tensor.squeeze(0).numpy()
    quality = float(quality_tensor.squeeze(0).item())

    params = {
        name: round(float(cell_np[i]), 3)
        for i, name in enumerate(PARAM_NAMES)
    }

    return CNNPrediction(
        params=params,
        quality_score=round(quality, 4),
        disagreement=False,
        flag_reason="",
    )


def compare_xds_and_cnn(
    xds_params:     dict,
    cnn_prediction: CNNPrediction,
    tol_lengths:    float = 5.0,
    tol_angles:     float = 5.0,
) -> CNNPrediction:
    """
    Compare XDS cell parameters against the CNN prediction.

    For each of the six unit cell parameters, checks whether XDS and the
    CNN agree within the given tolerance:
      - lengths (a, b, c)         : tolerance in Angstroms (default 5.0)
      - angles (alpha, beta, gamma): tolerance in degrees   (default 5.0)

    If any parameter differs by more than the tolerance, the returned
    CNNPrediction has disagreement=True and flag_reason describes which
    parameters disagreed and by how much.

    Disagreement does not mean the dataset is wrong -- it is a flag that
    something unusual is happening and the dataset deserves a closer look.

    Returns the cnn_prediction unchanged if it has no valid predictions.
    """
    # Nothing to compare if CNN did not produce valid predictions
    if not cnn_prediction.is_valid():
        return cnn_prediction

    tolerances = dict(
        a=tol_lengths, b=tol_lengths, c=tol_lengths,
        alpha=tol_angles, beta=tol_angles, gamma=tol_angles,
    )
    units = dict(
        a="A", b="A", c="A",
        alpha="deg", beta="deg", gamma="deg",
    )

    disagreements: list[str] = []

    for name in PARAM_NAMES:
        try:
            xds_f = float(xds_params.get(name, "n/a"))
            cnn_f = float(cnn_prediction.params.get(name, "n/a"))
        except (TypeError, ValueError):
            # One or both values are 'n/a' -- skip this parameter
            continue

        diff = abs(xds_f - cnn_f)
        if diff > tolerances[name]:
            disagreements.append(
                f"{name}: XDS={xds_f:.2f}, CNN={cnn_f:.2f}, "
                f"diff={diff:.2f} {units[name]} (tol={tolerances[name]})"
            )

    if disagreements:
        return CNNPrediction(
            params=cnn_prediction.params,
            quality_score=cnn_prediction.quality_score,
            disagreement=True,
            flag_reason="Disagreement -- " + "; ".join(disagreements),
        )

    # Everything agrees within tolerance
    return CNNPrediction(
        params=cnn_prediction.params,
        quality_score=cnn_prediction.quality_score,
        disagreement=False,
        flag_reason="",
    )
