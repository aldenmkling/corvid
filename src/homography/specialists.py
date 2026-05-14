"""Slim specialists runner — extracted from rectify.py so dataset-building
scripts can import it without dragging in the whole homography stack
(grid_solver_v2, h_tracker, line_fit, etc.).

Loads two SMP UNet (mit_b0) specialists and returns binary masks at frame
resolution:
    - line specialist (grayscale input, 2-ch output: yard, side)
    - hash specialist (RGB input, 1-ch output)

Number specialist lives in painted_numbers.predict_mask (not duplicated here).

Usage:
    from src.homography.specialists import (
        LINE_WEIGHTS, HASH_WEIGHTS, run_specialists,
    )
    yard, side, hash_ = run_specialists(frame_bgr, LINE_WEIGHTS, HASH_WEIGHTS,
                                          device_str="cuda")
"""
import os

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINE_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_stage2_last.pth")
HASH_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_hash_round3_last.pth")
NUMBER_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_numbers_last.pth")

# Unified mask model — single mit_b0 U-Net producing all 4 channels at once.
# Production: v8 (matches/beats specialists on yard, side, hash). Symlinked at
# models/unet_unified_default/.
UNIFIED_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_unified_default/best.pth")

UNET_INPUT_H, UNET_INPUT_W = 512, 896
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Same thresholds rectify.py uses (sourced from grid_solver_v2 / line_fit).
YARD_THRESH = 0.5
SIDE_THRESH = 0.5
HASH_THRESH = 0.40


def _preprocess(frame_bgr: np.ndarray, grayscale: bool):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UNET_INPUT_W, UNET_INPUT_H))
    if grayscale:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([g, g, g], axis=-1)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


_MODEL_CACHE = {}


def _load_smp_unet(weights: str, classes: int, device: torch.device):
    key = (weights, classes, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                  in_channels=3, classes=classes, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.to(device).eval()
    _MODEL_CACHE[key] = m
    return m


@torch.no_grad()
def run_specialists(frame: np.ndarray, line_weights: str, hash_weights: str,
                     device_str: str = "mps"):
    """Two forward passes: line UNet (grayscale, 2ch) + hash UNet (RGB, 1ch).
    Returns (yard_mask, side_mask, hash_mask) as binary uint8 masks at frame
    resolution."""
    device = torch.device(device_str)
    line_model = _load_smp_unet(line_weights, classes=2, device=device)
    hash_model = _load_smp_unet(hash_weights, classes=1, device=device)
    h0, w0 = frame.shape[:2]

    t_line = _preprocess(frame, grayscale=True).to(device)
    p_line = torch.sigmoid(line_model(t_line))[0].cpu().numpy()
    yard = (p_line[0] > YARD_THRESH).astype(np.uint8)
    side = (p_line[1] > SIDE_THRESH).astype(np.uint8)

    t_hash = _preprocess(frame, grayscale=False).to(device)
    p_hash = torch.sigmoid(hash_model(t_hash))[0, 0].cpu().numpy()
    hash_ = (p_hash > HASH_THRESH).astype(np.uint8)

    yard = cv2.resize(yard, (w0, h0), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w0, h0), interpolation=cv2.INTER_NEAREST)
    hash_ = cv2.resize(hash_, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return yard, side, hash_
