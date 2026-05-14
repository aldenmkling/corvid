"""Train the dense per-pixel field-coord regression model.

Architecture:
  - smp.Segformer encoder (default mit_b1) + native MLP decoder
  - Input: 7 channels = 3 RGB + 4 specialist sigmoid probabilities
    (yard, side, hash, number) — soft confidences, NOT binarized
  - Output: 4 channels at 1/4 resolution = (mu_x, mu_y, log_var_x, log_var_y)
  - Loss: aleatoric uncertainty L1 (Kendall & Gal 2017) — model learns its
    own per-pixel confidence.

Training data:
  - Manifest from output/dense_field_pool/manifest.json (the global pool)
  - decisions.json filters to user-Y'd entries
  - Per entry: source MP4 + frame_idx + H matrix (manifest-stored)
  - Specialists are run ONCE per entry to cache (rgb uint8,
    4-channel float16 sigmoid probability masks)
  - GT dense field coords computed on-the-fly from H

Augmentations (training only):
  - Random crop + resize (50% prob, scale 0.4-1.0) — synthesizes
    underconstrained tight-zoom pairs from wide shots
  - Color jitter on RGB
  - Mask noise: per-channel dilate/erode/patch-erase/blob-inject/dropout
  - NO horizontal flip (would invert NGS_x)

Validation:
  - Held-out game(s), no game-leak
  - Per-pixel L1 in yards
  - Per-pixel inlier rate at <0.5 yd, <1 yd

Usage:
    python scripts/training/train_dense_regression.py \\
        --pool-dir output/dense_field_pool \\
        --out-dir models/dense_regression_v1 \\
        --device cuda --epochs 50
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import segmentation_models_pytorch as smp

from src.homography.specialists import (
    LINE_WEIGHTS, HASH_WEIGHTS, NUMBER_WEIGHTS, UNIFIED_WEIGHTS,
    UNET_INPUT_H, UNET_INPUT_W,
    IMAGENET_MEAN_NP, IMAGENET_STD_NP,
)
from src.homography.field_model import FIELD_WIDTH


# ── Constants ──
# Input dims must be divisible by 32 for SegFormer/U-Net stride alignment.
# 704 = 32 * 22, 1280 = 32 * 40. Cache frames are 720×1280; the dataset
# resizes to (INPUT_H, INPUT_W) on load.
INPUT_H, INPUT_W = 704, 1280
OUTPUT_H, OUTPUT_W = 176, 320      # 1/4 resolution
NGS_X_MAX = 120.0                   # yards (full field incl. endzones)
NGS_Y_MAX = float(FIELD_WIDTH)     # 53.33 yards
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Dense field-coord ground truth from H ──
def make_undistort_grid(K, dist, out_h=OUTPUT_H, out_w=OUTPUT_W,
                          src_h=INPUT_H, src_w=INPUT_W):
    """Pre-compute the per-(K, dist) undistortion grid: for each output cell
    (i, j), the corresponding undistorted source-pixel (x, y). Use once at
    dataset __init__ time and reuse on every __getitem__ — saves ~3 ms/sample
    of cv2.undistortPoints calls.

    Returns (out_h, out_w, 2) float32 with (sx_und, sy_und)."""
    ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    sx = (xs + 0.5) * (src_w / out_w) - 0.5
    sy = (ys + 0.5) * (src_h / out_h) - 0.5
    pts2 = np.stack([sx.ravel(), sy.ravel()], axis=1).astype(np.float64)
    K_np = np.asarray(K, dtype=np.float64)
    dist_np = np.asarray(dist, dtype=np.float64)
    und = cv2.undistortPoints(
        pts2.reshape(-1, 1, 2), K_np, dist_np, P=K_np
    ).reshape(-1, 2)
    return und.reshape(out_h, out_w, 2).astype(np.float32)


def field_coords_from_H(H, out_h=OUTPUT_H, out_w=OUTPUT_W,
                          src_h=INPUT_H, src_w=INPUT_W,
                          K=None, dist=None, undistort_grid=None):
    """For each (i, j) in output grid, compute the corresponding source
    pixel, undistort it, then warp through H to get (NGS_x, NGS_y).

    H maps **undistorted** source pixel → NGS yards.

    Either pass `undistort_grid` (pre-computed via `make_undistort_grid`)
    for fast lookup, OR pass `K`/`dist` and the function will undistort
    inline. Without any of these the source grid goes directly to H —
    only correct for distortion-free cameras (broadcast All-22 has k1
    ≈0.1 → 0.1-0.4 yd avg GT error if skipped).

    Returns (out_h, out_w, 2) float32 with (x_yd, y_yd)."""
    if undistort_grid is not None:
        und = undistort_grid.reshape(-1, 2).astype(np.float64)
        pts = np.concatenate([und, np.ones((len(und), 1))], axis=1)
    else:
        ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
        sx = (xs + 0.5) * (src_w / out_w) - 0.5
        sy = (ys + 0.5) * (src_h / out_h) - 0.5
        pts2 = np.stack([sx.ravel(), sy.ravel()], axis=1).astype(np.float64)
        if K is not None and dist is not None:
            K_np = np.asarray(K, dtype=np.float64)
            dist_np = np.asarray(dist, dtype=np.float64)
            pts2 = cv2.undistortPoints(
                pts2.reshape(-1, 1, 2), K_np, dist_np, P=K_np
            ).reshape(-1, 2)
        pts = np.concatenate([pts2, np.ones((len(pts2), 1))], axis=1)
    field = (H @ pts.T).T                                                  # N×3
    field = field[:, :2] / np.clip(field[:, 2:3], 1e-9, None)
    return field.reshape(out_h, out_w, 2).astype(np.float32)


def valid_mask_from_gt(gt):
    """Return 1 where (NGS_x, NGS_y) is inside [0, NGS_X_MAX] × [0, NGS_Y_MAX]."""
    x, y = gt[..., 0], gt[..., 1]
    return ((x >= 0) & (x <= NGS_X_MAX) & (y >= 0) & (y <= NGS_Y_MAX)).astype(np.float32)


# ── Unified mask sigmoid extraction (no thresholding) ──
# Single forward pass through the v8 unified mit_b0 U-Net (3in/4out) instead of
# 3 separate specialist calls. The unified model now matches/beats specialists
# on all hand-labeled metrics and is ~3× faster.
_UNIFIED_MODEL = {}

def _load_unified(weights, device):
    key = (weights, str(device))
    if key in _UNIFIED_MODEL:
        return _UNIFIED_MODEL[key]
    m = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                  in_channels=3, classes=4, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.to(device).eval()
    _UNIFIED_MODEL[key] = m
    return m


def _preprocess_for_unified(rgb_bgr):
    """RGB input only (no grayscale variant — unified takes RGB for all 4 channels)."""
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UNET_INPUT_W, UNET_INPUT_H))
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def specialist_probs(frame_bgr, device_str, weights=UNIFIED_WEIGHTS):
    """Returns 4-channel sigmoid probability maps (yard, side, hash, num) at
    original frame resolution. NO thresholding applied — the dense regression
    model gets the unified model's soft confidence."""
    device = torch.device(device_str)
    h0, w0 = frame_bgr.shape[:2]
    m = _load_unified(weights, device)
    x = _preprocess_for_unified(frame_bgr).to(device)
    probs = torch.sigmoid(m(x))[0].cpu().numpy()    # (4, H', W')
    out = np.zeros((h0, w0, 4), dtype=np.float32)
    for ci in range(4):
        out[..., ci] = cv2.resize(probs[ci], (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float16)


# ── Caching specialists' outputs once per entry ──
def cache_entry(entry, clips_root, cache_dir, device):
    """Run all 4 specialists on this entry's source frame and save
    (rgb, masks_4ch_probs). Probabilities (sigmoid, float16) are saved
    instead of binary so the dense regression model gets the specialist's
    confidence info."""
    cp = os.path.join(cache_dir, f"{entry['id']}.npz")
    if os.path.exists(cp):
        return True
    clip_path = os.path.join(clips_root, entry["clip"])
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(entry["frame_idx"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    masks = specialist_probs(frame, device)        # HxWx4 float16 in [0, 1]
    np.savez_compressed(cp, rgb=frame, masks=masks)
    return True


def precache_dataset(entries, clips_root, cache_dir, device, verbose=True):
    """Iterate entries, call cache_entry. Skips already-cached. Single
    pass, since specialists are loaded once at first call (module-level
    cache inside src.homography modules)."""
    os.makedirs(cache_dir, exist_ok=True)
    n_done = n_skipped = n_failed = 0
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        if os.path.exists(os.path.join(cache_dir, f"{e['id']}.npz")):
            n_skipped += 1
            continue
        ok = cache_entry(e, clips_root, cache_dir, device)
        if ok:
            n_done += 1
        else:
            n_failed += 1
        if verbose and i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  cache [{i}/{len(entries)}] new={n_done} cached={n_skipped} "
                  f"failed={n_failed}  ({elapsed:.0f}s, {elapsed/max(1,n_done):.1f}s/new)",
                  flush=True)
    print(f"Cache: {n_done} new, {n_skipped} resumed, {n_failed} failed "
          f"({time.time()-t0:.0f}s)")


# ── Augmentations ──
def aug_color_jitter(rgb, brightness=0.2, contrast=0.2, saturation=0.15):
    """Mild ImageNet-style color jitter on a uint8 BGR frame."""
    img = rgb.astype(np.float32)
    if brightness > 0:
        img = img * random.uniform(1 - brightness, 1 + brightness)
    if contrast > 0:
        m = img.mean()
        img = (img - m) * random.uniform(1 - contrast, 1 + contrast) + m
    if saturation > 0:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)[..., None]
        img = gray + (img - gray) * random.uniform(1 - saturation, 1 + saturation)
    return np.clip(img, 0, 255).astype(np.uint8)


def aug_random_crop_resize(rgb, masks, gt, valid, scale_range=(0.4, 1.0)):
    """Random sub-window of input, resize back to original input shape.
    Synthesizes 'tight-zoom' training pairs from wide shots while keeping
    field-coord GT correct."""
    h, w = rgb.shape[:2]
    s = random.uniform(*scale_range)
    crop_h = max(1, int(h * s))
    crop_w = max(1, int(w * s))
    y0 = random.randint(0, h - crop_h)
    x0 = random.randint(0, w - crop_w)
    rgb_c = rgb[y0:y0+crop_h, x0:x0+crop_w]
    masks_c = masks[y0:y0+crop_h, x0:x0+crop_w]
    rgb_r = cv2.resize(rgb_c, (w, h), interpolation=cv2.INTER_LINEAR)
    # Linear interp preserves probability soft edges; nearest would
    # introduce aliasing on the soft mask values
    masks_r = cv2.resize(masks_c, (w, h), interpolation=cv2.INTER_LINEAR)
    # GT is at output resolution — scale crop region accordingly
    out_h, out_w = gt.shape[:2]
    sy = out_h / h
    sx = out_w / w
    oy0 = int(round(y0 * sy))
    oy1 = int(round((y0 + crop_h) * sy))
    ox0 = int(round(x0 * sx))
    ox1 = int(round((x0 + crop_w) * sx))
    oy1 = max(oy0 + 1, oy1)
    ox1 = max(ox0 + 1, ox1)
    gt_c = gt[oy0:oy1, ox0:ox1]
    valid_c = valid[oy0:oy1, ox0:ox1]
    gt_r = cv2.resize(gt_c, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid_c, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return rgb_r, masks_r, gt_r, valid_r


def aug_mask_noise(masks, p_patch_erase=0.2, p_blob_inject=0.15,
                    p_channel_dropout=0.05, p_mult_scale=0.4,
                    mult_range=(0.6, 1.0), p_gauss_noise=0.3,
                    gauss_sigma=0.05):
    """Per-channel perturbations on PROBABILITY masks (float in [0,1]) that
    mimic real specialist failures.

    - patch erase   → wipe a random rectangle to 0   (specialist missed a region)
    - blob inject   → set a small disk to 1.0        (specialist hallucinated)
    - mult scale    → scale entire channel by 0.6-1.0 (specialist less confident)
    - gauss noise   → add N(0, sigma) per pixel       (general probability jitter)
    - channel drop  → entire channel to 0            (rare full failure)
    """
    out = masks.astype(np.float32, copy=True)
    H, W, C = out.shape
    for ch in range(C):
        m = out[..., ch]
        if random.random() < p_patch_erase:
            ph = random.randint(20, 80)
            pw = random.randint(20, 80)
            y0 = random.randint(0, max(1, H - ph))
            x0 = random.randint(0, max(1, W - pw))
            m[y0:y0+ph, x0:x0+pw] = 0.0
        if random.random() < p_blob_inject:
            cx = random.randint(0, W - 1)
            cy = random.randint(0, H - 1)
            r = random.randint(3, 12)
            disk = np.zeros((H, W), dtype=np.float32)
            cv2.circle(disk, (cx, cy), r, 1.0, -1)
            m = np.maximum(m, disk)
        if random.random() < p_mult_scale:
            m = m * random.uniform(*mult_range)
        if random.random() < p_gauss_noise:
            m = m + np.random.normal(0, gauss_sigma, m.shape).astype(np.float32)
            m = np.clip(m, 0.0, 1.0)
        if random.random() < p_channel_dropout:
            m[:] = 0.0
        out[..., ch] = m
    return out


# ── Dataset ──
class DenseFieldDataset(Dataset):
    """Builds 7-channel input per frame.

    input_mode='v1' (legacy): 3 RGB + 4 v8 sigmoid prob channels (yard, side,
        hash, num) — continuous values in [0,1].
    input_mode='v2': 3 RGB + 3 binary thresholded masks (yard>0.5, side>0.5,
        hash>0.5) + 1 number-NGS_x label channel (per-pixel NGS_x value at
        painted-number locations, 0 elsewhere). The number channel is loaded
        from `v2_input_dir/<id>.npz` (key 'number_ngs_x').
    """
    def __init__(self, entries, clips_root, cache_dir, augment=False,
                 crop_prob=0.5, input_mode="v1", v2_input_dir=None,
                 aux_gt_dir=None, intrinsics_by_clip=None):
        self.entries = entries
        self.clips_root = clips_root
        self.cache_dir = cache_dir
        self.augment = augment
        self.crop_prob = crop_prob
        self.input_mode = input_mode
        self.v2_input_dir = v2_input_dir
        self.aux_gt_dir = aux_gt_dir   # if set, also returns canonical aux GT
        # clip → {"K": 3x3, "dist": list}. Required to undistort source
        # pixels before applying H — without it the GT is silently wrong.
        self.intrinsics_by_clip = intrinsics_by_clip or {}
        # Pre-compute one undistort lookup grid per unique (K, dist) pair so
        # __getitem__ doesn't have to call cv2.undistortPoints every sample
        # (~3 ms × batch × 2189 entries / epoch otherwise).
        # Only compute grids for clips actually referenced in `entries` —
        # otherwise we pre-allocate ~7 MB × #unique-clips of float buffer
        # that's never touched (full pool has 1300+ clips → ~10 GB OOM).
        self._und_grids = {}
        clips_used = {e["clip"] for e in entries}
        for clip in clips_used:
            intr = self.intrinsics_by_clip.get(clip)
            if not intr:
                continue
            key = self._und_key(intr["K"], intr["dist"])
            if key not in self._und_grids:
                self._und_grids[key] = make_undistort_grid(
                    intr["K"], intr["dist"])
        if input_mode == "v2" and v2_input_dir is None:
            raise ValueError("input_mode='v2' requires --v2-input-dir")

    @staticmethod
    def _und_key(K, dist):
        # Tuple of focal+pp+k1 is enough to deduplicate clips that share
        # intrinsics (broadcast all-22 typically uses fixed K and just
        # differs in k1 across clips).
        K = np.asarray(K).flatten()
        dist = np.asarray(dist).flatten()
        return (round(float(K[0]), 4), round(float(K[2]), 4),
                round(float(K[5]), 4), round(float(dist[0]), 6))

    def _grid_for_clip(self, clip):
        intr = self.intrinsics_by_clip.get(clip)
        if not intr:
            return None
        return self._und_grids.get(self._und_key(intr["K"], intr["dist"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        cp = os.path.join(self.cache_dir, f"{e['id']}.npz")
        d = np.load(cp)
        rgb = d["rgb"].copy()        # HxWx3 uint8 BGR
        masks = d["masks"].copy().astype(np.float32)    # HxWx4 in [0,1]
        H = np.array(e["H"], dtype=np.float64)
        und_grid = self._grid_for_clip(e["clip"])
        gt = field_coords_from_H(H, undistort_grid=und_grid)
        valid = valid_mask_from_gt(gt)
        if self.input_mode == "v2":
            v2_path = os.path.join(self.v2_input_dir, f"{e['id']}.npz")
            d2 = np.load(v2_path)
            number_ngs_x = d2["number_ngs_x"].astype(np.float32)   # HxW
            # Stack into masks-like 5th channel for joint augmentation:
            #   masks[..., 0..3] = v8 probs; masks[..., 4] = number_ngs_x
            masks = np.concatenate([masks, number_ngs_x[..., None]], axis=-1)
        if self.augment:
            if random.random() < self.crop_prob:
                rgb, masks, gt, valid = aug_random_crop_resize(rgb, masks, gt, valid)
            rgb = aug_color_jitter(rgb)
            # mask noise should only apply to v8 prob channels (ch 0..3), not
            # the number NGS_x label channel (which would corrupt its meaning)
            masks_for_aug = masks[..., :4]
            masks_for_aug = aug_mask_noise(masks_for_aug)
            masks[..., :4] = masks_for_aug
        # Resize to model input size
        if rgb.shape[:2] != (INPUT_H, INPUT_W):
            rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            masks = cv2.resize(masks, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        # To tensors. RGB BGR→RGB → ImageNet normalize → CHW
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_rgb = (rgb_rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb_rgb.transpose(2, 0, 1))  # 3xHxW

        if self.input_mode == "v1":
            masks_t = torch.from_numpy(masks.transpose(2, 0, 1).astype(np.float32))  # 4xHxW
            x = torch.cat([rgb_t, masks_t], dim=0)  # 7xHxW
        else:
            # v2: 3 binary thresholded + 1 number_ngs_x → 4 ch concat with rgb
            yard_b = (masks[..., 0] > 0.5).astype(np.float32)
            side_b = (masks[..., 1] > 0.5).astype(np.float32)
            hash_b = (masks[..., 2] > 0.5).astype(np.float32)
            num_x = masks[..., 4]   # NGS_x label, kept as-is
            v2_stack = np.stack([yard_b, side_b, hash_b, num_x], axis=0)
            v2_t = torch.from_numpy(v2_stack.astype(np.float32))   # 4xHxW
            x = torch.cat([rgb_t, v2_t], dim=0)   # 7xHxW
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1))            # 2xHxW
        valid_t = torch.from_numpy(valid).unsqueeze(0)            # 1xHxW
        if self.aux_gt_dir is not None:
            aux_path = os.path.join(self.aux_gt_dir, f"{e['id']}.npz")
            d_aux = np.load(aux_path)
            aux_gt = torch.from_numpy(d_aux["aux_gt"].astype(np.float32))   # 4xCANxCANW
            return x, gt_t, valid_t, aux_gt
        return x, gt_t, valid_t


# ── Model ──
# Aux head output: canonical NGS top-down grid (Option B). The model has to
# learn to undistort + rectify the input frame's specialist masks (yard,
# side, hash, num) into NGS yards. This is a stronger structural prior than
# Option B1 (image-space "where lines should be"): now the aux REQUIRES the
# encoder to know the lens model + camera pose well enough to land each
# detection in its NGS cell.
# Grid covers NGS_x ∈ [0, 120] × NGS_y ∈ [0, 53.33] at 1 yd/cell.
AUX_OUT_H = 54          # ceil(53.33)
AUX_OUT_W = 120
AUX_CHANNELS = 4        # yard, side, hash, num


def build_model(encoder_name="mit_b1", in_channels=7, classes=4,
                  decoder="unet", decoder_segmentation_channels=256):
    """Build the dense regression model.
        decoder='unet' (default): U-Net with skip connections.
        decoder='segformer': SegFormer's MLP decoder.
        decoder_segmentation_channels: width of the SegFormer decoder MLPs
            and concat conv. Default 256 (smp default). Bump to 512/768/
            1024 for noticeably more decoder capacity. Ignored for U-Net.
    Output 4 channels: (mu_x_raw, mu_y_raw, log_var_x, log_var_y); activations
    applied in `split_outputs()`."""
    common = dict(encoder_name=encoder_name, encoder_weights="imagenet",
                   in_channels=in_channels, classes=classes)
    if decoder == "unet":
        return smp.Unet(**common)
    if decoder == "segformer":
        return smp.Segformer(
            decoder_segmentation_channels=decoder_segmentation_channels,
            **common)
    raise ValueError(f"unknown decoder: {decoder}")


class DenseRegressionWithAux(torch.nn.Module):
    """Wraps the main dense regression model and adds an auxiliary
    canonical-NGS-space "rectified-feature-map" output head (Option B).

    Architecture: shared encoder + main SegFormer decoder (regression) +
    parallel SegFormer decoder for aux (4-channel canonical-NGS-space
    binary mask, AUX_OUT_H × AUX_OUT_W). Both decoders consume the same
    encoder features.

    The aux task: take the input (distorted broadcast frame's RGB +
    specialist masks) and predict, at each cell of a top-down 54×120 NGS
    canonical grid, whether the v8 specialist would have detected
    yardline / sideline / hash / number content there — i.e., the
    rectified specialist mask. Forces the encoder to internalize:
      - the lens distortion (k1, k2)
      - the camera pose / homography
      - the canonical positions of each feature in NGS yards
    so it can map detected pixels through (undistort + H) onto the
    canonical grid.

    Inference: typically only the main head's output is used. The aux
    output is available as a diagnostic ("show me the model's rectified
    view of this frame").
    """
    def __init__(self, encoder_name="mit_b1", in_channels=7, decoder="segformer",
                 decoder_segmentation_channels=256,
                 aux_segmentation_channels=128):
        super().__init__()
        self.main = build_model(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=4, decoder=decoder,
            decoder_segmentation_channels=decoder_segmentation_channels)
        # Build a parallel SegFormer-style decoder + head for aux output.
        # Same architecture as main but 4 binary-mask classes instead of 4
        # regression channels.
        from segmentation_models_pytorch.decoders.segformer.decoder import (
            SegformerDecoder)
        from segmentation_models_pytorch.base import SegmentationHead
        encoder_channels = self.main.encoder.out_channels
        self.aux_decoder = SegformerDecoder(
            encoder_channels=encoder_channels,
            encoder_depth=5,
            segmentation_channels=aux_segmentation_channels,
        )
        # Aux output is small (54×120 NGS cells), so we don't bother with the
        # SegmentationHead's stock 4× upsampling — we interpolate directly
        # from decoder-native res (1/4 input) down to AUX_OUT.
        self.aux_head = SegmentationHead(
            in_channels=aux_segmentation_channels,
            out_channels=AUX_CHANNELS,
            kernel_size=3, upsampling=1,
        )

    def forward(self, x, return_aux=False):
        # Single encoder pass; both heads see the same features.
        self.main.check_input_shape(x)
        features = self.main.encoder(x)
        # Main regression head
        decoder_output = self.main.decoder(features)
        main_out = self.main.segmentation_head(decoder_output)
        if return_aux:
            aux_dec_out = self.aux_decoder(features)
            aux_out = self.aux_head(aux_dec_out)   # B x AUX_CH x 176 x 320
            # Resize to canonical NGS grid (AUX_OUT_H x AUX_OUT_W = 54 x 120)
            if aux_out.shape[-2:] != (AUX_OUT_H, AUX_OUT_W):
                aux_out = F.interpolate(aux_out,
                                         size=(AUX_OUT_H, AUX_OUT_W),
                                         mode="bilinear", align_corners=False)
            return main_out, aux_out
        return main_out


def split_outputs(out):
    """out: B x 4 x H x W → (mu_x, mu_y, log_var_x, log_var_y) all B x H x W
    with appropriate activations applied."""
    mu_x = torch.sigmoid(out[:, 0]) * NGS_X_MAX
    mu_y = torch.sigmoid(out[:, 1]) * NGS_Y_MAX
    # Bound log-variance to [-6, 6] via tanh*6 so optimizer stays sane
    log_var_x = torch.tanh(out[:, 2]) * 6.0
    log_var_y = torch.tanh(out[:, 3]) * 6.0
    return mu_x, mu_y, log_var_x, log_var_y


# ── Loss ──
def maybe_coarse_pool(out, gt, valid, coarse_h, coarse_w):
    """If (coarse_h, coarse_w) > 0, average-pool both prediction and GT to
    that resolution. Validity is max-pooled (a coarse cell is valid if any
    fine pixel within it is valid). Used for macro-only training where the
    model only needs to get cell-mean NGS coords right, not per-pixel."""
    if coarse_h <= 0 or coarse_w <= 0:
        return out, gt, valid
    out_c = F.adaptive_avg_pool2d(out, (coarse_h, coarse_w))
    gt_c = F.adaptive_avg_pool2d(gt, (coarse_h, coarse_w))
    valid_c = F.adaptive_max_pool2d(valid, (coarse_h, coarse_w))
    return out_c, gt_c, valid_c


def aleatoric_l1_loss(out, gt, valid):
    """Per-pixel aleatoric L1:
        L = |mu - gt| * exp(-log_var) + 0.5 * log_var
    Averaged over valid pixels."""
    mu_x, mu_y, lv_x, lv_y = split_outputs(out)
    gt_x = gt[:, 0]; gt_y = gt[:, 1]
    inv_var_x = torch.exp(-lv_x)
    inv_var_y = torch.exp(-lv_y)
    loss_x = torch.abs(mu_x - gt_x) * inv_var_x + 0.5 * lv_x
    loss_y = torch.abs(mu_y - gt_y) * inv_var_y + 0.5 * lv_y
    valid_s = valid.squeeze(1)
    loss = (loss_x + loss_y) * valid_s
    return loss.sum() / valid_s.sum().clamp(min=1)


def gaussian_blur_2d(x, sigma):
    """Apply a 2D Gaussian blur to (B, C, H, W) tensor `x` with current
    sigma (in cells). Re-normalized so a single point hit peaks at 1.0
    after blur (so binary GT stays in [0, 1] post-blur).
    """
    if sigma <= 0:
        return x
    # Build a 1D Gaussian kernel large enough to cover ±3σ.
    radius = max(1, int(3 * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1,
                          device=x.device, dtype=x.dtype)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()                       # 1D kernel, sums to 1
    # Separable conv: blur along W then along H. Per-channel (depthwise).
    C = x.shape[1]
    kx = g.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    out = F.conv2d(x, kx, padding=(0, radius), groups=C)
    out = F.conv2d(out, ky, padding=(radius, 0), groups=C)
    # Re-normalize so an isolated point hit (originally 1.0) peaks at 1.0
    # again after the blur. The peak weight after a separable Gaussian is
    # the product of center 1D weights = (1/sum)^2 of the actual peak ...
    # equivalently: g[radius]^2.
    peak = (g[radius]) ** 2
    out = out / max(peak.item(), 1e-9)
    return out.clamp(0.0, 1.0)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0,
                         reduction="mean"):
    """Sigmoid focal loss (Lin et al. 2017). For sparse-positive binary
    targets where vanilla BCE collapses to predicting all zeros.
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t weights positives vs negatives and (1-p_t)^gamma
    down-weights well-classified examples (which are mostly the
    abundant negatives).
    """
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets,
                                              reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ── Validation at coarse resolution (for coarse-loss training) ──
@torch.no_grad()
def val_metrics_coarse(model, loader, device, coarse_h, coarse_w):
    """Same as val_metrics but pools both pred and GT to (coarse_h, coarse_w)
    before measuring L1. Tells us how well the model predicts cell-mean NGS."""
    model.eval()
    sum_l1 = 0.0; sum_n = 0.0
    for batch in loader:
        x, gt, valid = batch[0], batch[1], batch[2]   # ignore aux_gt if present
        x = x.to(device); gt = gt.to(device); valid = valid.to(device)
        out = model(x)
        if out.shape[-2:] != gt.shape[-2:]:
            out = F.interpolate(out, size=gt.shape[-2:], mode="bilinear",
                                align_corners=False)
        out_c, gt_c, valid_c = maybe_coarse_pool(out, gt, valid, coarse_h, coarse_w)
        mu_x, mu_y, _, _ = split_outputs(out_c)
        err = torch.abs(mu_x - gt_c[:, 0]) + torch.abs(mu_y - gt_c[:, 1])
        valid_s = valid_c.squeeze(1)
        sum_l1 += (err * valid_s).sum().item()
        sum_n += valid_s.sum().item()
    return {"l1_yd": sum_l1 / max(1, sum_n)}


# ── Validation metric: per-pixel L1 in yards ──
@torch.no_grad()
def val_metrics(model, loader, device):
    model.eval()
    sum_l1 = 0.0
    sum_n = 0.0
    inlier_05 = 0.0
    inlier_1 = 0.0
    sum_log_var = 0.0
    for batch in loader:
        x, gt, valid = batch[0], batch[1], batch[2]   # ignore aux_gt if present
        x = x.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        out = model(x)
        # Output may be at lower resolution than GT — interpolate if needed
        if out.shape[-2:] != gt.shape[-2:]:
            out = F.interpolate(out, size=gt.shape[-2:], mode="bilinear",
                                align_corners=False)
        mu_x, mu_y, lv_x, lv_y = split_outputs(out)
        err_x = torch.abs(mu_x - gt[:, 0])
        err_y = torch.abs(mu_y - gt[:, 1])
        err = err_x + err_y
        valid_s = valid.squeeze(1)
        n = valid_s.sum().item()
        sum_l1 += (err * valid_s).sum().item()
        inlier_05 += (((err_x < 0.5) & (err_y < 0.5)).float() * valid_s).sum().item()
        inlier_1 += (((err_x < 1.0) & (err_y < 1.0)).float() * valid_s).sum().item()
        sum_log_var += ((lv_x + lv_y) * 0.5 * valid_s).sum().item()
        sum_n += n
    if sum_n == 0:
        return {}
    return {
        "l1_yd": sum_l1 / sum_n,
        "inlier_lt_0.5yd": inlier_05 / sum_n,
        "inlier_lt_1yd": inlier_1 / sum_n,
        "mean_log_var": sum_log_var / sum_n,
    }


# ── Train/val split: hold out one full game ──
def split_by_game(entries, val_game=None, val_frac=0.1, seed=42):
    by_game = defaultdict(list)
    for e in entries:
        by_game[e["clip"].split("/")[0]].append(e)
    games = sorted(by_game.keys())
    if val_game is None:
        # Pick a single game to hold out — the smallest-but-not-tiny one
        sizes = sorted([(len(by_game[g]), g) for g in games])
        target = int(len(entries) * val_frac)
        # Smallest game whose size is at least target
        chosen = next((g for sz, g in sizes if sz >= target), sizes[-1][1])
        val_game = chosen
    train, val = [], []
    for g in games:
        (val if g == val_game else train).extend(by_game[g])
    return train, val, val_game


# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=os.path.join(PROJECT_ROOT, "output/dense_field_pool"))
    ap.add_argument("--manifest-file", default=None,
                    help="Path to a pre-filtered manifest JSON with 'entries' "
                         "list. Bypasses --pool-dir's decisions.json filter. "
                         "E.g. data/h_pool_and_intrinsics.json "
                         "(2500 H-verified frames).")
    ap.add_argument("--clips-root", default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "models/dense_regression_v1"))
    ap.add_argument("--encoder", default="mit_b1",
                    choices=["mit_b0", "mit_b1", "mit_b2", "mit_b3"])
    ap.add_argument("--decoder", default="segformer",
                    choices=["unet", "segformer"],
                    help="SegFormer's MLP decoder (default) is the right "
                         "choice for dense regression: output is smooth, the "
                         "MLP head mixes features from all 4 encoder stages "
                         "(wide receptive field for triangulating across "
                         "sparse markers), and it natively outputs at 1/4 "
                         "resolution (matches GT). U-Net is provided as an "
                         "alternative but not recommended here.")
    ap.add_argument("--decoder-seg-channels", type=int, default=256,
                    help="Width of the SegFormer decoder (4 MLP projections "
                         "+ concat conv). Default 256 (smp default). Bump "
                         "to 512/768/1024 for more decoder capacity. "
                         "Ignored when --decoder unet.")
    ap.add_argument("--aux-seg-channels", type=int, default=128,
                    help="Width of the aux SegFormer decoder. Default 128.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"],
                    help="Mixed-precision mode for the train forward+loss "
                         "(bf16 default — no GradScaler needed, ~2× speedup "
                         "on RTX 5090). Set 'off' to debug numerical issues.")
    ap.add_argument("--persistent-workers", action="store_true",
                    default=True,
                    help="Keep DataLoader worker processes alive across "
                         "epochs (avoids re-init overhead).")
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="DataLoader prefetch_factor (samples per worker).")
    ap.add_argument("--val-game", default=None,
                    help="Game id to hold out. Default: auto-pick.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cache", action="store_true",
                    help="Skip the precache step (assume cache is already populated).")
    ap.add_argument("--cache-only", action="store_true",
                    help="Run the precache step and exit (no training). "
                         "Useful for building the cache locally before "
                         "shipping to a training pod.")
    ap.add_argument("--init-from", default=None,
                    help="Path to a checkpoint to load weights from before "
                         "training starts (for resuming after a crash). "
                         "Loads model weights only — optimizer state, epoch "
                         "counter, and LR schedule are reset.")
    ap.add_argument("--input-mode", default="v1", choices=["v1", "v2"],
                    help="v1: 3 RGB + 4 v8 sigmoid prob channels (legacy). "
                         "v2: 3 RGB + 3 binary masks + 1 number-NGS_x label.")
    ap.add_argument("--v2-input-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression_v2_inputs"),
                    help="Per-frame number-NGS_x cache for input_mode=v2.")
    ap.add_argument("--loss-coarse-h", type=int, default=0,
                    help="If >0, avg-pool BOTH pred and GT to (loss_coarse_h, "
                         "loss_coarse_w) before computing aleatoric loss. Used "
                         "for coarse-resolution macro-only training (phase 1).")
    ap.add_argument("--loss-coarse-w", type=int, default=0)
    ap.add_argument("--aux-head", action="store_true",
                    help="Add an auxiliary image-space 'canonical-feature' "
                         "prediction head (Option B1). Forces encoder to "
                         "learn structural field facts via a SegFormer "
                         "decoder predicting where features SHOULD be.")
    ap.add_argument("--aux-gt-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression_canonical_aux"),
                    help="Per-frame canonical-NGS-space aux GT cache "
                         "(4 x AUX_OUT_H x AUX_OUT_W uint8). Built by "
                         "build_canonical_ngs_aux_gt.py — projects each "
                         "v8 specialist mask pixel through "
                         "(cv2.undistortPoints + H) into a top-down NGS "
                         "grid. Replaces the older image-space aux GT.")
    ap.add_argument("--aux-loss-weight", type=float, default=0.5,
                    help="Weight on the aux loss (relative to main).")
    ap.add_argument("--aux-loss-type", default="bce",
                    choices=["bce", "focal"],
                    help="bce (default, all-zero collapse on sparse "
                         "positives) or focal (sigmoid focal, robust "
                         "to class imbalance).")
    ap.add_argument("--focal-alpha", type=float, default=0.25,
                    help="alpha in sigmoid focal loss (positive class "
                         "weight). Default 0.25.")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="gamma in sigmoid focal loss (down-weights "
                         "well-classified examples). Default 2.0.")
    ap.add_argument("--aux-blur-sigma-start", type=float, default=3.0,
                    help="Aux GT Gaussian blur σ at epoch 0 (in NGS cells). "
                         "Broader blobs early give the model gradient "
                         "toward near-misses. Set 0 to disable curriculum.")
    ap.add_argument("--aux-blur-sigma-end", type=float, default=0.0,
                    help="Aux GT blur σ at the final epoch. 0.0 = the "
                         "original binary point GT (the actual task). "
                         "Curriculum tightens precision by the end.")
    ap.add_argument("--aux-pos-weights", default="11.6,202,269,94",
                    help="Comma-separated per-channel pos_weight for BCE "
                         "aux loss (yard, side, hash, num). Computed as "
                         "(1-p)/p from binary positive rates (yard 7.9%%, "
                         "side 0.49%%, hash 0.37%%, num 1.05%%). Balances "
                         "gradient so sparse hash/num classes don't "
                         "collapse to predicting all zeros. Set to 'none' "
                         "to disable per-class weighting.")
    ap.add_argument("--aux-blur-decay", default="linear",
                    choices=["linear", "quadratic", "cubic"],
                    help="Curriculum sigma decay shape. Linear: equal time "
                         "at each sigma. Quadratic / cubic: drops fast early, "
                         "spends more epochs near sigma_end (good when most "
                         "learning happens near binary).")
    ap.add_argument("--main-loss-weight", type=float, default=1.0,
                    help="Weight on main aleatoric L1 loss. Set to 0 for "
                         "aux-only pretraining (skips main forward to save "
                         "compute).")
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="eta_min for CosineAnnealingLR. 0 (default) ends "
                         "at LR=0; set to ~1e-5 for fine-tuning regimes "
                         "where you want non-zero LR through all epochs.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load manifest ──
    if args.manifest_file:
        manifest = json.load(open(args.manifest_file))
        entries = manifest["entries"]
        print(f"Manifest: {args.manifest_file} → {len(entries)} entries "
              f"(pre-filtered, no decisions.json applied)")
    else:
        manifest = json.load(open(os.path.join(args.pool_dir, "manifest.json")))
        decisions_path = os.path.join(args.pool_dir, "decisions.json")
        decisions = json.load(open(decisions_path)) if os.path.exists(decisions_path) else {}
        print(f"Manifest: {len(manifest['entries'])} entries, "
              f"decisions: {len(decisions)}")
        entries = [e for e in manifest["entries"] if decisions.get(e["id"]) == "y"]
        print(f"After Y/N filter: {len(entries)} accepted entries")
    if not entries:
        sys.exit("No entries to process.")

    # ── Per-clip intrinsics (K, dist) — required to compute correct GT ──
    intrinsics_by_clip = manifest.get("intrinsics_by_clip", {})
    if not intrinsics_by_clip:
        print("WARNING: manifest has no 'intrinsics_by_clip' — GT will be "
              "computed without lens undistortion. Run "
              "scripts/data_prep/inject_intrinsics_into_manifest.py to fix.")
    else:
        print(f"Loaded intrinsics for {len(intrinsics_by_clip)} clips")

    # ── Train/val split ──
    train_entries, val_entries, val_game = split_by_game(
        entries, val_game=args.val_game, val_frac=0.1, seed=args.seed)
    print(f"Split: train={len(train_entries)}  val={len(val_entries)}  "
          f"(val game = {val_game})")

    # ── Cache specialist outputs once per entry ──
    if not args.skip_cache:
        print(f"\nPre-caching specialists for {len(entries)} entries -> {args.cache_dir}")
        precache_dataset(entries, args.clips_root, args.cache_dir, args.device)
    if args.cache_only:
        print("--cache-only set. Done.")
        return

    # ── Datasets + loaders ──
    aux_gt_dir = args.aux_gt_dir if args.aux_head else None
    train_ds = DenseFieldDataset(train_entries, args.clips_root, args.cache_dir,
                                   augment=True, input_mode=args.input_mode,
                                   v2_input_dir=args.v2_input_dir,
                                   aux_gt_dir=aux_gt_dir)
    val_ds = DenseFieldDataset(val_entries, args.clips_root, args.cache_dir,
                                 augment=False, input_mode=args.input_mode,
                                 v2_input_dir=args.v2_input_dir,
                                 aux_gt_dir=aux_gt_dir)
    print(f"Input mode: {args.input_mode}")
    if args.loss_coarse_h > 0 and args.loss_coarse_w > 0:
        print(f"Coarse loss: avg-pool to {args.loss_coarse_h}x{args.loss_coarse_w}")
    if args.aux_head:
        print(f"Aux head ENABLED: canonical-NGS-space {AUX_OUT_H}x{AUX_OUT_W} "
              f"BCE × weight {args.aux_loss_weight}")
    persistent = args.persistent_workers and args.num_workers > 0
    pf = args.prefetch_factor if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=pf)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=pf)

    # ── Model + optim ──
    device = torch.device(args.device)
    if args.aux_head:
        model = DenseRegressionWithAux(
            encoder_name=args.encoder,
            in_channels=7,
            decoder=args.decoder,
            decoder_segmentation_channels=args.decoder_seg_channels,
            aux_segmentation_channels=args.aux_seg_channels,
        ).to(device)
    else:
        model = build_model(
            encoder_name=args.encoder, in_channels=7, classes=4,
            decoder=args.decoder,
            decoder_segmentation_channels=args.decoder_seg_channels,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    aux_str = " (+ aux head)" if args.aux_head else ""
    print(f"\nModel: {args.decoder.upper()}({args.encoder}, in_ch=7, "
          f"out_ch=4){aux_str} — {n_params:.1f}M params")

    if args.init_from:
        print(f"Loading init weights from {args.init_from} ...")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # If we're loading a vanilla SegFormer/Unet checkpoint into a
        # DenseRegressionWithAux wrapper, prepend 'main.' to keys so they
        # land in the .main submodule. The aux_decoder + aux_head will be
        # randomly initialized (via strict=False).
        if isinstance(model, DenseRegressionWithAux):
            sample_key = next(iter(sd.keys()))
            if not sample_key.startswith("main."):
                sd = {f"main.{k}": v for k, v in sd.items()}
                strict = False
                print("  remapped vanilla checkpoint to .main submodule "
                      "(aux head will be randomly initialized)")
            else:
                strict = True
        else:
            strict = True
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if missing:
            print(f"  missing keys: {len(missing)} (e.g. {missing[0] if missing else None})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
        print(f"  loaded (epoch={ckpt.get('epoch', '?')}, "
              f"val_l1={ckpt.get('val_l1_yd', '?')})")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    # ── AMP setup ──
    # bf16 has the same exponent range as fp32 (no overflow risk → no
    # GradScaler needed). RTX 5090 (Blackwell) has dedicated bf16 throughput,
    # so this is roughly 2× speedup over fp32 with no convergence concerns.
    use_amp = (args.amp != "off") and (args.device == "cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    print(f"AMP: {'bf16' if (use_amp and amp_dtype==torch.bfloat16) else ('fp16' if use_amp else 'off')}")

    # ── Per-channel pos_weight for BCE aux (handles class imbalance) ──
    pos_weights_t = None
    if args.aux_head and args.aux_loss_type == "bce" and \
            args.aux_pos_weights.lower() not in ("", "none", "off"):
        try:
            pw = [float(x) for x in args.aux_pos_weights.split(",")]
            assert len(pw) == AUX_CHANNELS, \
                f"--aux-pos-weights expects {AUX_CHANNELS} comma-separated"
            pos_weights_t = torch.tensor(pw, device=device,
                                          dtype=torch.float32).view(1, -1, 1, 1)
        except Exception as e:
            print(f"WARNING: failed to parse --aux-pos-weights: {e}; "
                  f"falling back to unweighted BCE.")

    # ── Train ──
    best_val_l1 = float("inf")
    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()    # truncate
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        aux_loss_sum = 0.0
        n_batches = 0
        # Curriculum sigma for aux GT.
        # Linear: equal time at each sigma. Quadratic / cubic: drops fast
        # early and spends more epochs near sigma_end (good when the
        # near-binary regime is where actual learning happens).
        if args.epochs > 1:
            t_frac = epoch / max(1, args.epochs - 1)
        else:
            t_frac = 1.0
        if args.aux_blur_decay == "quadratic":
            f = (1 - t_frac) ** 2
        elif args.aux_blur_decay == "cubic":
            f = (1 - t_frac) ** 3
        else:    # linear
            f = 1 - t_frac
        aux_blur_sigma = (args.aux_blur_sigma_start * f +
                            args.aux_blur_sigma_end * (1 - f))
        if args.aux_head and epoch == 0:
            print(f"Aux blur curriculum ({args.aux_blur_decay}): sigma "
                  f"{args.aux_blur_sigma_start} → {args.aux_blur_sigma_end} "
                  f"over {args.epochs} epochs")
            if pos_weights_t is not None:
                print(f"Aux pos_weight (yard, side, hash, num): "
                      f"{pos_weights_t.flatten().tolist()}")
        if epoch == 0:
            print(f"Main loss weight: {args.main_loss_weight}  "
                  f"Aux loss weight: {args.aux_loss_weight}  "
                  f"LR: {args.lr} → {args.lr_min} (cosine)")
        for batch in train_loader:
            if args.aux_head:
                x, gt, valid, aux_gt = batch
                aux_gt = aux_gt.to(device, non_blocking=True)
            else:
                x, gt, valid = batch
            x = x.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if args.aux_head:
                    out, aux_out = model(x, return_aux=True)
                else:
                    out = model(x)
                # Main loss (only computed if its weight > 0; saves compute
                # in aux-only pretraining mode where main_loss_weight=0)
                if args.main_loss_weight > 0:
                    if out.shape[-2:] != gt.shape[-2:]:
                        out_resized = F.interpolate(
                            out, size=gt.shape[-2:],
                            mode="bilinear", align_corners=False)
                    else:
                        out_resized = out
                    out_loss, gt_loss, valid_loss = maybe_coarse_pool(
                        out_resized, gt, valid,
                        args.loss_coarse_h, args.loss_coarse_w)
                    main_loss = aleatoric_l1_loss(
                        out_loss, gt_loss, valid_loss)
                    loss = args.main_loss_weight * main_loss
                else:
                    # Tensor zero on the device, requires_grad-free.
                    loss = torch.zeros((), device=device)
                if args.aux_head:
                    # Apply current curriculum blur to the binary aux GT.
                    # Done in fp32 outside autocast for numerical safety.
                    aux_gt_blurred = gaussian_blur_2d(
                        aux_gt.float(), aux_blur_sigma)
                    if args.aux_loss_type == "focal":
                        aux_loss = sigmoid_focal_loss(
                            aux_out, aux_gt_blurred,
                            alpha=args.focal_alpha, gamma=args.focal_gamma)
                    else:
                        aux_loss = F.binary_cross_entropy_with_logits(
                            aux_out, aux_gt_blurred,
                            pos_weight=pos_weights_t)
                    loss = loss + args.aux_loss_weight * aux_loss
                    aux_loss_sum += aux_loss.item()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item()
            n_batches += 1
        sched.step()
        train_loss = loss_sum / max(1, n_batches)
        elapsed = time.time() - t0

        avg_aux = aux_loss_sum / max(1, n_batches)
        # ── Val ── (always at fine resolution; coarse loss is just for training)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            v = val_metrics(model, val_loader, device)
            coarse_str = ""
            if args.loss_coarse_h > 0:
                v_coarse = val_metrics_coarse(
                    model, val_loader, device,
                    args.loss_coarse_h, args.loss_coarse_w)
                coarse_str = f" | coarse({args.loss_coarse_h}x{args.loss_coarse_w}) L1={v_coarse['l1_yd']:.3f}yd"
                v.update({f"coarse_{k}": vv for k, vv in v_coarse.items()})
        aux_str = f" | aux={avg_aux:.4f}" if args.aux_head else ""
        print(f"Epoch {epoch+1:2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val L1={v.get('l1_yd', float('nan')):.3f}yd  "
              f"<.5yd={v.get('inlier_lt_0.5yd', 0)*100:.1f}%  "
              f"<1yd={v.get('inlier_lt_1yd', 0)*100:.1f}%"
              f"{coarse_str}{aux_str}  "
              f"({elapsed:.0f}s)", flush=True)

        with open(log_path, "a") as f:
            json.dump({"epoch": epoch+1, "train_loss": train_loss, **v,
                       "lr": sched.get_last_lr()[0]}, f)
            f.write("\n")

        # ── Checkpoint ──
        ckpt = {
            "model_state_dict": model.state_dict(),
            "encoder": args.encoder,
            "in_channels": 7, "classes": 4,
            "epoch": epoch + 1,
            "val_l1_yd": v.get("l1_yd"),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if v.get("l1_yd", float("inf")) < best_val_l1:
            best_val_l1 = v["l1_yd"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"  ↑ new best (val L1 = {best_val_l1:.3f}yd)")

    print(f"\nDone. Best val L1: {best_val_l1:.3f} yd")
    print(f"Checkpoints + log -> {args.out_dir}")


if __name__ == "__main__":
    main()
