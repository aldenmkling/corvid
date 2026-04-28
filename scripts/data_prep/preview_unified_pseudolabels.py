#!/usr/bin/env python3
"""Sanity-check cross-predicted pseudo-labels for the unified dataset.

For a sample of frames from each side of the cross-prediction:
  - hash dataset frame: real hash mask + pseudo yard/side from line UNet
  - line dataset frame: real yard/side + pseudo hash from hash UNet
Builds an overlay panel showing source + composited 3-channel mask
(yard=red, side=green, hash=blue) per frame.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import segmentation_models_pytorch as smp

INPUT_H, INPUT_W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LINE_THRESH_YARD = 0.5
LINE_THRESH_SIDE = 0.5
HASH_THRESH = 0.5


def preprocess(img_bgr, grayscale: bool):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_W, INPUT_H))
    if grayscale:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([g, g, g], axis=-1)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def load_unet(weights, n_classes, device):
    m = smp.Unet("mit_b0", encoder_weights=None, classes=n_classes, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return m.to(device).eval()


def predict_line(model, img, device):
    with torch.no_grad():
        t = preprocess(img, grayscale=True).to(device)
        p = torch.sigmoid(model(t))[0].cpu().numpy()       # (2, H, W)
    yard = (p[0] > LINE_THRESH_YARD).astype(np.uint8)
    side = (p[1] > LINE_THRESH_SIDE).astype(np.uint8)
    h, w = img.shape[:2]
    yard = cv2.resize(yard, (w, h), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w, h), interpolation=cv2.INTER_NEAREST)
    return yard, side


def predict_hash(model, img, device):
    with torch.no_grad():
        t = preprocess(img, grayscale=False).to(device)
        p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    h, w = img.shape[:2]
    mask = (p > HASH_THRESH).astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_3ch(img, yard, side, hash_, alpha=0.55):
    """Red=yard, Green=side, Blue=hash."""
    out = img.copy().astype(np.float32)
    layers = [
        (yard,  np.array([60, 60, 230],   dtype=np.float32)),  # red (BGR)
        (side,  np.array([60, 230, 60],   dtype=np.float32)),  # green
        (hash_, np.array([230, 60, 60],   dtype=np.float32)),  # blue
    ]
    for m, c in layers:
        if m is None: continue
        mask = m > 0
        out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_mit_b0_gray_best.pth"))
    ap.add_argument("--hash-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_hash_round3_last.pth"))
    ap.add_argument("--hash-dataset", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/round3/train"))
    ap.add_argument("--line-dataset", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/train"))
    ap.add_argument("--out-panel", default=os.path.join(
        PROJECT_ROOT, "output/unified_pseudolabels_preview.jpg"))
    ap.add_argument("--n-each", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    line_model = load_unet(args.line_weights, n_classes=2, device=device)
    hash_model = load_unet(args.hash_weights, n_classes=1, device=device)

    rows = []

    # Side A: hash frames → real hash + pseudo line
    hash_imgs = sorted([f for f in os.listdir(os.path.join(args.hash_dataset, "images"))
                         if f.endswith(".jpg") and not f.startswith("._")])
    for fname in hash_imgs[::max(1, len(hash_imgs) // args.n_each)][:args.n_each]:
        img = cv2.imread(os.path.join(args.hash_dataset, "images", fname))
        stem = os.path.splitext(fname)[0]
        real_hash = cv2.imread(os.path.join(args.hash_dataset, "masks", stem + ".png"),
                                cv2.IMREAD_GRAYSCALE)
        real_hash = (real_hash > 127).astype(np.uint8)
        pseudo_yard, pseudo_side = predict_line(line_model, img, device)
        ov = overlay_3ch(img, pseudo_yard, pseudo_side, real_hash)
        rows.append(("hash→pseudo line", fname, img, ov))

    # Side B: line frames → real line + pseudo hash
    line_imgs = sorted([f for f in os.listdir(os.path.join(args.line_dataset, "images"))
                         if f.endswith(".jpg") and not f.startswith("._")])
    for fname in line_imgs[::max(1, len(line_imgs) // args.n_each)][:args.n_each]:
        img = cv2.imread(os.path.join(args.line_dataset, "images", fname))
        stem = os.path.splitext(fname)[0]
        line_mask_bgr = cv2.imread(os.path.join(args.line_dataset, "masks", stem + ".png"))
        # B=empty, G=side, R=yard (per train_unet_lines docstring)
        real_yard = (line_mask_bgr[..., 2] > 127).astype(np.uint8)
        real_side = (line_mask_bgr[..., 1] > 127).astype(np.uint8)
        pseudo_hash = predict_hash(hash_model, img, device)
        ov = overlay_3ch(img, real_yard, real_side, pseudo_hash)
        rows.append(("line→pseudo hash", fname, img, ov))

    # Build panel: source | overlay per row, height-normalized
    target_h = 280
    cells_rows = []
    for label, fname, src, ov in rows:
        s = target_h / src.shape[0]
        src_r = cv2.resize(src, (int(src.shape[1] * s), target_h))
        ov_r = cv2.resize(ov, (int(ov.shape[1] * s), target_h))
        row = np.hstack([src_r, ov_r])
        hdr = np.full((24, row.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(hdr, f"[{label}] {fname}  src | overlay (yard=R, side=G, hash=B)",
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cells_rows.append(np.vstack([hdr, row]))

    max_w = max(r.shape[1] for r in cells_rows)
    padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), mode="constant")
                for r in cells_rows]
    grid = np.vstack(padded)
    os.makedirs(os.path.dirname(args.out_panel), exist_ok=True)
    cv2.imwrite(args.out_panel, grid)
    print(f"  wrote {args.out_panel}  shape={grid.shape}")


if __name__ == "__main__":
    main()
