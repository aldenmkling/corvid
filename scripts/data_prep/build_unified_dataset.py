#!/usr/bin/env python3
"""Merge the line + hash datasets into a unified 3-channel mask dataset.

For each frame, fill missing channels via cross-prediction:
  - hash dataset frame  (real hash)  → run line UNet for pseudo yard/side
  - line dataset frame  (real line)  → run hash UNet for pseudo hash

Output mask format (BGR PNG, same convention as build_line_dataset):
   B = hash  (formerly empty channel — now used for hash)
   G = side
   R = yard

Output:
  data/unified/train/{images,masks}/   ← 906 = 650 + 256
  data/unified/valid/{images,masks}/   ← 65 (line val + pseudo hash)
"""

import argparse
import json
import os
import shutil
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


@torch.no_grad()
def predict_line(model, img, device):
    t = preprocess(img, grayscale=True).to(device)
    p = torch.sigmoid(model(t))[0].cpu().numpy()
    yard = (p[0] > LINE_THRESH_YARD).astype(np.uint8)
    side = (p[1] > LINE_THRESH_SIDE).astype(np.uint8)
    h, w = img.shape[:2]
    yard = cv2.resize(yard, (w, h), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w, h), interpolation=cv2.INTER_NEAREST)
    return yard, side


@torch.no_grad()
def predict_hash(model, img, device):
    t = preprocess(img, grayscale=False).to(device)
    p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    h, w = img.shape[:2]
    return cv2.resize((p > HASH_THRESH).astype(np.uint8),
                      (w, h), interpolation=cv2.INTER_NEAREST)


def stack_bgr(yard, side, hash_):
    """Encode 3 binary masks as a BGR PNG: B=hash, G=side, R=yard."""
    h, w = yard.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 0] = hash_ * 255
    out[..., 1] = side * 255
    out[..., 2] = yard * 255
    return out


def process_hash_frame(fname, hash_dir, line_model, device):
    """Hash dataset frame → real hash + pseudo line."""
    img = cv2.imread(os.path.join(hash_dir, "images", fname))
    stem = os.path.splitext(fname)[0]
    real_hash = cv2.imread(os.path.join(hash_dir, "masks", stem + ".png"),
                            cv2.IMREAD_GRAYSCALE)
    real_hash = (real_hash > 127).astype(np.uint8)
    yard, side = predict_line(line_model, img, device)
    return img, stack_bgr(yard, side, real_hash)


def process_line_frame(fname, line_dir, hash_model, device):
    """Line dataset frame → real line + pseudo hash."""
    img = cv2.imread(os.path.join(line_dir, "images", fname))
    stem = os.path.splitext(fname)[0]
    line_bgr = cv2.imread(os.path.join(line_dir, "masks", stem + ".png"))
    real_yard = (line_bgr[..., 2] > 127).astype(np.uint8)
    real_side = (line_bgr[..., 1] > 127).astype(np.uint8)
    pseudo_hash = predict_hash(hash_model, img, device)
    return img, stack_bgr(real_yard, real_side, pseudo_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_mit_b0_gray_best.pth"))
    ap.add_argument("--hash-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_hash_round3_last.pth"))
    ap.add_argument("--hash-train", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/round3/train"))
    ap.add_argument("--line-train", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/train"))
    ap.add_argument("--line-valid", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/valid"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/unified"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    line_model = load_unet(args.line_weights, n_classes=2, device=device)
    hash_model = load_unet(args.hash_weights, n_classes=1, device=device)

    out_train_img = os.path.join(args.out_dir, "train/images")
    out_train_msk = os.path.join(args.out_dir, "train/masks")
    out_valid_img = os.path.join(args.out_dir, "valid/images")
    out_valid_msk = os.path.join(args.out_dir, "valid/masks")
    for d in (out_train_img, out_train_msk, out_valid_img, out_valid_msk):
        os.makedirs(d, exist_ok=True)

    n_hash = n_line = n_val = 0

    # --- TRAIN: hash frames ---
    hash_imgs = sorted([f for f in os.listdir(os.path.join(args.hash_train, "images"))
                         if f.endswith(".jpg") and not f.startswith("._")])
    print(f"  processing {len(hash_imgs)} hash-side train frames...")
    for fname in hash_imgs:
        img, mask_bgr = process_hash_frame(fname, args.hash_train, line_model, device)
        cv2.imwrite(os.path.join(out_train_img, fname), img)
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_train_msk, stem + ".png"), mask_bgr)
        n_hash += 1

    # --- TRAIN: line frames ---
    line_imgs = sorted([f for f in os.listdir(os.path.join(args.line_train, "images"))
                         if f.endswith(".jpg") and not f.startswith("._")])
    print(f"  processing {len(line_imgs)} line-side train frames...")
    for fname in line_imgs:
        img, mask_bgr = process_line_frame(fname, args.line_train, hash_model, device)
        cv2.imwrite(os.path.join(out_train_img, fname), img)
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_train_msk, stem + ".png"), mask_bgr)
        n_line += 1

    # --- VALID: line valid frames (real line + pseudo hash) ---
    valid_imgs = sorted([f for f in os.listdir(os.path.join(args.line_valid, "images"))
                          if f.endswith(".jpg") and not f.startswith("._")])
    print(f"  processing {len(valid_imgs)} line-valid frames...")
    for fname in valid_imgs:
        img, mask_bgr = process_line_frame(fname, args.line_valid, hash_model, device)
        cv2.imwrite(os.path.join(out_valid_img, fname), img)
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_valid_msk, stem + ".png"), mask_bgr)
        n_val += 1

    print()
    print(f"  train: {n_hash + n_line}  ({n_hash} hash-side + {n_line} line-side)")
    print(f"  valid: {n_val}  (line-valid + pseudo hash)")
    print(f"  out:   {args.out_dir}/")


if __name__ == "__main__":
    main()
