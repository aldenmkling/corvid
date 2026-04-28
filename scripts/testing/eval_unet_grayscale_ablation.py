#!/usr/bin/env python3
"""Inference-only ablation: how much does the existing RGB-trained UNet
suffer when fed grayscale-replicated input? Tells us how much the model
actually relies on color (without retraining).

Loads models/unet_line_round3_best.pth, runs both RGB and grayscale inputs
on the line val set, reports per-class F1.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, grayscale: bool):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.stack([gray, gray, gray], axis=-1)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def gt_masks(mask_path):
    """Returns 2-channel binary mask (yard, side) at INPUT_H × INPUT_W."""
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    yard = (mask[..., 2] > 127).astype(np.uint8)   # R channel
    side = (mask[..., 1] > 127).astype(np.uint8)   # G channel
    return yard, side


def f1_at(pred_logits, gt_yard, gt_side, thresh=0.5):
    """Pixel-level F1 per channel."""
    prob = torch.sigmoid(pred_logits)[0].cpu().numpy()  # (2, H, W)
    pred_y = (prob[0] > thresh).astype(np.uint8)
    pred_s = (prob[1] > thresh).astype(np.uint8)
    out = {}
    for name, p, g in [("yard", pred_y, gt_yard), ("side", pred_s, gt_side)]:
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        out[name] = (P, R, F1)
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None,
                     in_channels=3, classes=2)
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    img_dir = os.path.join(VAL_DIR, "images")
    mask_dir = os.path.join(VAL_DIR, "masks")
    fids = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if f.endswith(".jpg") and not f.startswith("._")])
    print(f"  val frames: {len(fids)}")

    agg = {"rgb": {"yard": [0, 0, 0], "side": [0, 0, 0]},
           "gray": {"yard": [0, 0, 0], "side": [0, 0, 0]}}

    with torch.no_grad():
        for fid in fids:
            img_bgr = cv2.imread(os.path.join(img_dir, f"{fid}.jpg"))
            if img_bgr is None: continue
            gt_y, gt_s = gt_masks(os.path.join(mask_dir, f"{fid}.png"))
            for mode, is_gray in [("rgb", False), ("gray", True)]:
                tensor = preprocess(img_bgr, is_gray).to(device)
                logits = model(tensor)
                f1 = f1_at(logits, gt_y, gt_s)
                for cls in ("yard", "side"):
                    p, r, _ = f1[cls]
                    # Aggregate TP/FP/FN at pixel level for global F1
                    pass

    # Re-run aggregating TP/FP/FN globally, not averaging per-frame.
    tot = {"rgb": {"yard": [0, 0, 0], "side": [0, 0, 0]},
           "gray": {"yard": [0, 0, 0], "side": [0, 0, 0]}}
    with torch.no_grad():
        for fid in fids:
            img_bgr = cv2.imread(os.path.join(img_dir, f"{fid}.jpg"))
            if img_bgr is None: continue
            gt_y, gt_s = gt_masks(os.path.join(mask_dir, f"{fid}.png"))
            for mode, is_gray in [("rgb", False), ("gray", True)]:
                tensor = preprocess(img_bgr, is_gray).to(device)
                logits = model(tensor)
                prob = torch.sigmoid(logits)[0].cpu().numpy()
                pred_y = (prob[0] > 0.5).astype(np.uint8)
                pred_s = (prob[1] > 0.5).astype(np.uint8)
                for cls, p, g in [("yard", pred_y, gt_y), ("side", pred_s, gt_s)]:
                    tot[mode][cls][0] += int(((p == 1) & (g == 1)).sum())
                    tot[mode][cls][1] += int(((p == 1) & (g == 0)).sum())
                    tot[mode][cls][2] += int(((p == 0) & (g == 1)).sum())

    print(f"\n  {'mode':>5} {'class':>5}  {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")
    for mode in ("rgb", "gray"):
        for cls in ("yard", "side"):
            tp, fp, fn = tot[mode][cls]
            P = tp / max(tp + fp, 1)
            R = tp / max(tp + fn, 1)
            F1 = 2 * P * R / max(P + R, 1e-9)
            print(f"  {mode:>5} {cls:>5}  {P:>6.3f} {R:>6.3f} {F1:>6.3f}")


if __name__ == "__main__":
    main()
