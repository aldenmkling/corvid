#!/usr/bin/env python3
"""Run the trained UNet on a handful of val frames and save side-by-side
[prediction | ground-truth] overlays so we can eyeball where the model
is getting things right/wrong."""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
OUT_DIR = os.path.join(PROJECT_ROOT, "output/unet_predictions")


def load_model(device):
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=2, activation=None)
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()


def preprocess(img_bgr):
    """Return (tensor, original_shape)."""
    h0, w0 = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normed, (2, 0, 1))).unsqueeze(0).float()
    return tensor, (h0, w0)


def make_overlay(frame, yard_mask, side_mask):
    """Blend cyan (yard) and yellow (side) onto a copy of the BGR frame."""
    overlay = frame.copy()
    color = np.zeros_like(overlay)
    color[yard_mask > 0] = (255, 255, 0)   # cyan
    color[side_mask > 0] = (0, 255, 255)   # yellow
    any_m = (yard_mask > 0) | (side_mask > 0)
    overlay[any_m] = cv2.addWeighted(overlay, 0.3, color, 0.7, 0)[any_m]
    return overlay


def label_strip(w, text, h=32):
    """A simple label banner to stick above each image."""
    strip = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


@torch.no_grad()
def run_one(model, device, img_path, mask_path, out_path, thresh=0.5):
    frame = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path)
    # Builder uses BGR channel 2 = yard (R), 1 = side (G)
    gt_yard = (gt_mask[..., 2] > 127).astype(np.uint8) * 255
    gt_side = (gt_mask[..., 1] > 127).astype(np.uint8) * 255

    tensor, (h0, w0) = preprocess(frame)
    logits = model(tensor.to(device))
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # (2, H, W)
    pred_yard = (probs[0] > thresh).astype(np.uint8) * 255
    pred_side = (probs[1] > thresh).astype(np.uint8) * 255
    pred_yard = cv2.resize(pred_yard, (w0, h0), interpolation=cv2.INTER_NEAREST)
    pred_side = cv2.resize(pred_side, (w0, h0), interpolation=cv2.INTER_NEAREST)

    pred_vis = make_overlay(frame, pred_yard, pred_side)
    gt_vis = make_overlay(frame, gt_yard, gt_side)

    strip_h = 32
    top = np.hstack([label_strip(w0, "PREDICTION", strip_h),
                     label_strip(w0, "GROUND TRUTH", strip_h)])
    bottom = np.hstack([pred_vis, gt_vis])
    combined = np.vstack([top, bottom])
    cv2.imwrite(out_path, combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="frames to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")
    model = load_model(device)

    images = sorted(f for f in os.listdir(os.path.join(VAL_DIR, "images"))
                    if f.endswith(".jpg") and not f.startswith("._"))
    rng = random.Random(args.seed)
    rng.shuffle(images)
    picks = images[: args.n]

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in picks:
        fid = os.path.splitext(fname)[0]
        img_path = os.path.join(VAL_DIR, "images", fname)
        mask_path = os.path.join(VAL_DIR, "masks", f"{fid}.png")
        out_path = os.path.join(OUT_DIR, f"{fid}__pred_vs_gt.jpg")
        run_one(model, device, img_path, mask_path, out_path, args.thresh)
        print(f"  {fid}")

    print(f"\nsaved {len(picks)} comparisons to {OUT_DIR}/")


if __name__ == "__main__":
    main()
