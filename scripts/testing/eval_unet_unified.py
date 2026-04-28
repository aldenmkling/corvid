#!/usr/bin/env python3
"""Compare the unified 3-channel UNet against specialist models.

Reports:
  - Line pixel F1 (yard + side) on data/line_detection/valid (vs real GT)
    against the line-specialist baseline.
  - Hash keypoint F1 on data/field_keypoints/valid (vs real GT keypoints)
    against the hash-specialist baseline. Match radius = 8 px in input
    coords (~11 image px).

Run after training completes:
  .venv/bin/python scripts/testing/eval_unet_unified.py \\
    --weights models/unet_unified_best.pth
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

INPUT_H, INPUT_W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HASH_MATCH_RADIUS = 8.0       # input-coord px (~11 image px = hash-mark size)
MIN_CC_AREA = 4


def preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_W, INPUT_H))
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def load_unified(weights, device):
    m = smp.Unet("mit_b0", encoder_weights=None, classes=3, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return m.to(device).eval()


def cc_centroids(prob_mask, threshold):
    bin_mask = (prob_mask >= threshold).astype(np.uint8)
    if not bin_mask.any():
        return []
    n, _, stats, cents = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_CC_AREA:
            continue
        cx, cy = cents[k]
        out.append((float(cx), float(cy)))
    return out


def match_keypoints(pred, gt, radius):
    if not pred and not gt: return 0, 0, 0
    if not pred:           return 0, 0, len(gt)
    if not gt:             return 0, len(pred), 0
    used = [False] * len(gt); tp = 0
    for px, py in pred:
        best_d, best_j = float("inf"), -1
        for j, (gx, gy) in enumerate(gt):
            if used[j]: continue
            d = np.hypot(px - gx, py - gy)
            if d < best_d: best_d, best_j = d, j
        if best_j >= 0 and best_d <= radius:
            used[best_j] = True; tp += 1
    return tp, len(pred) - tp, sum(1 for u in used if not u)


def eval_line_pixel_f1(model, device, val_dir, threshold=0.5):
    """Pixel-level F1 on real-GT line val. Aggregates global TP/FP/FN."""
    img_dir = os.path.join(val_dir, "images")
    mask_dir = os.path.join(val_dir, "masks")
    fids = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir)
                    if f.endswith(".jpg") and not f.startswith("._")])
    tot = {"yard": [0, 0, 0], "side": [0, 0, 0]}
    with torch.no_grad():
        for fid in fids:
            img = cv2.imread(os.path.join(img_dir, f"{fid}.jpg"))
            mask_bgr = cv2.imread(os.path.join(mask_dir, f"{fid}.png"))
            mask_bgr = cv2.resize(mask_bgr, (INPUT_W, INPUT_H),
                                    interpolation=cv2.INTER_NEAREST)
            gt_yard = (mask_bgr[..., 2] > 127).astype(np.uint8)
            gt_side = (mask_bgr[..., 1] > 127).astype(np.uint8)
            t = preprocess(img).to(device)
            probs = torch.sigmoid(model(t))[0].cpu().numpy()      # (3, H, W)
            pred_yard = (probs[0] > threshold).astype(np.uint8)
            pred_side = (probs[1] > threshold).astype(np.uint8)
            for cls, p, g in [("yard", pred_yard, gt_yard),
                              ("side", pred_side, gt_side)]:
                tot[cls][0] += int(((p == 1) & (g == 1)).sum())
                tot[cls][1] += int(((p == 1) & (g == 0)).sum())
                tot[cls][2] += int(((p == 0) & (g == 1)).sum())
    out = {}
    for cls in ("yard", "side"):
        tp, fp, fn = tot[cls]
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F = 2 * P * R / max(P + R, 1e-10)
        out[cls] = (P, R, F)
    return out


def eval_hash_keypoint_f1(model, device, kp_val_dir, thresholds):
    """Hash channel: CC centroids matched to real GT keypoints. Sweeps
    thresholds and returns best F1."""
    with open(os.path.join(kp_val_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    cache = []
    with torch.no_grad():
        for ann in coco["annotations"]:
            info = images_by_id[ann["image_id"]]
            frame = cv2.imread(os.path.join(kp_val_dir, "images", info["file_name"]))
            if frame is None: continue
            h, w = frame.shape[:2]
            t = preprocess(frame).to(device)
            probs = torch.sigmoid(model(t))[0].cpu().numpy()
            hash_prob = probs[2]                                  # (H_in, W_in)
            gt = [(p["x"] / w * INPUT_W, p["y"] / h * INPUT_H)
                  for p in ann["points"] if p["channel"] == 1]
            cache.append((hash_prob, gt))

    print()
    print(f"  ── Hash keypoint F1 (input coords, radius=8 px) ──")
    print(f"  {'thresh':>7}  {'tp':>5} {'fp':>5} {'fn':>5}  "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    results = []
    for t in thresholds:
        ttp = tfp = tfn = 0
        for prob, gt in cache:
            preds = cc_centroids(prob, t)
            tp, fp, fn = match_keypoints(preds, gt, HASH_MATCH_RADIUS)
            ttp += tp; tfp += fp; tfn += fn
        P = ttp / max(ttp + tfp, 1)
        R = ttp / max(ttp + tfn, 1)
        F = 2 * P * R / max(P + R, 1e-10)
        results.append((t, P, R, F))
        print(f"  {t:>7.2f}  {ttp:>5} {tfp:>5} {tfn:>5}  "
              f"{P:>6.3f} {R:>6.3f} {F:>6.3f}")
    best = max(results, key=lambda r: r[3])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--line-valid", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/valid"))
    ap.add_argument("--hash-valid", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/valid"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device else
               torch.device("mps" if torch.backends.mps.is_available()
                             else ("cuda" if torch.cuda.is_available() else "cpu")))
    print(f"  device: {device}")
    print(f"  loading: {args.weights}")
    model = load_unified(args.weights, device)

    # ── Line pixel F1 ──
    print()
    print(f"  ── Line pixel F1 (real GT, threshold=0.5) ──")
    line = eval_line_pixel_f1(model, device, args.line_valid)
    print(f"  {'cls':>5}  {'P':>6} {'R':>6} {'F1':>6}")
    for cls in ("yard", "side"):
        P, R, F = line[cls]
        print(f"  {cls:>5}  {P:>6.3f} {R:>6.3f} {F:>6.3f}")

    # ── Hash keypoint F1 ──
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.70, 0.80]
    best_hash = eval_hash_keypoint_f1(model, device, args.hash_valid, thresholds)

    # ── Summary ──
    print()
    print(f"  ── Summary ──")
    print(f"  yard pixel F1 = {line['yard'][2]:.3f}  "
          f"(specialist: 0.86)")
    print(f"  side pixel F1 = {line['side'][2]:.3f}  "
          f"(specialist: 0.87)")
    print(f"  hash keypoint best F1 = {best_hash[3]:.3f}  "
          f"thresh={best_hash[0]:.2f}  "
          f"(specialist: 0.953)")


if __name__ == "__main__":
    main()
