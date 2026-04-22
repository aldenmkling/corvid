#!/usr/bin/env python3
"""
Evaluate fine-tuned HRNet on the validation set at multiple confidence thresholds.

Computes precision/recall per channel (sideline, hash) and overall, sweeping
thresholds to find the operating point tradeoff.
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy import ndimage
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.keypoint_detector import FieldKeypointDetector, HRNetKeypointModel
from src.homography.keypoint_schema import NUM_CHANNELS

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "field_keypoints", "valid")
INPUT_H, INPUT_W = 512, 896
HEATMAP_H, HEATMAP_W = 256, 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Match radius for peak-to-GT matching (in heatmap coords)
MATCH_RADIUS = 4.0


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def extract_peaks(heatmap, threshold):
    """Extract peaks (one per connected component) from a heatmap."""
    mask = heatmap >= threshold
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    peaks = []
    for comp_id in range(1, n + 1):
        comp_mask = labels == comp_id
        vals = heatmap * comp_mask
        idx = vals.argmax()
        y, x = idx // heatmap.shape[1], idx % heatmap.shape[1]
        peaks.append((float(x), float(y)))
    return peaks


def gt_points_for_image(ann, img_w, img_h):
    """Return per-channel list of GT points in heatmap coords."""
    per_ch = {0: [], 1: []}
    for p in ann["points"]:
        # scale from original image coords to heatmap coords
        hx = p["x"] / img_w * HEATMAP_W
        hy = p["y"] / img_h * HEATMAP_H
        ch = p["channel"]
        if ch in per_ch:
            per_ch[ch].append((hx, hy))
    return per_ch


def match_peaks_to_gt(pred_peaks, gt_peaks, radius):
    """Greedy matching: each pred matches to closest unused gt within radius.

    Returns (tp, fp, fn).
    """
    if not pred_peaks and not gt_peaks:
        return 0, 0, 0
    if not pred_peaks:
        return 0, 0, len(gt_peaks)
    if not gt_peaks:
        return 0, len(pred_peaks), 0

    used_gt = [False] * len(gt_peaks)
    tp = 0

    for px, py in pred_peaks:
        # Find closest unused gt
        best_d = float('inf')
        best_j = -1
        for j, (gx, gy) in enumerate(gt_peaks):
            if used_gt[j]:
                continue
            d = np.hypot(px - gx, py - gy)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= radius:
            used_gt[best_j] = True
            tp += 1

    fp = len(pred_peaks) - tp
    fn = sum(1 for u in used_gt if not u)
    return tp, fp, fn


def main():
    print(f"Loading weights: {WEIGHTS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HRNetKeypointModel(num_channels=NUM_CHANNELS)
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load validation annotations
    with open(os.path.join(VAL_DIR, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    img_dir = os.path.join(VAL_DIR, "images")

    # Run inference on all val images, collect heatmaps + gt
    print(f"\nRunning inference on {len(coco['annotations'])} val images...")

    # Store per-image: (heatmaps, gt_per_channel, img_w, img_h)
    cache = []
    with torch.no_grad():
        for i, ann in enumerate(coco["annotations"]):
            img_info = images_by_id[ann["image_id"]]
            img_path = os.path.join(img_dir, img_info["file_name"])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            tensor = preprocess(frame).to(device)
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits[0]).cpu().numpy()

            gt = gt_points_for_image(ann, w, h)
            cache.append((heatmaps, gt))

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(coco['annotations'])}")

    # Sweep thresholds
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    print("\nThreshold sweep (match radius = 4 heatmap px ≈ 20 image px):")
    print(f"{'thresh':>7}  {'tp':>6} {'fp':>6} {'fn':>6}  {'prec':>6} {'recall':>6} {'F1':>6}   |   "
          f"{'side P':>6} {'side R':>6}  {'hash P':>6} {'hash R':>6}")

    results = []
    for thresh in thresholds:
        total_tp = total_fp = total_fn = 0
        per_ch = {0: [0, 0, 0], 1: [0, 0, 0]}

        for heatmaps, gt in cache:
            for ch in [0, 1]:
                peaks = extract_peaks(heatmaps[ch], thresh)
                tp, fp, fn = match_peaks_to_gt(peaks, gt[ch], MATCH_RADIUS)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                per_ch[ch][0] += tp
                per_ch[ch][1] += fp
                per_ch[ch][2] += fn

        prec = total_tp / max(total_tp + total_fp, 1)
        rec = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)

        side_p = per_ch[0][0] / max(per_ch[0][0] + per_ch[0][1], 1)
        side_r = per_ch[0][0] / max(per_ch[0][0] + per_ch[0][2], 1)
        hash_p = per_ch[1][0] / max(per_ch[1][0] + per_ch[1][1], 1)
        hash_r = per_ch[1][0] / max(per_ch[1][0] + per_ch[1][2], 1)

        results.append((thresh, prec, rec, f1))
        print(f"{thresh:>7.2f}  {total_tp:>6} {total_fp:>6} {total_fn:>6}  "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}   |   "
              f"{side_p:>6.3f} {side_r:>6.3f}  {hash_p:>6.3f} {hash_r:>6.3f}")

    # Best F1
    best = max(results, key=lambda r: r[3])
    print(f"\nBest F1 = {best[3]:.3f} at threshold {best[0]} (P={best[1]:.3f}, R={best[2]:.3f})")


if __name__ == "__main__":
    main()
