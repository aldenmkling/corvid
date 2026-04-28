#!/usr/bin/env python3
"""Score each predicted hash mask against GT keypoint annotations and
rank frames by per-frame F1.

For each train frame:
  - Take CC centroids of the predicted hash mask  (= predicted hashes)
  - Match each prediction to nearest unused GT keypoint within
    --match-radius (image-space pixels)
  - Compute per-frame TP/FP/FN/P/R/F1

Output: CSV ranked by F1 desc, plus an overall distribution summary.
Used as a triage shortcut so we can auto-pick high-quality frames for
training without reviewing every overlay manually.
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_CC_AREA = 4
MATCH_RADIUS_PX = 12        # image-space pixels


def cc_centroids(mask: np.ndarray):
    """Centroids of every CC larger than MIN_CC_AREA pixels."""
    bin_mask = (mask > 127).astype(np.uint8)
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


def match(pred, gt, radius):
    """Greedy nearest-unused match. Returns (tp, fp, fn, matched_distances)."""
    if not pred and not gt: return 0, 0, 0, []
    if not pred:            return 0, 0, len(gt), []
    if not gt:              return 0, len(pred), 0, []
    used = [False] * len(gt)
    tp = 0
    matched_d = []
    for px, py in pred:
        best_d, best_j = float("inf"), -1
        for j, (gx, gy) in enumerate(gt):
            if used[j]: continue
            d = float(np.hypot(px - gx, py - gy))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= radius:
            used[best_j] = True
            tp += 1
            matched_d.append(best_d)
    return tp, len(pred) - tp, sum(1 for u in used if not u), matched_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint-dir", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train"))
    ap.add_argument("--mask-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/predicted_round1_shifted/masks"))
    ap.add_argument("--out-csv", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/triage_round2/auto_scores.csv"))
    ap.add_argument("--match-radius", type=float, default=MATCH_RADIUS_PX,
                    help="Image-space pixel radius for matching pred ↔ GT")
    args = ap.parse_args()

    with open(os.path.join(args.keypoint_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    rows = []
    for ann in coco["annotations"]:
        info = images_by_id[ann["image_id"]]
        gt = [(p["x"], p["y"]) for p in ann["points"] if p["channel"] == 1]
        if not gt:
            continue
        stem = os.path.splitext(info["file_name"])[0]
        mask_path = os.path.join(args.mask_dir, stem + ".png")
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        pred = cc_centroids(mask)
        tp, fp, fn, matched_d = match(pred, gt, args.match_radius)
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-10)
        rows.append({
            "frame": info["file_name"],
            "n_gt": len(gt),
            "n_pred": len(pred),
            "tp": tp, "fp": fp, "fn": fn,
            "p": round(P, 3), "r": round(R, 3), "f1": round(F1, 3),
            "mean_offset_px": (round(float(np.mean(matched_d)), 2)
                                if matched_d else ""),
        })

    rows.sort(key=lambda r: (r["f1"], -(r["fp"] + r["fn"])), reverse=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    f1s = np.array([r["f1"] for r in rows])
    print(f"  scored {len(rows)} frames  (match radius {args.match_radius:.0f} px)")
    print(f"  F1 percentiles:  "
          f"p10={np.percentile(f1s, 10):.3f}  "
          f"p25={np.percentile(f1s, 25):.3f}  "
          f"p50={np.percentile(f1s, 50):.3f}  "
          f"p75={np.percentile(f1s, 75):.3f}  "
          f"p90={np.percentile(f1s, 90):.3f}")
    print(f"  thresholded counts:")
    for thr in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        n = int((f1s >= thr).sum())
        pct = 100 * n / len(rows)
        print(f"    F1 >= {thr:.2f}:  {n:>4} frames ({pct:.1f}%)")
    print(f"  out: {args.out_csv}")


if __name__ == "__main__":
    main()
