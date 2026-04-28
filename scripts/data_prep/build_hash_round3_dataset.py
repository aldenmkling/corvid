#!/usr/bin/env python3
"""Build round-3 hash mask training set automatically.

For each train frame:
  1. Take the round-2 model-predicted (shifted) hash mask.
  2. Drop CCs that don't match any GT hash keypoint within --match-radius
     (kills FPs in training labels).
  3. For GT hash keypoints with no matching CC, render a homography-scaled
     filled pill (rotated ellipse) at the keypoint:
       long axis  ≈ 6 × w_yl px  (real-world 24" / 4" yardline = 6×)
       short axis ≈ 1 × w_yl px  (real-world 4")
       angle      = blend of perp-to-yardline and image-horizontal
     This seeds prediction at missed locations using real hash dimensions
     without forcing a rectangular rule-based shape into training.

Outputs per-frame masks + copies source images into round3 train/.
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/data_prep"))

from generate_hash_masks import (
    estimate_yardline_direction,
    estimate_yardline_width_perpendicular,
)
from src.homography.grid_solver_v2 import run_unet


MATCH_RADIUS_PX = 12        # image-space px
MIN_CC_AREA = 4
DEFAULT_W_YL = 5            # fallback when yardline not detected at GT
SEED_TILT = 0.6             # 0=strict perp-to-yardline, 1=image-horizontal
HASH_LENGTH_K = 2.5         # pill long axis / yardline width.
                             # Real hash is 6×, but pill is a SEED — model
                             # learns to extend to natural shape from context.
HASH_WIDTH_K = 0.4          # pill short axis / yardline width


def cc_with_pixels(mask: np.ndarray):
    bin_mask = (mask > 127).astype(np.uint8)
    if not bin_mask.any():
        return [], None
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_CC_AREA:
            continue
        cx, cy = cents[k]
        out.append((k, float(cx), float(cy)))
    return out, labels


def match_ccs(ccs, gt, radius):
    """Distance-first greedy bipartite match: each GT keeps only its closest
    CC within `radius`; other nearby CCs are dropped as FPs (de-dups
    multi-CC clusters around a single hash).

    Returns (keep_flag_per_cc, used_gt_flag).
    """
    keep = [False] * len(ccs)
    used = [False] * len(gt)
    pairs = []   # (distance, cc_idx, gt_idx)
    for i, (_, cx, cy) in enumerate(ccs):
        for j, (gx, gy) in enumerate(gt):
            d = float(np.hypot(cx - gx, cy - gy))
            if d <= radius:
                pairs.append((d, i, j))
    pairs.sort()
    for _, i, j in pairs:
        if keep[i] or used[j]:
            continue
        keep[i] = True
        used[j] = True
    return keep, used


def render_pill(mask: np.ndarray, x: float, y: float, yl_bin: np.ndarray,
                tilt: float):
    """Filled rotated ellipse at (x, y) sized to local yardline width."""
    xi, yi = int(round(x)), int(round(y))
    along = estimate_yardline_direction(yl_bin, xi, yi)
    w_yl = estimate_yardline_width_perpendicular(yl_bin, xi, yi, along)
    if w_yl is None or w_yl < 2 or w_yl > 30:
        w_yl = DEFAULT_W_YL

    # Pill long axis = blend of perp-to-yardline and image-horizontal.
    yperp = np.array([-along[1], along[0]])
    if yperp[0] < 0:
        yperp = -yperp
    target = np.array([1.0, 0.0])
    long_dir = (1 - tilt) * yperp + tilt * target
    long_dir /= max(np.linalg.norm(long_dir), 1e-9)
    angle_deg = float(np.degrees(np.arctan2(long_dir[1], long_dir[0])))

    half_long = max(2, int(round(0.5 * HASH_LENGTH_K * w_yl)))
    half_short = max(1, int(round(0.5 * HASH_WIDTH_K * w_yl)))
    cv2.ellipse(mask, (xi, yi), (half_long, half_short),
                 angle_deg, 0, 360, 255, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint-dir", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train"))
    ap.add_argument("--mask-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/predicted_round1_shifted/masks"))
    ap.add_argument("--unet-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_round3_best.pth"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/round3"))
    ap.add_argument("--excluded-frames", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/excluded_frames.txt"),
                    help="Text file listing filenames to skip (one per line; "
                         "'#' starts a comment).")
    ap.add_argument("--match-radius", type=float, default=MATCH_RADIUS_PX)
    ap.add_argument("--seed-tilt", type=float, default=SEED_TILT)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    excluded = set()
    if os.path.exists(args.excluded_frames):
        with open(args.excluded_frames) as f:
            for line in f:
                name = line.split("#", 1)[0].strip()
                if name:
                    excluded.add(name)
        if excluded:
            print(f"  excluding {len(excluded)} frames")

    img_out = os.path.join(args.out_dir, "train/images")
    mask_out = os.path.join(args.out_dir, "train/masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    with open(os.path.join(args.keypoint_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    n_frames = n_kept = n_dropped = n_pills = n_excluded = 0
    for ann in coco["annotations"]:
        info = images_by_id[ann["image_id"]]
        if info["file_name"] in excluded:
            n_excluded += 1
            continue
        gt = [(p["x"], p["y"]) for p in ann["points"] if p["channel"] == 1]
        if not gt: continue
        stem = os.path.splitext(info["file_name"])[0]
        mask_path = os.path.join(args.mask_dir, stem + ".png")
        img_path = os.path.join(args.keypoint_dir, "images", info["file_name"])
        if not (os.path.exists(mask_path) and os.path.exists(img_path)):
            continue

        frame = cv2.imread(img_path)
        pred = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = frame.shape[:2]

        ccs, labels = cc_with_pixels(pred)
        keep, used = match_ccs(ccs, gt, args.match_radius)

        new_mask = np.zeros((h, w), dtype=np.uint8)
        if labels is not None:
            for (lab, _, _), k in zip(ccs, keep):
                if k:
                    new_mask[labels == lab] = 255

        n_kept += sum(keep)
        n_dropped += sum(1 for k in keep if not k)

        # Seed unmatched GT keypoints with homography-scaled pills.
        if not all(used):
            yl_mask, _ = run_unet(frame, args.unet_weights, device=args.device)
            yl_bin = (yl_mask > 0).astype(np.uint8)
            for j, u in enumerate(used):
                if u: continue
                render_pill(new_mask, gt[j][0], gt[j][1], yl_bin,
                             tilt=args.seed_tilt)
                n_pills += 1

        shutil.copy2(img_path, os.path.join(img_out, info["file_name"]))
        cv2.imwrite(os.path.join(mask_out, stem + ".png"), new_mask)
        n_frames += 1

    print(f"  frames:        {n_frames}  (excluded {n_excluded})")
    print(f"  kept CCs:      {n_kept}  (matched GT)")
    print(f"  dropped CCs:   {n_dropped}  (FP — no GT in radius)")
    print(f"  seeded pills:  {n_pills}  (GT keypoints model missed)")
    print(f"  out:           {args.out_dir}/train/")


if __name__ == "__main__":
    main()
