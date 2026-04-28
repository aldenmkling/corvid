#!/usr/bin/env python3
"""Preview the round-3 hash mask builder on a stratified sample of frames.

Picks frames across the F1 spectrum (top, middle, bottom of auto_scores.csv)
and renders 4-panel rows for each:
   source | predicted-shifted (red) | round-3 (red, kept blue, pills green) | GT keypoints
Saves a single stitched panel JPG.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/data_prep"))

from build_hash_round3_dataset import (
    cc_with_pixels, match_ccs, render_pill,
    MATCH_RADIUS_PX, SEED_TILT,
)
from src.homography.grid_solver_v2 import run_unet


def overlay(img, mask, color, alpha=0.55):
    out = img.copy().astype(np.float32)
    m = mask > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def render_round3_with_breakdown(frame, pred_mask, gt, yl_bin, match_radius, tilt):
    """Same logic as build_hash_round3_dataset but returns (kept_only, pill_only)
    so we can color them separately in the viz."""
    h, w = frame.shape[:2]
    ccs, labels = cc_with_pixels(pred_mask)
    keep, used = match_ccs(ccs, gt, match_radius)
    kept = np.zeros((h, w), dtype=np.uint8)
    if labels is not None:
        for (lab, _, _), k in zip(ccs, keep):
            if k:
                kept[labels == lab] = 255
    pills = np.zeros((h, w), dtype=np.uint8)
    for j, u in enumerate(used):
        if u: continue
        render_pill(pills, gt[j][0], gt[j][1], yl_bin, tilt=tilt)
    return kept, pills, keep, used


def crop_to_field(frame, mask_extent_pad=80, gt=None, kept=None, pills=None):
    """Tighten viz crop to area with mask + GT activity (skip empty sky/sidelines)."""
    h, w = frame.shape[:2]
    activity = np.zeros((h, w), dtype=np.uint8)
    if kept is not None: activity |= kept
    if pills is not None: activity |= pills
    if activity.any():
        ys, xs = np.where(activity > 0)
        y0, y1 = max(0, ys.min() - mask_extent_pad), min(h, ys.max() + mask_extent_pad)
        x0, x1 = max(0, xs.min() - mask_extent_pad), min(w, xs.max() + mask_extent_pad)
    else:
        y0, y1, x0, x1 = 0, h, 0, w
    return (x0, y0, x1, y1)


def draw_gt_dots(img, gt, radius=3, color=(0, 255, 255)):
    out = img.copy()
    for x, y in gt:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/triage_round2/auto_scores.csv"))
    ap.add_argument("--keypoint-dir", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train"))
    ap.add_argument("--mask-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/predicted_round1_shifted/masks"))
    ap.add_argument("--unet-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_round3_best.pth"))
    ap.add_argument("--out-panel", default=os.path.join(
        PROJECT_ROOT, "output/hash_round3_preview.jpg"))
    ap.add_argument("--excluded-frames", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/excluded_frames.txt"))
    ap.add_argument("--n-each", type=int, default=4,
                    help="Number of frames to sample from each F1 tier")
    ap.add_argument("--bottom-n", type=int, default=None,
                    help="If set, show this many lowest-F1 frames (overrides "
                         "tier sampling).")
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

    # Load scores, group into tiers
    rows = []
    with open(args.scores_csv) as f:
        for r in csv.DictReader(f):
            if r["frame"] in excluded:
                continue
            r["f1"] = float(r["f1"])
            rows.append(r)
    rows.sort(key=lambda r: r["f1"])
    samples = []
    if args.bottom_n is not None:
        for r in rows[:args.bottom_n]:
            r["_tier"] = "bot"
            samples.append(r)
    else:
        n = len(rows)
        tiers = {
            "low":  rows[:n // 3],
            "mid":  rows[n // 3 : 2 * n // 3],
            "high": rows[2 * n // 3:],
        }
        for name, tier_rows in tiers.items():
            if not tier_rows: continue
            step = max(1, len(tier_rows) // args.n_each)
            picks = tier_rows[::step][:args.n_each]
            for r in picks:
                r["_tier"] = name
                samples.append(r)
    print(f"  {len(samples)} preview samples")

    # Load annotations for GT
    with open(os.path.join(args.keypoint_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    by_filename = {info["file_name"]: ann
                    for ann in coco["annotations"]
                    for info in [images_by_id[ann["image_id"]]]}

    panels = []
    for samp in samples:
        frame_name = samp["frame"]
        ann = by_filename.get(frame_name)
        if ann is None: continue
        gt = [(p["x"], p["y"]) for p in ann["points"] if p["channel"] == 1]
        stem = os.path.splitext(frame_name)[0]
        img_path = os.path.join(args.keypoint_dir, "images", frame_name)
        mask_path = os.path.join(args.mask_dir, stem + ".png")
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        frame = cv2.imread(img_path)
        pred = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        yl_mask, _ = run_unet(frame, args.unet_weights, device=args.device)
        yl_bin = (yl_mask > 0).astype(np.uint8)

        kept, pills, keep_flags, used_flags = render_round3_with_breakdown(
            frame, pred, gt, yl_bin, args.match_radius, args.seed_tilt)

        # Build the four panel cells
        panel_src = frame.copy()
        panel_pred = overlay(frame, pred, color=(60, 60, 230))
        panel_round3 = overlay(frame, kept, color=(220, 60, 60), alpha=0.55)
        panel_round3 = overlay(panel_round3, pills, color=(60, 220, 60), alpha=0.55)
        panel_gt = draw_gt_dots(frame, gt, radius=4, color=(0, 255, 255))

        # Crop to active area
        x0, y0, x1, y1 = crop_to_field(frame, gt=gt, kept=kept, pills=pills)
        crop = lambda im: im[y0:y1, x0:x1]
        cells = [crop(panel_src), crop(panel_pred), crop(panel_round3), crop(panel_gt)]
        # Resize to consistent height
        target_h = 200
        cells_r = []
        for c in cells:
            ch, cw = c.shape[:2]
            scale = target_h / ch
            cells_r.append(cv2.resize(c, (int(cw * scale), target_h)))
        # Pad to common per-cell width
        max_w = max(c.shape[1] for c in cells_r)
        cells_p = [np.pad(c, ((0, 0), (0, max_w - c.shape[1]), (0, 0)),
                            mode="constant") for c in cells_r]
        row = np.hstack(cells_p)
        # Header
        n_kept = sum(keep_flags); n_drop = sum(1 for k in keep_flags if not k)
        n_pill = sum(1 for u in used_flags if not u)
        header = np.full((26, row.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(header,
                    f"[{samp['_tier']}] {frame_name}  F1={samp['f1']:.2f}  "
                    f"src | predicted | kept(red)+pills(green) | GT(yellow)  "
                    f"(kept={n_kept} dropped={n_drop} pills={n_pill})",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, row]))

    # Pad rows to common width and stack vertically
    if not panels:
        print("  no panels generated")
        return
    max_w = max(p.shape[1] for p in panels)
    padded = [np.pad(p, ((0, 0), (0, max_w - p.shape[1]), (0, 0)),
                       mode="constant") for p in panels]
    grid = np.vstack(padded)
    os.makedirs(os.path.dirname(args.out_panel), exist_ok=True)
    cv2.imwrite(args.out_panel, grid)
    print(f"  wrote {args.out_panel}  shape={grid.shape}")


if __name__ == "__main__":
    main()
