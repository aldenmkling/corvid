#!/usr/bin/env python3
"""Post-processing precision/recall evaluation for the HRNet + grid-solver pipeline.

Raw HRNet P/R measures how well the model places peaks on real features.
This script measures what actually gets emitted to downstream code (the grid
solver filters HRNet output heavily — dropping singletons that don't fit the
grid, noise that doesn't land on either row line, etc.). For homography, what
matters is the precision of what gets through.

On the val set (default: data/field_keypoints/valid/):
  - For each frame, run the full pipeline → list of emitted correspondences.
  - Match each emission to the nearest GT keypoint of the same class
    (hash or sideline) within `--match-tol-px`.
  - Count TP / FP / FN per frame and aggregate.

Outputs:
  - per-frame CSV with counts + metrics
  - overall + per-type (far_hash / near_hash / sideline) summary

Note: GT labels only distinguish `hash_intersection` vs `sideline_intersection`.
They don't distinguish far vs near row, so a `far_hash` emission matches any
hash GT (and same for `near_hash`). We still report per-emission-type counts
so you can see where filtering hurts most.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver import (  # noqa: E402
    run_hrnet, extract_peaks, build_yard_lines,
    HASH_THRESH, SIDELINE_THRESH,
)

CHANNEL_SIDELINE = 0
CHANNEL_HASH = 1


def load_gt(ann_path):
    """Load val annotations into {image_id: {'file_name', 'w', 'h', 'points'}}."""
    with open(ann_path) as f:
        d = json.load(f)
    by_img = {}
    for img in d["images"]:
        by_img[img["id"]] = {
            "file_name": img["file_name"],
            "w": img.get("width", 1280),
            "h": img.get("height", 720),
            "points": [],
        }
    for ann in d["annotations"]:
        if ann["image_id"] not in by_img:
            continue
        for p in ann["points"]:
            if not p.get("visible", True):
                continue
            by_img[ann["image_id"]]["points"].append({
                "x": float(p["x"]),
                "y": float(p["y"]),
                "channel": int(p["channel"]),
            })
    return by_img


def pipeline_emissions(frame, weights, device):
    """Run full pipeline and return (emissions, n_paired_groups) where
    emissions is list of (x, y, emit_type) and n_paired_groups is the count of
    non-singleton yard-line groups produced by the grid solver.
    """
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, weights, device=device)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    yard_lines, _used_sideline = build_yard_lines(
        hash_pxs, hash_confs, sideline_pxs, sideline_confs,
    )
    emissions = []
    n_paired = 0
    for yl in yard_lines:
        if not yl.get("singleton"):
            n_paired += 1
        # Skip singletons whose grid_fit_ok is False (rejected post-grid)
        if yl.get("singleton") and not yl.get("grid_fit_ok", False):
            continue
        fh = yl.get("far_hash")
        nh = yl.get("near_hash")
        sl = yl.get("sideline")
        if fh is not None:
            emissions.append((float(fh[0]), float(fh[1]), "far_hash"))
        if nh is not None:
            emissions.append((float(nh[0]), float(nh[1]), "near_hash"))
        if sl is not None:
            emissions.append((float(sl[0]), float(sl[1]), "sideline"))
    return emissions, n_paired


def match_emissions(emissions, gt_points, tol_px):
    """Greedy nearest-unmatched matching with class-agreement constraint.

    Emissions of type 'far_hash' or 'near_hash' match GT points with channel=HASH.
    Emissions of type 'sideline' match GT points with channel=SIDELINE.

    Returns (tp_list, fp_list, fn_list, matches) where each item is a dict with
    coordinates and types for downstream analysis.
    """
    # Index GT by class
    gt_hash = [dict(p, _idx=i, _used=False) for i, p in enumerate(gt_points)
               if p["channel"] == CHANNEL_HASH]
    gt_side = [dict(p, _idx=i, _used=False) for i, p in enumerate(gt_points)
               if p["channel"] == CHANNEL_SIDELINE]

    tp, fp, fn, matches = [], [], [], []

    # For each emission, look up candidate GT of same class and take closest.
    # Sort emissions by distance-to-best-gt (ascending) to resolve contention:
    # closer emission claims first.
    candidates = []  # (best_dist, emission_idx, gt_ref_idx_in_same_class_list)
    for ei, (x, y, et) in enumerate(emissions):
        gt_list = gt_hash if et in ("far_hash", "near_hash") else gt_side
        best_d = float("inf")
        best_j = None
        for j, g in enumerate(gt_list):
            d = np.hypot(x - g["x"], y - g["y"])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= tol_px:
            candidates.append((best_d, ei, et, best_j))

    # Greedy: assign in order of smallest distance, skipping if gt or emission
    # is already taken.
    candidates.sort()
    claimed_emissions = set()
    for d, ei, et, j in candidates:
        if ei in claimed_emissions:
            continue
        gt_list = gt_hash if et in ("far_hash", "near_hash") else gt_side
        if gt_list[j]["_used"]:
            continue
        claimed_emissions.add(ei)
        gt_list[j]["_used"] = True
        tp.append({
            "emit_x": emissions[ei][0], "emit_y": emissions[ei][1],
            "emit_type": et, "gt_x": gt_list[j]["x"], "gt_y": gt_list[j]["y"],
            "gt_channel": gt_list[j]["channel"], "dist": d,
        })
        matches.append((ei, j, et))

    for ei, (x, y, et) in enumerate(emissions):
        if ei in claimed_emissions:
            continue
        fp.append({"x": x, "y": y, "type": et})
    for g in gt_hash + gt_side:
        if not g["_used"]:
            fn.append({"x": g["x"], "y": g["y"], "channel": g["channel"]})

    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir",
                        default=os.path.join(PROJECT_ROOT, "data",
                                             "field_keypoints", "valid"))
    parser.add_argument("--weights",
                        default=os.path.join(PROJECT_ROOT, "models",
                                             "hrnet_finetuned_last.pth"))
    parser.add_argument("--match-tol-px", type=float, default=15.0,
                        help="Pixel tolerance for TP match.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv",
                        default=os.path.join(PROJECT_ROOT, "output",
                                             "pipeline_pr_per_frame.csv"))
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Cap for dry runs (0 = all).")
    args = parser.parse_args()

    ann_path = os.path.join(args.val_dir, "annotations.json")
    img_dir = os.path.join(args.val_dir, "images")
    print(f"Loading {ann_path}...")
    gt = load_gt(ann_path)
    image_ids = sorted(gt.keys())
    if args.max_frames:
        image_ids = image_ids[:args.max_frames]
    print(f"Evaluating {len(image_ids)} frames on device={args.device}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    agg = {"TP": 0, "FP": 0, "FN": 0}
    by_type = {
        "far_hash": {"TP": 0, "FP": 0},
        "near_hash": {"TP": 0, "FP": 0},
        "sideline": {"TP": 0, "FP": 0},
    }
    by_gt_class = {"hash": {"TP": 0, "FN": 0},
                   "sideline": {"TP": 0, "FN": 0}}
    # Pipeline-outcome buckets (per frame)
    tier_full = 0         # ≥4 emissions → full homography solvable
    tier_delta = 0        # 2-3 emissions → similarity-delta usable
    tier_grid_only = 0    # <2 emissions but ≥1 paired group → grid exists
    tier_empty = 0        # no paired groups and <2 emissions

    rows = []
    t0 = time.time()
    for n, img_id in enumerate(image_ids):
        meta = gt[img_id]
        img_path = os.path.join(img_dir, meta["file_name"])
        if not os.path.exists(img_path):
            print(f"  [{n+1}/{len(image_ids)}] missing: {img_path}")
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{n+1}/{len(image_ids)}] unreadable: {img_path}")
            continue
        emissions, n_paired = pipeline_emissions(frame, args.weights, args.device)
        tp, fp, fn = match_emissions(emissions, meta["points"], args.match_tol_px)
        ne = len(emissions)
        if ne >= 4:
            tier_full += 1
        elif ne >= 2:
            tier_delta += 1
        elif n_paired >= 1:
            tier_grid_only += 1
        else:
            tier_empty += 1
        agg["TP"] += len(tp)
        agg["FP"] += len(fp)
        agg["FN"] += len(fn)
        for t in tp:
            by_type[t["emit_type"]]["TP"] += 1
            cls = "hash" if t["gt_channel"] == CHANNEL_HASH else "sideline"
            by_gt_class[cls]["TP"] += 1
        for f in fp:
            by_type[f["type"]]["FP"] += 1
        for f in fn:
            cls = "hash" if f["channel"] == CHANNEL_HASH else "sideline"
            by_gt_class[cls]["FN"] += 1

        rows.append({
            "image_id": img_id,
            "file_name": meta["file_name"],
            "n_emissions": ne,
            "n_paired_groups": n_paired,
            "n_gt": len(meta["points"]),
            "TP": len(tp),
            "FP": len(fp),
            "FN": len(fn),
        })
        if (n + 1) % 10 == 0 or n == 0:
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (len(image_ids) - n - 1)
            print(f"  [{n+1}/{len(image_ids)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-frame CSV: {args.output_csv}")

    def fmt(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    p, r, f1 = fmt(agg["TP"], agg["FP"], agg["FN"])
    print(f"\n=== OVERALL (match tol {args.match_tol_px}px) ===")
    print(f"  TP={agg['TP']}  FP={agg['FP']}  FN={agg['FN']}")
    print(f"  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

    total = len(image_ids)
    print(f"\n=== Pipeline outcome per frame (n={total}) ===")
    print(f"  Full H solvable (≥4 emissions)     : {tier_full:3d}  ({tier_full/total*100:4.1f}%)")
    print(f"  Similarity-delta only (2-3 em)     : {tier_delta:3d}  ({tier_delta/total*100:4.1f}%)")
    print(f"  Grid computed but <2 em            : {tier_grid_only:3d}  ({tier_grid_only/total*100:4.1f}%)")
    print(f"  Nothing usable                     : {tier_empty:3d}  ({tier_empty/total*100:4.1f}%)")

    print(f"\n=== By emission type ===")
    for k, v in by_type.items():
        p_t = v["TP"] / (v["TP"] + v["FP"]) if (v["TP"] + v["FP"]) else 0.0
        print(f"  {k:10s}  TP={v['TP']:4d}  FP={v['FP']:4d}  Precision={p_t:.3f}")

    print(f"\n=== By GT class (recall) ===")
    for k, v in by_gt_class.items():
        r_c = v["TP"] / (v["TP"] + v["FN"]) if (v["TP"] + v["FN"]) else 0.0
        print(f"  {k:10s}  TP={v['TP']:4d}  FN={v['FN']:4d}  Recall={r_c:.3f}")


if __name__ == "__main__":
    main()
