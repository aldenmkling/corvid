#!/usr/bin/env python3
"""
Tighten the self-supervised pool using two quality signals:

1. Per-line whiteness. For each projected yard line (21 of them) and
   sideline (2), sample pixels along its length (clipped to in-frame
   extent), compute the fraction above the white threshold. Store per-line
   means. Pass the frame only if the worst in-frame yard line is above
   --min-line-whiteness (default 0.70).

2. Sideline-intersection detection presence. HRNet's channel 0 fires at
   sideline × yard-line intersection points. For each of the 42 possible
   field intersections, project through H_inv + forward-distort. If the
   projected pixel lands in-frame, require an HRNet sideline-channel peak
   within --side-detect-radius pixels.

Runs HRNet once per frame (uses saved H+k1+k2 from the pool for
projection). Writes an augmented summary CSV.
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver import run_hrnet, extract_peaks
from src.homography.apply_homography import field_to_pixel
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
)

HRNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
SIDELINE_THRESH = 0.30           # HRNet peak threshold for sideline channel
WHITE_THRESH = 180               # grayscale pixel value considered "painted"
LINE_SAMPLES = 400               # samples per projected line
LINE_WIDTH = 5                   # window width for whiteness sampling

IN_FRAME_PAD = 20                # pixels; project point is "in frame" within this pad
SIDE_DETECT_RADIUS = 15          # pixel radius for matching HRNet peak to projected intersection


def forward_distort(xy_und, fx, fy, cx, cy, k1, k2):
    """Brown-Conrady forward: undistorted → distorted."""
    x_n = (xy_und[:, 0] - cx) / fx
    y_n = (xy_und[:, 1] - cy) / fy
    r2 = x_n * x_n + y_n * y_n
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.stack([x_n * factor * fx + cx, y_n * factor * fy + cy], axis=1)


def in_frame_runs(pts, h, w, pad):
    in_b = (pts[:, 0] >= -pad) & (pts[:, 0] < w + pad) & \
           (pts[:, 1] >= -pad) & (pts[:, 1] < h + pad)
    runs = []
    start = None
    for i, ok in enumerate(in_b):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    if start is not None and len(in_b) - start >= 2:
        runs.append((start, len(in_b)))
    return runs


def line_whiteness(gray, H_inv, intr, field_pts, width=LINE_WIDTH):
    """Mean whiteness over the in-frame portion of a projected polyline.

    Returns (fraction, n_sampled). None if no in-frame samples."""
    h, w = gray.shape
    und_pts = field_to_pixel(field_pts, H_inv)
    runs = in_frame_runs(und_pts, h, w, pad=5)
    if not runs:
        return None, 0
    r = width // 2
    vals = []
    for s, e in runs:
        seg_und = und_pts[s:e]
        seg_dist = forward_distort(seg_und, **intr)
        seg_dist = np.round(seg_dist).astype(np.int32)
        for (x, y) in seg_dist:
            if 0 <= x < w and 0 <= y < h:
                x0 = max(0, x - r); x1 = min(w, x + r + 1)
                y0 = max(0, y - r); y1 = min(h, y + r + 1)
                vals.append(gray[y0:y1, x0:x1].max())
    if not vals:
        return None, 0
    vals = np.array(vals)
    return float((vals >= WHITE_THRESH).mean()), len(vals)


def project_intersection(field_xy, H_inv, intr, h, w, pad=IN_FRAME_PAD):
    """Project a single field point through H_inv + forward distortion.
    Return (x, y) pixel coords and whether it lies within frame+pad."""
    und = field_to_pixel(np.array([field_xy]), H_inv)[0]
    dst = forward_distort(np.array([und]), **intr)[0]
    in_f = (-pad <= dst[0] < w + pad) and (-pad <= dst[1] < h + pad)
    return dst, in_f


def process_frame(frame, H, k1, k2, device):
    """Compute per-line whiteness + sideline-intersection detection presence.

    Returns dict with:
      yard_white_min, yard_white_mean: min/mean across in-frame yard lines
      yard_n_inframe: how many yard lines had any in-frame samples
      side_isect_total: # sideline-yardline intersections projected in-frame
      side_isect_covered: # of those covered by an HRNet sideline peak
    """
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    intr = dict(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0, k1=float(k1), k2=float(k2))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H_inv = np.linalg.inv(H)

    # --- Per-line whiteness ---
    y_samp = np.linspace(0.0, FIELD_WIDTH, LINE_SAMPLES)
    yard_whites = []
    for x in YARD_LINE_POSITIONS:
        field_pts = np.stack([np.full_like(y_samp, x), y_samp], axis=1)
        frac, n = line_whiteness(gray, H_inv, intr, field_pts)
        if frac is not None:
            yard_whites.append(frac)
    x_samp = np.linspace(0.0, FIELD_LENGTH, LINE_SAMPLES)
    side_whites = []
    for yv in (0.0, FIELD_WIDTH):
        field_pts = np.stack([x_samp, np.full_like(x_samp, yv)], axis=1)
        frac, n = line_whiteness(gray, H_inv, intr, field_pts)
        if frac is not None:
            side_whites.append(frac)

    # --- HRNet sideline peaks (channel 0) ---
    heatmaps = run_hrnet(frame, HRNET_WEIGHTS, device=device)
    sideline_pxs, _ = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))

    # --- Per-sideline coverage ---
    # For each sideline (near/far): collect all its in-frame yard-line
    # intersections. The sideline is "covered" if (a) none of its intersections
    # project in-frame (not visible), or (b) at least one in-frame intersection
    # has an HRNet sideline peak within SIDE_DETECT_RADIUS.
    # We do NOT require every intersection to be covered individually.
    per_sideline = {}
    for side_name, yv in (("near", 0.0), ("far", FIELD_WIDTH)):
        in_frame_projs = []
        for x in YARD_LINE_POSITIONS:
            proj, in_f = project_intersection((x, yv), H_inv, intr, h, w)
            if in_f:
                in_frame_projs.append(proj)
        n_in = len(in_frame_projs)
        n_cov = 0
        if in_frame_projs and len(sideline_pxs) > 0:
            for p in in_frame_projs:
                d = np.linalg.norm(sideline_pxs - p[None, :], axis=1)
                if d.min() <= SIDE_DETECT_RADIUS:
                    n_cov += 1
        covered = (n_in == 0) or (n_cov >= 1)
        per_sideline[side_name] = dict(n_in_frame=n_in, n_covered=n_cov, covered=covered)

    return dict(
        yard_white_min=float(min(yard_whites)) if yard_whites else None,
        yard_white_mean=float(np.mean(yard_whites)) if yard_whites else None,
        yard_n_inframe=len(yard_whites),
        side_white_min=float(min(side_whites)) if side_whites else None,
        side_n_inframe=len(side_whites),
        n_side_peaks=len(sideline_pxs),
        near_side_in_frame=per_sideline["near"]["n_in_frame"],
        near_side_covered=int(per_sideline["near"]["covered"]),
        far_side_in_frame=per_sideline["far"]["n_in_frame"],
        far_side_covered=int(per_sideline["far"]["covered"]),
        both_sides_covered=int(per_sideline["near"]["covered"]
                               and per_sideline["far"]["covered"]),
    )


def grab_frame(mp4, idx):
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main(args):
    with open(os.path.join(args.pool_dir, "summary.csv")) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    t0 = time.time()
    n_done = 0
    for r in rows:
        if r["solved"] != "True":
            out_rows.append({**r})
            continue
        mp4 = os.path.join(args.clips_dir, r["game"], r["play"], f"{r['angle']}.mp4")
        frame = grab_frame(mp4, int(r["frame_idx"]))
        if frame is None:
            out_rows.append({**r})
            continue
        try:
            with open(os.path.join(args.pool_dir, r["h_path"]), "rb") as f:
                hd = pickle.load(f)
            stats = process_frame(frame, hd["H"], hd["k1"], hd["k2"], args.device)
        except Exception as e:
            print(f"  ERROR on {r['frame_id']}: {e}")
            stats = {}
        out_rows.append({**r, **stats})
        n_done += 1
        if n_done % 200 == 0:
            fps = n_done / (time.time() - t0)
            eta = (len(rows) - n_done) / max(fps, 0.01) / 60
            print(f"  [{n_done}/{len(rows)}] {fps:.2f} fps  ETA {eta:.0f} min")

    # Normalize all rows to the same field set
    all_keys = set()
    for r in out_rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    out_path = os.path.join(args.pool_dir, "summary_hquality.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\naugmented summary → {out_path}")

    # Summary stats
    def _f(v): return float(v) if v not in (None, "", "None") else None
    processed = [r for r in out_rows if r.get("yard_white_min") not in (None, "", "None")]
    print(f"\nprocessed: {len(processed)} frames")
    if processed:
        print("Filter pass counts:")
        for min_white in (0.60, 0.70, 0.80):
            for require_side in (False, True):
                n = 0
                for r in processed:
                    ywm = _f(r.get("yard_white_min"))
                    if ywm is None or ywm < min_white:
                        continue
                    if require_side:
                        total = int(r.get("side_isect_total", 0) or 0)
                        cov = int(r.get("side_isect_covered", 0) or 0)
                        if total > 0 and cov < total:
                            continue
                    n += 1
                tag = "min_white + side_coverage" if require_side else "min_white only"
                print(f"  {tag}, min={min_white}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir",
                    default=os.path.join(PROJECT_ROOT, "output/self_sup_pool_10k"))
    ap.add_argument("--clips-dir",
                    default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--device", default="mps",
                    help='"cpu", "cuda", "mps"')
    main(ap.parse_args())
