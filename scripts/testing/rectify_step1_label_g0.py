#!/usr/bin/env python3
"""Step 1 of the new rectify workflow: show first frame of a clip with
all detected yardlines labeled g0, g1, ... (g0 = leftmost in image),
plus the hash row lines and intersection keypoints from rebuild v2.

User picks one clip, runs this, looks at the labeled image, and tells
us what NGS-x value `g0` corresponds to. We feed that into step 2 to
build correspondences and solve homographies for the whole clip.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/testing"))

from src.homography.grid_solver_v2 import (
    group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

from rebuild_step4_hashes_v2 import total_mse, ransac_line
from rectify_step2_per_frame import run_specialists, LINE_WEIGHTS, HASH_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(
        PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4"))
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output/rebuild/rectify_step1_g0.jpg"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  failed to read {args.clip}"); return
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard, side, hash_ = run_specialists(frame, LINE_WEIGHTS, HASH_WEIGHTS, args.device)
    yl = group_yardline_pixels_cc(yard)
    sl = group_sideline_pixels(side)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()

    # Yardline fits + leftmost-first ordering (by x at frame midline)
    yl_fits = []
    for g in yl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append((float(a), float(b),
                          float(ys.min()), float(ys.max())))
    # Reference y for ordering: middle of frame after undistortion
    y_ref = h / 2.0
    yl_fits_with_x = [(a + b * y_ref, a, b, ymin, ymax)
                       for (a, b, ymin, ymax) in yl_fits]
    yl_fits_with_x.sort(key=lambda t: t[0])     # leftmost first → g0
    yl_fits_sorted = [(a, b, ymin, ymax) for (_, a, b, ymin, ymax) in yl_fits_with_x]

    # Hash row lines via the v2 RANSAC path
    ys_h, xs_h = np.where(hash_ > 0)
    hash_pts = np.column_stack([xs_h, ys_h]).astype(np.float64)
    hash_u = undistort_points(hash_pts, intr) if len(hash_pts) else np.empty((0, 2))
    m_far = c_far = m_near = c_near = None
    far_mask = near_mask = None
    if len(hash_u) >= 30:
        m1, c1, in1 = ransac_line(hash_u, inlier_dist=2.0)
        if m1 is not None:
            rem = hash_u[~in1]
            m2, c2, in2 = ransac_line(rem, inlier_dist=2.0)
            if m2 is not None:
                far_pts_mask = in1.copy()
                rem_idx = np.where(~in1)[0]
                near_pts_mask = np.zeros(len(hash_u), dtype=bool)
                near_pts_mask[rem_idx[in2]] = True
                if hash_u[far_pts_mask, 1].mean() > hash_u[near_pts_mask, 1].mean():
                    far_pts_mask, near_pts_mask = near_pts_mask, far_pts_mask
                    m_far, c_far = m2, c2
                    m_near, c_near = m1, c1
                else:
                    m_far, c_far = m1, c1
                    m_near, c_near = m2, c2
                far_mask = far_pts_mask
                near_mask = near_pts_mask

    # Render
    canvas = frame_u.copy()
    # Hash mask pixels colored by row
    if far_mask is not None:
        overlay = canvas.copy()
        pts_int = hash_u.astype(np.int32)
        in_bounds = ((pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) &
                      (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h))
        for mask, color in [(far_mask, (60, 60, 255)),
                              (near_mask, (255, 80, 80)),
                              (~(far_mask | near_mask), (120, 120, 120))]:
            m_in = mask & in_bounds
            if m_in.any():
                p = pts_int[m_in]
                overlay[p[:, 1], p[:, 0]] = color
        canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
        # Row lines
        xs_line = np.linspace(0, w - 1, 200)
        for (m, c, color, label) in [(m_far, c_far, (60, 60, 255), "far"),
                                       (m_near, c_near, (255, 80, 80), "near")]:
            ys_line = m * xs_line + c
            cv2.polylines(canvas, [np.stack([xs_line, ys_line], axis=1).astype(np.int32)],
                          False, color, 1, cv2.LINE_AA)

    # Yardlines + g-index labels
    for g_idx, (a, b, ymin, ymax) in enumerate(yl_fits_sorted):
        ys = np.linspace(ymin, ymax, 200)
        xs = a + b * ys
        # Color the g0 line green to make it pop
        col = (60, 220, 60) if g_idx == 0 else (220, 220, 220)
        thick = 3 if g_idx == 0 else 1
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, col, thick, cv2.LINE_AA)
        # Label text near top of line
        x_top = a + b * (ymin + 30)
        cv2.putText(canvas, f"g{g_idx}", (int(x_top) - 12, int(ymin) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"g{g_idx}", (int(x_top) - 12, int(ymin) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # Intersection points (sanity check)
    if m_far is not None:
        for j, (a, b, ymin, ymax) in enumerate(yl_fits_sorted):
            for role, m, c in (("far", m_far, c_far), ("near", m_near, c_near)):
                denom = 1.0 - b * m
                if abs(denom) < 1e-6:
                    continue
                y = (c + a * m) / denom
                x = a + b * y
                if not (0 <= x <= w and 0 <= y <= h):
                    continue
                col = (60, 60, 255) if role == "far" else (255, 80, 80)
                cv2.circle(canvas, (int(round(x)), int(round(y))), 5, col, -1)
                cv2.circle(canvas, (int(round(x)), int(round(y))), 7, (0, 0, 0), 1)

    cv2.putText(canvas,
                f"k1={k1:+.4f}  yl={len(yl_fits_sorted)}  "
                f"g0=LEFTMOST yardline (green).  "
                f"Tell me what NGS-x value g0 is.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, os.path.relpath(args.clip, PROJECT_ROOT),
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"  yardlines: {len(yl_fits_sorted)}  k1={k1:+.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
