#!/usr/bin/env python3
"""Step 2b: explicit MSE comparison for distortion calibration.

For each (k1, k2) setting, compute total MSE = sum over all lines of
(perpendicular distance from each undistorted pixel to its line's best-fit
straight line)². Tells us how well distortion calibration is straightening
the lines vs. how much the residual is irreducible noise.

Three regimes tested:
  1. k1=0, k2=0 (no calibration, baseline)
  2. Optimize k1 only (k2=0). Sweep + scipy.
  3. Optimize (k1, k2) jointly. scipy.

Reports MSE for each — the relative reductions tell us how much real signal
distortion calibration captures vs. noise floor from UNet mask thickness.
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize, minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")


def total_mse(line_pts: list[np.ndarray], line_kinds: list[str],
              intr: CameraIntrinsics) -> float:
    """Σ (perpendicular distance)² across all line pixels after undistortion."""
    total_sq = 0.0
    n_total = 0
    for p, kind in zip(line_pts, line_kinds):
        p_u = undistort_points(p.astype(np.float64), intr)
        if kind == "yardline":
            ys, xs = p_u[:, 1], p_u[:, 0]
            b, a = np.polyfit(ys, xs, 1)
            resid = (xs - (a + b * ys)) / np.sqrt(1 + b * b)
        else:
            xs, ys = p_u[:, 0], p_u[:, 1]
            b, a = np.polyfit(xs, ys, 1)
            resid = (ys - (a + b * xs)) / np.sqrt(1 + b * b)
        total_sq += float((resid ** 2).sum())
        n_total += len(p)
    return total_sq, n_total


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    focal = float(max(h, w))

    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl_groups = group_yardline_pixels_cc(yard_mask)
    sl_groups = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl_groups] + [g.pixels for g in sl_groups]
    line_kinds = ["yardline"] * len(yl_groups) + ["sideline"] * len(sl_groups)
    print(f"  {len(yl_groups)} yardlines + {len(sl_groups)} sidelines")
    print(f"  total pixels: {sum(len(p) for p in line_pts)}")

    # Subsample uniformly to keep optimizer fast (~50 pts per line).
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    n_sub = sum(len(p) for p in line_pts_sub)
    print(f"  subsample: {n_sub} pixels for optimization\n")

    cx, cy = w / 2.0, h / 2.0

    def mse_at(k1, k2, pts=line_pts_sub):
        intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=k2)
        total_sq, n = total_mse(pts, line_kinds, intr)
        return total_sq / n

    def mse_full(k1, k2):
        return mse_at(k1, k2, pts=line_pts)

    # ── 1. Baseline: no calibration ──
    mse_baseline_sub = mse_at(0.0, 0.0)
    mse_baseline_full = mse_full(0.0, 0.0)
    print(f"  [1] No calibration (k1=0, k2=0):")
    print(f"      MSE (subsample) = {mse_baseline_sub:.4f} px²,  "
          f"sqrt(MSE) = {np.sqrt(mse_baseline_sub):.3f} px")
    print(f"      MSE (full)      = {mse_baseline_full:.4f} px²,  "
          f"sqrt(MSE) = {np.sqrt(mse_baseline_full):.3f} px\n")

    # ── 2. Optimize k1 only ──
    t0 = time.time()
    res1 = minimize_scalar(lambda k1: mse_at(k1, 0.0),
                            bounds=(-0.5, 0.5), method="bounded",
                            options={"xatol": 1e-4})
    t_k1 = (time.time() - t0) * 1000
    k1_only = float(res1.x)
    mse_k1_sub = mse_at(k1_only, 0.0)
    mse_k1_full = mse_full(k1_only, 0.0)
    print(f"  [2] Optimize k1 only (k2=0): {t_k1:.0f} ms, {res1.nit} iters")
    print(f"      best k1 = {k1_only:+.4f}")
    print(f"      MSE (subsample) = {mse_k1_sub:.4f} px²  → reduction: "
          f"{(1 - mse_k1_sub/mse_baseline_sub)*100:.1f}%")
    print(f"      MSE (full)      = {mse_k1_full:.4f} px²  → reduction: "
          f"{(1 - mse_k1_full/mse_baseline_full)*100:.1f}%\n")

    # ── 3. Optimize (k1, k2) jointly ──
    t0 = time.time()
    res2 = minimize(lambda k: mse_at(k[0], k[1]), x0=[k1_only, 0.0],
                     method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4})
    t_k1k2 = (time.time() - t0) * 1000
    k1_joint, k2_joint = float(res2.x[0]), float(res2.x[1])
    mse_joint_sub = mse_at(k1_joint, k2_joint)
    mse_joint_full = mse_full(k1_joint, k2_joint)
    print(f"  [3] Optimize (k1, k2): {t_k1k2:.0f} ms, {res2.nit} iters")
    print(f"      best (k1, k2) = ({k1_joint:+.4f}, {k2_joint:+.4f})")
    print(f"      MSE (subsample) = {mse_joint_sub:.4f} px²  → reduction: "
          f"{(1 - mse_joint_sub/mse_baseline_sub)*100:.1f}%")
    print(f"      MSE (full)      = {mse_joint_full:.4f} px²  → reduction: "
          f"{(1 - mse_joint_full/mse_baseline_full)*100:.1f}%\n")

    print(f"  Summary (sqrt MSE on FULL pixel set):")
    print(f"    no cal:    {np.sqrt(mse_baseline_full):.3f} px")
    print(f"    k1 only:   {np.sqrt(mse_k1_full):.3f} px  ({(1 - mse_k1_full/mse_baseline_full)*100:.1f}% MSE reduction)")
    print(f"    k1 + k2:   {np.sqrt(mse_joint_full):.3f} px  ({(1 - mse_joint_full/mse_baseline_full)*100:.1f}% MSE reduction)")


if __name__ == "__main__":
    main()
