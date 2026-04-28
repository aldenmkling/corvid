#!/usr/bin/env python3
"""Step 2c: how variable is k1 across frames in a clip?

For each sampled frame:
  1. Run UNet → CC grouping → line pixel sets
  2. Optimize k1 (k2=0) via scipy.minimize_scalar
  3. Report (frame_idx, k1, MSE at frame's own k1)

Then compare: what would MSE be if we used FRAME 0's k1 on every frame?
That tells us whether caching k1 from bootstrap is OK or whether per-frame
recalibration adds meaningful accuracy.
"""

import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")


def total_mse(line_pts, line_kinds, intr):
    total_sq = 0.0
    n = 0
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
        n += len(p)
    return total_sq / max(n, 1)


def process_frame(frame, focal):
    h, w = frame.shape[:2]
    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    if not line_pts:
        return None
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]

    cx, cy = w / 2.0, h / 2.0

    def mse_at(k1, pts):
        intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
        return total_mse(pts, line_kinds, intr)

    res = minimize_scalar(lambda k1: mse_at(k1, line_pts_sub),
                           bounds=(-0.5, 0.5), method="bounded",
                           options={"xatol": 1e-4})
    return {
        "k1": float(res.x),
        "line_pts": line_pts,
        "line_kinds": line_kinds,
        "n_yard": len(yl),
        "n_side": len(sl),
        "mse_at_self_full": mse_at(float(res.x), line_pts),
        "mse_at_zero_full": mse_at(0.0, line_pts),
    }


def main():
    cap = cv2.VideoCapture(CLIP)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  clip: {n_frames} frames @ {fps:.1f}fps")

    # Sample 1 frame per second across the clip.
    sample_indices = list(range(0, n_frames, int(round(fps))))
    print(f"  sampling {len(sample_indices)} frames at 1-sec interval")
    print()

    results = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        focal = float(max(h, w))
        r = process_frame(frame, focal)
        if r is None:
            continue
        r["frame_idx"] = idx
        r["t"] = idx / fps
        r["focal"] = focal
        results.append(r)

    cap.release()

    if not results:
        print("  no results")
        return

    # Print per-frame: k1, MSE at own k1, MSE at frame-0's k1.
    k1_frame0 = results[0]["k1"]
    cx, cy = w / 2.0, h / 2.0
    focal = results[0]["focal"]
    print(f"  frame 0 k1: {k1_frame0:+.4f}\n")

    print(f"  {'frame':>5}  {'t(s)':>5}  {'#lns':>4}  {'k1_opt':>8}  "
          f"{'rmse@opt':>9}  {'rmse@k1₀':>9}  {'rmse@0':>7}  "
          f"{'Δ_cache':>8}")
    print("  " + "-" * 80)
    rmse_opt_arr = []
    rmse_cached_arr = []
    rmse_zero_arr = []
    for r in results:
        intr_cached = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                          k1=k1_frame0, k2=0.0)
        mse_cached = total_mse(r["line_pts"], r["line_kinds"], intr_cached)
        rmse_opt = np.sqrt(r["mse_at_self_full"])
        rmse_cached = np.sqrt(mse_cached)
        rmse_zero = np.sqrt(r["mse_at_zero_full"])
        delta = rmse_cached - rmse_opt
        rmse_opt_arr.append(rmse_opt)
        rmse_cached_arr.append(rmse_cached)
        rmse_zero_arr.append(rmse_zero)
        print(f"  {r['frame_idx']:>5}  {r['t']:>5.1f}  "
              f"{r['n_yard']+r['n_side']:>4}  {r['k1']:>+8.4f}  "
              f"{rmse_opt:>9.3f}  {rmse_cached:>9.3f}  "
              f"{rmse_zero:>7.3f}  {delta:>+8.3f}")

    print("  " + "-" * 80)
    k1_arr = np.array([r["k1"] for r in results])
    print(f"\n  k1 stats: mean={k1_arr.mean():+.4f}  median={np.median(k1_arr):+.4f}  "
          f"std={k1_arr.std():.4f}  range=[{k1_arr.min():+.4f}, {k1_arr.max():+.4f}]")
    print(f"  RMSE summary across all sampled frames:")
    print(f"    no calibration:    {np.mean(rmse_zero_arr):.3f} px")
    print(f"    cached k1 (frame 0): {np.mean(rmse_cached_arr):.3f} px")
    print(f"    per-frame optimal k1: {np.mean(rmse_opt_arr):.3f} px")
    cost_of_caching = np.mean(rmse_cached_arr) - np.mean(rmse_opt_arr)
    print(f"  cost of caching frame-0 k1: +{cost_of_caching:.3f} px RMSE on average")


if __name__ == "__main__":
    main()
