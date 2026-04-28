#!/usr/bin/env python3
"""Step 3 of the rebuild: linear fits drawn on the undistorted frame.

Pipeline:
  1. UNet → masks
  2. CC grouping → line pixel sets
  3. Calibrate k1 (1D scipy minimize_scalar on subsampled pixels)
  4. Undistort frame + line pixels
  5. Fit a straight line through each undistorted line group
  6. Render: undistorted frame with ONLY the fitted lines drawn (no pixels,
     no mask overlay). Sanity check that the fitted lines run along the
     visible field lines.
"""

import os
import sys
import time

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
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step3_linear_fits.jpg")

# Distinct colors per line (one for yardlines, brighter for sidelines)
COLORS_YL = [
    (0, 255, 255), (0, 165, 255), (0, 100, 255), (255, 0, 255),
    (255, 100, 0), (255, 255, 0), (100, 255, 100), (0, 255, 100),
    (200, 200, 200), (180, 105, 255), (50, 150, 255), (255, 50, 50),
]
COLORS_SL = [(255, 255, 255), (200, 200, 0)]


def total_mse(line_pts, line_kinds, intr):
    total_sq = 0.0; n = 0
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


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    # Step 1+2: UNet + CC.
    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    print(f"  {len(yl)} yardlines + {len(sl)} sidelines, "
          f"{sum(len(p) for p in line_pts)} total pixels")

    # Step 3: 1D k1 fit on subsampled pixels.
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    t0 = time.time()
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    print(f"  k1 = {k1:+.4f}  ({(time.time()-t0)*1000:.0f} ms)")

    # Step 4: undistort frame + line pixels with this k1.
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()

    # Step 5: linear fit through each undistorted line group.
    line_fits: list[tuple[float, float, str, np.ndarray]] = []
    for p, kind in zip(line_pts, line_kinds):
        pts_u = undistort_points(p.astype(np.float64), intr)
        if kind == "yardline":
            ys, xs = pts_u[:, 1], pts_u[:, 0]
            b, a = np.polyfit(ys, xs, 1)        # x = a + b·y
        else:
            xs, ys = pts_u[:, 0], pts_u[:, 1]
            b, a = np.polyfit(xs, ys, 1)        # y = a + b·x
        line_fits.append((float(a), float(b), kind, pts_u))

    # Step 6: draw ONLY the linear fits on the undistorted frame.
    canvas = frame_u.copy()
    for i, (a, b, kind, pts_u) in enumerate(line_fits):
        if kind == "yardline":
            color = COLORS_YL[i % len(COLORS_YL)]
            ymin = float(pts_u[:, 1].min())
            ymax = float(pts_u[:, 1].max())
            ys = np.linspace(ymin, ymax, 200)
            xs = a + b * ys
        else:
            j = i - sum(1 for f in line_fits if f[2] == "yardline" and line_fits.index(f) < i)
            color = COLORS_SL[j % len(COLORS_SL)]
            xmin = float(pts_u[:, 0].min())
            xmax = float(pts_u[:, 0].max())
            xs = np.linspace(xmin, xmax, 200)
            ys = a + b * xs
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, color, 2)

    cv2.putText(canvas, f"k1={k1:+.4f}  {len(yl)} yardlines + {len(sl)} sidelines",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, canvas)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
