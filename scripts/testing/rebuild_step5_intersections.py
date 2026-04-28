#!/usr/bin/env python3
"""Step 5 of the rebuild: sideline × yardline intersections.

In undistorted pixel space:
  yardline: x = a_y + b_y · y      (parameterized by y, near-vertical)
  sideline: y = a_s + b_s · x      (parameterized by x, near-horizontal)

Intersection (closed form):
  x = (a_y + b_y · a_s) / (1 − b_y · b_s)
  y = a_s + b_s · x

Pipeline reuse:
  1+2+3. UNet → CC groups → k1 calibration (same as step 3)
  4. Linear fit per yardline (x = a_y + b_y · y) and per sideline
     (y = a_s + b_s · x) in undistorted space.
  5. Compute pairwise intersections; keep those that fall inside the
     yardline's y-range AND the sideline's x-range (no extrapolation).
  6. Render: undistorted frame + linear yardlines + linear sidelines +
     intersection points (yellow dots).
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
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step5_intersections.jpg")

EXTRAP_FRAC = 0.20    # allow 20% extension beyond observed line range


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

    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    print(f"  {len(yl)} yardlines + {len(sl)} sidelines")

    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    print(f"  k1 = {k1:+.4f}")

    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()

    # Yardline fits: x = a_y + b_y · y; track y-range for clipping.
    yl_fits = []
    for g in yl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append({"a": float(a), "b": float(b),
                         "ymin": float(ys.min()), "ymax": float(ys.max())})

    # Sideline fits: y = a_s + b_s · x; track x-range for clipping.
    sl_fits = []
    for g in sl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        xs, ys = pts_u[:, 0], pts_u[:, 1]
        b, a = np.polyfit(xs, ys, 1)
        sl_fits.append({"a": float(a), "b": float(b),
                         "xmin": float(xs.min()), "xmax": float(xs.max())})

    # ── 5. Closed-form intersections; gate by image bounds AND ±20%
    # extrapolation of each line's observed range. Linear fits drift further
    # from truth as you leave the support; a fractional cap bounds that.
    intersections = []   # (x, y, yi, si, ok, extrap_yl_frac, extrap_sl_frac)
    for yi, yfit in enumerate(yl_fits):
        yspan = yfit["ymax"] - yfit["ymin"]
        ymin_ext = yfit["ymin"] - EXTRAP_FRAC * yspan
        ymax_ext = yfit["ymax"] + EXTRAP_FRAC * yspan
        for si, sfit in enumerate(sl_fits):
            xspan = sfit["xmax"] - sfit["xmin"]
            xmin_ext = sfit["xmin"] - EXTRAP_FRAC * xspan
            xmax_ext = sfit["xmax"] + EXTRAP_FRAC * xspan

            denom = 1.0 - yfit["b"] * sfit["b"]
            if abs(denom) < 1e-9:
                continue
            x = (yfit["a"] + yfit["b"] * sfit["a"]) / denom
            y = sfit["a"] + sfit["b"] * x

            in_image = 0 <= x <= w - 1 and 0 <= y <= h - 1
            in_yl = ymin_ext <= y <= ymax_ext
            in_sl = xmin_ext <= x <= xmax_ext
            ok = in_image and in_yl and in_sl

            # How far beyond observed range, as fraction of span.
            if y < yfit["ymin"]:
                extrap_yl = (yfit["ymin"] - y) / max(yspan, 1.0)
            elif y > yfit["ymax"]:
                extrap_yl = (y - yfit["ymax"]) / max(yspan, 1.0)
            else:
                extrap_yl = 0.0
            if x < sfit["xmin"]:
                extrap_sl = (sfit["xmin"] - x) / max(xspan, 1.0)
            elif x > sfit["xmax"]:
                extrap_sl = (x - sfit["xmax"]) / max(xspan, 1.0)
            else:
                extrap_sl = 0.0

            intersections.append((x, y, yi, si, ok, extrap_yl, extrap_sl))

    n_in = sum(1 for *_, ok, _, _ in intersections if ok)
    print(f"  {len(intersections)} candidates, {n_in} pass "
          f"(image + ≤{EXTRAP_FRAC*100:.0f}% extrap)")
    for (x, y, yi, si, ok, ey, es) in intersections:
        why = "OK" if ok else "REJ"
        print(f"    yl[{yi}] × sl[{si}] @ ({x:7.1f}, {y:7.1f})  "
              f"extrap yl={ey*100:4.0f}% sl={es*100:4.0f}%  {why}")

    # ── 6. Render ──
    canvas = frame_u.copy()

    for yfit in yl_fits:
        ys = np.linspace(yfit["ymin"], yfit["ymax"], 200)
        xs = yfit["a"] + yfit["b"] * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (180, 180, 180), 1, cv2.LINE_AA)
    for sfit in sl_fits:
        xs = np.linspace(sfit["xmin"], sfit["xmax"], 200)
        ys = sfit["a"] + sfit["b"] * xs
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (200, 200, 0), 1, cv2.LINE_AA)

    for (x, y, yi, si, ok, ey, es) in intersections:
        in_image = 0 <= x <= w - 1 and 0 <= y <= h - 1
        if not in_image:
            continue
        color = (0, 255, 255) if ok else (80, 80, 200)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 7, color, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 9, (0, 0, 0), 1)

    cv2.putText(canvas,
                f"k1={k1:+.4f}  yl={len(yl)}  sl={len(sl)}  "
                f"intersections: {n_in}/{len(intersections)} in-frame",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"yellow=accepted (extrap≤{EXTRAP_FRAC*100:.0f}%)  "
                f"red=rejected (extrap>{EXTRAP_FRAC*100:.0f}% or off-frame)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, canvas)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
