#!/usr/bin/env python3
"""Step 2 of the rebuild: FAST closed-form k1 estimate from line pixels.

Math:
  For a straight world line `x_world = a + b·y_world` projected through
  small radial distortion δ = k1·r², the resulting image curve has a
  deg-2 coefficient (in centered y-coord u = y_pix − cy) given by:

      c2_image ≈ ξ · k1 · K2 + O(k2)

  where ξ = a − cx + b·cy (line's signed offset from optical axis at the
  image-vertical-center) and K2 = (b² + 1) / f².

  Inverting: k1_est = c2_image / (ξ · K2). One polyfit + scalar division
  per line gives an estimate. Median across lines is the final value.

  k2 is dropped because the contribution to c2 is via 2K0K2 + K1² which
  vanishes near the optical center and is small elsewhere; if needed we'd
  read it off c4 with a similar formula. For now k1 alone is enough since
  c2 captures the dominant radial term.

Visualization:
  - source (distorted)
  - source + colored line pixels (per-line RMSE in distorted space)
  - undistorted (k1 applied)
  - undistorted + colored line pixels + best-fit straight lines (per-line
    RMSE in undistorted space) — should drop dramatically if calibration
    worked.

Output: per-line RMSE before/after, total wall-time for calibration.
"""

import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step2_distortion.jpg")

COLORS = [
    (0, 255, 255), (0, 165, 255), (0, 100, 255), (255, 0, 255),
    (255, 100, 0), (255, 255, 0), (100, 255, 100), (0, 255, 100),
    (200, 200, 200), (180, 105, 255), (50, 150, 255), (255, 50, 50),
]


def perpendicular_rmse(pts: np.ndarray, kind: str) -> float:
    """RMSE of perpendicular distances from pts to their best-fit straight line."""
    if kind == "yardline":
        ys, xs = pts[:, 1], pts[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        residuals = (xs - (a + b * ys)) / np.sqrt(1 + b * b)
    else:
        xs, ys = pts[:, 0], pts[:, 1]
        b, a = np.polyfit(xs, ys, 1)
        residuals = (ys - (a + b * xs)) / np.sqrt(1 + b * b)
    return float(np.sqrt(np.mean(residuals ** 2)))


def fit_line(pts: np.ndarray, kind: str) -> tuple[float, float]:
    if kind == "yardline":
        ys, xs = pts[:, 1], pts[:, 0]
        b, a = np.polyfit(ys, xs, 1)
    else:
        xs, ys = pts[:, 0], pts[:, 1]
        b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def calibrate_k1_closed_form(line_pts: list[np.ndarray], line_kinds: list[str],
                              frame_shape: tuple[int, int], focal: float) -> float:
    """Fast analytical k1 estimate via deg-2 polynomial fit + linear inversion.

    Each line gives one k1 estimate; we return the median for robustness.
    """
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    estimates: list[float] = []
    for pts, kind in zip(line_pts, line_kinds):
        if len(pts) < 3:
            continue
        if kind == "yardline":
            # x = c0 + c1·u + c2·u²  where u = y_pix − cy
            u = pts[:, 1].astype(np.float64) - cy
            x = pts[:, 0].astype(np.float64)
            c2_u, c1_u, c0_u = np.polyfit(u, x, 2)
            # Line's linear params: x_world = a + b·y_world
            # In u-space: x_pix = c0_u + c1_u·u (linear part). At u=0 (y=cy):
            #   c0_u = a + b·cy  →  a = c0_u − b·cy
            #   c1_u = b
            b = float(c1_u)
            a_intercept_at_y0 = float(c0_u) - b * cy
            xi = a_intercept_at_y0 - cx + b * cy   # = c0_u - cx
            if abs(xi) < 5.0:
                continue   # line passes near optical center → singular
            K2 = (b * b + 1.0) / (focal * focal)
            # c2_u ≈ ξ · k1 · K2  →  k1 ≈ c2_u / (ξ · K2)
            estimates.append(float(c2_u) / (xi * K2))
        else:  # sideline: y = c0 + c1·v + c2·v²  where v = x_pix − cx
            v = pts[:, 0].astype(np.float64) - cx
            y = pts[:, 1].astype(np.float64)
            c2_v, c1_v, c0_v = np.polyfit(v, y, 2)
            # Line: y_world = a + b·x_world
            # In v-space: y_pix = c0_v + c1_v·v.
            #   c0_v = a + b·cx; c1_v = b
            b = float(c1_v)
            xi = float(c0_v) - cy + b * cx          # = c0_v - cy + b·cx
            if abs(xi) < 5.0:
                continue
            K2 = (b * b + 1.0) / (focal * focal)
            estimates.append(float(c2_v) / (xi * K2))
    if not estimates:
        return 0.0
    return float(np.median(estimates))


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
    print(f"  step 1: {len(yl_groups)} yardlines, {len(sl_groups)} sidelines")

    # Distorted-space RMSE.
    print("\n  per-line RMSE in distorted (raw) image:")
    rmse_dist = [perpendicular_rmse(p, k) for p, k in zip(line_pts, line_kinds)]
    for i, (p, k, r) in enumerate(zip(line_pts, line_kinds, rmse_dist)):
        print(f"    {k:>9} {i}: n={len(p):>5}  RMSE={r:>5.2f} px")

    # Fast closed-form k1 (initial guess).
    t0 = time.time()
    k1_init = calibrate_k1_closed_form(line_pts, line_kinds, (h, w), focal)
    closed_form_ms = (time.time() - t0) * 1000
    print(f"\n  closed-form k1 = {k1_init:+.4f}  ({closed_form_ms:.1f} ms)")

    # Refine (k1, k2) with scipy on subsampled pixels, starting from closed-form k1.
    from scipy.optimize import minimize

    SUBSAMPLE = 50
    line_pts_sub = [p[::max(1, len(p) // SUBSAMPLE)] for p in line_pts]
    print(f"  subsample → {sum(len(p) for p in line_pts_sub)} total points "
          f"across {len(line_pts_sub)} lines")

    def residual_sum(k):
        intr_ = CameraIntrinsics(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0,
                                   k1=float(k[0]), k2=float(k[1]))
        total = 0.0
        for p, kk in zip(line_pts_sub, line_kinds):
            p_u = undistort_points(p.astype(np.float64), intr_)
            if kk == "yardline":
                ys, xs = p_u[:, 1], p_u[:, 0]
                b, a = np.polyfit(ys, xs, 1)
                resid = (xs - (a + b * ys)) / np.sqrt(1 + b * b)
            else:
                xs, ys = p_u[:, 0], p_u[:, 1]
                b, a = np.polyfit(xs, ys, 1)
                resid = (ys - (a + b * xs)) / np.sqrt(1 + b * b)
            total += float((resid ** 2).sum())
        return total

    t0 = time.time()
    res = minimize(residual_sum, x0=[k1_init, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-3})
    refine_ms = (time.time() - t0) * 1000
    k1, k2 = float(res.x[0]), float(res.x[1])
    print(f"  refined k1 = {k1:+.4f}, k2 = {k2:+.4f}  ({refine_ms:.1f} ms, "
          f"{res.nit} iters)")

    elapsed_ms = closed_form_ms + refine_ms
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0, k1=k1, k2=k2)

    # Undistort line pixels and recompute RMSE.
    line_pts_u = [undistort_points(p.astype(np.float64), intr) for p in line_pts]
    rmse_undist = [perpendicular_rmse(p, k) for p, k in zip(line_pts_u, line_kinds)]
    print("\n  per-line RMSE after undistortion:")
    for i, (k, rd, ru) in enumerate(zip(line_kinds, rmse_dist, rmse_undist)):
        arrow = "↓" if ru < rd else "↑"
        print(f"    {k:>9} {i}: {rd:>5.2f} → {ru:>5.2f} px  {arrow}")
    print(f"\n  mean RMSE: {np.mean(rmse_dist):.2f} → {np.mean(rmse_undist):.2f} px")

    # ── Build viz (4-up vertical) ──
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        frame_u = cv2.undistort(frame, K, dist_coeffs)
    else:
        frame_u = frame.copy()

    def overlay(panel, pts, color, alpha=0.55):
        xs = pts[:, 0].astype(np.int32); ys = pts[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[valid], ys[valid]
        panel[ys, xs] = (1 - alpha) * panel[ys, xs] + alpha * np.array(color, dtype=np.float32)

    panel_a = frame.copy()
    cv2.putText(panel_a, "source (distorted)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    panel_b = frame.copy().astype(np.float32)
    for i, p in enumerate(line_pts):
        overlay(panel_b, p, COLORS[i % len(COLORS)])
    panel_b = panel_b.clip(0, 255).astype(np.uint8)
    cv2.putText(panel_b, f"line groups (mean RMSE={np.mean(rmse_dist):.2f} px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    panel_c = frame_u.copy()
    cv2.putText(panel_c, f"undistorted (k1={k1:+.4f}, k2={k2:+.4f}, {elapsed_ms:.1f}ms)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    panel_d = frame_u.copy().astype(np.float32)
    for i, p in enumerate(line_pts_u):
        overlay(panel_d, p, COLORS[i % len(COLORS)])
    panel_d = panel_d.clip(0, 255).astype(np.uint8)
    for i, (p, k_) in enumerate(zip(line_pts_u, line_kinds)):
        a, b = fit_line(p, k_)
        if k_ == "yardline":
            ymin, ymax = float(p[:, 1].min()), float(p[:, 1].max())
            ys = np.linspace(ymin, ymax, 200); xs = a + b * ys
        else:
            xmin, xmax = float(p[:, 0].min()), float(p[:, 0].max())
            xs = np.linspace(xmin, xmax, 200); ys = a + b * xs
        cv2.polylines(panel_d, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, COLORS[i % len(COLORS)], 2)
    cv2.putText(panel_d, f"undistorted + linear fits (mean RMSE={np.mean(rmse_undist):.2f} px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    full = np.vstack([panel_a, panel_b, panel_c, panel_d])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, full)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
