#!/usr/bin/env python3
"""Render the same frame's rectified output at different rerank_top_n values.

Side-by-side visualization for: do we actually need top_n=250, or does
something smaller already produce a correct H? Includes CPU pure-search as
a reference.

Usage:
    .venv/bin/python scripts/testing/viz_top_n_sweep.py \
        --frame data/line_detection/valid/images/2024091501_play_128_sideline_f000085.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    solve_grid, run_unet, run_hash_w18,
    fit_homography_from_result, calibrate_distortion_from_result,
)
from src.homography.field_model import FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR
from src.homography.distortion import undistort_points

UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
W18 = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output/top_n_sweep")


def render_rectified(frame, H, k1, k2, intrinsics,
                      panel_w=600, panel_h=400, margin_yd=5.0,
                      max_extent_yd=80.0):
    """Render rectified frame to a FIXED canvas size so panels compare visually.
    Auto-sizes the world bbox to fit the visible region but caps it at
    max_extent_yd so a broken H doesn't blow up the field-of-view.
    """
    H_src, W_src = frame.shape[:2]
    corners_px = np.array([[0, 0], [W_src, 0], [W_src, H_src], [0, H_src]],
                          dtype=np.float64)
    corners_undist = undistort_points(corners_px, intrinsics)
    ones = np.ones((4, 1))
    corners_field = (H @ np.hstack([corners_undist, ones]).T).T
    corners_field = corners_field[:, :2] / corners_field[:, 2:3]
    x_min = float(corners_field[:, 0].min()) - margin_yd
    x_max = float(corners_field[:, 0].max()) + margin_yd
    y_min = float(corners_field[:, 1].min()) - margin_yd
    y_max = float(corners_field[:, 1].max()) + margin_yd

    # Cap the field-of-view to prevent broken H from showing nothing useful.
    # If H is broken, the bbox can be huge — clip around its center.
    w_yd_full = x_max - x_min
    h_yd_full = y_max - y_min
    if w_yd_full > max_extent_yd:
        cx_yd = 0.5 * (x_min + x_max)
        x_min, x_max = cx_yd - max_extent_yd / 2, cx_yd + max_extent_yd / 2
    if h_yd_full > max_extent_yd / 2:
        cy_yd = 0.5 * (y_min + y_max)
        y_min, y_max = cy_yd - max_extent_yd / 4, cy_yd + max_extent_yd / 4

    w_yd = x_max - x_min
    h_yd = y_max - y_min
    w_out = panel_w
    h_out = panel_h
    yd_per_px_x = w_yd / w_out
    yd_per_px_y = h_yd / h_out

    xs_field = np.linspace(x_min, x_max, w_out)
    ys_field = np.linspace(y_max, y_min, h_out)
    gx, gy = np.meshgrid(xs_field, ys_field)
    field_pts = np.stack([gx.ravel(), gy.ravel(), np.ones_like(gx.ravel())], axis=1)

    H_inv = np.linalg.inv(H)
    src_undist = (H_inv @ field_pts.T).T
    src_undist = src_undist[:, :2] / src_undist[:, 2:3]

    x_n = (src_undist[:, 0] - intrinsics.cx) / intrinsics.fx
    y_n = (src_undist[:, 1] - intrinsics.cy) / intrinsics.fy
    r2 = x_n ** 2 + y_n ** 2
    factor = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d = x_n * factor * intrinsics.fx + intrinsics.cx
    y_d = y_n * factor * intrinsics.fy + intrinsics.cy
    map_x = x_d.reshape(h_out, w_out).astype(np.float32)
    map_y = y_d.reshape(h_out, w_out).astype(np.float32)

    margin = 0.5 * max(H_src, W_src)
    src_x_u = src_undist[:, 0].reshape(h_out, w_out)
    src_y_u = src_undist[:, 1].reshape(h_out, w_out)
    factor_2d = factor.reshape(h_out, w_out)
    in_frame = ((map_x >= 0) & (map_x < W_src) &
                (map_y >= 0) & (map_y < H_src) &
                (src_x_u >= -margin) & (src_x_u < W_src + margin) &
                (src_y_u >= -margin) & (src_y_u < H_src + margin) &
                (factor_2d > 0))
    map_x[~in_frame] = -1.0
    map_y[~in_frame] = -1.0

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 50, 20))

    def x_to_px(x_yd):
        return int(round((x_yd - x_min) / yd_per_px_x))
    def y_to_px(y_yd):
        return int(round((y_max - y_yd) / yd_per_px_y))
    x_lo = int(np.ceil(x_min / 5.0)) * 5
    x_hi = int(np.floor(x_max / 5.0)) * 5
    for x_yd in range(x_lo, x_hi + 1, 5):
        x_px = x_to_px(x_yd)
        if 0 <= x_px < w_out:
            color = (255, 255, 255) if x_yd % 10 == 0 else (180, 180, 180)
            cv2.line(warped, (x_px, 0), (x_px, h_out - 1), color, 1)
            cv2.putText(warped, f"{x_yd:+d}", (x_px + 3, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for y_yd, color in [
        (0.0, (255, 255, 255)),
        (FIELD_WIDTH, (255, 255, 255)),
        (HASH_Y_NEAR, (0, 200, 200)),
        (HASH_Y_FAR, (0, 200, 200)),
    ]:
        if y_min <= y_yd <= y_max:
            y_px = y_to_px(y_yd)
            cv2.line(warped, (0, y_px), (w_out - 1, y_px), color, 1)
    return warped


PANEL_W, PANEL_H = 600, 400


def render_one(frame, label, H, k1, k2, intr, n_in, err, vp):
    if H is None:
        rect = np.full((PANEL_H, PANEL_W, 3), 30, dtype=np.uint8)
        cv2.putText(rect, "NO H", (PANEL_W // 2 - 60, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 60, 200), 3, cv2.LINE_AA)
        err_s = "nan"
        n_in_s = "0"
    else:
        rect = render_rectified(frame, H, k1, k2, intr,
                                 panel_w=PANEL_W, panel_h=PANEL_H)
        err_s = f"{err:.3f}yd"
        n_in_s = str(n_in)

    msg = f"{label}  in={n_in_s}  err={err_s}  vp=({vp[0]:.0f},{vp[1]:.0f})"
    bar = np.zeros((36, PANEL_W, 3), dtype=np.uint8)
    cv2.putText(bar, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, rect])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--top-ns", type=int, nargs="+", default=[20, 50, 100, 250])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fid = os.path.splitext(os.path.basename(args.frame))[0]
    frame = cv2.imread(args.frame)
    yard, side = run_unet(frame, UNET, device=args.device)
    hp, hc = run_hash_w18(frame, W18, device=args.device)

    # CPU baseline for reference
    res_cpu = solve_grid(yard, side, hp, hc, frame_shape=frame.shape[:2],
                          use_gpu_vp=False)
    k1, k2 = calibrate_distortion_from_result(res_cpu, frame_shape=frame.shape[:2])
    h_cpu = fit_homography_from_result(res_cpu, base_ngs_x=0.0, distortion=(k1, k2),
                                        frame_shape=frame.shape[:2])
    panels = [render_one(frame, "CPU pure", h_cpu["H"], k1, k2, h_cpu["intrinsics"],
                          h_cpu["n_inliers"], h_cpu["mean_err_yd"],
                          res_cpu.vp or (0, 0))]

    for n in args.top_ns:
        res = solve_grid(yard, side, hp, hc, frame_shape=frame.shape[:2],
                         use_gpu_vp=True, vp_rerank_top_n=n,
                         vp_device=args.device)
        k1, k2 = calibrate_distortion_from_result(res, frame_shape=frame.shape[:2])
        h_g = fit_homography_from_result(res, base_ngs_x=0.0, distortion=(k1, k2),
                                          frame_shape=frame.shape[:2])
        panels.append(render_one(
            frame, f"GPU top_n={n}", h_g["H"], k1, k2, h_g["intrinsics"],
            h_g["n_inliers"], h_g["mean_err_yd"], res.vp or (0, 0),
        ))

    # Stack panels horizontally (all are PANEL_W wide, same height).
    grid = np.hstack(panels)

    # Source frame on top, resized to panel-grid width.
    grid_w = grid.shape[1]
    src_h = int(round(frame.shape[0] * grid_w / frame.shape[1]))
    src_h = min(src_h, 360)
    src_resized = cv2.resize(frame, (grid_w, src_h))
    full = np.vstack([src_resized, grid])

    out_path = os.path.join(args.out, f"{fid}__top_n_sweep.jpg")
    cv2.imwrite(out_path, full)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
