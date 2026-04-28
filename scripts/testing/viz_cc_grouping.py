#!/usr/bin/env python3
"""Render val frames with CC-based yardline groupings overlaid.

For each frame: [src + UNet mask | CC groupings color-coded + polynomial fits].
Per yardline group, all merged-fragment pixels share a color, and the fitted
polynomial is drawn through them.

Usage:
    .venv/bin/python scripts/testing/viz_cc_grouping.py --n 8
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    group_yardline_pixels_cc, group_yardline_pixels,
    fit_line_polynomial, eval_poly,
    run_unet, UNET_YARD_THRESH, UNET_SIDE_THRESH,
)

VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output/cc_grouping")

GROUP_COLORS = [
    (0, 255, 255),     # yellow
    (0, 165, 255),     # orange
    (0, 100, 255),     # red-orange
    (255, 0, 255),     # magenta
    (255, 100, 0),     # blue
    (255, 255, 0),     # cyan
    (100, 255, 100),   # light green
    (0, 255, 100),     # green
    (200, 200, 200),   # gray
    (180, 105, 255),   # pink
    (50, 150, 255),
    (255, 50, 50),
]


def color(i):
    return GROUP_COLORS[i % len(GROUP_COLORS)]


def overlay_pixels(canvas, pts, color, alpha=0.55):
    xs = pts[:, 0].astype(np.int32)
    ys = pts[:, 1].astype(np.int32)
    h, w = canvas.shape[:2]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    orig = canvas[ys, xs].astype(np.float32)
    blended = orig * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    canvas[ys, xs] = blended.astype(np.uint8)


def draw_poly(canvas, line_obj, color, n_pts=300, thickness=2):
    if line_obj.poly is None:
        return
    if line_obj.kind == "yardline":
        ymin = float(line_obj.pixels[:, 1].min())
        ymax = float(line_obj.pixels[:, 1].max())
        ys = np.linspace(ymin, ymax, n_pts)
        xs = eval_poly(line_obj, ys)
    else:
        xmin = float(line_obj.pixels[:, 0].min())
        xmax = float(line_obj.pixels[:, 0].max())
        xs = np.linspace(xmin, xmax, n_pts)
        ys = eval_poly(line_obj, xs)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], False, color, thickness)


def render_one(frame, yard_mask, side_mask, title):
    h, w = frame.shape[:2]

    # Left panel: source + UNet mask overlay
    left = frame.copy()
    left[yard_mask > 0] = (0.4 * left[yard_mask > 0]
                            + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
    left[side_mask > 0] = (0.4 * left[side_mask > 0]
                            + 0.6 * np.array([0, 255, 255])).astype(np.uint8)

    # Right panel: CC groupings
    right = frame.copy()
    yl_groups = group_yardline_pixels_cc(yard_mask)
    for i, lo in enumerate(yl_groups):
        c = color(i)
        overlay_pixels(right, lo.pixels, c, alpha=0.5)
        draw_poly(right, lo, c, thickness=2)

    # Stats text
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"yardlines: {len(yl_groups)}",
    ]
    for i, lo in enumerate(yl_groups):
        rmse = lo.residual_rmse
        rmse_s = f"{rmse:.2f}" if rmse is not None else "?"
        lines.append(f"  yl{i}: n={len(lo.pixels):4d} rmse={rmse_s} "
                     f"peak_x={lo.peak_coord:.0f}")
    y0 = 26
    for k, t in enumerate(lines):
        cv2.putText(right, t, (10, y0 + k * 20), font, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(right, t, (10, y0 + k * 20), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    strip = np.zeros((40, w * 2, 3), dtype=np.uint8)
    cv2.putText(strip, title, (10, 28), font, 0.7, (255, 255, 255),
                1, cv2.LINE_AA)
    return np.vstack([strip, np.hstack([left, right])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images = sorted(f for f in os.listdir(os.path.join(VAL_DIR, "images"))
                    if f.endswith(".jpg") and not f.startswith("._"))
    rng = random.Random(args.seed)
    rng.shuffle(images)
    picks = images[: args.n]

    for fname in picks:
        fid = os.path.splitext(fname)[0]
        frame = cv2.imread(os.path.join(VAL_DIR, "images", fname))
        yard, side = run_unet(frame, UNET_WEIGHTS, device=args.device,
                               yard_thresh=UNET_YARD_THRESH,
                               side_thresh=UNET_SIDE_THRESH)
        img = render_one(frame, yard, side, fid)
        out_path = os.path.join(args.out, f"{fid}__cc.jpg")
        cv2.imwrite(out_path, img)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
