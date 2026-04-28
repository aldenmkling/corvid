#!/usr/bin/env python3
"""Diagnostic viz for grid_solver_v2 — per-line grouping + polynomial fits.

Renders a side-by-side [UNet-mask | pixels-colored-by-group-with-poly-overlay]
for each selected frame so we can eyeball whether grouping + fitting is right.

Uses the GT masks from data/line_detection/valid (so the viz isolates the
grouping/fitting logic from UNet detection quality). Run with --use-unet to
swap in real UNet inference for end-to-end testing.

Usage:
    .venv/bin/python scripts/testing/test_grid_solver_v2.py \
        --n 6 --out output/grid_solver_v2/
    .venv/bin/python scripts/testing/test_grid_solver_v2.py \
        --use-unet --weights models/unet_line_round2_best.pth
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
    GridSolverResult,
    group_yardline_pixels,
    group_sideline_pixels,
    fit_line_polynomial,
    eval_poly,
    solve_grid,
    run_unet,
    run_hash_w18,
    UNET_YARD_THRESH, UNET_SIDE_THRESH,
)


VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output/grid_solver_v2")

UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
W18_WEIGHTS = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")

# Distinct colors for group IDs (BGR)
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
]


def group_color(i: int) -> tuple[int, int, int]:
    return GROUP_COLORS[i % len(GROUP_COLORS)]


def draw_group_pixels(canvas: np.ndarray, pixels: np.ndarray, color, alpha=0.5):
    """Paint the pixels of a grouped line onto canvas (with simple alpha blend)."""
    xs = pixels[:, 0].astype(np.int32)
    ys = pixels[:, 1].astype(np.int32)
    h, w = canvas.shape[:2]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    orig = canvas[ys, xs].astype(np.float32)
    blended = orig * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    canvas[ys, xs] = blended.astype(np.uint8)


def draw_poly(canvas, line_obj, color, n_pts=400, thickness=2):
    """Draw the fitted polynomial as a polyline across the pixel extent."""
    if line_obj.poly is None:
        return
    if line_obj.kind == "yardline":
        ymin = float(line_obj.pixels[:, 1].min())
        ymax = float(line_obj.pixels[:, 1].max())
        ys = np.linspace(ymin, ymax, n_pts)
        xs = eval_poly(line_obj, ys)
    else:  # sideline
        xmin = float(line_obj.pixels[:, 0].min())
        xmax = float(line_obj.pixels[:, 0].max())
        xs = np.linspace(xmin, xmax, n_pts)
        ys = eval_poly(line_obj, xs)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)


def draw_cross(canvas, pt, color, size=12, thickness=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(canvas, (x - size, y), (x + size, y), color, thickness)
    cv2.line(canvas, (x, y - size), (x, y + size), color, thickness)


def draw_keypoint(canvas, pt, color, label=None, radius=8):
    """Filled dot with black outline + optional label."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(canvas, (x, y), radius + 2, (0, 0, 0), -1)
    cv2.circle(canvas, (x, y), radius, color, -1)
    if label:
        cv2.putText(canvas, label, (x + radius + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + radius + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def render_frame(frame: np.ndarray, yard_mask: np.ndarray, side_mask: np.ndarray,
                 title: str, hash_pxs: np.ndarray | None = None,
                 hash_confs: np.ndarray | None = None) -> np.ndarray:
    """Build the [mask overlay | group+keypoints overlay] panel for one frame."""
    from src.homography.grid_solver_v2 import (
        solve_grid, Yardline,
    )

    h, w = frame.shape[:2]

    # Left panel: raw mask overlay (cyan=yard, yellow=side), hashes as white dots
    left = frame.copy()
    left[yard_mask > 0] = (0.4 * left[yard_mask > 0] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
    left[side_mask > 0] = (0.4 * left[side_mask > 0] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
    if hash_pxs is not None:
        for p in hash_pxs:
            draw_cross(left, p, (255, 255, 255), size=6, thickness=1)

    # Right panel: end-to-end grid solve w/ polys + hashes + sideline intersections
    right = frame.copy()

    hashes = hash_pxs if hash_pxs is not None else np.zeros((0, 2))
    result = solve_grid(yard_mask, side_mask, hashes,
                        hash_confs=hash_confs, frame_shape=frame.shape[:2])

    # Color map: one color per yardline's grid_pos (but grid_pos isn't set yet
    # for all groups, so use list index for now)
    for i, yl in enumerate(result.yardlines):
        c = group_color(i)
        draw_group_pixels(right, yl.line.pixels, c, alpha=0.4)
        draw_poly(right, yl.line, c, thickness=2)
        # Paired / singleton hashes
        if yl.far_hash is not None:
            draw_keypoint(right, yl.far_hash, c, label="F")
        if yl.near_hash is not None:
            draw_keypoint(right, yl.near_hash, c, label="N")
        # Sideline intersections
        if yl.far_sideline is not None:
            draw_keypoint(right, yl.far_sideline, c, label="fS")
        if yl.near_sideline is not None:
            draw_keypoint(right, yl.near_sideline, c, label="nS")

    # Sidelines themselves (drawn with distinct palette offset)
    sl_colors = [(255, 255, 255), (200, 200, 0)]
    for j, sl in enumerate([result.far_sideline, result.near_sideline]):
        if sl is None:
            continue
        c = sl_colors[j]
        draw_group_pixels(right, sl.pixels, c, alpha=0.3)
        draw_poly(right, sl, c, thickness=2)

    # Draw unassigned hash detections (outliers) in grey
    assigned = set()
    for yl in result.yardlines:
        for pt in (yl.far_hash, yl.near_hash):
            if pt is not None:
                assigned.add(tuple(round(float(x), 1) for x in pt))
    if hash_pxs is not None:
        for p in hash_pxs:
            key = tuple(round(float(x), 1) for x in p)
            if key not in assigned:
                draw_cross(right, p, (128, 128, 128), size=8, thickness=1)

    # Annotate
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"yardlines: {len(result.yardlines)}  sidelines: "
        f"{(result.far_sideline is not None) + (result.near_sideline is not None)}",
    ]
    n_assigned = sum(1 for yl in result.yardlines
                      if yl.far_hash is not None or yl.near_hash is not None)
    lines.append(f"hashes: {len(hash_pxs) if hash_pxs is not None else 0} "
                  f"({n_assigned} assigned)")
    for i, yl in enumerate(result.yardlines):
        rmse = yl.line.residual_rmse
        rmse_s = f"{rmse:.2f}" if rmse is not None else "?"
        f_s = "F" if yl.far_hash is not None else "."
        n_s = "N" if yl.near_hash is not None else "."
        fs_s = "fS" if yl.far_sideline is not None else ".."
        ns_s = "nS" if yl.near_sideline is not None else ".."
        lines.append(f"  yl{i}: n={len(yl.line.pixels):4d} rmse={rmse_s} "
                     f"[{f_s}{n_s} {fs_s} {ns_s}]")
    y0 = 26
    for k, text in enumerate(lines):
        cv2.putText(right, text, (10, y0 + k * 20), font, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(right, text, (10, y0 + k * 20), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Title strip
    strip = np.zeros((40, w * 2, 3), dtype=np.uint8)
    cv2.putText(strip, title, (10, 28), font, 0.8, (255, 255, 255),
                1, cv2.LINE_AA)

    combined = np.vstack([strip, np.hstack([left, right])])
    return combined


def iter_val_frames(use_unet: bool, weights: str, device: str, n: int, seed: int,
                    run_hashes: bool = True, hash_weights: str = W18_WEIGHTS):
    """Yield (frame_bgr, yard_mask, side_mask, hash_pxs, hash_confs, title)."""
    images = sorted(f for f in os.listdir(os.path.join(VAL_DIR, "images"))
                    if f.endswith(".jpg") and not f.startswith("._"))
    rng = random.Random(seed)
    rng.shuffle(images)
    picks = images[:n]
    for fname in picks:
        fid = os.path.splitext(fname)[0]
        img_path = os.path.join(VAL_DIR, "images", fname)
        frame = cv2.imread(img_path)
        if use_unet:
            yard_mask, side_mask = run_unet(frame, weights, device=device,
                                             yard_thresh=UNET_YARD_THRESH,
                                             side_thresh=UNET_SIDE_THRESH)
            src = "unet"
        else:
            mask_path = os.path.join(VAL_DIR, "masks", f"{fid}.png")
            m = cv2.imread(mask_path)
            yard_mask = (m[..., 2] > 127).astype(np.uint8)
            side_mask = (m[..., 1] > 127).astype(np.uint8)
            src = "gt"

        if run_hashes:
            hash_pxs, hash_confs = run_hash_w18(frame, hash_weights, device=device)
        else:
            hash_pxs, hash_confs = np.zeros((0, 2)), np.zeros(0)

        yield frame, yard_mask, side_mask, hash_pxs, hash_confs, f"{fid}  ({src})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--use-unet", action="store_true",
                    help="Run UNet inference instead of using GT masks")
    ap.add_argument("--weights", default=UNET_WEIGHTS)
    ap.add_argument("--hash-weights", default=W18_WEIGHTS)
    ap.add_argument("--no-hashes", action="store_true",
                    help="Skip hash detection (yardline grouping only)")
    ap.add_argument("--device", default="mps",
                    help="cpu|cuda|mps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for frame, yard_mask, side_mask, hash_pxs, hash_confs, title in iter_val_frames(
        args.use_unet, args.weights, args.device, args.n, args.seed,
        run_hashes=not args.no_hashes, hash_weights=args.hash_weights,
    ):
        img = render_frame(frame, yard_mask, side_mask, title,
                           hash_pxs=hash_pxs, hash_confs=hash_confs)
        fid = title.split(" ")[0]
        out_path = os.path.join(args.out, f"{fid}__v2_groups.jpg")
        cv2.imwrite(out_path, img)
        print(f"  wrote {out_path}")

    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
