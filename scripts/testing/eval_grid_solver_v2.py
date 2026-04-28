#!/usr/bin/env python3
"""Quick end-to-end eval of grid_solver_v2 on val frames.

Runs UNet + W18 + grid solve + distortion + homography on each frame;
reports #correspondences, #inliers, (k1, k2), and reprojection RMSE.

Also optionally side-by-side renders a rectified top-down for sanity.

Usage:
    .venv/bin/python scripts/testing/eval_grid_solver_v2.py --n 8 --seed 42
    .venv/bin/python scripts/testing/eval_grid_solver_v2.py --rectify --n 4
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import solve_frame_full
from src.homography.field_model import FIELD_LENGTH, FIELD_WIDTH


VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
W18_WEIGHTS = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output/grid_solver_v2_eval")


def render_rectified(frame, H, k1, k2, intrinsics,
                     yd_per_px: float = 0.1, margin_yd: float = 5.0,
                     max_canvas_px: int = 2400):
    """Top-down view via H (after lens undistort), auto-sized to the frame.

    Without an NGS anchor, x is in field-yards but its origin is whatever the
    grid solver assigned to grid_pos=0; we don't force it onto the 0–100yd
    field. Canvas extent comes from projecting the four image corners through
    H, then padding by `margin_yd`.
    """
    from src.homography.distortion import undistort_points

    H_src, W_src = frame.shape[:2]

    # Project image corners → field to get the visible bbox.
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

    # Cap canvas size — extreme distortion can blow up the bbox.
    w_yd = x_max - x_min
    h_yd = y_max - y_min
    scale = 1.0 / yd_per_px
    if max(w_yd, h_yd) * scale > max_canvas_px:
        scale = max_canvas_px / max(w_yd, h_yd)
    w_out = max(64, int(round(w_yd * scale)))
    h_out = max(64, int(round(h_yd * scale)))
    yd_per_px_x = w_yd / w_out
    yd_per_px_y = h_yd / h_out

    # Build the field-coord grid spanning the visible bbox. y runs high → low
    # so far sideline (large y) ends up at top, matching broadcast orientation.
    xs_field = np.linspace(x_min, x_max, w_out)
    ys_field = np.linspace(y_max, y_min, h_out)
    gx, gy = np.meshgrid(xs_field, ys_field)
    field_pts = np.stack([gx.ravel(), gy.ravel(), np.ones_like(gx.ravel())], axis=1)

    H_inv = np.linalg.inv(H)
    src_undist = (H_inv @ field_pts.T).T
    src_undist = src_undist[:, :2] / src_undist[:, 2:3]

    # Redistort: r' = r(1 + k1 r² + k2 r⁴), applied in normalized coords.
    # At large r, the factor can flip sign or blow up — that's pure extrapolation
    # outside the calibrated region. Detect and mask those before remap.
    x_n = (src_undist[:, 0] - intrinsics.cx) / intrinsics.fx
    y_n = (src_undist[:, 1] - intrinsics.cy) / intrinsics.fy
    r2 = x_n ** 2 + y_n ** 2
    factor = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d = x_n * factor * intrinsics.fx + intrinsics.cx
    y_d = y_n * factor * intrinsics.fy + intrinsics.cy
    map_x = x_d.reshape(h_out, w_out).astype(np.float32)
    map_y = y_d.reshape(h_out, w_out).astype(np.float32)

    # Mask out-of-frame lookups. Three failure modes to catch:
    #   1. Redistorted lookup outside frame bounds → trivial mask.
    #   2. Negative redistortion factor (r so large that 1+k1r²+k2r⁴ < 0,
    #      flipping the radial direction → bogus wrap-around).
    #   3. Undistorted lookup wildly outside the source frame (even if
    #      redistortion folded it back inside). Cap at a generous margin.
    H_src, W_src = frame.shape[:2]
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

    # Grid overlay every 5yd in x, near/far hash + sidelines in y.
    def x_to_px(x_yd: float) -> int:
        return int(round((x_yd - x_min) / yd_per_px_x))

    def y_to_px(y_yd: float) -> int:
        return int(round((y_max - y_yd) / yd_per_px_y))

    x_lo = int(np.ceil(x_min / 5.0)) * 5
    x_hi = int(np.floor(x_max / 5.0)) * 5
    for x_yd in range(x_lo, x_hi + 1, 5):
        x_px = x_to_px(x_yd)
        if 0 <= x_px < w_out:
            color = (255, 255, 255) if x_yd % 10 == 0 else (180, 180, 180)
            cv2.line(warped, (x_px, 0), (x_px, h_out - 1), color, 1)
            cv2.putText(warped, f"{x_yd:+d}", (x_px + 3, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Horizontal references: y=0 (near sideline), y=FIELD_WIDTH (far),
    # y=HASH_Y_NEAR/FAR. Drawn only if within visible bbox.
    from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--base-ngs-x", type=float, default=0.0,
                    help="Arbitrary anchor (reprojection err is anchor-invariant)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rectify", action="store_true",
                    help="Render side-by-side [frame | rectified top-down]")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images = sorted(f for f in os.listdir(os.path.join(VAL_DIR, "images"))
                    if f.endswith(".jpg") and not f.startswith("._"))
    rng = random.Random(args.seed)
    rng.shuffle(images)
    picks = images[: args.n]

    print(f"{'frame':<55}  {'ncor':>4}  {'inlr':>4}  {'k1':>7}  {'k2':>7}  "
          f"{'mean':>5}  {'med':>5}  {'max':>5}")
    print("-" * 108)

    for fname in picks:
        fid = os.path.splitext(fname)[0]
        frame = cv2.imread(os.path.join(VAL_DIR, "images", fname))
        result, homo = solve_frame_full(
            frame, UNET_WEIGHTS, W18_WEIGHTS, base_ngs_x=args.base_ngs_x,
            device=args.device,
        )
        ncor = homo["n_correspondences"]
        inlr = homo["n_inliers"]
        k1 = homo.get("k1", 0.0)
        k2 = homo.get("k2", 0.0)
        mean = homo["mean_err_yd"]
        med = homo["median_err_yd"]
        mx = homo["max_err_yd"]
        print(f"{fid:<55}  {ncor:4d}  {inlr:4d}  {k1:+.4f}  {k2:+.4f}  "
              f"{mean:5.2f}  {med:5.2f}  {mx:5.2f}")

        if args.rectify and homo["H"] is not None:
            warped = render_rectified(frame, homo["H"], k1, k2, homo["intrinsics"])
            h = min(frame.shape[0], warped.shape[0])
            fr = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
            wr = cv2.resize(warped, (int(warped.shape[1] * h / warped.shape[0]), h))
            combined = np.hstack([fr, wr])
            out_path = os.path.join(args.out, f"{fid}__rectify.jpg")
            cv2.imwrite(out_path, combined)


if __name__ == "__main__":
    main()
