#!/usr/bin/env python3
"""
Build a 2-class line-segmentation dataset from the self-supervised pool.

For each filtered frame:
  1. Grab the raw DISTORTED frame from its source clip.
  2. Project field-model yard lines and sidelines through H_inv.
  3. FORWARD-DISTORT each projected point via (k1, k2) so the drawn polyline
     follows the true painted curvature.
  4. Rasterize a 2-channel mask (yard line = R channel, sideline = G channel).

Output directory layout (stratified 80/20 by game, matching our other datasets):
  <out-dir>/
    train/images/<frame_id>.jpg
    train/masks/<frame_id>.png         # 3-channel: R=yard, G=side, B=unused
    valid/images/<frame_id>.jpg
    valid/masks/<frame_id>.png
    manifest.json                      # filter params, split sizes, game counts

Labels are drawn on UNdistorted coordinates' forward-distorted image-space
positions. Lines will appear curved in the output masks, matching what the
raw frame shows.
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.apply_homography import field_to_pixel
from src.homography.field_model import YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH

DEFAULT_POOL = os.path.join(PROJECT_ROOT, "output/self_sup_pool_10k")
DEFAULT_CLIPS = os.path.join(PROJECT_ROOT, "videos/clips")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data/line_detection")

SAMPLES_PER_LINE = 500     # polyline density; high enough to stay smooth after distortion
VAL_FRACTION = 0.20
RNG_SEED = 42

# --- Line-thickness rendering ---
# Yard lines: width adapts per line to match the actual painted 4-inch width
# (~0.111 yd) in projected pixels. Sidelines: fixed skinny — we render the
# INNER edge of the painted strip (which is what matters for homography),
# not the whole paint band.
YARD_PAINT_WIDTH_YD = 4.0 / 36.0     # 4 inches in yards = 0.111
YARD_THICKNESS_MIN = 2
YARD_THICKNESS_MAX = 15
SIDE_THICKNESS = 3

# --- Snap-to-white refinement (yard lines only; sidelines stay locked) ---
SNAP_MAX_SHIFT = 15         # max perpendicular shift in pixels
SNAP_MIN_WHITE = 200        # grayscale threshold for "painted" pixel
SNAP_LINE_HALFWIDTH = 2     # half-width for both-edges trigger check
POLY_DEGREE = 4             # polynomial fit order after snap

# --- Filter (on summary_hquality.csv) ---
DEFAULT_MIN_YARD_WHITE = 0.70
DEFAULT_REQUIRE_SIDES_COVERED = True


def forward_distort(xy_undistorted, intr):
    """Apply Brown-Conrady forward distortion to a (N,2) array of undistorted
    pixel coords. Returns distorted pixel coords."""
    fx, fy, cx, cy, k1, k2 = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["k1"], intr["k2"]
    # Normalize
    x_n = (xy_undistorted[:, 0] - cx) / fx
    y_n = (xy_undistorted[:, 1] - cy) / fy
    r2 = x_n * x_n + y_n * y_n
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    # Apply distortion in normalized space, then map back to pixels
    xd = x_n * factor * fx + cx
    yd = y_n * factor * fy + cy
    return np.stack([xd, yd], axis=1)


def draw_polyline_on_mask(mask, pixel_pts, thickness):
    h, w = mask.shape
    pts = np.clip(pixel_pts, -10 * max(h, w), 10 * max(h, w)).astype(np.int32)
    cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False,
                  color=255, thickness=thickness, lineType=cv2.LINE_AA)


def _in_bounds_runs(pts, h, w, pad):
    """Return list of (start, end) index ranges where pts are in-bounds.

    A run is contiguous and has at least 2 samples."""
    in_b = (pts[:, 0] >= -pad) & (pts[:, 0] < w + pad) & \
           (pts[:, 1] >= -pad) & (pts[:, 1] < h + pad)
    runs = []
    start = None
    for i, ok in enumerate(in_b):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    if start is not None and len(in_b) - start >= 2:
        runs.append((start, len(in_b)))
    return runs


def snap_polyline_to_white(pts, gray, max_shift=SNAP_MAX_SHIFT,
                            min_white=SNAP_MIN_WHITE,
                            line_halfwidth=SNAP_LINE_HALFWIDTH):
    """Snap each polyline sample perpendicular to its local tangent onto
    nearest paint. Trigger: move when either edge of the drawn line is off
    paint. Scan: take the first offset where the CENTER hits paint. Keeps
    aligned lines stable; pulls misaligned ones onto paint up to max_shift.
    """
    n = len(pts)
    new_pts = pts.copy().astype(np.float64)
    if n < 3:
        return new_pts
    h, w = gray.shape
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(t)
        if norm < 1e-6:
            continue
        tx, ty = t / norm
        nx, ny = -ty, tx

        def sample(d):
            sx = int(round(pts[i, 0] + d * nx))
            sy = int(round(pts[i, 1] + d * ny))
            if 0 <= sx < w and 0 <= sy < h:
                return int(gray[sy, sx])
            return -1

        def edges_on_white(d):
            return (sample(d - line_halfwidth) >= min_white and
                    sample(d + line_halfwidth) >= min_white)

        # Both edges on paint → don't move.
        if edges_on_white(0):
            continue

        # Scan for nearest offset where center is on paint.
        found_d = None
        for step in range(1, max_shift + 1):
            if sample(-step) >= min_white:
                found_d = -step
                break
            if sample(+step) >= min_white:
                found_d = +step
                break
        if found_d is not None:
            new_pts[i] = pts[i] + found_d * np.array([nx, ny])
    return new_pts


def polyfit_smooth(pts, degree=POLY_DEGREE):
    """Fit a smooth low-degree polynomial through the polyline using the
    principal axis as parameter. Preserves endpoints. Handles the gently-
    curved-by-lens-distortion shape of NFL yard lines in image space.
    """
    if len(pts) < degree + 2:
        return pts
    c = pts.mean(axis=0)
    centered = pts - c
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    t = centered @ d
    r = centered @ n
    coeffs = np.polyfit(t, r, degree)
    r_fit = np.polyval(coeffs, t)
    reconstructed = t[:, None] * d[None, :] + r_fit[:, None] * n[None, :]
    return reconstructed + c


def _draw_field_line(mask, field_pts, H_inv, intr, thickness, snap_gray=None):
    """Project, clip to in-image runs, forward-distort, optionally snap +
    polyfit-smooth (yard lines only), and draw each run.

    When snap_gray is None, we render the raw projection (used for sidelines).
    """
    h, w = mask.shape
    undist_px = field_to_pixel(field_pts, H_inv)
    for s, e in _in_bounds_runs(undist_px, h, w, pad=50):
        seg_und = undist_px[s:e]
        seg_dist = forward_distort(seg_und, intr)
        if snap_gray is not None:
            seg_dist = snap_polyline_to_white(seg_dist, snap_gray)
            seg_dist = polyfit_smooth(seg_dist)
        draw_polyline_on_mask(mask, seg_dist, thickness)


def yard_line_thickness(x_yd, H_inv, intr):
    """Per-yard-line pixel thickness matching the actual painted width.

    Projects two field points 0.111 yd apart (the 4-inch paint width), both
    on this yard line, at the line's vertical midpoint. Distance between the
    projected distorted pixels = paint width in that region of the image.
    """
    c = np.array([[x_yd - YARD_PAINT_WIDTH_YD / 2.0, FIELD_WIDTH / 2.0],
                   [x_yd + YARD_PAINT_WIDTH_YD / 2.0, FIELD_WIDTH / 2.0]])
    und = field_to_pixel(c, H_inv)
    dist = forward_distort(und, intr)
    w = int(round(float(np.linalg.norm(dist[0] - dist[1]))))
    return max(YARD_THICKNESS_MIN, min(YARD_THICKNESS_MAX, w))


def render_masks(frame, H, k1, k2, snap=True):
    """Render 2-channel line masks in DISTORTED image coordinates.

    `frame` is the full BGR frame (needed for snap). Yard lines get snap +
    polyfit refinement and are drawn at a per-frame adaptive thickness that
    matches the actual painted width. Sidelines are drawn skinny at the
    inner edge (field-side); no snap (avoids bench/ad whites).

    Returns (yard_mask, side_mask) — each uint8 (H, W) with painted lines = 255.
    """
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    intr = dict(fx=focal, fy=focal, cx=w / 2.0, cy=h / 2.0, k1=float(k1), k2=float(k2))
    H_inv = np.linalg.inv(H)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if snap else None

    yard = np.zeros((h, w), dtype=np.uint8)
    side = np.zeros((h, w), dtype=np.uint8)

    # Yard lines: snap-to-white + polyfit, thickness per-line matches local paint width
    y_samp = np.linspace(0.0, FIELD_WIDTH, SAMPLES_PER_LINE)
    for x in YARD_LINE_POSITIONS:
        field_pts = np.stack([np.full_like(y_samp, x), y_samp], axis=1)
        thickness = yard_line_thickness(x, H_inv, intr)
        _draw_field_line(yard, field_pts, H_inv, intr, thickness,
                          snap_gray=gray)

    # Sidelines: raw projection, skinny — we mark only the field-side edge
    x_samp = np.linspace(0.0, FIELD_LENGTH, SAMPLES_PER_LINE)
    for yv in (0.0, FIELD_WIDTH):
        field_pts = np.stack([x_samp, np.full_like(x_samp, yv)], axis=1)
        _draw_field_line(side, field_pts, H_inv, intr, SIDE_THICKNESS)

    return yard, side


def grab_frame(mp4, idx):
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def load_pool(summary_csv):
    rows = list(csv.DictReader(open(summary_csv)))
    return [r for r in rows if r.get("solved") == "True"
            and r.get("yard_white_min") not in (None, "", "None")]


def filter_pool(rows, min_yard_white, require_sides_covered):
    """Filter via the per-line whiteness + sideline coverage signals
    computed by scripts/data_prep/filter_h_quality.py."""
    out = []
    for r in rows:
        try:
            ywm = float(r["yard_white_min"])
        except (KeyError, ValueError, TypeError):
            continue
        if ywm < min_yard_white:
            continue
        if require_sides_covered:
            if int(r.get("both_sides_covered", "0") or "0") != 1:
                continue
        out.append(r)
    return out


def stratified_split_by_game(rows, val_fraction, rng):
    """Stratify so every game with >=5 samples contributes val frames proportionally."""
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)

    train_rows, val_rows = [], []
    for game, game_rows in sorted(by_game.items()):
        rng.shuffle(game_rows)
        n_val = max(1, round(len(game_rows) * val_fraction))
        val_rows.extend(game_rows[:n_val])
        train_rows.extend(game_rows[n_val:])
    return train_rows, val_rows


def write_split(rows, pool_dir, clips_dir, split_dir):
    img_dir = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    n_written, n_failed = 0, 0
    for r in rows:
        mp4 = os.path.join(clips_dir, r["game"], r["play"], f"{r['angle']}.mp4")
        frame = grab_frame(mp4, int(r["frame_idx"]))
        if frame is None:
            n_failed += 1; continue

        with open(os.path.join(pool_dir, r["h_path"]), "rb") as f:
            hd = pickle.load(f)
        yard, side = render_masks(frame, hd["H"], hd["k1"], hd["k2"])

        # Pack yard + side into R and G channels. cv2.imwrite assumes BGR
        # input, so combined[..., 2] becomes the RED channel on disk.
        combined = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        combined[..., 2] = yard   # → R channel in saved PNG
        combined[..., 1] = side   # → G channel in saved PNG

        fid = r["frame_id"]
        cv2.imwrite(os.path.join(img_dir, f"{fid}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(mask_dir, f"{fid}.png"), combined)
        n_written += 1

        if n_written % 100 == 0:
            print(f"  wrote {n_written}...")

    return n_written, n_failed


def main(args):
    rows = load_pool(os.path.join(args.pool_dir, "summary_hquality.csv"))
    filtered = filter_pool(rows, args.min_yard_white, args.require_sides_covered)
    print(f"pool: {len(rows)} processed → {len(filtered)} filtered "
          f"(yard_white_min>={args.min_yard_white}, "
          f"require_sides_covered={args.require_sides_covered})")

    if len(filtered) < 20:
        print("too few frames after filter; aborting"); return

    rng = random.Random(RNG_SEED)
    train_rows, val_rows = stratified_split_by_game(filtered, VAL_FRACTION, rng)
    print(f"split: {len(train_rows)} train, {len(val_rows)} val")

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nwriting train split...")
    n_train, f_train = write_split(train_rows, args.pool_dir, args.clips_dir,
                                     os.path.join(args.out_dir, "train"))
    print("\nwriting valid split...")
    n_val, f_val = write_split(val_rows, args.pool_dir, args.clips_dir,
                                 os.path.join(args.out_dir, "valid"))

    per_game = defaultdict(lambda: {"train": 0, "val": 0})
    for r in train_rows:
        per_game[r["game"]]["train"] += 1
    for r in val_rows:
        per_game[r["game"]]["val"] += 1

    manifest = {
        "filter": {
            "min_yard_white": args.min_yard_white,
            "require_sides_covered": args.require_sides_covered,
        },
        "snap": {
            "max_shift_px": SNAP_MAX_SHIFT,
            "min_white": SNAP_MIN_WHITE,
            "line_halfwidth_px": SNAP_LINE_HALFWIDTH,
            "polyfit_degree": POLY_DEGREE,
            "yard_lines_snapped": True,
            "sidelines_snapped": False,
        },
        "pool_size": len(rows),
        "filtered_size": len(filtered),
        "train": n_train, "train_failed": f_train,
        "valid": n_val, "valid_failed": f_val,
        "yard_thickness": f"per-line, paint-width-matched "
                          f"({YARD_THICKNESS_MIN}-{YARD_THICKNESS_MAX} px clamp)",
        "side_thickness_px": SIDE_THICKNESS,
        "samples_per_line": SAMPLES_PER_LINE,
        "mask_format": "3ch PNG, R=yard, G=side, B=0 (BGR convention)",
        "per_game_counts": dict(per_game),
    }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ndone. manifest → {args.out_dir}/manifest.json")
    print(f"train: {n_train} written, {f_train} failed")
    print(f"valid: {n_val} written, {f_val} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=DEFAULT_POOL)
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--min-yard-white", type=float, default=DEFAULT_MIN_YARD_WHITE,
                    help="Per-line min whiteness threshold from filter_h_quality output")
    ap.add_argument("--require-sides-covered", action="store_true",
                    default=DEFAULT_REQUIRE_SIDES_COVERED)
    ap.add_argument("--no-require-sides-covered", action="store_false",
                    dest="require_sides_covered")
    main(ap.parse_args())
