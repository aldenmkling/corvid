#!/usr/bin/env python3
"""Mine grouped-mask crops + NGS-position labels for the painted-number
classifier.

For each clip with known g0_NGS_x, sample N frames, run the rectify pass-1
pipeline (line/hash/number UNets + grouping + side classification), and for
each painted-number group:
  1. Derive the absolute NGS class label from g0 + 5·g_index:
       10L, 20L, 30L, 40L, 50, 40R, 30R, 20R, 10R
     Skip groups on 5-yardlines (no painted number, label=None).
  2. Build a binary mask containing ONLY this group's pixels (zero everything
     else in the frame).
  3. Crop to the group's bbox + small margin.
  4. Pad to square (preserves digit aspect ratio).
  5. Resize to 64×64.
  6. Save to data/number_classifier/round1/<label>/<stem>.png

Same per-clip bootstrap + per-frame logic as mine_yardline_numbers.py;
single-line additions wire in the painted_numbers.process_frame call to
get groups and the per-group crop emission. Inference at runtime will use
the same crop preprocessing — only difference is we know g0 here, so we
can derive the label rather than predict it.
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography import painted_numbers
from src.homography.distortion import CameraIntrinsics
from src.homography.grid_solver_v2 import group_yardline_pixels_cc

from src.homography.yardline_tracker import (
    YardlineTracker, group_sideline_pixels_cc as group_sideline_pixels,
)
from src.homography.line_fit import (
    total_mse, fit_yardline_undistorted, fit_sideline_undistorted,
)

from src.homography.rectify import (
    run_specialists, LINE_WEIGHTS, HASH_WEIGHTS, NUMBER_WEIGHTS,
    detect_hash_rows,
    HashRowTracker,
    NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL, YD_PER_GRID,
)


# ── Tunables ───────────────────────────────────────────────────────────────
CROP_SIZE = 64       # 64×64 input to mit_b0 classifier
MARGIN_PX = 5        # padding around the group's bbox before pad+resize


# ── Clip registry (same as mine_yardline_numbers.py) ───────────────────────
CLIPS = [
    # Round 1
    ("2019092204", "play_001",  45),
    ("2019092204", "play_087",  45),
    ("2019102712", "play_046",  20),
    ("2024090801", "play_001",  35),
    ("2024090801", "play_050",  30),
    ("2024090801", "play_115",  15),
    ("2024102701", "play_001",  40),
    ("2024102701", "play_044",  30),
    ("2024111001", "play_060",  70),
    # Round 2
    ("2024090802", "play_020",  10),
    ("2024090802", "play_120",  25),
    ("2024091501", "play_030",  55),
    ("2024092201", "play_040",  30),
    ("2024092201", "play_130",  45),
    ("2019092204", "play_120",  30),
    ("2024111001", "play_030",  10),
    # Round 3
    ("2019092204", "play_050",  40),
    ("2019102712", "play_010",  40),
    ("2019102712", "play_100",  30),
    ("2024090801", "play_020",  35),
    ("2024091501", "play_080",  15),
    ("2024092201", "play_080",  10),
    ("2024102701", "play_120",  10),
    # Targeted top-up for sparse classes (10R, 20R)
    ("2024091501", "play_041",  90),
    # Holdout-game (2024100601) plays added for font diversity
    ("2024100601", "play_028",  85),
    ("2024100601", "play_160",  85),
    ("2024100601", "play_075",  40),
    ("2024100601", "play_110",  35),
]


def derive_label(g0_ngs_x, g_index):
    """NGS-position class label for the yardline at g_index. Returns None
    for 5-yardlines (no painted number) and goal lines / off-field."""
    x_ngs = g0_ngs_x + YD_PER_GRID * g_index
    yd_from_left = x_ngs - 10.0
    if abs(round(yd_from_left / 10.0) * 10.0 - yd_from_left) > 0.5:
        return None                                   # 5-yard line
    if yd_from_left < 9.5 or yd_from_left > 90.5:
        return None                                   # goal line / off-field
    num = int(round(min(yd_from_left, 100 - yd_from_left)))
    if num == 50:
        return "50"
    return f"{num}{'L' if x_ngs < 60.0 else 'R'}"


def crop_group_mask_to_square(group, image_h, image_w,
                                 margin_px=MARGIN_PX, size=CROP_SIZE):
    """Binary mask of ONLY this group's pixels → crop bbox+margin → pad to
    square preserving aspect ratio → resize to size×size. Returns uint8."""
    xs = group["xs_all"]
    ys = group["ys_all"]
    if len(xs) == 0:
        return None

    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[ys, xs] = 255

    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(image_w, int(xs.max()) + margin_px + 1)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(image_h, int(ys.max()) + margin_px + 1)
    crop = mask[y0:y1, x0:x1]

    h_c, w_c = crop.shape
    if h_c > w_c:
        pad = h_c - w_c
        crop = np.pad(crop, ((0, 0), (pad // 2, pad - pad // 2)),
                       mode='constant')
    elif w_c > h_c:
        pad = w_c - h_c
        crop = np.pad(crop, ((pad // 2, pad - pad // 2), (0, 0)),
                       mode='constant')

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ── Per-clip pipeline state ────────────────────────────────────────────────
def bootstrap_clip(clip_path, g_min, g_max, device):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    ok, frame0 = cap.read()
    if not ok:
        cap.release(); return None
    h, w = frame0.shape[:2]
    focal = float(max(h, w)); cx, cy = w / 2.0, h / 2.0

    yard0, side0, _ = run_specialists(
        frame0, LINE_WEIGHTS, HASH_WEIGHTS, device)
    yl0 = group_yardline_pixels_cc(yard0)
    sl0 = group_sideline_pixels(side0)
    line_pts = [g.pixels for g in yl0] + [g.pixels for g in sl0]
    line_kinds = ["yardline"] * len(yl0) + ["sideline"] * len(sl0)
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)

    yl_tracker = YardlineTracker(g_min=g_min, g_max=g_max, frame_h=h)
    fits_yl0 = [fit_yardline_undistorted(g.pixels, intr) for g in yl0]
    yl_tracker.init_from(fits_yl0, cy)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return {
        "cap": cap, "h": h, "w": w, "K": K, "dist": dist, "k1": k1, "intr": intr,
        "yl_tracker": yl_tracker,
        "hash_tracker": HashRowTracker(image_w=w),
        "number_tracker": painted_numbers.NumberSideTracker(),
    }


def step_frame_get_groups(state, frame, device):
    """Run UNets + grouping + side classification. Returns (g_index, groups,
    cc_pixels). cc_pixels and groups follow painted_numbers.process_frame."""
    intr = state["intr"]; w = state["w"]; h = state["h"]
    K = state["K"]; dist = state["dist"]; k1 = state["k1"]

    yard, side, hash_ = run_specialists(
        frame, LINE_WEIGHTS, HASH_WEIGHTS, device)
    yl = group_yardline_pixels_cc(yard)
    sl = group_sideline_pixels(side)
    fits_yl = [fit_yardline_undistorted(g.pixels, intr) for g in yl]
    fits_sl = [fit_sideline_undistorted(g.pixels, intr) for g in sl]

    if state["yl_tracker"].last_fit:
        fits_kept, g_index, _, _ = state["yl_tracker"].update(fits_yl, h / 2.0)
    else:
        init = state["yl_tracker"].init_from(fits_yl, h / 2.0)
        if init is None:
            fits_kept, g_index = [], np.array([], dtype=int)
        else:
            fits_kept, g_index, _ = init

    rows_raw = detect_hash_rows(hash_, intr)
    rows = state["hash_tracker"].observe(rows_raw)

    num_mask_d = painted_numbers.predict_mask(frame, NUMBER_WEIGHTS, device)
    if abs(k1) > 1e-6:
        num_mask_u = cv2.undistort(num_mask_d, K, dist)
        num_mask_u = (num_mask_u > 127).astype(np.uint8) * 255
    else:
        num_mask_u = num_mask_d

    _, num_dbg = painted_numbers.process_frame(
        num_mask_u, fits_kept, rows, fits_sl, g_index, h, w,
        state["number_tracker"])
    return g_index, num_dbg["groups"], num_dbg["cc_pixels"]


def process_clip(game, play, g0_ngs_x, n_random, seed, out_root, device):
    clip_path = os.path.join(PROJECT_ROOT,
                                f"videos/clips/{game}/{play}/sideline.mp4")
    if not os.path.exists(clip_path):
        print(f"  [skip] missing {clip_path}"); return 0, {}
    g_min = int((NGS_X_LEFT_GOAL - g0_ngs_x) / YD_PER_GRID)
    g_max = int((NGS_X_RIGHT_GOAL - g0_ngs_x) / YD_PER_GRID)
    state = bootstrap_clip(clip_path, g_min, g_max, device)
    if state is None:
        print(f"  [skip] bootstrap failed: {clip_path}"); return 0, {}
    cap = state["cap"]
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rng = np.random.default_rng(seed + abs(hash(f"{game}/{play}")) % (2**32))
    target = min(n_random, n_frames)
    frame_indices = sorted(rng.choice(n_frames, size=target,
                                          replace=False).tolist())
    target_set = set(frame_indices)
    last_target = max(frame_indices)

    n_emitted = 0
    label_counts = defaultdict(int)
    h, w = state["h"], state["w"]
    for fi in range(last_target + 1):
        ok, frame = cap.read()
        if not ok:
            break
        g_index, groups, _ = step_frame_get_groups(state, frame, device)
        if fi not in target_set:
            continue
        for grp in groups:
            yl_idx = grp.get("yardline_idx", -1)
            if yl_idx < 0 or yl_idx >= len(g_index):
                continue
            side = grp.get("side")
            if side not in ("near", "far"):
                continue
            g = int(g_index[yl_idx])
            label = derive_label(g0_ngs_x, g)
            if label is None:
                continue
            crop = crop_group_mask_to_square(grp, h, w)
            if crop is None:
                continue
            out_dir = os.path.join(out_root, label)
            os.makedirs(out_dir, exist_ok=True)
            stem = f"{game}_{play}_f{fi:04d}_g{g:+03d}_{side}"
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), crop)
            n_emitted += 1
            label_counts[label] += 1
    cap.release()
    print(f"  {game}/{play}: emitted {n_emitted} crops "
          f"[{', '.join(f'{k}:{v}' for k, v in sorted(label_counts.items()))}]")
    return n_emitted, label_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT,
                                                    "data/number_classifier/round1"))
    ap.add_argument("--n-random", type=int, default=20,
                     help="random frames per clip (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--single-clip", action="append", default=None,
                     help="game/play (repeat for multiple); restricts to listed clips")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clips = CLIPS
    if args.single_clip:
        wanted = set(args.single_clip)
        clips = [c for c in CLIPS if f"{c[0]}/{c[1]}" in wanted]
        if not clips:
            print(f"no matching clips"); return

    total = 0
    overall = defaultdict(int)
    for game, play, g0 in clips:
        n, counts = process_clip(game, play, g0, args.n_random, args.seed,
                                    args.out, args.device)
        total += n
        for k, v in counts.items():
            overall[k] += v
    print(f"\ntotal: {total} crops across {len(clips)} clips → {args.out}")
    print("class breakdown:")
    for k in ('10L', '20L', '30L', '40L', '50', '40R', '30R', '20R', '10R'):
        print(f"  {k:4s} {overall[k]:5d}")


if __name__ == "__main__":
    main()
