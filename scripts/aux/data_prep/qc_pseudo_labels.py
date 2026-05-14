"""Random-sample y/n inspection of farmed pseudo-labels.

Picks N (clip, frame) pairs at random across data/pseudo_labels/*.npz,
shows source + rectified view side-by-side with field grid + projected
tokens overlaid. You press:
  y  → accept (label looks correct)
  n  → reject (label looks wrong / bad H)
  s  → skip (unclear, exclude from count)
  q  → quit early

Saves per-sample decisions to data/pseudo_labels_qc.json with summary
stats. Aborts training-readiness if accept rate < threshold.

Usage:
  python scripts/data_prep/qc_pseudo_labels.py
  python scripts/data_prep/qc_pseudo_labels.py --n 50 --seed 42
  python scripts/data_prep/qc_pseudo_labels.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)

SRC_W, SRC_H = 1280, 720          # canonical clip resolution
CANVAS_YDS_PER_PX = 1 / 8         # 8 px per yard
CANVAS_W = int(FIELD_LENGTH / CANVAS_YDS_PER_PX)   # 960
CANVAS_H = int(FIELD_WIDTH / CANVAS_YDS_PER_PX)    # 426


def field_to_canvas(field_xy):
    """Map NGS coords (yards) to canvas pixel coords. Y is flipped so
    near sideline (y=0) is at bottom (matches rectify.py convention)."""
    x_yd, y_yd = field_xy[..., 0], field_xy[..., 1]
    px = x_yd / CANVAS_YDS_PER_PX
    py = CANVAS_H - (y_yd / CANVAS_YDS_PER_PX)
    return np.stack([px, py], axis=-1)


def render_rectified(frame_bgr, H):
    """Warp source frame into NGS-yards canvas. Returns CANVAS_H × CANVAS_W
    BGR image with field grid drawn on top."""
    # H maps source pixels → NGS yards. To warp source → canvas pixels
    # we compose with the field_to_canvas affine.
    A = np.array([
        [1.0 / CANVAS_YDS_PER_PX, 0.0, 0.0],
        [0.0, -1.0 / CANVAS_YDS_PER_PX, CANVAS_H],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_full = A @ H
    warped = cv2.warpPerspective(frame_bgr, H_full, (CANVAS_W, CANVAS_H))
    # Field grid overlay.
    grid = warped.copy()
    # Sidelines (y=0, y=FIELD_WIDTH) — white.
    for y in (0.0, FIELD_WIDTH):
        p0 = field_to_canvas(np.array([0.0, y]))
        p1 = field_to_canvas(np.array([FIELD_LENGTH, y]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (255, 255, 255), 1, cv2.LINE_AA)
    # Yardlines every 5 yd in [10, 110] — green.
    for x in range(10, 115, 5):
        p0 = field_to_canvas(np.array([x, 0.0]))
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH]))
        color = (0, 255, 0) if x % 10 == 0 else (0, 180, 0)
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 color, 1, cv2.LINE_AA)
    # Hash lines — cyan.
    for y in (HASH_Y_NEAR, HASH_Y_FAR):
        p0 = field_to_canvas(np.array([10.0, y]))
        p1 = field_to_canvas(np.array([110.0, y]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (255, 255, 0), 1, cv2.LINE_AA)
    # Endzone lines (x=0, x=120) — yellow.
    for x in (0.0, FIELD_LENGTH):
        p0 = field_to_canvas(np.array([x, 0.0]))
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (0, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(warped, 0.7, grid, 0.3, 0)


def annotate_source(frame_bgr, tokens, type_idx, true_class):
    """Draw token centroids on source. Color by type:
       0 (yard) green, 1 (side) orange, 2 (num) yellow, 3 (hash) cyan."""
    out = frame_bgr.copy()
    palette = {
        0: (0, 200, 0),     # yard
        1: (0, 165, 255),   # side
        2: (0, 255, 255),   # num
        3: (255, 200, 0),   # hash
    }
    for i in range(len(tokens)):
        cx = float(tokens[i, 4]) * SRC_W
        cy = float(tokens[i, 5]) * SRC_H
        col = palette.get(int(type_idx[i]), (200, 200, 200))
        cv2.circle(out, (int(cx), int(cy)), 4, col, -1, cv2.LINE_AA)
        cv2.circle(out, (int(cx), int(cy)), 6, (0, 0, 0), 1, cv2.LINE_AA)
        if int(true_class[i]) >= 0:
            cv2.putText(out, str(int(true_class[i])),
                       (int(cx) + 8, int(cy) - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def npz_to_clip_path(npz_name):
    """2019092204_play_005_sideline → videos/clips/2019092204/play_005/sideline.mp4"""
    stem = npz_name.replace(".npz", "")
    parts = stem.split("_")
    # game = parts[0], play_<n> = parts[1] + "_" + parts[2], rest = sideline/endzone
    return os.path.join(PROJECT_ROOT, "videos", "clips",
                         parts[0], f"{parts[1]}_{parts[2]}",
                         f"{'_'.join(parts[3:])}.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/pseudo_labels")
    ap.add_argument("--manifest",
                   default="data/h_pool_and_intrinsics.json",
                   help="for per-clip K/dist (H is in undistorted-pixel space)")
    ap.add_argument("--out", default="data/pseudo_labels_qc.json")
    ap.add_argument("--n", type=int, default=50,
                   help="how many frames to inspect")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                   help="resume from existing --out file, skip done frames")
    args = ap.parse_args()

    npz_dir = os.path.join(PROJECT_ROOT, args.npz_dir)
    out_path = os.path.join(PROJECT_ROOT, args.out)
    manifest_path = os.path.join(PROJECT_ROOT, args.manifest)
    with open(manifest_path) as fp:
        intr_by_clip = json.load(fp)["intrinsics_by_clip"]

    rng = random.Random(args.seed)
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    print(f"Found {len(files)} npz files.")

    # Sample (npz, in_array_idx) pairs. Each npz has different n_frames.
    samples = []
    seen = set()
    while len(samples) < args.n:
        f = rng.choice(files)
        if f.endswith(".npz") is False: continue
        d = np.load(os.path.join(npz_dir, f), allow_pickle=True)
        n = len(d["frame_idx"])
        if n == 0: continue
        i = rng.randrange(n)
        key = (f, i)
        if key in seen: continue
        seen.add(key)
        samples.append(key)

    # Load existing decisions if resuming.
    decisions = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path) as fp:
            saved = json.load(fp)
        decisions = saved.get("decisions", {})
        print(f"Resuming with {len(decisions)} prior decisions.")

    def save():
        n_y = sum(1 for v in decisions.values() if v == "y")
        n_n = sum(1 for v in decisions.values() if v == "n")
        n_s = sum(1 for v in decisions.values() if v == "s")
        denom = max(1, n_y + n_n)
        with open(out_path, "w") as fp:
            json.dump({
                "n_total": len(samples),
                "n_judged": len(decisions),
                "n_yes": n_y,
                "n_no": n_n,
                "n_skip": n_s,
                "accept_rate": n_y / denom,
                "decisions": decisions,
            }, fp, indent=2)

    for k, (npz_name, fi) in enumerate(samples, 1):
        key = f"{npz_name}#{fi}"
        if key in decisions: continue

        d = np.load(os.path.join(npz_dir, npz_name), allow_pickle=True)
        frame_idx = int(d["frame_idx"][fi])
        H = np.asarray(d["H_used"][fi], dtype=np.float64)
        tokens = np.asarray(d["tokens"][fi])
        type_idx = np.asarray(d["type_idx"][fi])
        true_class = np.asarray(d["true_class"][fi])
        h_source = str(d["h_source"][fi])

        clip_path = npz_to_clip_path(npz_name)
        if not os.path.exists(clip_path):
            print(f"  [missing] {clip_path}")
            decisions[key] = "s"; save(); continue

        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  [read-fail] {clip_path}#{frame_idx}")
            decisions[key] = "s"; save(); continue
        if frame.shape[1] != SRC_W or frame.shape[0] != SRC_H:
            frame = cv2.resize(frame, (SRC_W, SRC_H))

        # H is in UNDISTORTED-pixel space (matches farm_pseudo_labels.py).
        # Undistort the frame before warping or token centroids will land
        # in the wrong place.
        rel = os.path.relpath(clip_path, os.path.join(PROJECT_ROOT, "videos/clips"))
        intr = intr_by_clip.get(rel, {})
        K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)
        frame_u = cv2.undistort(frame, K, dist)

        try:
            rect = render_rectified(frame_u, H)
        except Exception as e:
            print(f"  [render-fail] {key}: {e}")
            decisions[key] = "s"; save(); continue
        src = annotate_source(frame_u, tokens, type_idx, true_class)

        # Composite: source on top, rectified below.
        # Scale rectified up to source width.
        scale = SRC_W / CANVAS_W
        rect_w = SRC_W
        rect_h = int(CANVAS_H * scale)
        rect_show = cv2.resize(rect, (rect_w, rect_h))
        composite = np.vstack([src, rect_show])
        # HUD.
        hud = (f"[{k}/{len(samples)}]  {npz_name}#{frame_idx}  "
               f"H={h_source}  n_tok={len(tokens)}  "
               f"y=accept  n=reject  s=skip  q=quit")
        cv2.putText(composite, hud, (10, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(composite, hud, (10, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("pseudo-label QC", composite)
        while True:
            k_pressed = cv2.waitKey(0) & 0xFF
            if k_pressed in (ord("y"), ord("Y")):
                decisions[key] = "y"; break
            if k_pressed in (ord("n"), ord("N")):
                decisions[key] = "n"; break
            if k_pressed in (ord("s"), ord("S")):
                decisions[key] = "s"; break
            if k_pressed in (ord("q"), ord("Q"), 27):
                save()
                print("Quitting early.")
                cv2.destroyAllWindows()
                _print_summary(decisions, samples)
                return
        save()

    cv2.destroyAllWindows()
    _print_summary(decisions, samples)


def _print_summary(decisions, samples):
    n_y = sum(1 for v in decisions.values() if v == "y")
    n_n = sum(1 for v in decisions.values() if v == "n")
    n_s = sum(1 for v in decisions.values() if v == "s")
    n_j = n_y + n_n + n_s
    denom = max(1, n_y + n_n)
    rate = n_y / denom
    print("\n" + "=" * 50)
    print(f"  Judged:       {n_j}/{len(samples)}")
    print(f"  Yes (accept): {n_y}")
    print(f"  No  (reject): {n_n}")
    print(f"  Skip:         {n_s}")
    print(f"  Accept rate:  {rate*100:.1f}%  ({n_y}/{n_y+n_n} non-skip)")
    print("=" * 50)
    if rate >= 0.95:
        print("  ✓ ≥95% — ready for training.")
    elif rate >= 0.90:
        print("  ~ 90-95% — proceed with caution. Inspect rejected samples.")
    else:
        print("  ✗ <90% — debug before training!")


if __name__ == "__main__":
    main()
