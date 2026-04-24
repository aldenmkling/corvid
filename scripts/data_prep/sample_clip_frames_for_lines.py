#!/usr/bin/env python3
"""
Self-supervised line-label source collection.

Samples random frames from clip videos (non-holdout games, sideline angle
only for now), runs the current HRNet + grid solver + findHomography
pipeline on each, and saves (H, reprojection error, correspondence count)
per frame. Output is the candidate pool for line-label rendering.

Usage:
  Dry run (200 frames):
    python scripts/data_prep/sample_clip_frames_for_lines.py --n-frames 200

  Full run (50k frames on pod):
    python scripts/data_prep/sample_clip_frames_for_lines.py --n-frames 50000

The script ONLY collects + evaluates. Label rendering is a separate script
(to run on the filtered subset).
"""

import argparse
import csv
import os
import pickle
import random
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines, run_hrnet, extract_peaks,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.apply_homography import field_to_pixel

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
DEFAULT_CLIPS_DIR = os.path.join(PROJECT_ROOT, "videos", "clips")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "output", "self_sup_pool")
HOLDOUT_GAMES = {"2024100601", "2024122201"}  # per MEMORY.md
HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30
DUMMY_BASE_NGS_X = 50.0


def enumerate_clips(clips_dir, angles=("sideline",)):
    """List all (game, play, angle, mp4_path) tuples, skipping holdouts."""
    clips = []
    for game in sorted(os.listdir(clips_dir)):
        if game in HOLDOUT_GAMES:
            continue
        game_dir = os.path.join(clips_dir, game)
        if not os.path.isdir(game_dir):
            continue
        for play in sorted(os.listdir(game_dir)):
            play_dir = os.path.join(game_dir, play)
            if not os.path.isdir(play_dir) or not play.startswith("play_"):
                continue
            for angle in angles:
                mp4 = os.path.join(play_dir, f"{angle}.mp4")
                if os.path.exists(mp4):
                    clips.append((game, play, angle, mp4))
    return clips


def grab_random_frame(mp4_path, rng):
    """Seek to a random frame in the mp4 and return the BGR image + its index."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None, -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return None, -1
    idx = rng.randrange(max(1, n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, -1
    return frame, idx


def solve_frame(frame, device):
    """Run HRNet + grid solver + findHomography on a single frame.

    Returns dict or None on failure.
    """
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, WEIGHTS, device=device)
    sideline_pxs, _ = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, _ = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))

    if len(hash_pxs) < 4:
        return None

    sideline_confs = np.ones(len(sideline_pxs))
    groups, _ = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, _ = groups_to_correspondences(groups, DUMMY_BASE_NGS_X)
    if len(pixel_pts) < 4:
        return None

    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]
    focal = float(max(h, w))
    if line_sets:
        k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w), focal)
    else:
        k1, k2 = 0.0, 0.0
    if abs(k1) > 1.0 or abs(k2) > 1.0:
        k1, k2 = 0.0, 0.0

    intr = CameraIntrinsics(fx=focal, fy=focal, cx=w/2.0, cy=h/2.0, k1=k1, k2=k2)
    pixel_pts_u = undistort_points(pixel_pts, intr)
    H, inlier_mask = cv2.findHomography(
        pixel_pts_u.astype(np.float64), field_pts.astype(np.float64),
        method=cv2.RANSAC, ransacReprojThreshold=1.5,
    )
    if H is None:
        return None

    # Reprojection error on undistorted inliers (the meaningful gate)
    H_inv = np.linalg.inv(H)
    proj = field_to_pixel(field_pts, H_inv)
    diffs = proj - pixel_pts_u
    mean_err = float(np.sqrt((diffs ** 2).sum(axis=1)).mean())

    return dict(
        H=H, k1=float(k1), k2=float(k2),
        n_corr=int(len(pixel_pts_u)),
        n_inliers=int(inlier_mask.sum()) if inlier_mask is not None else len(pixel_pts_u),
        mean_err=mean_err,
    )


def main(args):
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    import torch  # imported inside so import doesn't fail on devices we don't use
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    clips = enumerate_clips(args.clips_dir, angles=args.angles)
    print(f"{len(clips)} clips available (after holdout exclusion)")
    rng.shuffle(clips)

    # Sampling: one random frame per clip. For smaller pools we'd loop through
    # clips; for larger pools we may revisit the same clip. Keep it 1:1 for now.
    targets = clips[: args.n_frames] if args.n_frames <= len(clips) else \
              [clips[i % len(clips)] for i in range(args.n_frames)]

    rows = []
    hs_dir = os.path.join(args.output_dir, "H")
    os.makedirs(hs_dir, exist_ok=True)

    t0 = time.time()
    for i, (game, play, angle, mp4) in enumerate(targets):
        frame, idx = grab_random_frame(mp4, rng)
        if frame is None:
            rows.append(dict(game=game, play=play, angle=angle, frame_idx=-1,
                             n_corr=0, mean_err=None, solved=False))
            continue

        try:
            res = solve_frame(frame, device)
        except Exception as e:
            print(f"  [{i+1}/{len(targets)}] {game}/{play}/{angle} f{idx} ERROR: {e}")
            res = None

        row = dict(game=game, play=play, angle=angle, frame_idx=idx)
        if res is None:
            rows.append({**row, "n_corr": 0, "mean_err": None, "solved": False})
        else:
            # Stash H matrix keyed by a stable id so we don't re-run inference later
            fid = f"{game}_{play}_{angle}_f{idx:06d}"
            h_path = os.path.join(hs_dir, f"{fid}.pkl")
            with open(h_path, "wb") as f:
                pickle.dump({"H": res["H"], "k1": res["k1"], "k2": res["k2"]}, f)
            rows.append({**row, "n_corr": res["n_corr"], "n_inliers": res["n_inliers"],
                         "mean_err": res["mean_err"], "solved": True,
                         "frame_id": fid, "h_path": os.path.relpath(h_path, args.output_dir)})

        if (i + 1) % 25 == 0 or i + 1 == len(targets):
            solved = sum(1 for r in rows if r.get("solved"))
            dt = time.time() - t0
            print(f"  [{i+1}/{len(targets)}] solved {solved}, {dt:.0f}s elapsed "
                  f"({(i+1)/max(dt,1):.2f} fps)")

    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["game", "play", "angle", "frame_idx", "frame_id",
                           "n_corr", "n_inliers", "mean_err", "solved", "h_path"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ── Stats ──────────────────────────────────────────────────────────────
    solved = [r for r in rows if r.get("solved")]
    print(f"\n{len(solved)} / {len(rows)} frames solved")
    if solved:
        errs = np.array([r["mean_err"] for r in solved])
        ncs = np.array([r["n_corr"] for r in solved])
        print(f"  n_corr:  mean={ncs.mean():.1f}, median={int(np.median(ncs))}, "
              f"min={ncs.min()}, max={ncs.max()}")
        print(f"  err px:  mean={errs.mean():.2f}, median={np.median(errs):.2f}, "
              f"p90={np.percentile(errs, 90):.2f}")
        for n_min in (8, 10):
            for err_max in (0.3, 0.5, 1.0):
                passed = [r for r in solved
                          if r["n_corr"] >= n_min and r["mean_err"] < err_max]
                print(f"  n_corr>={n_min} & err<{err_max}px : "
                      f"{len(passed)} ({100*len(passed)/len(rows):.1f}% of sampled)")
    print(f"\nsummary → {csv_path}")
    print(f"H matrices → {hs_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--angles", nargs="+", default=["sideline"],
                    help="which camera angles to sample")
    ap.add_argument("--n-frames", type=int, default=200,
                    help="total frames to sample and evaluate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help='"cpu", "cuda", "mps", or "auto"')
    main(ap.parse_args())
