"""End-to-end inference pipeline: clip → per-frame player tracking CSV.

Stages:
  1. field_mapping  — UNet → tokens → encoder → number_refiner → token_labeler
                      → keypoints → solve H per frame → LOO filter → smooth
  2. player_detection — RF-DETR per frame (in undistorted image space)
  3. player_tracking — Kalman + multi-cue association in NGS yards
  4. team_classification — color PCA + median split, 11/11 forced
  5. trajectory_smoothing — per-track Sav-Gol before differentiating

Output CSV columns:
    frame_idx, track_id, x_yd, y_yd, team, in_bad_run

Usage:
    python -m src.pipeline --clip <path> --out <csv> [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter

from src.field_mapping.pipeline import FieldMappingPipeline
from src.field_mapping.homography import (
    HomographyTrackerLite, smooth_hs, loo_filter_and_replace,
    detect_bad_runs,
)
from src.player_detection.detector import RFDETRDetector
from src.player_tracking.tracker import PlayerTracker
from src.player_tracking.team_classifier import (
    select_long_tracks, classify_teams_color_pca,
)


# Pipeline parameters (production defaults).
RMSE_THR_YD = 0.30
LOO_THR_YD = 0.20
CARRY_STREAK_LOST = 3        # 3+ consecutive H-solver failures → clip lost
BAD_RUN_MIN_LEN = 5          # consec red frames → mark as bad-run stretch
SMOOTH_WINDOW_H = 7          # Sav-Gol window for H trajectory
SMOOTH_POLY_H = 2
SMOOTH_WINDOW_TRACK = 9      # Sav-Gol window for per-track NGS positions
SMOOTH_POLY_TRACK = 2

DETECTOR_CONF_THRESH = 0.3
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dot_pixel_from_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """Player position estimate: middle-x, 95% from top of box to bottom."""
    x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return np.array([0.5 * (x0 + x1), y0 + 0.95 * (y1 - y0)], dtype=np.float64)


def _project_via_H(pts_pixel: np.ndarray, H: np.ndarray) -> np.ndarray:
    if len(pts_pixel) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homo = np.column_stack([pts_pixel, np.ones(len(pts_pixel))])
    proj = (H @ homo.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return proj[:, :2] / w


def run_pipeline(clip_path: str, device,
                  K: np.ndarray | None = None,
                  dist: np.ndarray | None = None,
                  verbose: bool = True):
    """Run the full pipeline on a clip. Returns dict with per-track field-coord
    arrays + team labels + bad-run mask.

    Args:
        clip_path: path to sideline.mp4
        device: torch device (cuda/mps/cpu)
        K, dist: camera intrinsics. If None, falls back to K=I, dist=0
            (the H absorbs residual radial distortion).
        verbose: print progress.

    Returns:
        dict with:
          n_frames: total frames in clip
          cutoff:   number of frames before sustained tracking loss
          Hs:       list[N] of (3,3) smoothed H per frame (None where lost)
          long_track_ids: set of track_ids that survived the long-track filter
          dot_field: dict[track_id -> (N, 2) array of NGS yard positions
                       (smoothed). NaN where the track wasn't detected.]
          team_labels: dict[track_id -> 'team_A' | 'team_B' | 'unknown']
          bad_run_mask: bool list[N] — True where H was bridged across a long
                         red-frame run (likely clip-disqualification candidate)
          fps: source frames per second
    """
    if K is None: K = np.eye(3, dtype=np.float64)
    if dist is None: dist = np.zeros(5, dtype=np.float64)

    # Models (loaded once, used per frame).
    if verbose: print("Loading field-mapping pipeline ...")
    fm_pipe = FieldMappingPipeline(device, project_root=PROJECT_ROOT)
    if verbose: print("Loading detector + tracker ...")
    detector = RFDETRDetector(
        weights=os.path.join(PROJECT_ROOT, "models/rfdetr_best_ema.pth"),
        device=str(device), conf_thresh=DETECTOR_CONF_THRESH)
    tracker = PlayerTracker(device=str(device), frame_rate=30)
    h_tracker = HomographyTrackerLite()

    # ── Pass 1: per-frame H + detect + track ─────────────────────────────
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose: print(f"Clip: {n_total} frames @ {fps:.1f} fps")

    per_frame = []           # list of {H, method, rmse}
    track_results = []
    frames_u = []
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != 1280: fr = cv2.resize(fr, (1280, 720))
        fr_u = cv2.undistort(fr, K, dist)
        frames_u.append(fr_u)

        fm_out = fm_pipe(fr, K, dist)
        if fm_out is None:
            per_frame.append(None)
        else:
            res = h_tracker.update(fm_out["corrs"], frame_idx=fi)
            per_frame.append({"H": res["H"], "method": res["method"],
                              "rmse": res["rmse_yd"]})

        H_for_tracker = per_frame[-1]["H"] if per_frame[-1] else None
        dets = detector.detect(fr_u)
        tr = tracker.update(dets, fr_u, H=H_for_tracker, K=K, dist=dist)
        track_results.append(tr)

        if verbose and (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{fi+1}/{n_total}] {dt:.0f}s ({(fi+1)/dt:.1f} fps)")
    cap.release()
    if verbose: print(f"  pass 1 in {time.time()-t0:.0f}s")

    # ── Carry-streak cutoff ─────────────────────────────────────────────
    cutoff = len(per_frame)
    streak = 0; run_start = None
    for i, r in enumerate(per_frame):
        m = r["method"] if r else "none"
        if m == "carry":
            if streak == 0: run_start = i
            streak += 1
            if streak >= CARRY_STREAK_LOST:
                cutoff = run_start; break
        else:
            streak = 0; run_start = None
    if verbose: print(f"  cutoff: {cutoff}/{len(per_frame)}")

    # ── LOO filter + replacement + Sav-Gol smoothing on H trajectory ────
    Hs = [r["H"] if (r and r["H"] is not None) else None
          for r in per_frame[:cutoff]]
    rmses = [r["rmse"] if (r and r.get("rmse") is not None) else None
             for r in per_frame[:cutoff]]
    Hs_clean, red_mask, _ = loo_filter_and_replace(
        Hs, rmses=rmses,
        thr_loo_yd=LOO_THR_YD, thr_rmse_yd=RMSE_THR_YD)
    valid_idx = [i for i in range(cutoff) if Hs_clean[i] is not None]
    if len(valid_idx) >= SMOOTH_WINDOW_H:
        sm = smooth_hs([Hs_clean[i] for i in valid_idx],
                        window=SMOOTH_WINDOW_H, poly=SMOOTH_POLY_H)
        for vi, si in enumerate(valid_idx):
            Hs_clean[si] = sm[vi]
    bad_runs = detect_bad_runs(red_mask, min_length=BAD_RUN_MIN_LEN)
    bad_run_mask = [False] * cutoff
    for s, e in bad_runs:
        for k in range(s, e): bad_run_mask[k] = True
    if verbose and bad_runs:
        print(f"  bad runs (≥{BAD_RUN_MIN_LEN} consec): {bad_runs}")

    # ── Team classify long tracks ───────────────────────────────────────
    trajectories = tracker.trajectories
    long_track_ids = select_long_tracks(
        trajectories, min_meas_frac=0.5, n_valid_frames=cutoff)
    if verbose:
        print(f"  long tracks: {len(long_track_ids)}/{len(trajectories)}")
    team_labels, _ = classify_teams_color_pca(
        trajectories, clip_path, n_samples_per_track=12,
        long_track_ids=long_track_ids)

    # ── Per-track field positions (NGS yards) ───────────────────────────
    dot_field = {tid: np.full((cutoff, 2), np.nan, dtype=np.float64)
                 for tid in long_track_ids}
    for fi in range(cutoff):
        H = Hs_clean[fi]
        if H is None: continue
        for p in track_results[fi].players:
            if p.track_id not in long_track_ids: continue
            dp = _dot_pixel_from_xyxy(p.xyxy)
            fxy = _project_via_H(dp[None], H)[0]
            if np.isfinite(fxy).all():
                dot_field[p.track_id][fi] = fxy

    # Per-track Sav-Gol smoothing.
    for tid, arr in dot_field.items():
        valid = ~np.isnan(arr[:, 0])
        in_seg = False; start = 0; segs = []
        for i in range(cutoff):
            if valid[i] and not in_seg: start = i; in_seg = True
            elif not valid[i] and in_seg: segs.append((start, i)); in_seg = False
        if in_seg: segs.append((start, cutoff))
        for s, e in segs:
            L = e - s
            if L < SMOOTH_WINDOW_TRACK: continue
            w = SMOOTH_WINDOW_TRACK
            if w > L: w = L if L % 2 == 1 else L - 1
            if w < SMOOTH_POLY_TRACK + 2: continue
            arr[s:e, 0] = savgol_filter(arr[s:e, 0], w, SMOOTH_POLY_TRACK)
            arr[s:e, 1] = savgol_filter(arr[s:e, 1], w, SMOOTH_POLY_TRACK)

    return {
        "n_frames": n_total,
        "cutoff": cutoff,
        "Hs": Hs_clean,
        "long_track_ids": long_track_ids,
        "dot_field": dot_field,
        "team_labels": team_labels,
        "bad_run_mask": bad_run_mask,
        "fps": fps,
    }


def write_csv(result: dict, out_path: str):
    """Write per-frame, per-track NGS-yard positions to CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cutoff = result["cutoff"]
    rows = []
    for tid in sorted(result["long_track_ids"]):
        arr = result["dot_field"][tid]
        team = result["team_labels"].get(tid, "unknown")
        for fi in range(cutoff):
            x, y = arr[fi]
            if not np.isfinite(x) or not np.isfinite(y): continue
            rows.append((fi, tid, x, y, team, int(result["bad_run_mask"][fi])))
    with open(out_path, "w") as f:
        f.write("frame_idx,track_id,x_yd,y_yd,team,in_bad_run\n")
        for fi, tid, x, y, team, bad in rows:
            f.write(f"{fi},{tid},{x:.3f},{y:.3f},{team},{bad}\n")
    print(f"  → {out_path} ({len(rows)} rows)")


def _load_intrinsics(clip_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Look up K + dist from data/manifests/h_pool_and_intrinsics.json. Returns
    (K=I, dist=0) fallback if the clip isn't in the manifest."""
    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifests", "h_pool_and_intrinsics.json")
    if not os.path.exists(manifest_path):
        return np.eye(3, dtype=np.float64), np.zeros(5, dtype=np.float64)
    rel = os.path.relpath(clip_path, os.path.join(PROJECT_ROOT, "videos/clips"))
    intr = json.load(open(manifest_path))["intrinsics_by_clip"].get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)
    return K, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Path to sideline.mp4")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--device", default="cuda",
                    help="cuda | mps | cpu")
    args = ap.parse_args()

    clip_path = (args.clip if os.path.isabs(args.clip)
                 else os.path.join(PROJECT_ROOT, args.clip))
    if not os.path.exists(clip_path):
        sys.exit(f"clip not found: {clip_path}")

    device = torch.device(args.device)
    K, dist = _load_intrinsics(clip_path)

    result = run_pipeline(clip_path, device, K=K, dist=dist, verbose=True)
    write_csv(result, args.out)


if __name__ == "__main__":
    main()
