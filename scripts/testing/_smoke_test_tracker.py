#!/usr/bin/env python3
"""Smoke test for the field-coord Kalman tracker (Layer 1).

Wires the rectify pipeline's compute_homographies() output (per-frame H, K,
dist) into PlayerTracker.update() with RF-DETR detections, and reports a
quick sanity check on track count + length + mean field positions.

Run from project root:
    python scripts/testing/_smoke_test_tracker.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import RFDETRDetector, get_or_build_detection_cache
from src.tracker import PlayerTracker
from src.team_classifier import classify_teams_by_position
from src.homography.rectify import (
    compute_homographies, get_or_build_homography_cache,
)


CLIP = str(PROJECT_ROOT / "videos/clips/2024122201/play_114/sideline.mp4")
DETECTOR_WEIGHTS = str(PROJECT_ROOT / "models/rfdetr_best_ema.pth")


def main():
    if not os.path.exists(CLIP):
        print(f"smoke test skipped — clip not found: {CLIP}")
        return
    if not os.path.exists(DETECTOR_WEIGHTS):
        print(f"smoke test skipped — detector weights not found: {DETECTOR_WEIGHTS}")
        return

    print(f"[1/3] homographies for {os.path.relpath(CLIP, PROJECT_ROOT)}")
    t0 = time.time()
    homo = get_or_build_homography_cache(
        clip_path=CLIP,
        device="mps",
        verbose=True,
    )
    if homo is None:
        print("  failed to compute homographies")
        return
    Hs = homo["Hs"]
    K = homo["K"]
    dist = homo["dist"]
    valid_until = homo["valid_until"]
    n_frames = homo["n_frames"]
    print(f"  homographies: {valid_until}/{n_frames} valid frames "
          f"(took {time.time()-t0:.1f}s)")

    print(f"\n[2/3] loading or building detection cache for {n_frames} frames")
    dets_cached = get_or_build_detection_cache(
        clip_path=CLIP,
        weights=DETECTOR_WEIGHTS,
        device="mps",
        conf_thresh=0.3,
        verbose=True,
    )

    print(f"\n[2.5/3] running tracker over {n_frames} frames")
    tracker = PlayerTracker(
        device="mps",
        process_noise_yd=8.0,
        measurement_noise_yd=0.5,
        max_lost_frames=8,
        max_graveyard_frames=90,
        confidence_gate=0.3,
        frame_rate=30,
    )

    cap = cv2.VideoCapture(CLIP)
    fi = 0
    n_dets_total = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or fi >= n_frames:
            break
        H = Hs[fi] if fi < len(Hs) else None
        dets = dets_cached[fi]
        n_dets_total += len(dets)
        tracker.update(dets, frame, H=H, K=K, dist=dist)
        fi += 1
        if fi % 50 == 0:
            print(f"  frame {fi}/{n_frames}  "
                  f"active tracks={sum(1 for tr in tracker._tracks if tr.frames_since_update == 0)}  "
                  f"total tracks so far={len(tracker.trajectories)}")
    cap.release()
    print(f"  done in {time.time()-t0:.1f}s; "
          f"total detections={n_dets_total}; "
          f"total trajectories={len(tracker.trajectories)}")

    print(f"\n[3/3] track stats")
    trajs = tracker.get_trajectories()
    if not trajs:
        print("  no trajectories created — something is wrong")
        return

    # "Long" tracks = at least half the valid frames in the clip.
    long_thresh = max(1, valid_until // 2)
    lengths = []
    long_tracks = []
    for tid, traj in trajs.items():
        # Count points where there was an actual measurement (not interrupted).
        n_meas = sum(1 for p in traj.points if not p.interrupted)
        lengths.append(n_meas)
        if n_meas >= long_thresh:
            long_tracks.append((tid, traj, n_meas))
    median_len = float(np.median(lengths)) if lengths else 0.0

    print(f"  total tracks created: {len(trajs)}")
    print(f"  long tracks (n_meas >= {long_thresh}): {len(long_tracks)}")
    print(f"  median track length (measurements): {median_len:.0f}")
    print()
    print(f"  per long-track summary (track_id, n_meas, mean field_xy):")
    long_tracks.sort(key=lambda x: -x[2])
    for tid, traj, n_meas in long_tracks:
        meas_pts = np.array([p.field_xy for p in traj.points
                              if (p.field_xy is not None and not p.interrupted)])
        if len(meas_pts) == 0:
            continue
        mean_xy = meas_pts.mean(axis=0)
        x_min, x_max = meas_pts[:, 0].min(), meas_pts[:, 0].max()
        y_min, y_max = meas_pts[:, 1].min(), meas_pts[:, 1].max()
        print(f"    id={tid:3d}  n={n_meas:3d}  "
              f"mean=({mean_xy[0]:6.2f}, {mean_xy[1]:5.2f})  "
              f"x range [{x_min:6.2f}, {x_max:6.2f}]  "
              f"y range [{y_min:5.2f}, {y_max:5.2f}]")

    # Sanity check on bounds.
    all_xy = np.concatenate([
        np.array([p.field_xy for p in traj.points
                  if p.field_xy is not None and not p.interrupted])
        for _, traj, _ in long_tracks
    ]) if long_tracks else np.empty((0, 2))
    if len(all_xy) > 0:
        in_x = ((all_xy[:, 0] >= 10) & (all_xy[:, 0] <= 110)).mean() * 100
        in_y = ((all_xy[:, 1] >= 0) & (all_xy[:, 1] <= 53.33)).mean() * 100
        print(f"\n  long-track field bounds: "
              f"{in_x:.1f}% in x∈[10,110]yd, {in_y:.1f}% in y∈[0,53.33]yd")

    # ── Layer 4 sanity check: team classification ─────────────────────
    print(f"\n[4/4] team classification (Layer 4)")
    t0 = time.time()
    team_labels = classify_teams_by_position(
        trajectories=trajs,
        snap_frame_idx=0,
    )
    print(f"  classified {len(team_labels)} tracks in {time.time()-t0:.1f}s")
    n_a = sum(1 for v in team_labels.values() if v == "team_A")
    n_b = sum(1 for v in team_labels.values() if v == "team_B")
    n_unk = sum(1 for v in team_labels.values() if v == "unknown")
    print(f"  team_A: {n_a}   team_B: {n_b}   unknown: {n_unk}")

    long_ids = {tid for tid, _, _ in long_tracks}
    n_a_long = sum(1 for tid, lab in team_labels.items()
                   if lab == "team_A" and tid in long_ids)
    n_b_long = sum(1 for tid, lab in team_labels.items()
                   if lab == "team_B" and tid in long_ids)
    n_unk_long = sum(1 for tid, lab in team_labels.items()
                     if lab == "unknown" and tid in long_ids)
    print(f"  long tracks split: team_A={n_a_long}  "
          f"team_B={n_b_long}  unknown={n_unk_long}  "
          f"(expect ≈ 11/11/0)")


if __name__ == "__main__":
    main()
