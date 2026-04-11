#!/usr/bin/env python3
"""
Full inference pipeline: video → detection → tracking → homography → tracking CSV.

Processes a single play clip through:
  1. Frame extraction from video
  2. Player detection (RF-DETR or YOLO)
  3. Multi-object tracking (BoT-SORT)
  4. Homography mapping (pixel → field coordinates)
  5. Output tracking data in NGS-compatible format

Usage:
  python -m src.pipeline --video videos/clips/2019092204/play_001/sideline.mp4 \
                         --weights models/rfdetr_best.pt \
                         --output output/tracking/
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detections, create_detector
from src.tracker import PlayerTracker, TrackingResult
from src.smoothing import smooth_all_trajectories, SmoothedTrajectory


def extract_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Extract all frames from a video file.

    Returns:
        (frames, fps) — list of BGR frames and the video's frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def process_play(
    video_path: str,
    detector,
    tracker: PlayerTracker,
    homography_fn=None,
    verbose: bool = True,
) -> dict:
    """Process a single play clip through the full pipeline.

    Args:
        video_path: Path to the play clip (sideline or endzone MP4).
        detector: A detector instance (YOLODetector or RFDETRDetector).
        tracker: A PlayerTracker instance (freshly initialized for this play).
        homography_fn: Optional callable(frame) -> H matrix. If None, skips
                       field coordinate mapping and outputs pixel coords only.
        verbose: Print progress info.

    Returns:
        dict with keys:
            'trajectories': dict of track_id -> PlayerTrajectory
            'fps': float
            'n_frames': int
            'timings': dict of stage -> total seconds
    """
    if verbose:
        print(f"  Loading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timings = {"detect": 0.0, "track": 0.0, "homography": 0.0, "total": 0.0}
    t_start = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- Detection ---
        t0 = time.time()
        detections = detector.detect(frame)
        timings["detect"] += time.time() - t0

        # --- Tracking ---
        t0 = time.time()
        tracking_result = tracker.update(detections, frame)
        timings["track"] += time.time() - t0

        # --- Homography (field coordinate mapping) ---
        if homography_fn is not None:
            t0 = time.time()
            H = homography_fn(frame)
            if H is not None and len(tracking_result) > 0:
                foot_points = tracking_result.foot_points
                # Apply homography to foot points
                from scripts.homography.apply_homography import pixel_to_field, is_on_field
                field_coords = pixel_to_field(foot_points, H)
                on_field = is_on_field(field_coords)

                # Update trajectory points with field coordinates
                for i, player in enumerate(tracking_result.players):
                    traj = tracker.trajectories.get(player.track_id)
                    if traj and traj.points:
                        last_point = traj.points[-1]
                        if on_field[i]:
                            last_point.field_xy = field_coords[i].copy()
                        else:
                            last_point.interrupted = True
            timings["homography"] += time.time() - t0

        frame_idx += 1
        if verbose and frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_idx / elapsed
            print(f"    Frame {frame_idx}/{n_frames} "
                  f"({fps_actual:.1f} fps, "
                  f"{len(tracking_result)} players tracked)")

    cap.release()
    timings["total"] = time.time() - t_start

    if verbose:
        print(f"  Done: {frame_idx} frames in {timings['total']:.1f}s "
              f"({frame_idx / timings['total']:.1f} fps)")
        print(f"    Detection:  {timings['detect']:.1f}s "
              f"({timings['detect']/frame_idx*1000:.0f}ms/frame)")
        print(f"    Tracking:   {timings['track']:.1f}s "
              f"({timings['track']/frame_idx*1000:.0f}ms/frame)")
        if homography_fn:
            print(f"    Homography: {timings['homography']:.1f}s "
                  f"({timings['homography']/frame_idx*1000:.0f}ms/frame)")

    return {
        "trajectories": tracker.get_trajectories(),
        "fps": fps,
        "n_frames": frame_idx,
        "timings": timings,
    }


def trajectories_to_csv(
    trajectories: dict,
    output_path: str,
    fps: float,
    game_id: str = "",
    play_id: str = "",
    smoothed: dict[int, SmoothedTrajectory] | None = None,
):
    """Write tracking trajectories to a CSV file in NGS-compatible format.

    If smoothed trajectories are provided, outputs 10fps smoothed data with
    velocity/acceleration columns. Otherwise outputs raw 30fps data.

    Output columns match NGS tracking data structure:
        gameId, playId, frame, track_id, x, y, confidence, interrupted

    Field coordinates (x, y) are in NGS convention:
        x: 0-120 yards (end line to end line)
        y: 0-53.33 yards (sideline to sideline)

    Frames with no field coordinates (homography unavailable or off-field)
    will have NaN for x/y.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []

    if smoothed:
        # Output smoothed 10fps data
        for track_id, straj in sorted(smoothed.items()):
            for i in range(len(straj.frame_indices)):
                x = straj.field_xy[i, 0]
                y = straj.field_xy[i, 1]
                row = {
                    "gameId": game_id,
                    "playId": play_id,
                    "frame": int(straj.frame_indices[i]),
                    "time_s": round(float(straj.times[i]), 3),
                    "track_id": track_id,
                    "x": round(float(x), 2) if not np.isnan(x) else "",
                    "y": round(float(y), 2) if not np.isnan(y) else "",
                    "pixel_x": round(float(straj.pixel_xy[i, 0]), 1),
                    "pixel_y": round(float(straj.pixel_xy[i, 1]), 1),
                    "confidence": round(float(straj.confidence[i]), 3),
                    "interrupted": int(straj.interrupted[i]),
                }
                # Add derivative columns if available
                if straj.speed is not None:
                    row["speed_yds"] = round(float(straj.speed[i]), 2) if not np.isnan(straj.speed[i]) else ""
                if straj.velocity is not None:
                    row["vx"] = round(float(straj.velocity[i, 0]), 2) if not np.isnan(straj.velocity[i, 0]) else ""
                    row["vy"] = round(float(straj.velocity[i, 1]), 2) if not np.isnan(straj.velocity[i, 1]) else ""
                if straj.accel_magnitude is not None:
                    row["accel_yds"] = round(float(straj.accel_magnitude[i]), 2) if not np.isnan(straj.accel_magnitude[i]) else ""
                rows.append(row)
    else:
        # Output raw 30fps data
        for track_id, traj in sorted(trajectories.items()):
            for point in traj.points:
                frame_time = point.frame_idx / fps if fps > 0 else 0.0
                x = point.field_xy[0] if point.field_xy is not None else np.nan
                y = point.field_xy[1] if point.field_xy is not None else np.nan
                rows.append({
                    "gameId": game_id,
                    "playId": play_id,
                    "frame": point.frame_idx,
                    "time_s": round(frame_time, 3),
                    "track_id": track_id,
                    "x": round(float(x), 2) if not np.isnan(x) else "",
                    "y": round(float(y), 2) if not np.isnan(y) else "",
                    "pixel_x": round(float(point.pixel_xy[0]), 1),
                    "pixel_y": round(float(point.pixel_xy[1]), 1),
                    "confidence": round(point.confidence, 3),
                    "interrupted": int(point.interrupted),
                })

    # Sort by frame, then track_id
    rows.sort(key=lambda r: (r["frame"], r["track_id"]))

    if not rows:
        return 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run player tracking pipeline on a play clip"
    )
    parser.add_argument("--video", required=True, help="Path to play clip (MP4)")
    parser.add_argument("--weights", required=True, help="Detection model weights (.pt or .pth)")
    parser.add_argument("--output", default="output/tracking", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu, cuda, mps)")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--game-id", default="", help="Game ID for output CSV")
    parser.add_argument("--play-id", default="", help="Play ID for output CSV")
    parser.add_argument("--no-reid", action="store_true", help="Disable appearance-based ReID")
    parser.add_argument("--no-homography", action="store_true", help="Skip field coordinate mapping")
    parser.add_argument("--track-buffer", type=int, default=30, help="Frames to keep lost tracks alive")
    parser.add_argument("--confidence-gate", type=float, default=0.3,
                        help="Below this confidence, mark trajectory as interrupted")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw 30fps data without smoothing/downsampling")
    parser.add_argument("--smooth-window", type=int, default=200,
                        help="Smoothing window in ms (default 200)")
    args = parser.parse_args()

    # --- Initialize detector ---
    print(f"Initializing detector: {args.weights}")
    detector = create_detector(
        args.weights,
        device=args.device,
        conf_thresh=args.conf_thresh,
    )

    # --- Initialize tracker ---
    print(f"Initializing tracker (BoT-SORT, reid={'on' if not args.no_reid else 'off'})")
    tracker = PlayerTracker(
        device=args.device,
        with_reid=not args.no_reid,
        track_buffer=args.track_buffer,
        confidence_gate=args.confidence_gate,
    )

    # --- Homography (optional) ---
    homography_fn = None
    if not args.no_homography:
        # TODO: integrate homography per-frame computation
        # For now, homography requires manual yard line identification.
        # Will be wired in when automatic yard line ID is ready.
        print("  Homography: skipped (not yet wired for automatic mode)")

    # --- Run pipeline ---
    print(f"\nProcessing: {args.video}")
    result = process_play(
        video_path=args.video,
        detector=detector,
        tracker=tracker,
        homography_fn=homography_fn,
        verbose=True,
    )

    # --- Smoothing (30fps → 10fps) ---
    smoothed = None
    if not args.raw:
        print(f"\nSmoothing trajectories (window={args.smooth_window}ms, 30fps→10fps)...")
        smoothed = smooth_all_trajectories(
            trajectories=result["trajectories"],
            source_fps=result["fps"],
            target_fps=10.0,
            window_ms=args.smooth_window,
        )
        print(f"  {len(smoothed)} trajectories smoothed "
              f"(dropped {len(result['trajectories']) - len(smoothed)} short tracks)")

    # --- Output ---
    video_name = Path(args.video).stem
    suffix = "_tracking_raw" if args.raw else "_tracking"
    output_path = os.path.join(args.output, f"{video_name}{suffix}.csv")

    n_rows = trajectories_to_csv(
        trajectories=result["trajectories"],
        output_path=output_path,
        fps=result["fps"],
        game_id=args.game_id,
        play_id=args.play_id,
        smoothed=smoothed,
    )

    n_tracks = len(smoothed) if smoothed else len(result["trajectories"])
    out_fps = "10fps smoothed" if smoothed else f"{result['fps']:.0f}fps raw"
    print(f"\nOutput: {output_path}")
    print(f"  {n_tracks} players tracked, {n_rows} total data points")
    print(f"  {result['n_frames']} source frames, output at {out_fps}")


if __name__ == "__main__":
    main()
