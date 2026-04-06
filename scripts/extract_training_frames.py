#!/usr/bin/env python3
"""
Extract diverse training frames from play clips for YOLO fine-tuning.

Samples frames across plays, views, and time points to get good coverage
of different formations, play states, and camera angles.

Usage:
  python scripts/extract_training_frames.py [--num-frames 200]
"""

import argparse
import os
import random

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIRS = [
    os.path.join(PROJECT_ROOT, "videos", "clips", "2019092204"),   # Ravens @ Chiefs
    os.path.join(PROJECT_ROOT, "videos", "clips", "2019102712"),   # Chiefs vs Packers
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "images")


def extract_frame(video_path: str, pct: float):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Total frames to extract (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all play directories across all games
    views = ["sideline"]
    # Bias toward early/mid play when players are spread out and visible
    time_points = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85]

    # Build candidate list: (game_id, play_dir, view, pct)
    candidates = []
    total_plays = 0
    for clips_dir in CLIPS_DIRS:
        game_id = os.path.basename(clips_dir)
        play_dirs = sorted([
            d for d in os.listdir(clips_dir)
            if d.startswith("play_") and os.path.isdir(os.path.join(clips_dir, d))
        ])
        total_plays += len(play_dirs)
        for play_dir in play_dirs:
            for view in views:
                video_path = os.path.join(clips_dir, play_dir, f"{view}.mp4")
                if os.path.exists(video_path):
                    for pct in time_points:
                        candidates.append((game_id, play_dir, view, pct))

    print(f"Found {total_plays} plays across {len(CLIPS_DIRS)} games")

    print(f"Total candidate frames: {len(candidates)}")

    # Sample uniformly
    num = min(args.num_frames, len(candidates))
    selected = random.sample(candidates, num)
    selected.sort()

    print(f"Extracting {num} frames...")
    extracted = 0
    for game_id, play_dir, view, pct in selected:
        clips_dir = next(d for d in CLIPS_DIRS if os.path.basename(d) == game_id)
        video_path = os.path.join(clips_dir, play_dir, f"{view}.mp4")
        frame = extract_frame(video_path, pct)
        if frame is None:
            continue

        # Name: 2019092204_play001_sideline_30.jpg
        play_num = play_dir.replace("play_", "")
        out_name = f"{game_id}_play{play_num}_{view}_{int(pct * 100):02d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
        extracted += 1

    print(f"Extracted {extracted} frames to {OUTPUT_DIR}/")
    print(f"\nNext step: annotate these frames with bounding boxes.")
    print(f"Recommended tool: Label Studio or CVAT")
    print(f"  - Class 0: player (on-field)")
    print(f"  - Ignore sideline personnel, refs optional")


if __name__ == "__main__":
    main()
