#!/usr/bin/env python3
"""
Extract diverse sideline frames for field keypoint annotation.

Samples 300 frames across all 10 games, stratified by play phase
(pre-snap, mid-play, late) and field position. Sideline view only.

Usage:
    python scripts/extract_keypoint_frames.py [--num-frames 300]
"""

import argparse
import os
import random
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIPS_DIR = os.path.join(PROJECT_ROOT, "videos", "clips")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "field_keypoints", "annotation_images")

# All 10 games
GAME_IDS = [
    "2019092204", "2019102712",
    "2024090801", "2024090802", "2024091501",
    "2024092201", "2024100601", "2024102701",
    "2024111001", "2024122201",
]

# Time points within each play (fraction of total frames)
TIME_POINTS = [0.0, 0.15, 0.35, 0.55, 0.75, 0.90]


def extract_frame(video_path: str, pct: float):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pct)))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_plays(game_dir: str) -> list[str]:
    """Get sorted list of play directories."""
    plays = []
    for d in sorted(os.listdir(game_dir)):
        if d.startswith("play_"):
            sideline = os.path.join(game_dir, d, "sideline.mp4")
            if os.path.exists(sideline):
                plays.append(d)
    return plays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all available (game_id, play, time_pct) candidates
    candidates = []
    for game_id in GAME_IDS:
        game_dir = os.path.join(CLIPS_DIR, game_id)
        if not os.path.isdir(game_dir):
            print(f"Skipping {game_id} (not found)")
            continue
        plays = get_plays(game_dir)
        for play in plays:
            for pct in TIME_POINTS:
                candidates.append((game_id, play, pct))

    print(f"Total candidates: {len(candidates)} across {len(GAME_IDS)} games")

    # Sample with roughly equal representation per game
    random.shuffle(candidates)

    # Group by game and sample proportionally
    by_game = {}
    for c in candidates:
        by_game.setdefault(c[0], []).append(c)

    per_game = max(1, args.num_frames // len(by_game))
    selected = []

    for game_id, game_candidates in by_game.items():
        n = min(per_game, len(game_candidates))
        selected.extend(random.sample(game_candidates, n))

    # Fill remaining slots
    remaining = args.num_frames - len(selected)
    if remaining > 0:
        pool = [c for c in candidates if c not in selected]
        selected.extend(random.sample(pool, min(remaining, len(pool))))

    selected = selected[:args.num_frames]
    print(f"Selected {len(selected)} frames")

    # Extract frames
    extracted = 0
    manifest = []

    for game_id, play, pct in selected:
        video_path = os.path.join(CLIPS_DIR, game_id, play, "sideline.mp4")
        frame = extract_frame(video_path, pct)
        if frame is None:
            continue

        fname = f"{game_id}_{play}_sideline_{int(pct * 100):02d}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        manifest.append({
            "file_name": fname,
            "game_id": game_id,
            "play": play,
            "time_pct": pct,
        })
        extracted += 1

        if extracted % 50 == 0:
            print(f"  Extracted {extracted}/{len(selected)}")

    # Save manifest
    manifest_path = os.path.join(
        PROJECT_ROOT, "data", "field_keypoints", "manifest.json"
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Extracted {extracted} frames to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    # Per-game breakdown
    game_counts = {}
    for m in manifest:
        game_counts[m["game_id"]] = game_counts.get(m["game_id"], 0) + 1
    for gid, cnt in sorted(game_counts.items()):
        print(f"  {gid}: {cnt} frames")


if __name__ == "__main__":
    main()
