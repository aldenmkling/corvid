#!/usr/bin/env python3
"""
Convert all 60fps clips to 30fps in-place.

Scans videos/clips/ for any MP4 over 30fps and re-encodes at 30fps
using libx264 CRF 18 (visually lossless). Skips clips already at 30fps.
Processes clips in parallel with configurable workers.

Usage:
  python scripts/convert_clips_30fps.py              # dry run
  python scripts/convert_clips_30fps.py --execute    # actually convert
"""

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(PROJECT_ROOT, "videos", "clips")
TARGET_FPS = 30


def get_fps(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    raw = result.stdout.strip()
    if not raw or "/" not in raw:
        return 0.0
    num, den = raw.split("/")
    return int(num) / int(den) if int(den) > 0 else 0.0


def convert_clip(path: str) -> str:
    """Convert a single clip to 30fps in-place. Returns status string."""
    fps = get_fps(path)
    if fps <= 31:
        return f"SKIP (already {fps:.1f}fps): {path}"

    tmp_path = path + ".tmp.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", path,
            "-r", str(TARGET_FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            tmp_path,
        ], capture_output=True, check=True)

        # Replace original
        os.replace(tmp_path, path)
        return f"OK ({fps:.0f}→30fps): {path}"
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return f"FAIL: {path} — {e.stderr[:200] if e.stderr else 'unknown error'}"


def main():
    parser = argparse.ArgumentParser(description="Convert 60fps clips to 30fps")
    parser.add_argument("--execute", action="store_true", help="Actually convert (default is dry run)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    # Find all clips
    clips = []
    for root, dirs, files in os.walk(CLIPS_DIR):
        for f in files:
            if f.endswith(".mp4"):
                clips.append(os.path.join(root, f))

    clips.sort()
    print(f"Found {len(clips)} clips in {CLIPS_DIR}")

    # Check which need conversion
    to_convert = []
    for path in clips:
        fps = get_fps(path)
        if fps > 31:
            to_convert.append(path)

    print(f"  {len(to_convert)} clips at >30fps need conversion")
    print(f"  {len(clips) - len(to_convert)} clips already at 30fps")

    if not to_convert:
        print("Nothing to do.")
        return

    if not args.execute:
        print("\nDry run — pass --execute to convert.")
        return

    print(f"\nConverting {len(to_convert)} clips ({args.workers} workers)...")
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_clip, p): p for p in to_convert}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if "FAIL" in result:
                errors += 1
                print(f"  {result}")
            if completed % 100 == 0:
                print(f"  {completed}/{len(to_convert)} done...")

    print(f"\nDone! {completed - errors}/{len(to_convert)} converted, {errors} errors")


if __name__ == "__main__":
    main()
