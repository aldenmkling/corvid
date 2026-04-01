#!/usr/bin/env python3
"""
Detect where All-22 videos freeze (duplicate frames) and trim them.

Strategy: Sample frames every 1 second starting from the middle of the video.
Compare consecutive frames using mean absolute difference. When the difference
drops to near-zero for several consecutive samples, that's the freeze point.
Trim the video to just before the freeze using ffmpeg stream copy (no re-encode).
"""

import subprocess
import sys
import os
import tempfile
import numpy as np

# pip install opencv-python if not installed
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2


VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")

# Threshold: if mean absolute pixel difference between consecutive frames
# sampled 1s apart is below this, consider them identical
FREEZE_THRESHOLD = 1.0
# How many consecutive frozen samples needed to confirm freeze
FREEZE_CONFIRM = 5


def get_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def detect_freeze_point(path: str) -> float | None:
    """
    Find the timestamp where the video freezes.
    Returns the timestamp (seconds) of the last good frame, or None if no freeze.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {path}")
        return None

    duration = get_duration(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min), FPS: {fps:.1f}")

    # Start scanning from 60 minutes in (actual game content is likely < 90 min)
    scan_start = 60 * 60  # 60 minutes
    scan_step = 1.0  # check every 1 second

    prev_frame = None
    freeze_start = None
    freeze_count = 0

    t = scan_start
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample for speed
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_frame is not None:
            diff = np.mean(np.abs(gray - prev_frame))
            if diff < FREEZE_THRESHOLD:
                if freeze_count == 0:
                    freeze_start = t - scan_step
                freeze_count += 1
                if freeze_count >= FREEZE_CONFIRM:
                    cap.release()
                    print(f"  Freeze detected at {freeze_start:.0f}s "
                          f"({freeze_start/60:.1f} min)")
                    return freeze_start
            else:
                freeze_count = 0
                freeze_start = None

        prev_frame = gray
        t += scan_step

    cap.release()
    print("  No freeze detected")
    return None


def trim_video(input_path: str, end_time: float, output_path: str):
    """Trim video to end_time using ffmpeg stream copy (fast, no re-encode)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-t", str(end_time),
        "-c", "copy",
        output_path
    ]
    print(f"  Trimming to {end_time:.0f}s ({end_time/60:.1f} min)...")
    subprocess.run(cmd, capture_output=True, check=True)

    # Report sizes
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {orig_size:.0f} MB → {new_size:.0f} MB")


def main():
    videos = sorted(
        f for f in os.listdir(VIDEO_DIR)
        if f.endswith(".mp4")
    )

    if not videos:
        print("No MP4 files found in", VIDEO_DIR)
        sys.exit(1)

    print(f"Found {len(videos)} videos\n")

    for filename in videos:
        path = os.path.join(VIDEO_DIR, filename)
        print(f"Processing: {filename}")

        freeze_at = detect_freeze_point(path)
        if freeze_at is None:
            print(f"  Skipping (no freeze found)\n")
            continue

        # Add a small buffer before freeze point
        trim_point = freeze_at - 2  # 2 second buffer

        # Trim to temp file, then replace original
        trimmed_path = path + ".trimmed.mp4"
        trim_video(path, trim_point, trimmed_path)

        # Replace original with trimmed version
        os.replace(trimmed_path, path)

        # Verify new duration
        new_dur = get_duration(path)
        print(f"  New duration: {new_dur:.0f}s ({new_dur/60:.1f} min)\n")

    print("Done!")


if __name__ == "__main__":
    main()
