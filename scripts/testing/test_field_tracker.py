#!/usr/bin/env python3
"""Visual test of FieldTracker on a sideline play clip.

Draws tracked keypoints on each frame with IDs, confidence, and stats overlay.
Outputs annotated video to output/field_tracker_test.mp4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import argparse
import time
from src.homography.classical.field_tracker import FieldTracker


def draw_keypoints(frame, result, frame_idx):
    """Draw tracked keypoints and stats on frame."""
    overlay = frame.copy()

    for kp in result.keypoints:
        x, y = int(kp.pixel_xy[0]), int(kp.pixel_xy[1])

        # Color by confidence: green=1.0, yellow=0.5, red=0.0
        if kp.is_detected:
            color = (0, 255, 0)  # green for detected
        else:
            # Orange for flow-only
            color = (0, 165, 255)

        # Circle size by confidence
        radius = int(4 + 4 * kp.confidence)
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)

        # Label: ID and hash type
        label = f"{kp.id}:{kp.hash_type[0]}"
        cv2.putText(overlay, label, (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Stats overlay
    stats = [
        f"Frame {frame_idx}",
        f"Keypoints: {len(result.keypoints)}",
        f"Detected: {result.n_detected}  Flow: {result.n_flow_only}",
        f"New: {result.n_new}  Retired: {result.n_retired}",
    ]
    for i, line in enumerate(stats):
        cv2.putText(overlay, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(overlay, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Test FieldTracker on a play clip")
    parser.add_argument("--clip", type=str,
                        default="videos/clips/2019092204/play_010/sideline.mp4",
                        help="Path to sideline clip")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to process (0=all)")
    parser.add_argument("--output", type=str, default="output/field_tracker_test.mp4")
    parser.add_argument("--show-every", type=int, default=0,
                        help="Show frame every N frames (0=no display)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clip_path = os.path.join(project_root, args.clip) if not os.path.isabs(args.clip) else args.clip
    out_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Error: cannot open {clip_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total

    print(f"Input: {clip_path}")
    print(f"  {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"  Processing up to {max_frames} frames")
    print(f"Output: {out_path}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = FieldTracker()

    frame_idx = 0
    times = []

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = tracker.update(frame, frame_idx)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        annotated = draw_keypoints(frame, result, frame_idx)
        writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{min(max_frames, total)}: "
                  f"{len(result.keypoints)} kps ({result.n_detected} det, "
                  f"{result.n_flow_only} flow), {elapsed:.1f}ms")

        frame_idx += 1

    cap.release()
    writer.release()

    avg_ms = np.mean(times) if times else 0
    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Avg time per frame: {avg_ms:.1f}ms")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
