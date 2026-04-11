#!/usr/bin/env python3
"""
Test the ML-based field keypoint detector and homography pipeline.

Runs the trained HRNet model on play clips and generates annotated
visualization videos showing detected keypoints, computed homography,
and projected field grid.

Usage:
    python scripts/test_ml_homography.py --weights models/hrnet_best.pth \
        --clip videos/clips/2019092204/play_010/sideline.mp4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import argparse
import time

from src.homography.keypoint_detector import FieldKeypointDetector, KeypointDetection
from src.homography.keypoint_schema import KEYPOINTS, NUM_KEYPOINTS
from src.homography.compute_homography import compute_homography
from src.homography.apply_homography import pixel_to_field, field_to_pixel
from src.homography.field_model import YARD_LINE_POSITIONS, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR


# Color by keypoint type
TYPE_COLORS = {
    "near_sideline": (0, 0, 255),      # red
    "near_hash": (0, 255, 0),          # green
    "far_hash": (255, 0, 0),           # blue
    "far_sideline": (0, 255, 255),     # yellow
    "near_number": (255, 0, 255),      # magenta
    "far_number": (255, 128, 255),     # pink
    "endzone_corner": (0, 165, 255),   # orange
}


def draw_result(frame, detection, homography_result, frame_idx):
    """Draw detected keypoints and projected field grid on frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Draw detected keypoints
    for i in range(len(detection.pixel_xy)):
        x, y = int(detection.pixel_xy[i, 0]), int(detection.pixel_xy[i, 1])
        ki = detection.keypoint_ids[i]
        conf = detection.confidences[i]
        kp = KEYPOINTS[ki]

        color = TYPE_COLORS.get(kp["type"], (255, 255, 255))
        radius = int(3 + 5 * conf)
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)

        label = f"{kp['name'][:10]} {conf:.2f}"
        cv2.putText(overlay, label, (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw projected field grid if homography is available
    if homography_result is not None:
        H_inv = homography_result.H_inv

        # Project yard lines
        for yl_x in YARD_LINE_POSITIONS:
            pts = np.array([[yl_x, 0.0], [yl_x, FIELD_WIDTH]])
            px_pts = field_to_pixel(pts, H_inv)

            if np.all(np.isfinite(px_pts)):
                p1 = tuple(px_pts[0].astype(int))
                p2 = tuple(px_pts[1].astype(int))
                cv2.line(overlay, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

        # Project hash lines
        for hash_y in [HASH_Y_NEAR, HASH_Y_FAR]:
            pts = np.array([[10.0, hash_y], [110.0, hash_y]])
            px_pts = field_to_pixel(pts, H_inv)
            if np.all(np.isfinite(px_pts)):
                p1 = tuple(px_pts[0].astype(int))
                p2 = tuple(px_pts[1].astype(int))
                cv2.line(overlay, p1, p2, (80, 80, 80), 1, cv2.LINE_AA)

    # Stats overlay
    n_kps = len(detection.pixel_xy)
    reproj = homography_result.reprojection_error if homography_result else float('nan')
    n_inliers = homography_result.n_inliers if homography_result else 0

    stats = [
        f"Frame {frame_idx}",
        f"Keypoints: {n_kps}",
        f"Homography: {'OK' if homography_result else 'FAIL'}",
        f"Inliers: {n_inliers}, Reproj: {reproj:.1f}px",
    ]
    for i, line in enumerate(stats):
        cv2.putText(overlay, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(overlay, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Test ML homography pipeline")
    parser.add_argument("--weights", required=True, help="Path to HRNet checkpoint")
    parser.add_argument("--clip", default="videos/clips/2019092204/play_010/sideline.mp4")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", default="output/ml_homography_test.mp4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clip_path = os.path.join(project_root, args.clip) if not os.path.isabs(args.clip) else args.clip
    out_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load detector
    print(f"Loading model from {weights_path}...")
    detector = FieldKeypointDetector(weights_path, args.device, args.conf_thresh)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Error: cannot open {clip_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total

    print(f"Input: {clip_path} ({w}x{h} @ {fps:.1f}fps, {total} frames)")
    print(f"Output: {out_path}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    det_times = []
    hom_times = []
    n_success = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints
        t0 = time.perf_counter()
        detection = detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        det_times.append(det_ms)

        # Compute homography
        t0 = time.perf_counter()
        hom_result = None
        if len(detection.pixel_xy) >= 4:
            hom_result = compute_homography(
                detection.pixel_xy, detection.field_xy, ransac_threshold=5.0
            )
            if hom_result is not None:
                n_success += 1
        hom_ms = (time.perf_counter() - t0) * 1000
        hom_times.append(hom_ms)

        # Draw visualization
        annotated = draw_result(frame, detection, hom_result, frame_idx)
        writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{min(max_frames, total)}: "
                  f"{len(detection.pixel_xy)} kps, "
                  f"det={det_ms:.1f}ms, hom={hom_ms:.1f}ms, "
                  f"{'OK' if hom_result else 'FAIL'}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Detection: {np.mean(det_times):.1f}ms avg")
    print(f"Homography: {np.mean(hom_times):.1f}ms avg")
    print(f"Success rate: {n_success}/{frame_idx} ({100 * n_success / max(frame_idx, 1):.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
