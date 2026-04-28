#!/usr/bin/env python3
"""
Test the HomographyTracker across a sequence of frames from a clip.

Extracts frames at specified timestamps from a play clip video, bootstraps
the tracker on the first frame with a known anchor, then processes subsequent
frames. For each frame, reports the method used (full/delta/carry) and
reprojection error. Saves overlaid visualizations.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.tracker import HomographyTracker
from src.homography.distortion import undistort_points
from src.homography.apply_homography import field_to_pixel
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "unet_line_round2_best.pth")
HASH_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_w18_hash_round1_best.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "tracker_test")


def extract_frames(video_path, fractions):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frac in fractions:
        idx = max(0, min(total - 1, int(total * frac)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        frames.append((frac, idx, f if ret else None))
    cap.release()
    return frames


def undistort_frame(frame, intrinsics):
    h, w = frame.shape[:2]
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ])
    dist = np.array([intrinsics.k1, intrinsics.k2, 0, 0, 0])
    if abs(intrinsics.k1) < 1e-6 and abs(intrinsics.k2) < 1e-6:
        return frame
    return cv2.undistort(frame, K, dist)


def draw_overlay(frame, tracker, result, out_path):
    vis = undistort_frame(frame, tracker.intrinsics)
    h, w = vis.shape[:2]
    H_inv = result.H_inv

    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        fp = np.column_stack([np.full_like(ys, x), ys])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1, p2 = tuple(pp[i].astype(int)), tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1, p2 = tuple(pp[i].astype(int)), tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (0, 200, 200), 1, cv2.LINE_AA)

    for y in [0, FIELD_WIDTH]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1, p2 = tuple(pp[i].astype(int)), tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

    # Correspondences
    for i in range(len(result.pixel_pts_u)):
        det = tuple(result.pixel_pts_u[i].astype(int))
        pb = field_to_pixel(np.array([result.field_pts[i]]), H_inv)[0]
        proj = tuple(pb.astype(int))
        cv2.circle(vis, det, 5, (0, 0, 255), 2)
        cv2.circle(vis, proj, 3, (0, 255, 0), -1)
        cv2.line(vis, det, proj, (255, 255, 0), 1)

    # Title
    title = (f"Frame {result.frame_idx}  [{result.method.upper()}]  "
             f"n={result.n_correspondences}  "
             f"err={result.field_reproj_error_mean:.3f} yd")
    for off in [(10, 30), (11, 31)]:
        cv2.putText(vis, title, off, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=os.path.join(PROJECT_ROOT, "videos", "clips", "2019102712",
                             "play_001", "sideline.mp4"),
    )
    parser.add_argument("--anchor", type=float, default=35.0,
                        help="NGS x of grid_pos=0 on the first (bootstrap) frame.")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = extract_frames(args.video, args.fractions)

    tracker = HomographyTracker(UNET_WEIGHTS, HASH_WEIGHTS, device="mps")

    for i, (frac, idx, frame) in enumerate(frames):
        if frame is None:
            print(f"[{i}] frame at t={frac} failed to load")
            continue
        anchor = args.anchor if i == 0 else None
        result = tracker.process_frame(frame, anchor_ngs_x=anchor)

        print(f"[{i}] t={frac:.2f} frame#{idx}  "
              f"method={result.method:<5}  n={result.n_correspondences:>2}  "
              f"err={result.field_reproj_error_mean:.3f}yd", end="")
        if result.method == "delta":
            print(f"  scale={result.delta_scale:.3f} "
                  f"rot={result.delta_rotation_deg:.2f}° "
                  f"tr=({result.delta_translation_px[0]:.0f},"
                  f"{result.delta_translation_px[1]:.0f})")
        else:
            print()

        out_path = os.path.join(OUTPUT_DIR, f"frame_{i:02d}_t{int(frac*100):03d}.jpg")
        draw_overlay(frame, tracker, result, out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
