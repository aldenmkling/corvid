#!/usr/bin/env python3
"""Side-by-side comparison of frame undistortion under different distortion
fits (full vs subsampled).

Layout per frame (top→bottom):
  source | undistorted with full-resolution (k1, k2) | undistorted with sub=50 (k1, k2)
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    solve_grid, run_unet, run_hash_w18, calibrate_distortion_from_result,
)

UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")
W18 = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output/distortion_compare")


def undistort(frame, k1, k2, focal):
    h, w = frame.shape[:2]
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]],
                 dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(frame, K, dist)


def label_panel(img, text, color=(255, 255, 255)):
    bar = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", default=[
        "2024090802_play_126_sideline_f000368",   # k2 sign flip
        "2024091501_play_128_sideline_f000085",   # k2 sign flip
        "2024102701_play_148_sideline_f000439",   # large k1, k2
    ])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for fid in args.frames:
        path = os.path.join(PROJECT_ROOT, "data/line_detection/valid/images",
                             f"{fid}.jpg")
        frame = cv2.imread(path)
        if frame is None:
            print(f"  skip {fid}: not found")
            continue
        h, w = frame.shape[:2]
        focal = float(max(h, w))

        yard, side = run_unet(frame, UNET, device="mps")
        hp, hc = run_hash_w18(frame, W18, device="mps")
        result = solve_grid(yard, side, hp, hc, frame_shape=(h, w),
                             grouping_mode="cc")

        k1f, k2f = calibrate_distortion_from_result(
            result, frame_shape=(h, w), subsample=1)
        k1s, k2s = calibrate_distortion_from_result(
            result, frame_shape=(h, w), subsample=50)

        und_full = undistort(frame, k1f, k2f, focal)
        und_sub = undistort(frame, k1s, k2s, focal)

        # Resize each panel to fit nicely (cap width)
        target_w = 900
        def resize(img):
            r = target_w / img.shape[1]
            return cv2.resize(img, (target_w, int(img.shape[0] * r)))

        src = label_panel(resize(frame), f"{fid}  (source)")
        full_panel = label_panel(
            resize(und_full),
            f"undistort  k1={k1f:+.4f}  k2={k2f:+.4f}  (full subsample=1)",
        )
        sub_panel = label_panel(
            resize(und_sub),
            f"undistort  k1={k1s:+.4f}  k2={k2s:+.4f}  (subsample=50)",
        )

        full = np.vstack([src, full_panel, sub_panel])
        out_path = os.path.join(args.out, f"{fid}__distortion_compare.jpg")
        cv2.imwrite(out_path, full)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
