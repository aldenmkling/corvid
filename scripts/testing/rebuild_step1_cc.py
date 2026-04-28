#!/usr/bin/env python3
"""Step 1 of the rebuild: CC grouping on the first frame.

Runs UNet, then CC + collinearity merge for yardlines and sidelines.
Shows: source | yard mask + side mask | grouped yardlines (colored) +
grouped sidelines (colored). Each group gets a unique color.

Usage:
    python scripts/testing/rebuild_step1_cc.py
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels,
)

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step1_cc.jpg")

# 12 distinct colors for groups
COLORS = [
    (0, 255, 255),     # yellow
    (0, 165, 255),     # orange
    (0, 100, 255),     # red-orange
    (255, 0, 255),     # magenta
    (255, 100, 0),     # blue
    (255, 255, 0),     # cyan
    (100, 255, 100),   # light green
    (0, 255, 100),     # green
    (200, 200, 200),   # gray
    (180, 105, 255),   # pink
    (50, 150, 255),
    (255, 50, 50),
]


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"failed to read {CLIP}")
        return
    h, w = frame.shape[:2]
    print(f"  frame: {w}x{h}")

    # 1) UNet
    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    print(f"  yard_mask: {int(yard_mask.sum())} pixels")
    print(f"  side_mask: {int(side_mask.sum())} pixels")

    # 2) CC grouping
    yl_groups = group_yardline_pixels_cc(yard_mask)
    sl_groups = group_sideline_pixels(side_mask)
    print(f"  yardline groups: {len(yl_groups)}")
    print(f"  sideline groups: {len(sl_groups)}")

    # Build viz: 3-up vertical
    # (a) source
    panel_a = frame.copy()
    cv2.putText(panel_a, "source", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # (b) UNet masks overlaid (yard=cyan, side=yellow)
    panel_b = frame.copy().astype(np.float32)
    yard_b = yard_mask > 0
    side_b = side_mask > 0
    panel_b[yard_b] = 0.4 * panel_b[yard_b] + 0.6 * np.array([255, 255, 0], dtype=np.float32)
    panel_b[side_b] = 0.4 * panel_b[side_b] + 0.6 * np.array([0, 255, 255], dtype=np.float32)
    panel_b = panel_b.clip(0, 255).astype(np.uint8)
    cv2.putText(panel_b, "UNet masks (yard=cyan, side=yellow)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # (c) Groupings — each yardline + sideline gets a unique color
    panel_c = frame.copy().astype(np.float32)
    for i, lo in enumerate(yl_groups):
        c = COLORS[i % len(COLORS)]
        xs = lo.pixels[:, 0].astype(np.int32)
        ys = lo.pixels[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        panel_c[ys[valid], xs[valid]] = (
            0.4 * panel_c[ys[valid], xs[valid]]
            + 0.6 * np.array(c, dtype=np.float32)
        )
    # Sidelines drawn in distinct higher-contrast colors
    sideline_colors = [(255, 255, 255), (200, 200, 0)]
    for j, lo in enumerate(sl_groups):
        c = sideline_colors[j % len(sideline_colors)]
        xs = lo.pixels[:, 0].astype(np.int32)
        ys = lo.pixels[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        panel_c[ys[valid], xs[valid]] = (
            0.3 * panel_c[ys[valid], xs[valid]]
            + 0.7 * np.array(c, dtype=np.float32)
        )
    panel_c = panel_c.clip(0, 255).astype(np.uint8)

    cv2.putText(panel_c, f"CC groups: {len(yl_groups)} yardlines, {len(sl_groups)} sidelines",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    full = np.vstack([panel_a, panel_b, panel_c])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, full)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
