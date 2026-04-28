#!/usr/bin/env python3
"""Diagnostic: show raw HRNet-W18 hash detections + UNet line masks on
frame 0 of play_023 (2019092204). Helps see what the detectors are
actually emitting before any pipeline gating.
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, run_hash_w18, group_yardline_pixels_cc,
)
from scripts.testing.rebuild_full_clip_viz import (
    group_sideline_pixels as cc_group_sideline,
)

CLIP = os.path.join(PROJECT_ROOT,
                     "videos/clips/2019092204/play_023/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
HASH = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/debug_play023_frame0.jpg")


def main():
    cap = cv2.VideoCapture(CLIP)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n_total} frames")
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    print(f"  frame 0: {w}x{h}")

    # UNet
    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yard_px = int((yard_mask > 0).sum())
    side_px = int((side_mask > 0).sum())
    print(f"  UNet: yard_px={yard_px}  side_px={side_px}")

    yl_groups = group_yardline_pixels_cc(yard_mask)
    sl_groups = cc_group_sideline(side_mask)
    print(f"  CC: {len(yl_groups)} yardline groups, {len(sl_groups)} sideline groups")

    # HRNet hashes — try multiple thresholds to see what's there
    for thresh in (0.20, 0.30, 0.40, 0.45, 0.50):
        pxs, confs = run_hash_w18(frame, HASH, device="mps", conf_thresh=thresh)
        print(f"  HRNet @ thresh={thresh}: {len(pxs)} hashes  "
              f"(max_conf={float(confs.max()) if len(confs) else 0:.3f})")

    # Build viz: 2-panel
    # Top: original + UNet masks (cyan=yard, yellow=side)
    top = frame.astype(np.float32)
    ym = yard_mask > 0; sm = side_mask > 0
    top[ym] = 0.4 * top[ym] + 0.6 * np.array([255, 255, 0], dtype=np.float32)
    top[sm] = 0.4 * top[sm] + 0.6 * np.array([0, 255, 255], dtype=np.float32)
    top = top.clip(0, 255).astype(np.uint8)
    cv2.putText(top, f"UNet masks  yard={yard_px}px  side={side_px}px  "
                f"groups: {len(yl_groups)}yl + {len(sl_groups)}sl",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom: original + raw hashes at thresh=0.20 with confidence labels
    bottom = frame.copy()
    pxs, confs = run_hash_w18(frame, HASH, device="mps", conf_thresh=0.20)
    print()
    for i in range(len(pxs)):
        x, y = int(pxs[i, 0]), int(pxs[i, 1])
        cf = float(confs[i])
        # Color by confidence: green if >0.45 (production), amber 0.30–0.45,
        # gray below 0.30
        if cf >= 0.45:
            color = (100, 255, 100)
        elif cf >= 0.30:
            color = (0, 200, 255)
        else:
            color = (180, 180, 180)
        cv2.circle(bottom, (x, y), 6, color, -1)
        cv2.circle(bottom, (x, y), 8, (0, 0, 0), 1)
        cv2.putText(bottom, f"{cf:.2f}",
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)
        print(f"    hash[{i:>2}]  ({x:>4}, {y:>4})  conf={cf:.3f}")
    cv2.putText(bottom,
                f"raw HRNet hashes @ thresh=0.20  (green ≥0.45, amber ≥0.30)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    full = np.vstack([top, bottom])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, full)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
