#!/usr/bin/env python3
"""Sample frames around given timestamps from a clip and visualize UNet mask
output as: source frame | yard mask overlay (cyan) + side mask overlay (yellow).

Use for debugging frames where sideline detection fails.

Usage:
    python scripts/testing/viz_unet_at_timestamps.py \\
        --clip videos/clips/2019102712/play_011/sideline.mp4 \\
        --timestamps 13 14 15 16 17 \\
        --weights models/unet_line_round3_best.pth \\
        --out output/unet_debug
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import run_unet


def overlay_masks(frame, yard_mask, side_mask, alpha=0.55):
    """Overlay yard (cyan) + side (yellow) masks on the source frame."""
    out = frame.copy().astype(np.float32)
    cyan = np.array([255, 255, 0], dtype=np.float32)
    yellow = np.array([0, 255, 255], dtype=np.float32)
    yard_mask_b = yard_mask > 0
    side_mask_b = side_mask > 0
    out[yard_mask_b] = (1 - alpha) * out[yard_mask_b] + alpha * cyan
    out[side_mask_b] = (1 - alpha) * out[side_mask_b] + alpha * yellow
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--timestamps", type=float, nargs="+", required=True,
                    help="Seconds from clip start, e.g. 13 14 15 16 17")
    ap.add_argument("--weights",
                    default=os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output/unet_debug"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = os.path.splitext(os.path.basename(args.clip))[0]
    parent = os.path.basename(os.path.dirname(args.clip))
    tag = f"{parent}_{base}"
    print(f"  {tag}: {n_frames} frames @ {fps:.1f} fps")

    panels = []
    for t in args.timestamps:
        idx = int(round(t * fps))
        idx = max(0, min(idx, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  skipping t={t}s (frame {idx}, read failed)")
            continue

        yard, side = run_unet(frame, args.weights, device=args.device)

        h, w = frame.shape[:2]
        # Resize each panel to fit nicely.
        panel_w = 640
        panel_h = int(panel_w * h / w)

        src = cv2.resize(frame, (panel_w, panel_h))
        ovr = cv2.resize(overlay_masks(frame, yard, side), (panel_w, panel_h))

        # Stat strip
        n_yard = int((yard > 0).sum())
        n_side = int((side > 0).sum())
        bar = np.zeros((36, panel_w * 2, 3), dtype=np.uint8)
        msg = (f"t={t:.1f}s  f{idx}  "
               f"yard_pixels={n_yard}  side_pixels={n_side}")
        cv2.putText(bar, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        row = np.vstack([bar, np.hstack([src, ovr])])
        panels.append(row)
        print(f"  t={t:>5.1f}s  f{idx:>4}  yard={n_yard:>6}  side={n_side:>6}")

    cap.release()
    if panels:
        full = np.vstack(panels)
        out_path = os.path.join(args.out, f"{tag}_unet_at_timestamps.jpg")
        cv2.imwrite(out_path, full)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
