#!/usr/bin/env python3
"""Diagnostic: raw UNet masks (no CC grouping) overlaid on full-clip frames.

Lets us tell whether the late-clip degradation is from UNet missing the
lines (mask is blank) or from CC grouping rejecting valid pixels.

Yardline mask in cyan, sideline mask in yellow, no per-line color.
"""

import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import run_unet

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step_full_raw_masks.mp4")

YARD_COLOR = np.array([255, 255, 0], dtype=np.float32)    # cyan (BGR)
SIDE_COLOR = np.array([0, 255, 255], dtype=np.float32)    # yellow (BGR)


def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n_total} frames @ {fps:.1f}fps")

    ok, frame0 = cap.read()
    if not ok: print("read failed"); return
    h, w = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))

    for fi in range(n_total):
        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()
        yard_mask, side_mask = run_unet(frame, UNET, device="mps")
        canvas = frame.astype(np.float32)
        ym = yard_mask > 0
        sm = side_mask > 0
        canvas[ym] = 0.35 * canvas[ym] + 0.65 * YARD_COLOR
        canvas[sm] = 0.35 * canvas[sm] + 0.65 * SIDE_COLOR
        canvas = canvas.clip(0, 255).astype(np.uint8)
        cv2.putText(canvas,
                    f"frame {fi}  t={fi/fps:.2f}s  "
                    f"yard_px={int(ym.sum())}  side_px={int(sm.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        if fi % 30 == 0 or fi < 5:
            print(f"  frame {fi:>3}  yard_px={int(ym.sum())}  "
                  f"side_px={int(sm.sum())}  ({(time.time()-t0)*1000:.0f}ms)")

    cap.release(); writer.release()
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
