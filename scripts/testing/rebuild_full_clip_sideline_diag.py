#!/usr/bin/env python3
"""Side-by-side: raw sideline mask vs grouper output.

Left panel  — raw side_mask pixels (yellow), no grouping.
Right panel — group_sideline_pixels output. Pixels INSIDE any returned
              group are colored per-group; pixels in the raw mask but
              dropped by the grouper are highlighted in red.

Lets us see frame-by-frame exactly where the sideline grouper is throwing
out valid pixels.
"""

import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import run_unet, group_sideline_pixels

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step_full_sideline_diag.mp4")

GROUP_COLORS = [
    (0, 255, 255),   # yellow (BGR)
    (255, 200, 50),  # cyan-blue
    (200, 100, 255), # magenta
]
DROP_COLOR = (60, 60, 255)   # red — raw mask px the grouper kicked out
RAW_COLOR = (0, 255, 255)    # yellow — left panel raw


def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n} frames @ {fps:.1f}fps")

    ok, frame0 = cap.read()
    if not ok: print("read failed"); return
    h, w = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, fps, (w * 2, h))

    for fi in range(n):
        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()
        _, side_mask = run_unet(frame, UNET, device="mps")

        raw_pix = int((side_mask > 0).sum())
        groups = group_sideline_pixels(side_mask)
        kept_pix = sum(len(g.pixels) for g in groups)

        # ── Left: raw mask overlay ──
        left = frame.astype(np.float32)
        m = side_mask > 0
        left[m] = 0.35 * left[m] + 0.65 * np.array(RAW_COLOR, dtype=np.float32)
        left = left.clip(0, 255).astype(np.uint8)
        cv2.putText(left,
                    f"frame {fi}  raw side_mask pixels = {raw_pix}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ── Right: group output + dropped highlight ──
        right = frame.astype(np.float32)
        # Build a "kept" mask from group pixels.
        kept_mask = np.zeros_like(side_mask, dtype=bool)
        for k, g in enumerate(groups):
            xs = g.pixels[:, 0].astype(np.int32)
            ys = g.pixels[:, 1].astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            kept_mask[ys[valid], xs[valid]] = True
            color = np.array(GROUP_COLORS[k % len(GROUP_COLORS)], dtype=np.float32)
            right[ys[valid], xs[valid]] = (
                0.35 * right[ys[valid], xs[valid]] + 0.65 * color
            )
        # Dropped pixels: raw mask AND not in any group.
        dropped = (side_mask > 0) & ~kept_mask
        if dropped.any():
            right[dropped] = (
                0.35 * right[dropped] +
                0.65 * np.array(DROP_COLOR, dtype=np.float32)
            )
        right = right.clip(0, 255).astype(np.uint8)
        n_drop = int(dropped.sum())
        cv2.putText(right,
                    f"groups={len(groups)}  kept={kept_pix}  "
                    f"dropped(red)={n_drop}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(np.hstack([left, right]))

        if fi % 30 == 0 or fi < 5:
            print(f"  frame {fi:>3}  raw={raw_pix}  groups={len(groups)}  "
                  f"kept={kept_pix}  dropped={n_drop}  "
                  f"({(time.time()-t0)*1000:.0f}ms)")

    cap.release(); writer.release()
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
