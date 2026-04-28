#!/usr/bin/env python3
"""Frame-by-frame pipeline state dump for debugging the grid solver + track bank.

Runs HomographyTracker through a clip, then prints the full intermediate
state for each frame in a target window:
  - yardline groups: pixel y-range, peak_coord, grid_pos, residual, ok flag,
    which keypoints got attached (hashes + sideline intersections)
  - sideline groups: pixel x/y-range, polynomial coefficients
  - correspondences emitted (per kind: hash counts, sideline counts)
  - track bank state + H-validation result + method chosen

Usage:
    python scripts/testing/diag_pipeline_state.py \\
        --clip videos/clips/2019102712/play_011/sideline.mp4 \\
        --anchor 35.0 \\
        --t-start 14.5 --t-end 16.0
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.tracker import HomographyTracker
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH

UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
W18 = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")


def kind_from_field_y(fy):
    if fy < 1.0: return "side_near"
    if fy > FIELD_WIDTH - 1.0: return "side_far"
    return "near_hash" if abs(fy - HASH_Y_NEAR) < abs(fy - HASH_Y_FAR) else "far_hash"


def dump_frame_state(det, fr_result, frame_idx, t_sec):
    print(f"\n========== frame {frame_idx}  t={t_sec:.2f}s  "
          f"method={fr_result.method}  n={fr_result.n_correspondences}  "
          f"err={fr_result.field_reproj_error_mean:.3f}yd ==========")
    result = det["result"]
    print(f"  UNet: yard_pixels={int(det['yard_mask'].sum())}, "
          f"side_pixels={int(det['side_mask'].sum())}")
    print(f"  W18: {len(det['hash_pxs'])} hashes")

    # Yardlines
    print(f"\n  Yardlines ({len(result.yardlines)}):")
    for yl in result.yardlines:
        line = yl.line
        ymin, ymax = ((float(line.pixels[:, 1].min()),
                        float(line.pixels[:, 1].max()))
                      if line.pixels is not None else (-1, -1))
        npx = len(line.pixels) if line.pixels is not None else 0
        rmse = f"{line.residual_rmse:.2f}" if line.residual_rmse is not None else "?"
        fit_resid = (f"{yl.grid_fit_residual:.3f}"
                      if yl.grid_fit_residual is not None else "?")
        kp_str = ""
        kp_str += "NH" if yl.near_hash is not None else ".."
        kp_str += "/" + ("FH" if yl.far_hash is not None else "..")
        kp_str += "/" + ("nS" if yl.near_sideline is not None else "..")
        kp_str += "/" + ("fS" if yl.far_sideline is not None else "..")
        peak_str = f"{line.peak_coord:.0f}" if line.peak_coord is not None else "?"
        print(f"    gp={yl.grid_pos!s:>3} ok={str(yl.grid_fit_ok):>5} "
              f"y[{ymin:.0f},{ymax:.0f}] npx={npx:>5} rmse={rmse:>5} "
              f"peak={peak_str:>5} resid={fit_resid:>5} kp={kp_str}")

    # Sidelines
    print(f"\n  Sidelines:")
    for label, sl in (("far_sideline", result.far_sideline),
                       ("near_sideline", result.near_sideline)):
        if sl is None:
            print(f"    {label}: None")
            continue
        xmin, xmax = float(sl.pixels[:, 0].min()), float(sl.pixels[:, 0].max())
        ymin, ymax = float(sl.pixels[:, 1].min()), float(sl.pixels[:, 1].max())
        npx = len(sl.pixels)
        rmse = f"{sl.residual_rmse:.2f}" if sl.residual_rmse is not None else "?"
        print(f"    {label}: x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}] "
              f"npx={npx} rmse={rmse}")

    # Correspondences emitted
    diag = fr_result.diagnostics
    print(f"\n  Correspondences: n_real={diag.get('n_current_corr', '?')}, "
          f"n_coasted={diag.get('n_coasted_corr', '?')}")
    if fr_result.field_pts is not None and len(fr_result.field_pts) > 0:
        kinds = [kind_from_field_y(float(fp[1])) for fp in fr_result.field_pts]
        c = Counter(kinds)
        print(f"    by kind: {dict(c)}")

    # Track bank
    tb_stats = diag.get('track_bank')
    tb_validation = diag.get('track_bank_validation')
    tb_validated = diag.get('track_bank_validated')
    print(f"\n  Track bank: stats={tb_stats}")
    print(f"  Track bank validation: validated={tb_validated}, info={tb_validation}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--anchor", type=float, required=True)
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--t-end", type=float, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--linearize", action="store_true",
                    help="Run with linearize=True (deg-1 in undistorted space)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_start = int(round(args.t_start * fps))
    f_end = int(round(args.t_end * fps))
    print(f"clip: {args.clip}")
    print(f"  {n_frames} frames @ {fps:.1f}fps, dump window f{f_start}–f{f_end}")

    tracker = HomographyTracker(UNET, W18, device=args.device,
                                  linearize=args.linearize)

    # Patch _detect to stash the most recent det dict for diagnostic access.
    orig_detect = tracker._detect
    last_det = {}
    def patched_detect(frame):
        det = orig_detect(frame)
        last_det.clear(); last_det.update(det)
        return det
    tracker._detect = patched_detect

    for i in range(min(f_end + 1, n_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        anchor = args.anchor if i == 0 else None
        try:
            r = tracker.process_frame(frame, anchor_ngs_x=anchor)
        except Exception as e:
            print(f"  frame {i}: process error {e}")
            continue
        if f_start <= i <= f_end:
            dump_frame_state(last_det, r, i, i / fps)

    cap.release()


if __name__ == "__main__":
    main()
