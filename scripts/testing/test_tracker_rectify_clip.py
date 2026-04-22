#!/usr/bin/env python3
"""Run HomographyTracker across a play clip and produce:
  1. A first-frame viz with grid_pos labeled (for the user to anchor on).
  2. A rectified top-down video of every frame (after --anchor is provided).

Usage:
  # Step 1: look at first frame to pick an anchor
  python test_tracker_rectify_clip.py --clip videos/clips/.../sideline.mp4 --show-first

  # Step 2: render the rectified video
  python test_tracker_rectify_clip.py --clip ... --anchor 60.0 --output out.mp4
"""

import argparse
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.tracker import HomographyTracker
from src.homography.camera_model import undistort_points
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
    YARD_LINE_POSITIONS,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "tracker_rectify")
YD_PER_PX = 0.1


def show_first_frame(clip_path: str, out_path: str):
    """Render the first frame with HRNet detections labeled by grid_pos."""
    cap = cv2.VideoCapture(clip_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"failed to read {clip_path}")
        return
    tracker = HomographyTracker(WEIGHTS)
    det = tracker._detect(frame)
    groups = det["groups"]
    sideline_pxs = det["sideline_pxs"]

    vis = frame.copy()
    # Paired groups get ordered grid_pos from assign_grid_positions
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255), (80, 255, 255),
              (255, 150, 50), (150, 50, 255), (50, 255, 150),
              (200, 200, 200)]
    for g in groups:
        gp = g.get("grid_pos")
        if gp is None:
            continue
        color = colors[gp % len(colors)]
        fh, nh, sl = g.get("far_hash"), g.get("near_hash"), g.get("sideline")
        if g.get("singleton"):
            pt = fh or nh or sl
            if pt is None: continue
            cv2.drawMarker(vis, tuple(int(x) for x in pt), color,
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"g{gp}s", (int(pt[0])+8, int(pt[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            if nh and fh:
                cv2.line(vis, tuple(int(x) for x in nh),
                         tuple(int(x) for x in fh), color, 2)
                cv2.drawMarker(vis, tuple(int(x) for x in fh), color,
                               cv2.MARKER_CROSS, 16, 2)
                cv2.drawMarker(vis, tuple(int(x) for x in nh), color,
                               cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, f"g{gp}", (int(nh[0])+8, int(nh[1])+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if sl:
                cv2.circle(vis, tuple(int(x) for x in sl), 10, color, 2)

    # Legend with NGS anchor reference
    cv2.putText(vis, "Tell me: what NGS x is g0? (NGS: 10=leftGoal, 60=50yd, 110=rightGoal)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "Each g is 5 yd apart in field coords.",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"  saved {out_path}")
    # Also print identified grid_pos range
    gps = sorted([g["grid_pos"] for g in groups if g.get("grid_pos") is not None])
    if gps:
        print(f"  grid_pos range: g{gps[0]} through g{gps[-1]} ({len(set(gps))} distinct)")


def rectify_clip(clip_path: str, anchor: float, output_mp4: str,
                 fps_override: float = None, use_track_bank: bool = True,
                 smooth_window: int = 0, smooth_poly: int = 2):
    """Run tracker on every frame, warp each to top-down, write as MP4.

    If smooth_window > 0: run in two passes. Pass 1 runs tracker + caches
    frames + H. Pass 2 applies Savitzky-Golay to each H-matrix entry across
    time (zero-phase, offline) and renders rectified output with smoothed H.
    """
    cap = cv2.VideoCapture(clip_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {total} frames @ {fps} fps, {w}x{h}  smooth_window={smooth_window}")

    # Field output dimensions
    field_w = int(FIELD_LENGTH / YD_PER_PX)
    field_h = int(FIELD_WIDTH / YD_PER_PX)

    # Each output frame = side-by-side: [original] | [rectified]
    out_w = w + field_w
    out_h = max(h, field_h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (out_w, out_h))

    tracker = HomographyTracker(WEIGHTS, use_track_bank=use_track_bank)

    # Flip y so near sideline is at bottom
    S = np.array([[1.0 / YD_PER_PX, 0, 0],
                  [0, -1.0 / YD_PER_PX, float(field_h)],
                  [0, 0, 1]], dtype=np.float64)

    def field_to_rect_px(x_yd, y_yd):
        return (int(x_yd / YD_PER_PX), int(field_h - y_yd / YD_PER_PX))

    method_counts = {"full": 0, "delta": 0, "carry": 0}
    errs = []

    # ── Pass 1: run tracker, cache frames + H ──
    cached_frames = []   # list of undistorted BGR frames
    raw_Hs = []          # (N, 3, 3) — to be optionally smoothed
    meta = []            # per-frame method + n + err
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            anchor_arg = anchor if i == 0 else None
            result = tracker.process_frame(frame, anchor_ngs_x=anchor_arg)
        except Exception as e:
            print(f"  frame {i}: tracker error {e}")
            continue

        method_counts[result.method] += 1
        if result.field_reproj_error_mean == result.field_reproj_error_mean:
            errs.append(result.field_reproj_error_mean)

        # Undistort now, cache for pass 2
        K = np.array([
            [tracker.intrinsics.fx, 0, tracker.intrinsics.cx],
            [0, tracker.intrinsics.fy, tracker.intrinsics.cy],
            [0, 0, 1],
        ])
        dist_coeffs = np.array([tracker.intrinsics.k1, tracker.intrinsics.k2, 0, 0, 0])
        if abs(tracker.intrinsics.k1) > 1e-6 or abs(tracker.intrinsics.k2) > 1e-6:
            frame_u = cv2.undistort(frame, K, dist_coeffs)
        else:
            frame_u = frame.copy()

        cached_frames.append(frame_u)
        raw_Hs.append(result.H.copy())
        meta.append({
            "method": result.method,
            "n": result.n_correspondences,
            "err": (result.field_reproj_error_mean
                    if result.field_reproj_error_mean == result.field_reproj_error_mean
                    else None),
        })
    cap.release()
    print(f"  pass 1 done: cached {len(cached_frames)} frames")

    # ── Optional: Savitzky-Golay smoothing over H ──
    if smooth_window > 0 and len(raw_Hs) >= smooth_window:
        from scipy.signal import savgol_filter
        H_flat = np.stack([h.flatten() for h in raw_Hs], axis=0)  # (N, 9)
        # Normalize each row so H[2,2]=1 BEFORE smoothing (consistent scale)
        scales = H_flat[:, 8:9].copy()
        scales[np.abs(scales) < 1e-9] = 1.0
        H_flat_n = H_flat / scales
        # Savitzky-Golay along time axis
        H_smooth_flat = savgol_filter(H_flat_n, window_length=smooth_window,
                                      polyorder=min(smooth_poly, smooth_window - 1),
                                      axis=0, mode="nearest")
        # Re-apply original scale
        H_smooth_flat = H_smooth_flat * scales
        Hs = [H_smooth_flat[i].reshape(3, 3) for i in range(len(raw_Hs))]
        print(f"  applied Savitzky-Golay (window={smooth_window}, poly={smooth_poly})")
    else:
        Hs = raw_Hs

    # ── Pass 2: render with (possibly smoothed) H ──
    for i, (frame_u, H_use, m) in enumerate(zip(cached_frames, Hs, meta)):
        H_pixel_to_rect = S @ H_use
        rectified = cv2.warpPerspective(frame_u, H_pixel_to_rect, (field_w, field_h))

        # Overlay yard-line grid on rectified
        for x in np.arange(0, FIELD_LENGTH + 1, 5):
            p1 = field_to_rect_px(x, 0)
            p2 = field_to_rect_px(x, FIELD_WIDTH)
            cv2.line(rectified, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
            if int(x) % 10 == 0:
                cv2.putText(rectified, f"{int(x)}",
                            field_to_rect_px(x + 0.3, 2.5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for y in [HASH_Y_NEAR, HASH_Y_FAR]:
            cv2.line(rectified, field_to_rect_px(0, y),
                     field_to_rect_px(FIELD_LENGTH, y),
                     (0, 200, 200), 1, cv2.LINE_AA)
        for y in [0, FIELD_WIDTH]:
            cv2.line(rectified, field_to_rect_px(0, y),
                     field_to_rect_px(FIELD_LENGTH, y),
                     (255, 255, 255), 2)

        # Diagnostic text on rectified
        cv2.putText(rectified, f"f{i} {m['method']} n={m['n']}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if m['err'] is not None:
            cv2.putText(rectified, f"err={m['err']:.2f}yd",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Compose side-by-side, vertically centering each panel
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y0_left = (out_h - h) // 2
        canvas[y0_left:y0_left + h, :w] = frame_u
        y0_right = (out_h - field_h) // 2
        canvas[y0_right:y0_right + field_h, w:w + field_w] = rectified

        writer.write(canvas)
        if (i + 1) % 60 == 0:
            print(f"  [{i+1}/{len(cached_frames)}] render")

    writer.release()
    print(f"  methods: {method_counts}")
    if errs:
        print(f"  mean err on real obs: {np.mean(errs):.3f} yd ({len(errs)}/{total} frames)")
    print(f"  wrote {output_mp4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--anchor", type=float, default=None,
                        help="NGS x of grid_pos 0 on the first frame.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-first", action="store_true",
                        help="Just render the first-frame anchor viz.")
    parser.add_argument("--no-track-bank", action="store_true",
                        help="Disable KeypointTrackBank (for A/B comparison).")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="Savitzky-Golay window (odd, e.g. 15) over H "
                             "matrix across frames. 0 = disabled.")
    parser.add_argument("--smooth-poly", type=int, default=2)
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.clip))[0]
    parent = os.path.basename(os.path.dirname(args.clip))
    tag = f"{parent}_{base}"
    if args.show_first:
        out = os.path.join(OUTPUT_DIR, f"{tag}_first_anchor.jpg")
        show_first_frame(args.clip, out)
    else:
        if args.anchor is None:
            print("--anchor required when rendering full clip")
            return
        suffix = "_nobank" if args.no_track_bank else ""
        if args.smooth_window > 0:
            suffix += f"_sg{args.smooth_window}"
        out = args.output or os.path.join(
            OUTPUT_DIR, f"{tag}_rectified{suffix}.mp4")
        rectify_clip(args.clip, args.anchor, out,
                     use_track_bank=not args.no_track_bank,
                     smooth_window=args.smooth_window,
                     smooth_poly=args.smooth_poly)


if __name__ == "__main__":
    main()
