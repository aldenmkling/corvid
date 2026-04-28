#!/usr/bin/env python3
"""Step 7 of the rebuild: cross-frame yardline tracker.

Carries the grid index `g` across frames by tracking yardlines themselves
(not keypoints). Cheap because yardlines move smoothly with camera pan and
frame-to-frame Δx is far less than one unit spacing at 30 fps.

State per yardline: {g, last_x_at_center, last_frame}.

Per frame:
  1. UNet → CC groups → linear fits in undistorted space (k1 from frame 0).
  2. Compute each yardline's x_at_center (x value at y = h/2).
  3. For each detection, find the tracked yardline whose last x_at_center
     is closest. If |Δx| ≤ MATCH_FRAC × unit_px, inherit g.
  4. Yardlines without a match: assign a new g by snapping the detection's
     x to the integer grid implied by (frame-0 anchor, current unit_px).
     This handles new yardlines entering from screen edges and degenerate
     re-entries.
  5. Update state with new positions; yardlines not seen this frame retain
     their last position (so they can be re-matched if they reappear).

Output: a video with per-frame yardlines drawn and labeled with their
persistent g index. If tracking holds, labels stay consistent across the
clip even as the camera pans.
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step7_track.mp4")

N_FRAMES = 90                 # how many frames to process (3 s at 30fps)
MATCH_FRAC = 0.40             # |Δx| ≤ 40 % of unit_px to inherit g
COLORS = [
    (0, 255, 255), (0, 165, 255), (0, 100, 255), (255, 0, 255),
    (255, 100, 0), (255, 255, 0), (100, 255, 100), (0, 255, 100),
    (200, 200, 200), (180, 105, 255), (50, 150, 255), (255, 50, 50),
]


def total_mse(line_pts, line_kinds, intr):
    total_sq = 0.0; n = 0
    for p, kind in zip(line_pts, line_kinds):
        p_u = undistort_points(p.astype(np.float64), intr)
        if kind == "yardline":
            ys, xs = p_u[:, 1], p_u[:, 0]
            b, a = np.polyfit(ys, xs, 1)
            resid = (xs - (a + b * ys)) / np.sqrt(1 + b * b)
        else:
            xs, ys = p_u[:, 0], p_u[:, 1]
            b, a = np.polyfit(xs, ys, 1)
            resid = (ys - (a + b * xs)) / np.sqrt(1 + b * b)
        total_sq += float((resid ** 2).sum())
        n += len(p)
    return total_sq / max(n, 1)


def fit_yardlines_undistorted(yard_mask, intr):
    """CC + linear fit per yardline group, return list of dicts."""
    yl_groups = group_yardline_pixels_cc(yard_mask)
    fits = []
    for g in yl_groups:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        fits.append({"a": float(a), "b": float(b),
                     "ymin": float(ys.min()), "ymax": float(ys.max())})
    return fits


def assign_grid_initial(fits, cy):
    """Frame-0 grid assignment: leftmost = g=0, sequential snap by median Δx."""
    if len(fits) < 2:
        return None
    x_at_center = np.array([f["a"] + f["b"] * cy for f in fits])
    order = np.argsort(x_at_center)
    sorted_x = x_at_center[order]
    deltas = np.diff(sorted_x)
    unit_px = float(np.median(deltas))
    anchor_x = float(sorted_x[0])
    raw_g = (sorted_x - anchor_x) / unit_px
    g_sorted = np.round(raw_g).astype(int)
    g_index = np.zeros(len(fits), dtype=int)
    for k_, orig in enumerate(order):
        g_index[orig] = int(g_sorted[k_])
    return {"g_index": g_index, "x_at_center": x_at_center,
            "unit_px": unit_px, "anchor_x_g0": anchor_x}


def assign_grid_with_tracker(fits, cy, state):
    """Match each yardline to a tracked g by nearest x_at_center; new ones
    get a fresh g by snapping their x to the integer grid (frame-0 anchor,
    current unit_px).
    """
    x_at_center = np.array([f["a"] + f["b"] * cy for f in fits])

    # Update unit_px from the current frame's median Δx (camera zoom may
    # change it slightly across the clip).
    if len(x_at_center) >= 2:
        unit_px = float(np.median(np.diff(np.sort(x_at_center))))
    else:
        unit_px = state["unit_px"]
    match_thresh = MATCH_FRAC * unit_px

    g_index = np.full(len(fits), -10_000, dtype=int)
    used_g = set()
    # Greedy: order by descending nearest-tracked-distance — assign easy
    # matches first. Implement as: rank pairs (i, g) by |Δx|, accept if
    # neither i nor g already used.
    pairs = []
    for i, x in enumerate(x_at_center):
        for g, x_prev in state["last_x"].items():
            pairs.append((abs(x - x_prev), i, g))
    pairs.sort()
    for d, i, g in pairs:
        if d > match_thresh:
            break
        if g_index[i] != -10_000 or g in used_g:
            continue
        g_index[i] = g
        used_g.add(g)

    # Unmatched detections → snap to integer grid via frame-0 anchor.
    anchor_x = state["anchor_x_g0"]
    for i in range(len(fits)):
        if g_index[i] != -10_000:
            continue
        # The frame-0 anchor moves as the camera pans. Update it via any
        # tracked yardline this frame:
        anchor_now = None
        for g, x_prev in state["last_x"].items():
            if g in used_g:
                # Find the detected i for this g.
                hits = np.where(g_index == g)[0]
                if len(hits):
                    j = int(hits[0])
                    anchor_now = float(x_at_center[j] - g * unit_px)
                    break
        if anchor_now is None:
            anchor_now = anchor_x  # fallback: still use frame-0 anchor
        new_g = int(round((x_at_center[i] - anchor_now) / unit_px))
        # Avoid collision with already-used g.
        while new_g in used_g:
            # Shift toward the side the unmatched line lives on.
            shift = 1 if x_at_center[i] - (anchor_now + new_g * unit_px) > 0 else -1
            new_g += shift
        g_index[i] = new_g
        used_g.add(new_g)

    return {"g_index": g_index, "x_at_center": x_at_center,
            "unit_px": unit_px}


def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(N_FRAMES, n_total)

    # Frame 0: read, calibrate k1, set up state.
    ok, frame0 = cap.read()
    if not ok:
        print("failed to read frame 0"); return
    h, w = frame0.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask = run_unet(frame0, UNET, device="mps")
    yl_groups = group_yardline_pixels_cc(yard_mask)
    sl_groups = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl_groups] + [g.pixels for g in sl_groups]
    line_kinds = ["yardline"] * len(yl_groups) + ["sideline"] * len(sl_groups)
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)

    fits0 = fit_yardlines_undistorted(yard_mask, intr)
    init = assign_grid_initial(fits0, cy)
    if init is None:
        print("frame 0 has <2 yardlines — bailing"); return

    state = {
        "last_x": {int(g): float(x) for g, x in
                   zip(init["g_index"], init["x_at_center"])},
        "unit_px": init["unit_px"],
        "anchor_x_g0": init["anchor_x_g0"],
    }
    print(f"  frame 0: k1={k1:+.4f}  unit={state['unit_px']:.1f}px  "
          f"anchor={state['anchor_x_g0']:.1f}  "
          f"g_range=[{init['g_index'].min():+d},{init['g_index'].max():+d}]")

    # Set up writer.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

    def render(frame_u, fits, g_index, frame_idx, t_ms):
        canvas = frame_u.copy()
        for j, yf in enumerate(fits):
            g = int(g_index[j])
            color = COLORS[(g + 6) % len(COLORS)]
            ys = np.linspace(yf["ymin"], yf["ymax"], 200)
            xs = yf["a"] + yf["b"] * ys
            cv2.polylines(canvas,
                          [np.stack([xs, ys], axis=1).astype(np.int32)],
                          False, color, 2, cv2.LINE_AA)
            y_lab = max(yf["ymin"] + 30, 40)
            x_lab = yf["a"] + yf["b"] * y_lab
            cv2.putText(canvas, f"g={g:+d}",
                        (int(x_lab) - 22, int(y_lab)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"frame {frame_idx}  t={frame_idx/fps:.2f}s  "
                    f"unit={state['unit_px']:.0f}px  "
                    f"yl={len(fits)}  ({t_ms:.0f}ms)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    # Frame 0 viz.
    frame0_u = cv2.undistort(frame0, K, dist) if abs(k1) > 1e-6 else frame0.copy()
    writer.write(render(frame0_u, fits0, init["g_index"], 0, 0.0))

    # Cross-frame loop.
    g_history = [list(state["last_x"].keys())]
    for fi in range(1, n):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        yard_mask, _ = run_unet(frame, UNET, device="mps")
        fits = fit_yardlines_undistorted(yard_mask, intr)
        if len(fits) < 1:
            t_ms = (time.time() - t0) * 1000
            print(f"  frame {fi}: 0 yardlines"); continue
        upd = assign_grid_with_tracker(fits, cy, state)
        t_ms = (time.time() - t0) * 1000

        # Update state for tracked + new yardlines.
        new_last_x = dict(state["last_x"])  # carry forward unseen ones
        for j, g in enumerate(upd["g_index"]):
            new_last_x[int(g)] = float(upd["x_at_center"][j])
        state["last_x"] = new_last_x
        state["unit_px"] = upd["unit_px"]

        g_history.append(list(upd["g_index"]))
        gs_now = sorted(int(g) for g in upd["g_index"])
        print(f"  frame {fi:>3}  t={fi/fps:.2f}s  "
              f"yl={len(fits)}  g={gs_now}  unit={upd['unit_px']:.1f}  "
              f"({t_ms:.0f}ms)")

        frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
        writer.write(render(frame_u, fits, upd["g_index"], fi, t_ms))

    cap.release(); writer.release()
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
