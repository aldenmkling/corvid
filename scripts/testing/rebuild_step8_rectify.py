#!/usr/bin/env python3
"""Step 8 (rectify variant): per-frame warp the undistorted frame into the
field plane using H, producing a top-down 'NGS rectangle' view.

Output canvas: SCALE px / yd × NGS field [0, 120] × [0, 53.333]
             = 120 · SCALE wide × 53 · SCALE tall.

Side-by-side video: left = undistorted source w/ projected grid overlay,
right = rectified field view (top-down).
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.testing.rebuild_full_clip_viz import (
    YardlineTracker, SidelineTracker, HashClassifier,
    fit_yardline_linear, fit_sideline_linear, total_mse,
    hashes_with_roles, sideline_yardline_intersections,
    render_fits,
    G_MIN, G_MAX, G0_NGS_X, YD_PER_GRID,
    UNET, HASH,
)
from scripts.testing.rebuild_full_clip_viz import (
    group_sideline_pixels as cc_group_sideline,
)
from scripts.testing.rebuild_step8_homography import (
    build_correspondences, solve_h, HomographyTrackerLite,
    smooth_hs, detect_lost,
)
from src.homography.grid_solver_v2 import run_unet, group_yardline_pixels_cc
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH

SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
MIN_SUSTAINED_LOSS = 3

# Field canvas: 120 yd × 53.333 yd at 10 px/yd → 1200 × 533.
SCALE = 10                                       # px per yard
FIELD_LEN_YD = 120                               # NGS x: 0–120
FIELD_W_YD = FIELD_WIDTH                         # 53.333
FIELD_W_PX = int(round(FIELD_LEN_YD * SCALE))    # 1200
FIELD_H_PX = int(round(FIELD_W_YD * SCALE))      # 533


def rectify_frame(frame_u: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp undistorted source into NGS-rectangle coords at SCALE px/yd.

    Output orientation: y_field=0 at the BOTTOM (near sideline), y_field=
    FIELD_W at the TOP (far sideline). This matches the natural camera view
    where the far sideline is at the top of the frame.
    """
    if H is None:
        return np.zeros((FIELD_H_PX, FIELD_W_PX, 3), dtype=np.uint8)
    # Flip y so y_field=0 ends up at the bottom row, not the top.
    S = np.array([[SCALE, 0, 0],
                   [0, -SCALE, FIELD_H_PX],
                   [0, 0, 1]], dtype=np.float64)
    M = S @ H
    rect = cv2.warpPerspective(frame_u, M, (FIELD_W_PX, FIELD_H_PX))
    # Draw an authoritative field grid on top so we can see alignment:
    # yardlines every 5 yd, sidelines, hash rows.
    for x_yd in range(0, FIELD_LEN_YD + 1, 5):
        x_px = int(round(x_yd * SCALE))
        color = (220, 220, 220) if (x_yd % 10 == 0) else (140, 140, 140)
        cv2.line(rect, (x_px, 0), (x_px, FIELD_H_PX - 1), color, 1)
        if x_yd % 10 == 0 and 10 <= x_yd <= 110:
            cv2.putText(rect, str(x_yd),
                        (x_px + 3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # Hash rows (y_field flipped: row = FIELD_H_PX − y_yd*SCALE).
    for y_yd in (HASH_Y_NEAR, HASH_Y_FAR):
        y_px = int(round(FIELD_H_PX - y_yd * SCALE))
        cv2.line(rect, (0, y_px), (FIELD_W_PX - 1, y_px),
                 (90, 200, 90), 1, cv2.LINE_AA)
    # Sidelines.
    for y_yd in (0, FIELD_W_YD):
        y_px = int(round(FIELD_H_PX - y_yd * SCALE))
        cv2.line(rect, (0, max(0, min(FIELD_H_PX - 1, y_px))),
                 (FIELD_W_PX - 1, max(0, min(FIELD_H_PX - 1, y_px))),
                 (220, 220, 0), 2, cv2.LINE_AA)
    return rect


def run_clip(clip_path: str, g0_ngs_x: float, out_path: str,
             smooth_window: int = SMOOTH_WINDOW,
             smooth_poly: int = SMOOTH_POLY,
             min_sustained_loss: int = MIN_SUSTAINED_LOSS):
    """End-to-end pipeline on a single clip with two-pass smoothing + lost
    detection. Pass 1 collects raw H, methods, and per-frame metadata; we
    SG-smooth the H trajectory and detect the first sustained-carry run.
    Pass 2 re-reads the clip and renders with the smoothed H (frozen at
    last good frame for the lost tail)."""

    print(f"\n══ {os.path.basename(os.path.dirname(clip_path))} "
          f"(g0 = NGS x={g0_ngs_x}) ══")

    g_min = int((10.0 - g0_ngs_x) / YD_PER_GRID)
    g_max = int((110.0 - g0_ngs_x) / YD_PER_GRID)

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n_total} frames @ {fps:.1f}fps")

    ok, frame0 = cap.read()
    if not ok: return
    h, w = frame0.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask = run_unet(frame0, UNET, device="mps")
    yl_g0 = group_yardline_pixels_cc(yard_mask)
    sl_g0 = cc_group_sideline(side_mask)
    line_pts = [g.pixels for g in yl_g0] + [g.pixels for g in sl_g0]
    line_kinds = ["yardline"] * len(yl_g0) + ["sideline"] * len(sl_g0)
    sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    print(f"  k1 = {k1:+.4f}  g_min={g_min}  g_max={g_max}")

    sl_tracker = SidelineTracker(frame_w=w)
    yl_tracker = YardlineTracker(g_min=g_min, g_max=g_max, frame_h=h)
    hash_clf = HashClassifier()
    h_tracker = HomographyTrackerLite()

    sl_fits0_all = [fit_sideline_linear(g.pixels, intr) for g in sl_g0]
    sl_labels0 = sl_tracker.init_from(sl_fits0_all)
    yl_fits0_all = [fit_yardline_linear(g.pixels, intr) for g in yl_g0]
    bs0 = yl_tracker.init_from(yl_fits0_all, cy)
    if bs0 is None: print("frame 0 init failed"); return
    yl_fits0, g_index0, _ = bs0

    # ── PASS 1: process every frame, collect metadata + raw H ──
    frame_meta = []   # list of dicts with everything we need to render

    # Frame 0 — bootstrap, no H_prev available, falls to PCA-Otsu.
    hashes0 = hashes_with_roles(frame0, intr, yl_fits0, w, 0, hash_clf,
                                  H_prev=None)
    inters0 = sideline_yardline_intersections(yl_fits0, sl_labels0, w, h)
    corrs0 = build_correspondences(yl_fits0, g_index0, hashes0, inters0,
                                    g0_ngs_x=g0_ngs_x)
    r0 = h_tracker.update(corrs0, frame_idx=0)
    frame_meta.append({
        "yl_fits": yl_fits0, "sl_labels": sl_labels0,
        "g_index": g_index0, "hashes": hashes0, "inters": inters0,
        "H": r0["H"], "method": r0["method"],
        "n_corrs": r0["n_corrs"], "n_inliers": r0["n_inliers"],
        "rmse_yd": r0["rmse_yd"],
    })

    method_counts = {"full": 0, "delta": 0, "carry": 0, "none": 0}
    method_counts[r0["method"]] += 1

    for fi in range(1, n_total):
        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()
        yard_mask, side_mask = run_unet(frame, UNET, device="mps")
        yl_g_raw = group_yardline_pixels_cc(yard_mask)
        sl_g = cc_group_sideline(side_mask)
        sl_fits_all = [fit_sideline_linear(g.pixels, intr) for g in sl_g]
        sl_labels = sl_tracker.update(sl_fits_all)
        if not yl_g_raw:
            r = h_tracker.update([], frame_idx=fi)
            method_counts[r["method"]] += 1
            frame_meta.append({
                "yl_fits": [], "sl_labels": sl_labels, "g_index": np.zeros(0, dtype=int),
                "hashes": [], "inters": [],
                "H": r["H"], "method": r["method"],
                "n_corrs": 0, "n_inliers": None, "rmse_yd": None,
            })
            continue

        yl_fits_all = [fit_yardline_linear(g.pixels, intr) for g in yl_g_raw]
        yl_fits, g_index, _, _ = yl_tracker.update(yl_fits_all, cy)
        # Use the established H_prev to classify hashes (more robust than
        # PCA-Otsu and rejects outliers automatically).
        hashes = hashes_with_roles(frame, intr, yl_fits, w, fi, hash_clf,
                                     H_prev=h_tracker.H_prev)
        inters = sideline_yardline_intersections(yl_fits, sl_labels, w, h)
        corrs = build_correspondences(yl_fits, g_index, hashes, inters,
                                       g0_ngs_x=g0_ngs_x)
        r = h_tracker.update(corrs, frame_idx=fi)
        method_counts[r["method"]] += 1
        frame_meta.append({
            "yl_fits": yl_fits, "sl_labels": sl_labels, "g_index": g_index,
            "hashes": hashes, "inters": inters,
            "H": r["H"], "method": r["method"],
            "n_corrs": r["n_corrs"], "n_inliers": r["n_inliers"],
            "rmse_yd": r["rmse_yd"],
        })
        if fi % 30 == 0:
            t_ms = (time.time() - t0) * 1000
            rmse_s = f"{r['rmse_yd']:.3f}yd" if r["rmse_yd"] is not None else "—"
            print(f"  pass1 frame {fi:>3}  {r['method']:5s}  "
                  f"corrs={r['n_corrs']}  rmse={rmse_s}  ({t_ms:.0f}ms)")
    cap.release()

    # ── Detect lost frames: first sustained-carry run ──
    methods = [m["method"] for m in frame_meta]
    lost_from = detect_lost(methods, min_sustained_loss=min_sustained_loss)
    valid_until = lost_from if lost_from is not None else len(frame_meta)
    if lost_from is not None:
        print(f"  clip LOST from frame {lost_from} "
              f"({len(frame_meta) - lost_from} frames; "
              f"≥{min_sustained_loss} consecutive carries)")

    # ── Smooth Hs over the valid (non-lost) range ──
    Hs_raw = [m["H"] for m in frame_meta[:valid_until]
              if m["H"] is not None]
    if len(Hs_raw) >= smooth_window:
        Hs_smoothed = smooth_hs(Hs_raw, window=smooth_window, poly=smooth_poly)
        print(f"  applied SG smoothing (window={smooth_window}, "
              f"poly={smooth_poly}) on {len(Hs_smoothed)} frames")
    else:
        Hs_smoothed = Hs_raw
        print(f"  too few full-H frames ({len(Hs_raw)}) for window={smooth_window};"
              " skipping smooth")

    # Map smoothed Hs back into frame_meta. valid_until frames in order;
    # entries with H=None retain None.
    si = 0
    for i in range(valid_until):
        if frame_meta[i]["H"] is not None and si < len(Hs_smoothed):
            frame_meta[i]["H_smoothed"] = Hs_smoothed[si]
            si += 1
        else:
            frame_meta[i]["H_smoothed"] = frame_meta[i]["H"]
    frozen_H = (frame_meta[valid_until - 1]["H_smoothed"]
                if valid_until > 0 and frame_meta[valid_until - 1]["H_smoothed"] is not None
                else None)
    for i in range(valid_until, len(frame_meta)):
        frame_meta[i]["H_smoothed"] = frozen_H
        frame_meta[i]["lost"] = True

    # ── PASS 2: render ──
    rect_aspect = FIELD_W_PX / FIELD_H_PX
    right_h = h; right_w = int(round(rect_aspect * right_h))
    out_w = w + right_w; out_h = h

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    def compose(left, right_rect, frame_idx, rmse, n_corrs, n_in, method,
                lost):
        right = cv2.resize(right_rect, (right_w, right_h))
        if lost:
            method_label = "LOST"
            method_color = (60, 60, 60)
        else:
            method_label = method
            method_color = {"full": (100, 255, 100), "delta": (0, 200, 255),
                            "carry": (60, 60, 255),
                            "none": (60, 60, 60)}.get(method, (200, 200, 200))
        cv2.putText(right, f"{method_label}  {SCALE} px/yd  "
                    f"NGS [0,120]×[0,{FIELD_W_YD:.1f}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    method_color, 1, cv2.LINE_AA)
        rmse_s = f"{rmse:.3f}yd" if rmse is not None else "—"
        cv2.putText(right, f"frame {frame_idx}  corrs={n_corrs}  "
                    f"inliers={n_in}  rmse={rmse_s}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.hstack([left, right])

    cap = cv2.VideoCapture(clip_path)
    for fi in range(len(frame_meta)):
        ok, frame = cap.read()
        if not ok: break
        m = frame_meta[fi]
        frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
        is_lost = m.get("lost", False)
        if m["yl_fits"]:
            left = render_fits(frame_u, m["yl_fits"],
                                list(m["sl_labels"].values()),
                                m["g_index"], m["hashes"], m["inters"],
                                fi, fps, w, h)
        else:
            left = frame_u.copy()
            cv2.putText(left, f"frame {fi}: no yardlines",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        H_use = m["H_smoothed"] if not is_lost else None
        rect = rectify_frame(frame_u, H_use)
        n_in = m["n_inliers"] if m["n_inliers"] is not None else 0
        writer.write(compose(left, rect, fi, m["rmse_yd"],
                              m["n_corrs"], n_in, m["method"], is_lost))
    cap.release(); writer.release()

    rmses = [m["rmse_yd"] for m in frame_meta[:valid_until]
             if m["rmse_yd"] is not None]
    if rmses:
        rms = np.array(rmses)
        print(f"  reproj RMSE (full only): median={np.median(rms):.3f}yd  "
              f"max={rms.max():.3f}yd  ({len(rms)} full-H frames)")
    total = sum(method_counts.values())
    print(f"  method counts (of {total}): full={method_counts['full']}  "
          f"delta={method_counts['delta']}  carry={method_counts['carry']}  "
          f"none={method_counts['none']}  "
          f"lost_tail={len(frame_meta) - valid_until}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    clips_dir = os.path.join(PROJECT_ROOT, "videos/clips")
    out_dir = os.path.join(PROJECT_ROOT, "output/rebuild")

    # Permissive midfield default for clips with unknown anchor: G0=50 →
    # g_min=-8, g_max=+12. Absolute NGS x in the rectified view won't be
    # calibrated but tracking quality and continuity are what we're checking.
    test_clips = [
        ("2024111001/play_060", 50.0),    # had hash near/far flip
    ]
    for relpath, g0 in test_clips:
        clip = os.path.join(clips_dir, relpath, "sideline.mp4")
        if not os.path.exists(clip):
            print(f"  skip — missing {clip}")
            continue
        out_name = "step_smoketest_" + relpath.replace("/", "_") + ".mp4"
        run_clip(
            clip_path=clip, g0_ngs_x=g0,
            out_path=os.path.join(out_dir, out_name),
        )
