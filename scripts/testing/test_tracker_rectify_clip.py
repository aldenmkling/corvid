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
from src.homography.distortion import undistort_points
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
    YARD_LINE_POSITIONS,
)

UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "unet_line_round2_best.pth")
HASH_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_w18_hash_round1_best.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "tracker_rectify")
YD_PER_PX = 0.1

# Map kind ↔ field y for keypoint smoothing reverse lookup
_KIND_TO_FIELD_Y = {
    "sideline_near": 0.0,
    "sideline_far":  FIELD_WIDTH,
    "near_hash":     HASH_Y_NEAR,
    "far_hash":      HASH_Y_FAR,
}


def _kind_from_field_y(fy: float) -> str:
    if fy < 1.0:
        return "sideline_near"
    if fy > FIELD_WIDTH - 1.0:
        return "sideline_far"
    return "near_hash" if abs(fy - HASH_Y_NEAR) < abs(fy - HASH_Y_FAR) else "far_hash"


def _yard_slot_from_field_x(fx: float) -> int:
    return int(round((fx - 10.0) / 5.0))


def smooth_keypoint_tracks_and_resolve(meta, raw_Hs, window: int, poly: int):
    """Per-keypoint SG smoothing + per-frame H re-solve.

    Each detected keypoint has an offline-stable identity (kind, yard_slot)
    that lets us build a time series of its pixel position across the clip.
    Smoothing each track independently kills:
      - polynomial-coefficient noise on UNet line fits → sideline×yardline
        intersection wiggle
      - W18 hash heatmap variance
      - any per-frame outlier (one-frame bad poly fit)

    Returns (new_pixel_pts_u, new_field_pts, new_Hs) — each list of length
    len(meta). Entries are None where we couldn't form a valid solution.
    """
    from scipy.signal import savgol_filter

    n = len(meta)
    # tracks[(kind, slot)] = dict frame_idx -> (px, py)
    tracks: dict = {}
    for i, m in enumerate(meta):
        if m["pixel_pts_u"] is None or m["field_pts"] is None:
            continue
        for k in range(len(m["pixel_pts_u"])):
            fx, fy = float(m["field_pts"][k][0]), float(m["field_pts"][k][1])
            px, py = float(m["pixel_pts_u"][k][0]), float(m["pixel_pts_u"][k][1])
            key = (_kind_from_field_y(fy), _yard_slot_from_field_x(fx))
            tracks.setdefault(key, {})[i] = (px, py)

    smoothed: dict = {}
    n_tracks_smoothed = 0
    for key, frame_to_xy in tracks.items():
        indices = sorted(frame_to_xy.keys())
        if len(indices) < window:
            # Track too short for SG; pass through raw
            smoothed[key] = {i: frame_to_xy[i] for i in indices}
            continue

        # Build dense series across [i_min, i_max], linearly interp gaps.
        i_min, i_max = indices[0], indices[-1]
        idx_arr = np.array(indices, dtype=np.int64)
        xs = np.array([frame_to_xy[i][0] for i in indices])
        ys = np.array([frame_to_xy[i][1] for i in indices])
        full = np.arange(i_min, i_max + 1)
        dense_x = np.interp(full, idx_arr, xs)
        dense_y = np.interp(full, idx_arr, ys)

        # SG along time
        eff_window = window if (window <= len(dense_x)) else (
            len(dense_x) | 1)  # ensure odd, ≤ length
        eff_poly = min(poly, eff_window - 1)
        sm_x = savgol_filter(dense_x, window_length=eff_window,
                              polyorder=eff_poly, mode="nearest")
        sm_y = savgol_filter(dense_y, window_length=eff_window,
                              polyorder=eff_poly, mode="nearest")

        # Read out smoothed values only at originally-observed frames.
        smoothed[key] = {
            int(i): (float(sm_x[i - i_min]), float(sm_y[i - i_min]))
            for i in indices
        }
        n_tracks_smoothed += 1

    # Per-frame re-solve from smoothed keypoints
    new_pixel = [None] * n
    new_field = [None] * n
    new_Hs = [None] * n
    for i in range(n):
        kpts = []  # (px, py, fx, fy)
        for (kind, slot), frame_to_xy in smoothed.items():
            if i not in frame_to_xy:
                continue
            px, py = frame_to_xy[i]
            kpts.append((px, py,
                         10.0 + slot * 5.0,
                         _KIND_TO_FIELD_Y[kind]))
        if len(kpts) < 4:
            continue
        pxs = np.array([[k[0], k[1]] for k in kpts], dtype=np.float64)
        fxs = np.array([[k[2], k[3]] for k in kpts], dtype=np.float64)
        H, _ = cv2.findHomography(pxs, fxs, method=cv2.RANSAC,
                                   ransacReprojThreshold=1.5)
        if H is None:
            continue
        new_pixel[i] = pxs
        new_field[i] = fxs
        new_Hs[i] = H

    print(f"  keypoint smoothing: {len(tracks)} tracks "
          f"({n_tracks_smoothed} ≥window={window}), "
          f"{sum(1 for h in new_Hs if h is not None)}/{n} frames re-solved")
    return new_pixel, new_field, new_Hs


def show_first_frame(clip_path: str, out_path: str):
    """Render the first frame with HRNet detections labeled by grid_pos."""
    cap = cv2.VideoCapture(clip_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"failed to read {clip_path}")
        return
    tracker = HomographyTracker(UNET_WEIGHTS, HASH_WEIGHTS)
    det = tracker._detect(frame)
    result = det["result"]

    vis = frame.copy()
    # Yardlines get ordered grid_pos from grid_solver_v2 assign_grid_positions
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255), (80, 255, 255),
              (255, 150, 50), (150, 50, 255), (50, 255, 150),
              (200, 200, 200)]
    for yl in result.yardlines:
        if yl.grid_pos is None:
            continue
        color = colors[yl.grid_pos % len(colors)]
        fh, nh = yl.far_hash, yl.near_hash
        ns, fs = yl.near_sideline, yl.far_sideline
        ok_tag = "" if yl.grid_fit_ok else "?"

        if fh is not None and nh is not None:
            cv2.line(vis, tuple(int(x) for x in nh),
                     tuple(int(x) for x in fh), color, 2)
            cv2.drawMarker(vis, tuple(int(x) for x in fh), color,
                           cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(vis, tuple(int(x) for x in nh), color,
                           cv2.MARKER_CROSS, 16, 2)
            cv2.putText(vis, f"g{yl.grid_pos}{ok_tag}",
                        (int(nh[0])+8, int(nh[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            single = fh if fh is not None else nh
            if single is not None:
                cv2.drawMarker(vis, tuple(int(x) for x in single), color,
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, f"g{yl.grid_pos}s{ok_tag}",
                            (int(single[0])+8, int(single[1])+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for s in (ns, fs):
            if s is not None:
                cv2.circle(vis, tuple(int(x) for x in s), 10, color, 2)

    # Legend with NGS anchor reference
    cv2.putText(vis, "Tell me: what NGS x is g0? (NGS: 10=leftGoal, 60=50yd, 110=rightGoal)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "Each g is 5 yd apart in field coords.",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"  saved {out_path}")
    gps = sorted([yl.grid_pos for yl in result.yardlines
                  if yl.grid_pos is not None])
    if gps:
        print(f"  grid_pos range: g{gps[0]} through g{gps[-1]} ({len(set(gps))} distinct)")


def rectify_clip(clip_path: str, anchor: float, output_mp4: str,
                 fps_override: float = None, use_track_bank: bool = True,
                 bank_coast: bool = False,   # default off: see tracker.py note
                 smooth_window: int = 0, smooth_poly: int = 2,
                 keypoint_smooth_window: int = 0, keypoint_smooth_poly: int = 3,
                 device: str = "mps",
                 unet_weights: str = UNET_WEIGHTS,
                 hash_weights: str = HASH_WEIGHTS):
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

    tracker = HomographyTracker(unet_weights, hash_weights, device=device,
                                use_track_bank=use_track_bank,
                                track_bank_coast=bank_coast)

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
        # Cache the keypoints actually used in this frame's homography solve,
        # in UNDISTORTED pixel space (same space as frame_u).
        meta.append({
            "method": result.method,
            "n": result.n_correspondences,
            "err": (result.field_reproj_error_mean
                    if result.field_reproj_error_mean == result.field_reproj_error_mean
                    else None),
            "pixel_pts_u": (result.pixel_pts_u.copy()
                            if result.pixel_pts_u is not None
                            and len(result.pixel_pts_u) > 0 else None),
            "field_pts": (result.field_pts.copy()
                          if result.field_pts is not None
                          and len(result.field_pts) > 0 else None),
        })
    cap.release()
    print(f"  pass 1 done: cached {len(cached_frames)} frames")

    # ── Per-keypoint SG smoothing → re-solve H per frame ──
    # Replaces the SG-on-H block when keypoint_smooth_window > 0. Smooths each
    # (kind, yard_slot) track's pixel position across time, then refits H from
    # smoothed keypoints — kills the polynomial-coefficient + intersection-
    # extrapolation jitter without flexing the projective constraint.
    if keypoint_smooth_window > 0 and len(meta) >= keypoint_smooth_window:
        new_pixel, new_field, new_Hs = smooth_keypoint_tracks_and_resolve(
            meta, raw_Hs, keypoint_smooth_window, keypoint_smooth_poly,
        )
        for i in range(len(meta)):
            if new_Hs[i] is not None:
                raw_Hs[i] = new_Hs[i]
                meta[i]["pixel_pts_u"] = new_pixel[i]
                meta[i]["field_pts"] = new_field[i]
                # Recompute err on smoothed correspondences
                projected = (raw_Hs[i] @ np.hstack([
                    new_pixel[i],
                    np.ones((len(new_pixel[i]), 1)),
                ]).T).T
                projected = projected[:, :2] / projected[:, 2:3]
                meta[i]["err"] = float(np.mean(np.linalg.norm(
                    projected - new_field[i], axis=1)))
                # Smoothed-kpt re-solve effectively makes this a 'full' frame
                if meta[i]["method"] == "carry":
                    meta[i]["method"] = "full"

    # ── Carry imputation ──
    # Carry mode copies H_prev, which is wasteful — we have future frames too.
    # For each carry frame, if it's bracketed by good frames within a small
    # gap, linearly interpolate H. Remaining real-carry frames = unrecoverable
    # (end-of-clip failure).
    CARRY_IMPUTE_MAX_GAP = 8  # frames — interpolate across gaps up to this size

    def _is_good_for_impute(i):
        m = meta[i]
        return m["method"] in ("full", "delta")

    n_imputed = 0
    for i in range(len(meta)):
        if meta[i]["method"] != "carry":
            continue
        j_before = i - 1
        while j_before >= 0 and not _is_good_for_impute(j_before):
            j_before -= 1
        j_after = i + 1
        while j_after < len(meta) and not _is_good_for_impute(j_after):
            j_after += 1
        if j_before < 0 or j_after >= len(meta):
            continue  # no bracket on one side
        if (j_after - j_before) > CARRY_IMPUTE_MAX_GAP:
            continue  # gap too wide to trust interpolation
        alpha = (i - j_before) / float(j_after - j_before)
        # Linear interp on normalized H entries
        hA = raw_Hs[j_before]
        hB = raw_Hs[j_after]
        sa = hA[2, 2] if abs(hA[2, 2]) > 1e-9 else 1.0
        sb = hB[2, 2] if abs(hB[2, 2]) > 1e-9 else 1.0
        h_interp = (1 - alpha) * (hA / sa) + alpha * (hB / sb)
        h_interp = h_interp * ((1 - alpha) * sa + alpha * sb)
        raw_Hs[i] = h_interp
        meta[i]["imputed"] = True
        n_imputed += 1
    if n_imputed:
        print(f"  carry imputation: replaced {n_imputed} frames via neighbor interp")

    # ── Err-based similarity fallback ──
    # For frames where err spikes (i.e. the full-H fit is internally inconsistent
    # and/or badly constrained), REPLACE the raw H with a similarity update from
    # the nearest preceding "good" frame. Similarity = 4 DOF (scale, rotation,
    # translation), can't express the projective slanting that high-err frames
    # usually exhibit.
    errs_arr = np.array([m["err"] if m["err"] is not None else np.nan
                         for m in meta])
    full_errs = errs_arr[[m["method"] == "full" for m in meta]]
    finite_full = full_errs[np.isfinite(full_errs)]
    if len(finite_full) > 5:
        baseline_err = float(np.median(finite_full))
    else:
        baseline_err = 0.12
    bad_threshold = max(0.25, 2.5 * baseline_err)
    print(f"  err baseline={baseline_err:.3f} yd, bad_threshold={bad_threshold:.3f} yd")

    n_replaced = 0
    for i in range(len(meta)):
        e = meta[i]["err"]
        m = meta[i]
        if m["method"] != "full":
            continue
        if e is None or not np.isfinite(e) or e <= bad_threshold:
            continue
        # Find nearest preceding good frame
        j = i - 1
        while j >= 0:
            ej = meta[j]["err"]
            if (meta[j]["method"] == "full" and ej is not None
                    and np.isfinite(ej) and ej <= bad_threshold):
                break
            j -= 1
        if j < 0:
            continue  # no good reference yet; skip
        # Need current frame's correspondences to compute similarity update.
        pix_i = m.get("pixel_pts_u")
        fld_i = m.get("field_pts")
        if pix_i is None or fld_i is None or len(pix_i) < 2:
            continue
        H_ref = raw_Hs[j]
        H_ref_inv = np.linalg.inv(H_ref)
        # Predict where these field points were in the reference frame (pixels)
        prev_pix = []
        for f in fld_i:
            fh = np.array([f[0], f[1], 1.0])
            p = H_ref_inv @ fh
            prev_pix.append([p[0] / p[2], p[1] / p[2]])
        prev_pix = np.array(prev_pix, dtype=np.float64)
        # Fit similarity prev_pix → current pix
        M, _ = cv2.estimateAffinePartial2D(
            prev_pix, pix_i.astype(np.float64), method=cv2.LMEDS,
        )
        if M is None:
            continue
        S_mat = np.vstack([M, [0, 0, 1]])
        raw_Hs[i] = H_ref @ np.linalg.inv(S_mat)
        n_replaced += 1
    if n_replaced:
        print(f"  err-based similarity fallback replaced {n_replaced} frames "
              f"(err > {bad_threshold:.2f} yd)")

    # ── Optional: Savitzky-Golay smoothing over H ──
    # Skipped when keypoint smoothing was applied — smoothing twice over-blurs.
    if keypoint_smooth_window > 0:
        Hs = raw_Hs
    elif smooth_window > 0 and len(raw_Hs) >= smooth_window:
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

    # ── Detect "clip lost" state retrospectively.
    # Once we hit sustained loss (≥ MIN_SUSTAINED_LOSS consecutive real carries),
    # the whole rest of the clip is LOST — even if detections come back later.
    # Reason: H_prev has drifted during the lost window, and downstream grid_pos
    # assignments depend on it, so "recovered" frames are anchored to a
    # compromised reference.
    MIN_SUSTAINED_LOSS = 3

    def _real_carry(m):
        return m["method"] == "carry" and not m.get("imputed")

    is_lost = [False] * len(meta)
    lost_from = None
    consec_carry = 0
    for i, m in enumerate(meta):
        if _real_carry(m):
            consec_carry += 1
        else:
            consec_carry = 0
        if consec_carry >= MIN_SUSTAINED_LOSS:
            lost_from = i - consec_carry + 1
            break

    if lost_from is not None:
        for i in range(lost_from, len(meta)):
            is_lost[i] = True
        print(f"  clip LOST from frame {lost_from} "
              f"({len(meta) - lost_from} frames) — first sustained loss")

    # Freeze H at last good (non-lost) frame and reuse after lost starts.
    frozen_H = None
    if lost_from is not None and lost_from > 0:
        frozen_H = Hs[lost_from - 1].copy()

    # ── Pass 2: render with (possibly smoothed) H ──
    def _kind_color(fy):
        """Color by keypoint kind (y in field coords)."""
        if fy < 1.0:
            return (255, 200, 80)    # sideline_near — cyan
        if fy > FIELD_WIDTH - 1.0:
            return (80, 200, 255)    # sideline_far — orange
        if abs(fy - HASH_Y_FAR) < abs(fy - HASH_Y_NEAR):
            return (80, 255, 80)     # far_hash — green
        return (80, 80, 255)         # near_hash — red

    for i, (frame_u, H_use, m) in enumerate(zip(cached_frames, Hs, meta)):
        # Draw the keypoints used in the homography on the source panel.
        # These are in undistorted pixel space (matches frame_u).
        frame_vis = frame_u.copy()
        if (not is_lost[i] and m.get("pixel_pts_u") is not None
                and m.get("field_pts") is not None):
            for px, py, field_xy in zip(
                m["pixel_pts_u"][:, 0], m["pixel_pts_u"][:, 1], m["field_pts"]
            ):
                color = _kind_color(float(field_xy[1]))
                cv2.drawMarker(
                    frame_vis, (int(px), int(py)), color,
                    cv2.MARKER_CROSS, 14, 2,
                )

        # If clip is lost, freeze H and overlay "LOST" on rectified panel.
        if is_lost[i] and frozen_H is not None:
            H_use = frozen_H
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

        if is_lost[i]:
            # Big red "LOST" overlay across the rectified panel
            overlay = rectified.copy()
            cv2.rectangle(overlay, (0, 0), (field_w, field_h), (0, 0, 120), -1)
            rectified = cv2.addWeighted(overlay, 0.45, rectified, 0.55, 0)
            cv2.putText(rectified, "TRACKING LOST",
                        (field_w // 2 - 180, field_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                        cv2.LINE_AA)

        # Compose side-by-side, vertically centering each panel
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y0_left = (out_h - h) // 2
        canvas[y0_left:y0_left + h, :w] = frame_vis
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
    parser.add_argument("--bank-coast", action="store_true",
                        help="Enable coasted correspondences from the track "
                             "bank. Off by default — coasting can create "
                             "a feedback loop with H_prev-drift.")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="Savitzky-Golay window (odd, e.g. 15) over H "
                             "matrix across frames. 0 = disabled. Skipped "
                             "when --keypoint-smooth-window > 0.")
    parser.add_argument("--smooth-poly", type=int, default=2)
    parser.add_argument("--keypoint-smooth-window", type=int, default=0,
                        help="Per-keypoint SG window (odd, e.g. 31) over each "
                             "(kind, yard_slot) track's pixel position. H is "
                             "re-solved per frame from smoothed keypoints. "
                             "Replaces --smooth-window when set.")
    parser.add_argument("--keypoint-smooth-poly", type=int, default=3)
    parser.add_argument("--device", default="mps",
                        choices=["cpu", "cuda", "mps"],
                        help="Torch device for HRNet inference.")
    parser.add_argument("--unet-weights", default=UNET_WEIGHTS,
                        help="Path to UNet line-detection weights")
    parser.add_argument("--hash-weights", default=HASH_WEIGHTS,
                        help="Path to W18 hash-detection weights")
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
        if args.keypoint_smooth_window > 0:
            suffix += f"_kpsg{args.keypoint_smooth_window}"
        elif args.smooth_window > 0:
            suffix += f"_sg{args.smooth_window}"
        out = args.output or os.path.join(
            OUTPUT_DIR, f"{tag}_rectified{suffix}.mp4")
        rectify_clip(args.clip, args.anchor, out,
                     use_track_bank=not args.no_track_bank,
                     bank_coast=args.bank_coast,
                     smooth_window=args.smooth_window,
                     smooth_poly=args.smooth_poly,
                     keypoint_smooth_window=args.keypoint_smooth_window,
                     keypoint_smooth_poly=args.keypoint_smooth_poly,
                     device=args.device,
                     unet_weights=args.unet_weights,
                     hash_weights=args.hash_weights)


if __name__ == "__main__":
    main()
