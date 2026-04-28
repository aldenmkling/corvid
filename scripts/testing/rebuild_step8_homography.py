#!/usr/bin/env python3
"""Step 8: per-frame homography fit using RANSAC, no smoothing.

Builds on the full-clip viz pipeline (UNet + CC + line fits + grid + tracker).
Per frame, builds a unified correspondence list:
  (px_x, px_y, x_ngs, y_field, label)
where:
  - hash points: (px, (G0_NGS_X + 5·g, HASH_Y_NEAR or HASH_Y_FAR))
  - sideline×yardline: (px, (G0_NGS_X + 5·g, 0 if near else FIELD_WIDTH))

Solves H via cv2.findHomography RANSAC. Renders:
  - undistorted frame + field grid (every 5 yd, both sidelines, hash rows)
    projected back into image coords. If H is right, projected grid lines
    sit on top of the visible markings.
  - per-frame log of #correspondences, RANSAC inliers, reprojection RMSE.
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Pull all the heavy lifting from the full-clip viz module.
from scripts.testing.rebuild_full_clip_viz import (
    YardlineTracker, SidelineTracker, yardline_in_bounds,
    fit_yardline_linear, fit_sideline_linear, total_mse,
    hashes_with_roles, sideline_yardline_intersections,
    color_for_g, SIDELINE_COLOR,
    G_MIN, G_MAX, G0_NGS_X, YD_PER_GRID, EXTRAP_FRAC,
    UNET, HASH,
)
from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH
from src.homography.keypoint_track_bank import KeypointTrackBank
from src.homography.apply_homography import pixel_to_field, field_to_pixel

# Use the CC-based sideline grouper from the full-clip viz module.
from scripts.testing.rebuild_full_clip_viz import (
    group_sideline_pixels as cc_group_sideline,
)

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step_full_homography.mp4")

RANSAC_REPROJ_PX = 4.0
MIN_CORRS_FOR_H = 4    # cv2.findHomography minimum; 4 non-collinear pts → H
FULL_H_SANITY_MAX_YD = 2.0
DELTA_MAX_SCALE = 1.8
DELTA_MAX_ROT_DEG = 3.0
DELTA_MAX_TRANS_PX = 400
# H_prev-based pre-filter: drop correspondences whose pixel projects (via
# H_prev) to a field position more than this far from their claimed
# canonical field position. Catches false hashes + mis-classified hashes
# before they get into RANSAC. Generous enough to absorb 1 frame of camera
# motion at 30 fps.
PREFILTER_TOL_YD = 1.0


def build_correspondences(yl_fits, g_index, hashes, intersections,
                           g0_ngs_x: float = G0_NGS_X):
    """Return list of correspondence dicts:
        {pixel_u, field, kind, label}.
    Compatible with KeypointTrackBank's observe/validate_h.
    """
    corrs = []
    for (hx, hy, role, _conf, yli) in hashes:
        g = int(g_index[yli])
        x_ngs = g0_ngs_x + YD_PER_GRID * g
        y_f = HASH_Y_FAR if role == "far" else HASH_Y_NEAR
        kind = "far_hash" if role == "far" else "near_hash"
        corrs.append({
            "pixel_u": np.array([hx, hy], dtype=np.float64),
            "field": np.array([x_ngs, y_f], dtype=np.float64),
            "kind": kind,
            "label": f"{role}_hash@g{g:+d}",
        })
    for (x, y, yi, sl_label) in intersections:
        g = int(g_index[yi])
        x_ngs = g0_ngs_x + YD_PER_GRID * g
        y_f = 0.0 if sl_label == "near" else FIELD_WIDTH
        kind = f"sideline_{sl_label}"
        corrs.append({
            "pixel_u": np.array([x, y], dtype=np.float64),
            "field": np.array([x_ngs, y_f], dtype=np.float64),
            "kind": kind,
            "label": f"{sl_label}sl×g{g:+d}",
        })
    return corrs


def smooth_hs(Hs_raw, window: int = 7, poly: int = 2):
    """Savitzky-Golay smooth a list of 3×3 H matrices. Normalizes h[2,2]=1
    before smoothing, re-applies original scale after."""
    from scipy.signal import savgol_filter
    if not Hs_raw or window < 3 or len(Hs_raw) < window:
        return list(Hs_raw)
    H_flat = np.stack([h.flatten() for h in Hs_raw], axis=0)   # (N, 9)
    scales = H_flat[:, 8:9].copy()
    scales[np.abs(scales) < 1e-9] = 1.0
    H_flat_n = H_flat / scales
    sm = savgol_filter(H_flat_n, window_length=window,
                       polyorder=min(poly, window - 1),
                       axis=0, mode="nearest")
    sm = sm * scales
    return [sm[i].reshape(3, 3) for i in range(len(Hs_raw))]


def detect_lost(methods, min_sustained_loss: int = 3):
    """Return the index of the first frame in the FIRST sustained-carry run,
    or None if no such run exists. Once `min_sustained_loss` consecutive
    'carry' methods occur, the clip is treated as LOST from the run's start."""
    consec = 0
    for i, m in enumerate(methods):
        if m == "carry":
            consec += 1
            if consec >= min_sustained_loss:
                return i - consec + 1
        else:
            consec = 0
    return None


def solve_h(corrs):
    """RANSAC homography from undistorted-pixel → NGS field coords.
    Returns (H, inlier_mask, rmse_yd) or (None, None, None) on failure."""
    if len(corrs) < MIN_CORRS_FOR_H:
        return None, None, None
    src = np.array([c["pixel_u"] for c in corrs], dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array([c["field"] for c in corrs], dtype=np.float64).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=RANSAC_REPROJ_PX)
    if H is None:
        return None, None, None
    inlier = mask.ravel().astype(bool)
    src_h = np.column_stack([src.reshape(-1, 2), np.ones(len(corrs))])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj_xy - dst.reshape(-1, 2), axis=1)
    rmse_field = float(np.sqrt(np.mean(err[inlier] ** 2))) if inlier.any() else None
    return H, inlier, rmse_field


# ── Full / delta / carry tracker ────────────────────────────────────────

class HomographyTrackerLite:
    """Per-frame homography with full / delta / carry fallback + a track
    bank that validates candidate H matrices.

    full   = RANSAC on this frame's correspondences (preferred).
    delta  = similarity transform from prev-frame pixel positions of the
              same field points to this-frame pixel positions; used when
              full is unavailable or rejected.
    carry  = use H_prev unchanged when nothing better is available.

    No coasting: the bank only validates; it doesn't synthesize fake
    correspondences (per the comment in src/homography/tracker.py, coasting
    introduces a feedback loop).
    """

    def __init__(self):
        # Tighter bank thresholds than the defaults: catch single-frame
        # identity flips (was tol=1.0, bad_frac=0.34, min_obs=2).
        self.bank = KeypointTrackBank(
            h_validate_tol_yd=0.7,
            h_validate_bad_frac=0.20,
            min_obs_for_trust=3,
        )
        self.H_prev = None
        self.H_inv_prev = None

    def _solve_delta(self, corrs):
        """Similarity transform: prev pixel positions of these field points
        → current pixel positions. H_cur = H_prev @ inv(S)."""
        if self.H_prev is None or len(corrs) < 2:
            return None
        cur_px = np.array([c["pixel_u"] for c in corrs], dtype=np.float64)
        field = np.array([c["field"] for c in corrs], dtype=np.float64)
        prev_px = field_to_pixel(field, self.H_inv_prev)
        M, _ = cv2.estimateAffinePartial2D(
            prev_px, cur_px, method=cv2.LMEDS,
        )
        if M is None:
            return None
        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        if (scale > DELTA_MAX_SCALE or scale < 1.0 / DELTA_MAX_SCALE):
            return None
        if abs(rot_deg) > DELTA_MAX_ROT_DEG:
            return None
        if np.hypot(tx, ty) > DELTA_MAX_TRANS_PX:
            return None
        S = np.vstack([M, [0.0, 0.0, 1.0]])
        H = self.H_prev @ np.linalg.inv(S)
        H_inv = np.linalg.inv(H)
        return H, H_inv, {"scale": scale, "rot_deg": rot_deg,
                          "trans_px": (tx, ty)}

    def update(self, corrs, frame_idx):
        """Returns dict with H, method, n_corrs, n_inliers, rmse_yd, info."""
        n_in = len(corrs)
        # Pre-filter against H_prev: drop corrs whose pixel doesn't project
        # to its claimed canonical field position. Catches false hashes
        # and mis-classified hashes before they pollute the H solve.
        if self.H_prev is not None and corrs:
            kept = []
            n_dropped = 0
            for c in corrs:
                px = c["pixel_u"]
                f_via_prev = pixel_to_field(
                    px.reshape(1, 2), self.H_prev,
                )[0]
                if np.linalg.norm(f_via_prev - c["field"]) <= PREFILTER_TOL_YD:
                    kept.append(c)
                else:
                    n_dropped += 1
            corrs = kept
            if n_dropped:
                # store for diagnostics
                pass
        H, inl, rmse = solve_h(corrs) if corrs else (None, None, None)
        method = None
        info = {"n_input_corrs": n_in, "n_after_prefilter": len(corrs)}

        # Full path (with sanity + bank validation).
        if H is not None:
            ok = True
            if self.H_prev is not None:
                cur_px = np.array([c["pixel_u"] for c in corrs], dtype=np.float64)
                f_new = pixel_to_field(cur_px, H)
                f_prev = pixel_to_field(cur_px, self.H_prev)
                divergence = float(np.mean(np.linalg.norm(f_new - f_prev, axis=1)))
                info["divergence_yd"] = divergence
                if divergence > FULL_H_SANITY_MAX_YD:
                    ok = False
            if ok and self.H_prev is not None:
                valid, diag = self.bank.validate_h(H, corrs, frame_idx=frame_idx)
                info["bank"] = diag
                if not valid:
                    ok = False
            if ok:
                method = "full"

        # Delta fallback.
        if method is None and self.H_prev is not None and corrs:
            d = self._solve_delta(corrs)
            if d is not None:
                H, H_inv, dinfo = d
                method = "delta"
                info["delta"] = dinfo
                inl = None; rmse = None

        # Carry fallback.
        if method is None and self.H_prev is not None:
            H = self.H_prev; H_inv = self.H_inv_prev
            method = "carry"; inl = None; rmse = None

        # First frame, nothing to fall back to.
        if method is None:
            return {"H": None, "method": "none", "n_corrs": len(corrs),
                    "n_inliers": 0, "rmse_yd": None, "info": info}

        if method == "full":
            H_inv = np.linalg.inv(H)

        # Update state. Only observe on real-H frames (full or delta).
        if method in ("full", "delta"):
            self.bank.observe(corrs, frame_idx=frame_idx)
            self.bank.prune(frame_idx=frame_idx)
        self.H_prev = H
        self.H_inv_prev = H_inv

        return {
            "H": H, "H_inv": H_inv, "method": method,
            "n_corrs": len(corrs),
            "n_inliers": int(inl.sum()) if inl is not None else None,
            "rmse_yd": rmse,
            "info": info,
        }


def field_grid_lines():
    """Generate field-coordinate grid lines (in yards). Returns list of
    (label, points (N, 2))."""
    lines = []
    # Yardlines every 5 yards.
    for x_ngs in range(10, 111, 5):
        ys = np.linspace(0, FIELD_WIDTH, 50)
        xs = np.full_like(ys, float(x_ngs))
        lines.append((f"yl{x_ngs}", np.column_stack([xs, ys])))
    # Sidelines.
    xs = np.linspace(10, 110, 100)
    lines.append(("sl_near", np.column_stack([xs, np.zeros_like(xs)])))
    lines.append(("sl_far",
                   np.column_stack([xs, np.full_like(xs, FIELD_WIDTH)])))
    # Hash rows.
    lines.append(("hash_near",
                   np.column_stack([xs, np.full_like(xs, HASH_Y_NEAR)])))
    lines.append(("hash_far",
                   np.column_stack([xs, np.full_like(xs, HASH_Y_FAR)])))
    return lines


def project_field_to_image(H_pixel_to_field, field_pts):
    """field_pts: (N, 2). Returns (N, 2) image pixel coords."""
    H_inv = np.linalg.inv(H_pixel_to_field)
    field_h = np.column_stack([field_pts, np.ones(len(field_pts))])
    img_h = (H_inv @ field_h.T).T
    img_xy = img_h[:, :2] / img_h[:, 2:3]
    return img_xy


def render_h_overlay(frame_u, H, inliers, corrs, rmse, frame_idx, fps, w, h):
    canvas = frame_u.copy()
    if H is None:
        cv2.putText(canvas, f"frame {frame_idx} t={frame_idx/fps:.2f}s  "
                    f"NO H ({len(corrs)} corrs)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    # Project field grid into image and draw.
    for label, pts in field_grid_lines():
        img_pts = project_field_to_image(H, pts)
        # Clip to image bounds for clean drawing.
        in_view = ((img_pts[:, 0] > -50) & (img_pts[:, 0] < w + 50)
                   & (img_pts[:, 1] > -50) & (img_pts[:, 1] < h + 50))
        if not in_view.any():
            continue
        valid = img_pts[in_view].astype(np.int32)
        if label.startswith("yl"):
            color = (200, 200, 200)
            thickness = 1
        elif label.startswith("sl"):
            color = SIDELINE_COLOR
            thickness = 2
        else:   # hash row
            color = (120, 200, 120)
            thickness = 1
        cv2.polylines(canvas, [valid], False, color, thickness, cv2.LINE_AA)

    # Inlier vs outlier markers.
    if inliers is not None:
        for k, (px, _, _) in enumerate(corrs):
            color = (0, 255, 0) if inliers[k] else (0, 60, 255)
            cv2.circle(canvas, (int(px[0]), int(px[1])), 4, color, -1)

    n_in = int(inliers.sum()) if inliers is not None else 0
    cv2.putText(canvas,
                f"frame {frame_idx}  t={frame_idx/fps:.2f}s  "
                f"corrs={len(corrs)} (inliers {n_in})  "
                f"reproj RMSE={rmse:.2f}yd" if rmse is not None
                else f"frame {frame_idx}  no inliers",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                "gray=yardlines  yellow=sidelines  green=hash rows  "
                "(via projected H)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n_total} frames @ {fps:.1f}fps")

    ok, frame0 = cap.read()
    if not ok: print("read failed"); return
    h, w = frame0.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    # Frame 0: k1 calibration.
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
    print(f"  k1 = {k1:+.4f}")

    sl_tracker = SidelineTracker(frame_w=w)
    yl_tracker = YardlineTracker(g_min=G_MIN, g_max=G_MAX, frame_h=h)

    sl_fits0_all = [fit_sideline_linear(g.pixels, intr) for g in sl_g0]
    sl_labels0 = sl_tracker.init_from(sl_fits0_all)

    yl_pix_u_0 = [undistort_points(g.pixels.astype(np.float64), intr)
                  for g in yl_g0]
    yl_fits0_all = [fit_yardline_linear(g.pixels, intr) for g in yl_g0]
    keep_oob_0 = [yardline_in_bounds(p, sl_labels0) for p in yl_pix_u_0]
    yl_fits0_oob = [yl_fits0_all[i] for i in range(len(yl_fits0_all)) if keep_oob_0[i]]
    res0 = yl_tracker.init_from(yl_fits0_oob, cy)
    if res0 is None:
        print("frame 0 init failed"); return
    yl_fits0, g_index0, _ = res0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))

    frame0_u = cv2.undistort(frame0, K, dist) if abs(k1) > 1e-6 else frame0.copy()
    hashes0 = hashes_with_roles(frame0, intr, yl_fits0, w)
    inters0 = sideline_yardline_intersections(yl_fits0, sl_labels0, w, h)
    corrs0 = build_correspondences(yl_fits0, g_index0, hashes0, inters0)
    H0, inl0, rmse0 = solve_h(corrs0)
    writer.write(render_h_overlay(frame0_u, H0, inl0, corrs0, rmse0,
                                    0, fps, w, h))
    print(f"  frame 0: corrs={len(corrs0)}  inliers="
          f"{int(inl0.sum()) if inl0 is not None else 0}  "
          f"rmse={rmse0:.3f}yd" if rmse0 is not None else "  frame 0: no H")

    rmse_history = [rmse0] if rmse0 is not None else []

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
            blank = (cv2.undistort(frame, K, dist)
                     if abs(k1) > 1e-6 else frame.copy())
            cv2.putText(blank, f"frame {fi}: no yardlines", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            writer.write(blank); continue

        yl_pix_u = [undistort_points(g.pixels.astype(np.float64), intr)
                    for g in yl_g_raw]
        keep_oob = [yardline_in_bounds(p, sl_labels) for p in yl_pix_u]
        yl_g_oob = [yl_g_raw[i] for i in range(len(yl_g_raw)) if keep_oob[i]]
        yl_fits_oob = [fit_yardline_linear(g.pixels, intr) for g in yl_g_oob]

        yl_fits, g_index, _, n_rej = yl_tracker.update(yl_fits_oob, cy)
        hashes = hashes_with_roles(frame, intr, yl_fits, w)
        inters = sideline_yardline_intersections(yl_fits, sl_labels, w, h)
        corrs = build_correspondences(yl_fits, g_index, hashes, inters)
        H, inl, rmse = solve_h(corrs)
        if rmse is not None:
            rmse_history.append(rmse)

        frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
        writer.write(render_h_overlay(frame_u, H, inl, corrs, rmse,
                                        fi, fps, w, h))

        if fi % 10 == 0 or fi < 5:
            t_ms = (time.time() - t0) * 1000
            n_in = int(inl.sum()) if inl is not None else 0
            rmse_s = f"{rmse:.3f}yd" if rmse is not None else "—"
            print(f"  frame {fi:>3}  corrs={len(corrs)} (inl {n_in})  "
                  f"rmse={rmse_s}  ({t_ms:.0f}ms)")

    cap.release(); writer.release()
    if rmse_history:
        rms = np.array(rmse_history)
        print(f"\n  reproj RMSE: median={np.median(rms):.3f}yd  "
              f"mean={rms.mean():.3f}yd  "
              f"max={rms.max():.3f}yd  ({len(rms)} frames with H)")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
