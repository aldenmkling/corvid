#!/usr/bin/env python3
"""Step 6 of the rebuild: yardline grid + identity assignment within a frame.

Premise (vs old grid solver): yardline detections are reliable enough to be
the carrier of identity. Within a frame:
  1. Sort yardlines left-to-right by their x at the image-center y line
     (in undistorted space).
  2. Compute median Δx between adjacent yardlines = unit spacing
     (one 5-yard increment in pixels).
  3. Snap each yardline to the nearest integer multiple of unit spacing,
     measured from the leftmost yardline. The leftmost gets g = 0;
     all others get g > 0. This handles a single missed yardline
     gracefully — the remaining ones still get the right indices.
  4. Each hash inherits g from its assigned yardline + role (near/far).
  5. Each sideline×yardline intersection inherits g + sideline_id
     (broadcast sideline cameras: visible sideline = "far_sideline").

Output: list of correspondences (px_x, px_y, x_field, y_field, label).
Field coords are RELATIVE for now (g=0 at leftmost, no absolute anchor).
Add absolute anchoring in a separate clip-prelude step later (numeral OCR
or goal-line detection).
"""

import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, group_sideline_pixels, run_hash_w18,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
HASH = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step6_grid.jpg")

YD_PER_GRID = 5.0
MAX_DIST_PX = 12.0
HASH_CONF_FLOOR = 0.30
HASH_CONF_REL = 0.60
EXTRAP_FRAC = 0.20
MAD_K = 3.0
MIN_CLUSTER_GAP_PX = 30.0
PCA_S0_S1_MIN = 3.0
SIDELINE_ID = "far_sideline"          # broadcast sideline cam convention
SIDELINE_Y_FIELD = FIELD_WIDTH        # 53.333 yd


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


def otsu_split_1d(values):
    s = np.sort(values); N = len(s)
    best_score = -np.inf; best_t = float(s.mean())
    for j in range(1, N):
        mu1 = s[:j].mean(); mu2 = s[j:].mean()
        score = (j * (N - j) / N) * (mu1 - mu2) ** 2
        if score > best_score:
            best_score = score
            best_t = 0.5 * (s[j-1] + s[j])
    return best_t, float(best_score)


def fit_row_line(pts):
    if len(pts) == 1:
        return 0.0, float(pts[0, 1])
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return float(m), float(c)


def perp_dist(pts, m, c):
    return np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(1.0 + m * m)


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
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
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()

    print(f"  {len(yl_groups)} yardlines + {len(sl_groups)} sidelines, "
          f"k1={k1:+.4f}")

    # Yardline + sideline linear fits in undistorted space.
    yl_fits = []
    for g in yl_groups:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append({"a": float(a), "b": float(b),
                         "ymin": float(ys.min()), "ymax": float(ys.max())})
    sl_fits = []
    for g in sl_groups:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        xs, ys = pts_u[:, 0], pts_u[:, 1]
        b, a = np.polyfit(xs, ys, 1)
        sl_fits.append({"a": float(a), "b": float(b),
                         "xmin": float(xs.min()), "xmax": float(xs.max())})

    # ── Yardline grid: sort by x at image center, snap to integer spacing.
    x_at_center = np.array([yf["a"] + yf["b"] * cy for yf in yl_fits])
    order = np.argsort(x_at_center)
    sorted_x = x_at_center[order]
    if len(sorted_x) < 2:
        print("  <2 yardlines — cannot establish grid")
        return
    deltas = np.diff(sorted_x)
    unit_px = float(np.median(deltas))
    anchor_x = float(sorted_x[0])
    raw_g = (sorted_x - anchor_x) / unit_px
    g_idx_sorted = np.round(raw_g).astype(int)
    snap_resid = raw_g - g_idx_sorted

    # Map back to original yardline indices.
    g_index = np.zeros(len(yl_fits), dtype=int)
    for k_, orig in enumerate(order):
        g_index[orig] = int(g_idx_sorted[k_])

    print(f"  unit spacing = {unit_px:.1f} px,  "
          f"snap residuals (frac of unit): "
          f"min={snap_resid.min():+.2f}  max={snap_resid.max():+.2f}")
    for k_, orig in enumerate(order):
        print(f"    yl[{orig}]: x@cy={x_at_center[orig]:7.1f}  "
              f"g={g_index[orig]:+d}  resid={snap_resid[k_]:+.2f}")

    # ── Hashes ──
    hash_pxs_all, hash_confs_all = run_hash_w18(
        frame, HASH, device="mps", conf_thresh=HASH_CONF_FLOOR,
    )
    if len(hash_pxs_all) == 0:
        hash_pxs = np.zeros((0, 2)); hash_confs = np.zeros(0)
        rel_thresh = HASH_CONF_FLOOR
    else:
        max_conf = float(hash_confs_all.max())
        rel_thresh = max(HASH_CONF_FLOOR, HASH_CONF_REL * max_conf)
        keep = hash_confs_all >= rel_thresh
        hash_pxs = hash_pxs_all[keep]; hash_confs = hash_confs_all[keep]

    correspondences = []   # (px_x, px_y, x_field, y_field, label, color)

    if len(hash_pxs) > 0:
        hash_u = undistort_points(hash_pxs.astype(np.float64), intr)
        # Snap to nearest yardline by perp distance.
        perp = np.full((len(hash_u), len(yl_fits)), np.inf)
        for j, yf in enumerate(yl_fits):
            d = (hash_u[:, 0] - (yf["a"] + yf["b"] * hash_u[:, 1])) \
                / np.sqrt(1 + yf["b"] ** 2)
            perp[:, j] = np.abs(d)
        nearest_yl = np.argmin(perp, axis=1)
        matched = perp[np.arange(len(hash_u)), nearest_yl] < MAX_DIST_PX
        work_idx = np.where(matched)[0]

        if len(work_idx) >= 2:
            pts_work = hash_u[work_idx]
            center_pca = pts_work.mean(axis=0)
            _, S, Vt = np.linalg.svd(pts_work - center_pca, full_matrices=False)
            pc2 = Vt[1]
            proj_pc2 = (pts_work - center_pca) @ pc2
            ratio = float(S[0] / max(S[1], 1e-6))
            split_p, _ = otsu_split_1d(proj_pc2)
            side_a = proj_pc2 < split_p
            side_b = ~side_a
            if side_a.any() and side_b.any() and \
               pts_work[side_a, 1].mean() < pts_work[side_b, 1].mean():
                far_local, near_local = side_a, side_b
            else:
                far_local, near_local = side_b, side_a
            gap = (abs(proj_pc2[far_local].mean() - proj_pc2[near_local].mean())
                   if far_local.any() and near_local.any() else 0.0)

            if ratio >= PCA_S0_S1_MIN and gap >= MIN_CLUSTER_GAP_PX:
                m_far, c_far = fit_row_line(pts_work[far_local])
                m_near, c_near = fit_row_line(pts_work[near_local])
                d_far = perp_dist(pts_work, m_far, c_far)
                d_near = perp_dist(pts_work, m_near, c_near)

                def thr(d, mask):
                    if mask.sum() < 2: return 5.0
                    v = d[mask]; med = float(np.median(v))
                    mad = float(np.median(np.abs(v - med))) + 1e-6
                    return max(5.0, med + MAD_K * mad)
                far_thr = thr(d_far, far_local); near_thr = thr(d_near, near_local)

                # Per-yardline dedup: keep best-fit-to-row per (yl, role).
                from collections import defaultdict
                kept = {}     # (gi, role) → (i_local, dist)
                for i_local in range(len(work_idx)):
                    gi = int(nearest_yl[work_idx[i_local]])
                    if far_local[i_local] and d_far[i_local] <= far_thr:
                        key = (gi, "far"); dist_ = float(d_far[i_local])
                    elif near_local[i_local] and d_near[i_local] <= near_thr:
                        key = (gi, "near"); dist_ = float(d_near[i_local])
                    else:
                        continue
                    if key not in kept or dist_ < kept[key][1]:
                        kept[key] = (i_local, dist_)

                for (yl_i, role), (i_local, _) in kept.items():
                    g = g_index[yl_i]
                    px = hash_u[work_idx[i_local]]
                    x_field = YD_PER_GRID * g
                    y_field = HASH_Y_FAR if role == "far" else HASH_Y_NEAR
                    label = f"{role}_hash @ g={g:+d}"
                    color = (60, 60, 255) if role == "far" else (255, 80, 80)
                    correspondences.append(
                        (float(px[0]), float(px[1]), x_field, y_field, label, color)
                    )

    # ── Sideline × yardline intersections ──
    for yi, yf in enumerate(yl_fits):
        yspan = yf["ymax"] - yf["ymin"]
        for si, sf in enumerate(sl_fits):
            xspan = sf["xmax"] - sf["xmin"]
            denom = 1.0 - yf["b"] * sf["b"]
            if abs(denom) < 1e-9:
                continue
            x = (yf["a"] + yf["b"] * sf["a"]) / denom
            y = sf["a"] + sf["b"] * x
            if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
                continue
            if not (yf["ymin"] - EXTRAP_FRAC * yspan <= y
                    <= yf["ymax"] + EXTRAP_FRAC * yspan):
                continue
            if not (sf["xmin"] - EXTRAP_FRAC * xspan <= x
                    <= sf["xmax"] + EXTRAP_FRAC * xspan):
                continue
            g = g_index[yi]
            x_field = YD_PER_GRID * g
            y_field = SIDELINE_Y_FIELD
            label = f"sl × g={g:+d}"
            correspondences.append(
                (float(x), float(y), x_field, y_field, label, (0, 255, 255))
            )

    print(f"\n  total correspondences: {len(correspondences)}")
    for (px, py, xf, yf_, lab, _) in correspondences:
        print(f"    ({px:7.1f}, {py:7.1f})  →  field ({xf:+5.1f}, {yf_:5.2f})  {lab}")

    # ── Render ──
    canvas = frame_u.copy()
    for yf in yl_fits:
        ys = np.linspace(yf["ymin"], yf["ymax"], 200)
        xs = yf["a"] + yf["b"] * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (180, 180, 180), 1, cv2.LINE_AA)
    for sf in sl_fits:
        xs = np.linspace(sf["xmin"], sf["xmax"], 200)
        ys = sf["a"] + sf["b"] * xs
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (200, 200, 0), 1, cv2.LINE_AA)

    # Yardline g labels at top of each line.
    for j, yf in enumerate(yl_fits):
        g = g_index[j]
        y_lab = max(yf["ymin"] + 20, 30)
        x_lab = yf["a"] + yf["b"] * y_lab
        cv2.putText(canvas, f"g={g:+d}",
                    (int(x_lab) - 18, int(y_lab)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Correspondence dots + small labels.
    for (px, py, xf, yf_, lab, color) in correspondences:
        cv2.circle(canvas, (int(round(px)), int(round(py))), 6, color, -1)
        cv2.circle(canvas, (int(round(px)), int(round(py))), 8, (0, 0, 0), 1)
        cv2.putText(canvas, lab,
                    (int(px) + 9, int(py) - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    cv2.putText(canvas,
                f"k1={k1:+.4f}  yl={len(yl_groups)} (g {g_index.min():+d}…{g_index.max():+d})  "
                f"correspondences={len(correspondences)}  unit={unit_px:.0f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                "g=0 at leftmost yardline (relative anchor; absolute TBD)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, canvas)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
