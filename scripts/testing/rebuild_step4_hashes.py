#!/usr/bin/env python3
"""Step 4 of the rebuild: map hash detections to yardlines + classify near/far.

Pipeline:
  1. UNet → yard/side masks → CC groups → line pixels
  2. Calibrate k1 (1D scipy minimize_scalar)
  3. Undistort frame + line pixels; fit linear x=a+b·y per yardline
  4. Run W18 hash detector at low floor (0.30); filter via relative
     threshold = max(0.30, 0.6 · max_conf).
  5. Undistort hashes; snap to nearest yardline by perp distance ≤ 12 px.
  6. PCA on undistorted hashes → PC1=along-row, PC2=across-rows.
     1D Otsu on PC2 projection → initial near vs far. Robust to camera tilt.
     Guard: if PC1/PC2 singular-value ratio < 3, treat as single-row.
  7. Fit a 2D row line (y = m·x + c) per cluster.
  8. Reject hashes whose perpendicular distance to its row line exceeds
     median + 3·MAD (or absolute floor 5 px). Per-yardline same-role
     duplicates: keep the one closest to its row line.
  9. Render: undistorted frame, linear yardlines, hashes color-coded,
     and the two row lines drawn through the kept points.
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
    run_unet, group_yardline_pixels_cc, group_sideline_pixels, run_hash_w18,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
HASH = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step4_hashes.jpg")

MAX_DIST_PX = 12.0
HASH_CONF_FLOOR = 0.30
HASH_CONF_REL = 0.60          # min_conf = max(floor, REL · max_conf)
MAD_K = 3.0                   # outlier reject if proj_dist_to_median > MAD_K · MAD
MIN_CLUSTER_GAP_PX = 30.0     # below this, treat as single row, skip near/far


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


def otsu_split_1d(values: np.ndarray) -> tuple[float, float]:
    """1D Otsu split on a vector. Returns (split_value, between_var_score)."""
    s = np.sort(values)
    N = len(s)
    best_score = -np.inf; best_t = float(s.mean())
    for j in range(1, N):
        mu1 = s[:j].mean(); mu2 = s[j:].mean()
        score = (j * (N - j) / N) * (mu1 - mu2) ** 2
        if score > best_score:
            best_score = score
            best_t = 0.5 * (s[j-1] + s[j])
    return best_t, float(best_score)


def fit_row_line(pts: np.ndarray) -> tuple[float, float]:
    """Fit y = m·x + c through 2D pts (row lines run left-right in image).
    Returns (m, c). For a single point, slope=0, c=y.
    """
    if len(pts) == 1:
        return 0.0, float(pts[0, 1])
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return float(m), float(c)


def perp_dist_to_row(pts: np.ndarray, m: float, c: float) -> np.ndarray:
    """Perp distance from each pt to line y = m·x + c."""
    return np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(1.0 + m * m)


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    print(f"  {len(yl)} yardlines + {len(sl)} sidelines, "
          f"{sum(len(p) for p in line_pts)} px")

    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    print(f"  k1 = {k1:+.4f}")

    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()

    yl_fits = []
    for g in yl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append((float(a), float(b), float(ys.min()), float(ys.max())))

    # ── 4. Hash detection at low floor ───────────────────────────────────
    t0 = time.time()
    hash_pxs_all, hash_confs_all = run_hash_w18(
        frame, HASH, device="mps", conf_thresh=HASH_CONF_FLOOR,
    )
    print(f"  raw hashes @ floor={HASH_CONF_FLOOR}: {len(hash_pxs_all)}  "
          f"({(time.time()-t0)*1000:.0f} ms)")
    if len(hash_pxs_all) == 0:
        print("  no hashes — bailing")
        return

    # Relative confidence gate.
    max_conf = float(hash_confs_all.max())
    rel_thresh = max(HASH_CONF_FLOOR, HASH_CONF_REL * max_conf)
    keep_conf = hash_confs_all >= rel_thresh
    hash_pxs = hash_pxs_all[keep_conf]
    hash_confs = hash_confs_all[keep_conf]
    print(f"  max_conf={max_conf:.2f}  rel_thresh={rel_thresh:.2f}  "
          f"kept {keep_conf.sum()}/{len(hash_pxs_all)}")

    if len(hash_pxs) == 0:
        print("  no hashes after relative gate — bailing")
        return

    hash_u = undistort_points(hash_pxs.astype(np.float64), intr)

    # ── 5+6. Snap to yardline by perp distance ───────────────────────────
    perp = np.full((len(hash_u), len(yl_fits)), np.inf)
    for j, (a, b, _, _) in enumerate(yl_fits):
        d = (hash_u[:, 0] - (a + b * hash_u[:, 1])) / np.sqrt(1 + b * b)
        perp[:, j] = np.abs(d)
    nearest_yl = np.argmin(perp, axis=1)
    nearest_d = perp[np.arange(len(hash_u)), nearest_yl]
    matched = nearest_d < MAX_DIST_PX
    print(f"  matched (≤{MAX_DIST_PX}px): {matched.sum()}/{len(hash_u)}")

    # Drop unmatched up front for clustering.
    work_idx = np.where(matched)[0]
    if len(work_idx) < 2:
        print("  <2 matched hashes — can't cluster near/far")
        # fall through to render whatever we have
        role = ["unmatched"] * len(hash_u)
        for i in work_idx:
            role[i] = "near"  # arbitrary
    else:
        pts_work = hash_u[work_idx]

        # ── 7. PCA → 1D Otsu on PC2 (across-rows axis) ──
        # PC1 spans the along-row direction (many yardlines = wide spread);
        # PC2 is perpendicular and captures the near/far row gap. Rotation-
        # invariant, so this works even when camera tilt is high enough that
        # near and far hashes overlap in image-y.
        center_pca = pts_work.mean(axis=0)
        _, S, Vt = np.linalg.svd(pts_work - center_pca, full_matrices=False)
        pc1, pc2 = Vt[0], Vt[1]
        proj_pc2 = (pts_work - center_pca) @ pc2
        ratio = float(S[0] / max(S[1], 1e-6))
        split_p, fisher = otsu_split_1d(proj_pc2)
        side_a = proj_pc2 < split_p
        side_b = ~side_a
        # Sign convention: smaller image-y mean = far.
        if side_a.any() and side_b.any():
            if pts_work[side_a, 1].mean() < pts_work[side_b, 1].mean():
                far_local, near_local = side_a, side_b
            else:
                far_local, near_local = side_b, side_a
        else:
            far_local = np.zeros(len(pts_work), dtype=bool)
            near_local = np.zeros(len(pts_work), dtype=bool)
        gap = (abs(proj_pc2[far_local].mean() - proj_pc2[near_local].mean())
               if far_local.any() and near_local.any() else 0.0)
        pc1_deg = float(np.degrees(np.arctan2(pc1[1], pc1[0]))) % 180
        print(f"  PCA: PC1 axis = {pc1_deg:.0f}°  S0/S1 = {ratio:.1f}")
        print(f"  Otsu on PC2 @ p={split_p:.1f}  Fisher={fisher:.0f}  "
              f"gap={gap:.1f}px")

        role = ["unmatched"] * len(hash_u)

        if ratio < 3.0:
            print(f"  S0/S1 < 3 → PC1 not dominant, treating as SINGLE-ROW")
            for i in work_idx:
                role[i] = "single"
        elif gap < MIN_CLUSTER_GAP_PX:
            print(f"  gap < {MIN_CLUSTER_GAP_PX}px → SINGLE-ROW, "
                  f"skipping near/far split")
            for i in work_idx:
                role[i] = "single"
        else:
            # ── 8. Fit row line per cluster, MAD-reject by perp distance ──
            far_pts = pts_work[far_local]; near_pts = pts_work[near_local]
            m_far, c_far = fit_row_line(far_pts)
            m_near, c_near = fit_row_line(near_pts)
            print(f"  far row:  y = {m_far:+.3f}·x + {c_far:.1f}  "
                  f"({far_local.sum()} pts)")
            print(f"  near row: y = {m_near:+.3f}·x + {c_near:.1f}  "
                  f"({near_local.sum()} pts)")

            d_far = perp_dist_to_row(pts_work, m_far, c_far)
            d_near = perp_dist_to_row(pts_work, m_near, c_near)

            def mad_thresh(d_vals: np.ndarray, mask: np.ndarray) -> float:
                if mask.sum() < 2:
                    return 5.0
                vals = d_vals[mask]
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) + 1e-6
                return max(5.0, med + MAD_K * mad)

            far_thresh = mad_thresh(d_far, far_local)
            near_thresh = mad_thresh(d_near, near_local)
            print(f"  perp-dist thresholds: far≤{far_thresh:.1f}px, "
                  f"near≤{near_thresh:.1f}px")

            for i_local, gi in enumerate(work_idx):
                if far_local[i_local]:
                    role[gi] = ("far" if d_far[i_local] <= far_thresh
                                else "drop_outlier")
                else:
                    role[gi] = ("near" if d_near[i_local] <= near_thresh
                                else "drop_outlier")

            # Per-yardline dedup: 2+ same-role on same yardline → keep the
            # one closest to its row line.
            from collections import defaultdict
            for role_name, d_arr in (("far", d_far), ("near", d_near)):
                by_yl = defaultdict(list)
                for i_local, gi in enumerate(work_idx):
                    if role[gi] != role_name:
                        continue
                    by_yl[int(nearest_yl[gi])].append((gi, float(d_arr[i_local])))
                for items in by_yl.values():
                    if len(items) < 2:
                        continue
                    items.sort(key=lambda kv: kv[1])
                    for gi_drop, _ in items[1:]:
                        role[gi_drop] = "drop_dup"

    n_far = sum(1 for r in role if r == "far")
    n_near = sum(1 for r in role if r == "near")
    n_single = sum(1 for r in role if r == "single")
    n_drop_o = sum(1 for r in role if r == "drop_outlier")
    n_drop_d = sum(1 for r in role if r == "drop_dup")
    n_um = sum(1 for r in role if r == "unmatched")
    print(f"  result: {n_far} far, {n_near} near, {n_single} single-row, "
          f"{n_drop_o} drop(outlier), {n_drop_d} drop(dup), "
          f"{n_um} unmatched")

    # ── 9. Render ──────────────────────────────────────────────────────────
    canvas = frame_u.copy()

    for (a, b, ymin, ymax) in yl_fits:
        ys = np.linspace(ymin, ymax, 200)
        xs = a + b * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (180, 180, 180), 1)

    color_for = {
        "near": (255, 80, 80),     # blue (BGR)
        "far": (60, 60, 255),      # red
        "single": (0, 200, 255),   # orange
        "drop_outlier": (80, 80, 80),
        "drop_dup": (40, 40, 40),
        "unmatched": (180, 180, 180),
    }
    for i, (hx, hy) in enumerate(hash_u):
        c = color_for[role[i]]
        cv2.circle(canvas, (int(hx), int(hy)), 6, c, -1)
        cv2.circle(canvas, (int(hx), int(hy)), 8, (0, 0, 0), 1)
        cv2.putText(canvas, f"{hash_confs[i]:.2f}",
                    (int(hx) + 8, int(hy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    c, 1, cv2.LINE_AA)

    # Draw the two row-fit lines if we computed them.
    if (n_far + n_drop_o + n_drop_d) > 0 or n_near > 0:
        try:
            xs_line = np.linspace(0, w - 1, 200)
            if n_far + sum(1 for r in role if r == "drop_outlier") > 0:
                ys_far = m_far * xs_line + c_far
                cv2.polylines(canvas,
                              [np.stack([xs_line, ys_far], axis=1).astype(np.int32)],
                              False, (60, 60, 255), 1, cv2.LINE_AA)
            if n_near > 0:
                ys_near = m_near * xs_line + c_near
                cv2.polylines(canvas,
                              [np.stack([xs_line, ys_near], axis=1).astype(np.int32)],
                              False, (255, 80, 80), 1, cv2.LINE_AA)
        except NameError:
            pass

    cv2.putText(canvas,
                f"k1={k1:+.4f}  yl={len(yl)}  hashes: {n_far}far/{n_near}near "
                f"+ {n_drop_o + n_drop_d}drop + {n_um + n_single}other  "
                f"rel_thresh={rel_thresh:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue=near  red=far  orange=single-row  dark=drop",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, canvas)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
