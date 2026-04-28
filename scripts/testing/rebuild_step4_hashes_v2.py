#!/usr/bin/env python3
"""Step 4 v2: hash detection via line fits through unified UNet hash mask
pixels.

Differences from step4_hashes.py:
  - One unified UNet forward pass (replaces line UNet + W18 hash detector).
  - Hash row geometry comes from line fits through ALL undistorted hash
    mask pixels per row, not from centroids of W18 keypoints. With
    thousands of pixels per row vs ~16 centroids, the row line is much
    more stable.
  - Hash keypoints = intersection of (each yardline) × (each hash row).
    No per-keypoint detection; everything is geometry.

Pipeline:
  1. Unified UNet on raw frame → yard / side / hash binary masks.
  2. CC group yardline + sideline pixels (existing helpers).
  3. Calibrate k1 from line pixels (existing logic).
  4. Undistort line pixels and hash mask pixels.
  5. Fit linear yardline `x = a + b·y` per yardline CC.
  6. Project undistorted hash pixels onto perpendicular-to-mean-yardline
     axis; 1D Otsu split → near/far row clusters.
  7. Fit `y = m·x + c` per row through all cluster pixels.
  8. Each yardline × each row → intersection keypoint.
  9. Render: undistorted frame + yardlines + row lines + intersection dots.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import segmentation_models_pytorch as smp

from src.homography.grid_solver_v2 import (
    group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_unified_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step4_hashes_v2.jpg")

INPUT_H, INPUT_W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YARD_THRESH = 0.5
SIDE_THRESH = 0.5
HASH_THRESH = 0.5
MAX_HASH_PIXELS = 8000     # subsample if more (vectorized polyfit is fine,
                            #   undistort is the bottleneck)


def run_unified(frame, weights, device):
    """Returns (yard, side, hash) binary masks at frame resolution."""
    model = smp.Unet("mit_b0", encoder_weights=None, classes=3, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    h0, w0 = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_W, INPUT_H))
    x = rgb.astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)
    t = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(t))[0].cpu().numpy()        # (3, H_in, W_in)

    yard = (probs[0] > YARD_THRESH).astype(np.uint8)
    side = (probs[1] > SIDE_THRESH).astype(np.uint8)
    hash_ = (probs[2] > HASH_THRESH).astype(np.uint8)
    yard = cv2.resize(yard, (w0, h0), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w0, h0), interpolation=cv2.INTER_NEAREST)
    hash_ = cv2.resize(hash_, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return yard, side, hash_


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


def ransac_line(pts: np.ndarray, n_iters: int = 800,
                  inlier_dist: float = 2.0,
                  min_inliers: int = 30,
                  seed: int = 0) -> tuple[float | None, float | None, np.ndarray | None]:
    """Sequential RANSAC line fit. Returns (m, c, inlier_mask) for
    `y = m·x + c`, or (None, None, None) if no consensus.
    """
    n = len(pts)
    if n < 2:
        return None, None, None
    rng = np.random.RandomState(seed)
    best_count = 0
    best_in = None
    for _ in range(n_iters):
        i, j = rng.choice(n, 2, replace=False)
        dx = pts[j, 0] - pts[i, 0]
        if abs(dx) < 1e-3:
            continue
        m = (pts[j, 1] - pts[i, 1]) / dx
        c = pts[i, 1] - m * pts[i, 0]
        d = np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(1.0 + m * m)
        in_mask = d < inlier_dist
        cnt = int(in_mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_in = in_mask
    if best_in is None or best_count < min_inliers:
        return None, None, None
    in_pts = pts[best_in]
    m, c = np.polyfit(in_pts[:, 0], in_pts[:, 1], 1)
    return float(m), float(c), best_in


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  failed to read clip: {CLIP}"); return
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ── 1. Unified UNet ──────────────────────────────────────────────────
    t0 = time.time()
    yard_mask, side_mask, hash_mask = run_unified(frame, WEIGHTS, device)
    print(f"  unified UNet: {(time.time()-t0)*1000:.0f} ms  "
          f"yard={int(yard_mask.sum())}px  side={int(side_mask.sum())}px  "
          f"hash={int(hash_mask.sum())}px")

    # ── 2. CC groups ─────────────────────────────────────────────────────
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    print(f"  {len(yl)} yardlines + {len(sl)} sidelines")

    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)

    # ── 3. k1 calibration ────────────────────────────────────────────────
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

    # ── 4. Yardline fits in undistorted space ────────────────────────────
    yl_fits = []
    for g in yl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append((float(a), float(b),
                          float(ys.min()), float(ys.max())))

    # ── 5. Hash mask pixels, undistort, cluster, line-fit ────────────────
    ys_h, xs_h = np.where(hash_mask > 0)
    if len(xs_h) == 0:
        print("  no hash pixels — bailing"); return
    hash_pts = np.column_stack([xs_h, ys_h]).astype(np.float64)
    if len(hash_pts) > MAX_HASH_PIXELS:
        idx = np.random.RandomState(0).choice(len(hash_pts),
                                                MAX_HASH_PIXELS, replace=False)
        hash_pts = hash_pts[idx]
    hash_u = undistort_points(hash_pts, intr)
    print(f"  hash mask px: {len(hash_u)} (after subsample / undistort)")

    # Sequential RANSAC: fit line 1, drop its inliers, fit line 2 on the
    # remainder. Robust to outliers + axis-agnostic (no projection / Otsu
    # required, so we don't have to guess which axis separates the rows).
    m1, c1, in1 = ransac_line(hash_u, inlier_dist=2.0)
    if m1 is None:
        print("  RANSAC line 1 found no consensus — bailing"); return
    rem = hash_u[~in1]
    print(f"  line 1: y = {m1:+.4f}·x + {c1:.1f}  ({int(in1.sum())} inliers)")
    m2, c2, in2 = ransac_line(rem, inlier_dist=2.0)
    if m2 is None:
        print("  RANSAC line 2 found no consensus — single-row frame")
        return
    print(f"  line 2: y = {m2:+.4f}·x + {c2:.1f}  ({int(in2.sum())} inliers)")

    # Sign convention: smaller mean image-y = far row (sideline view).
    far_pts_mask = in1.copy()      # mask in original hash_u indexing
    rem_idx = np.where(~in1)[0]
    near_pts_mask = np.zeros(len(hash_u), dtype=bool)
    near_pts_mask[rem_idx[in2]] = True
    if hash_u[far_pts_mask, 1].mean() > hash_u[near_pts_mask, 1].mean():
        # line 1 was actually the near row — swap labels.
        far_pts_mask, near_pts_mask = near_pts_mask, far_pts_mask
        m_far, c_far = m2, c2
        m_near, c_near = m1, c1
    else:
        m_far, c_far = m1, c1
        m_near, c_near = m2, c2
    far_mask, near_mask = far_pts_mask, near_pts_mask
    print(f"  far row:  y = {m_far:+.4f}·x + {c_far:.1f}  "
          f"({far_mask.sum()} px)")
    print(f"  near row: y = {m_near:+.4f}·x + {c_near:.1f}  "
          f"({near_mask.sum()} px)")

    # ── 6. Intersections: yardline × row → keypoint ──────────────────────
    # yardline:  x = a + b·y     row: y = m·x + c
    # Solve: a + b·y = (y - c)/m → y = (c + a·m) / (1 − b·m)
    keypoints = []   # (yl_idx, role, x, y)
    for j, (a, b, ymin, ymax) in enumerate(yl_fits):
        for role, m, c in (("far", m_far, c_far), ("near", m_near, c_near)):
            denom = 1.0 - b * m
            if abs(denom) < 1e-6:
                continue
            y = (c + a * m) / denom
            x = a + b * y
            # Sanity: skip if intersection is way outside frame bounds.
            if not (0 <= x <= w and 0 <= y <= h):
                continue
            keypoints.append((j, role, float(x), float(y)))
    n_far_kp = sum(1 for _, r, _, _ in keypoints if r == "far")
    n_near_kp = sum(1 for _, r, _, _ in keypoints if r == "near")
    print(f"  intersections: {n_far_kp} far + {n_near_kp} near = "
          f"{len(keypoints)} keypoints")

    # ── 7. Render ────────────────────────────────────────────────────────
    canvas = frame_u.copy()

    for (a, b, ymin, ymax) in yl_fits:
        ys = np.linspace(ymin, ymax, 200)
        xs = a + b * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (180, 180, 180), 1)

    # Hash mask pixels colored by row (far=red, near=blue, dropped=gray).
    overlay = canvas.copy()
    pts_int = hash_u.astype(np.int32)
    in_bounds = ((pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) &
                  (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h))
    for mask, color in [(far_mask, (60, 60, 255)),
                          (near_mask, (255, 80, 80)),
                          (~(far_mask | near_mask), (120, 120, 120))]:
        m_in = mask & in_bounds
        if m_in.any():
            p = pts_int[m_in]
            overlay[p[:, 1], p[:, 0]] = color
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

    # Row lines drawn full width
    xs_line = np.linspace(0, w - 1, 200)
    for (m, c, color) in [(m_far, c_far, (60, 60, 255)),
                            (m_near, c_near, (255, 80, 80))]:
        ys_line = m * xs_line + c
        cv2.polylines(canvas, [np.stack([xs_line, ys_line], axis=1).astype(np.int32)],
                      False, color, 1, cv2.LINE_AA)

    # Intersection points
    for _, role, x, y in keypoints:
        col = (60, 60, 255) if role == "far" else (255, 80, 80)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, col, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (0, 0, 0), 1)

    cv2.putText(canvas,
                f"v2: unified UNet  k1={k1:+.4f}  yl={len(yl)}  "
                f"hash px={len(hash_u)}  intersections: "
                f"{n_far_kp} far + {n_near_kp} near",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue=near row  red=far row  cyan dots=hash mask px",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, canvas)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
