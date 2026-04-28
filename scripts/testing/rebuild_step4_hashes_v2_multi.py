#!/usr/bin/env python3
"""Run the v2 hash-row pipeline on a stratified sample of clips and stitch
the per-frame outputs into one panel.

Reuses everything from rebuild_step4_hashes_v2 — only diff is iterating
over clips and tiling results.
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/testing"))

import segmentation_models_pytorch as smp
from src.homography.grid_solver_v2 import (
    group_yardline_pixels_cc, group_sideline_pixels,
)
from src.homography.distortion import CameraIntrinsics, undistort_points

from rebuild_step4_hashes_v2 import (
    run_unified, total_mse, ransac_line,
    YARD_THRESH, SIDE_THRESH, HASH_THRESH, MAX_HASH_PIXELS, WEIGHTS,
)


def process_frame(frame, device):
    h, w = frame.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask, hash_mask = run_unified(frame, WEIGHTS, device)
    yl = group_yardline_pixels_cc(yard_mask)
    sl = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl] + [g.pixels for g in sl]
    line_kinds = ["yardline"] * len(yl) + ["sideline"] * len(sl)
    if not line_pts:
        return None

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

    yl_fits = []
    for g in yl:
        pts_u = undistort_points(g.pixels.astype(np.float64), intr)
        ys, xs = pts_u[:, 1], pts_u[:, 0]
        b, a = np.polyfit(ys, xs, 1)
        yl_fits.append((float(a), float(b),
                          float(ys.min()), float(ys.max())))

    ys_h, xs_h = np.where(hash_mask > 0)
    if len(xs_h) == 0:
        return {"frame_u": frame_u, "yl_fits": yl_fits, "k1": k1, "n_yl": len(yl),
                "msg": "no hash pixels"}
    hash_pts = np.column_stack([xs_h, ys_h]).astype(np.float64)
    if len(hash_pts) > MAX_HASH_PIXELS:
        idx = np.random.RandomState(0).choice(len(hash_pts),
                                                MAX_HASH_PIXELS, replace=False)
        hash_pts = hash_pts[idx]
    hash_u = undistort_points(hash_pts, intr)

    m1, c1, in1 = ransac_line(hash_u, inlier_dist=2.0)
    if m1 is None:
        return {"frame_u": frame_u, "yl_fits": yl_fits, "k1": k1, "n_yl": len(yl),
                "hash_u": hash_u, "msg": "RANSAC line 1 no consensus"}
    rem = hash_u[~in1]
    m2, c2, in2 = ransac_line(rem, inlier_dist=2.0)
    if m2 is None:
        return {"frame_u": frame_u, "yl_fits": yl_fits, "k1": k1, "n_yl": len(yl),
                "hash_u": hash_u, "msg": "single-row only"}

    far_pts_mask = in1.copy()
    rem_idx = np.where(~in1)[0]
    near_pts_mask = np.zeros(len(hash_u), dtype=bool)
    near_pts_mask[rem_idx[in2]] = True
    if hash_u[far_pts_mask, 1].mean() > hash_u[near_pts_mask, 1].mean():
        far_pts_mask, near_pts_mask = near_pts_mask, far_pts_mask
        m_far, c_far = m2, c2
        m_near, c_near = m1, c1
    else:
        m_far, c_far = m1, c1
        m_near, c_near = m2, c2

    keypoints = []
    for j, (a, b, ymin, ymax) in enumerate(yl_fits):
        for role, m, c in (("far", m_far, c_far), ("near", m_near, c_near)):
            denom = 1.0 - b * m
            if abs(denom) < 1e-6:
                continue
            y = (c + a * m) / denom
            x = a + b * y
            if not (0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]):
                continue
            keypoints.append((j, role, float(x), float(y)))

    return {
        "frame_u": frame_u, "yl_fits": yl_fits, "k1": k1, "n_yl": len(yl),
        "hash_u": hash_u, "far_mask": far_pts_mask, "near_mask": near_pts_mask,
        "m_far": m_far, "c_far": c_far, "m_near": m_near, "c_near": c_near,
        "keypoints": keypoints,
    }


def render(state):
    canvas = state["frame_u"].copy()
    h, w = canvas.shape[:2]
    for (a, b, ymin, ymax) in state["yl_fits"]:
        ys = np.linspace(ymin, ymax, 200)
        xs = a + b * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, (180, 180, 180), 1)
    if "msg" in state:
        cv2.putText(canvas, f"k1={state['k1']:+.4f}  yl={state['n_yl']}  "
                    f"({state['msg']})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    overlay = canvas.copy()
    pts_int = state["hash_u"].astype(np.int32)
    in_bounds = ((pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) &
                  (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h))
    for mask, color in [(state["far_mask"], (60, 60, 255)),
                          (state["near_mask"], (255, 80, 80)),
                          (~(state["far_mask"] | state["near_mask"]),
                           (120, 120, 120))]:
        m_in = mask & in_bounds
        if m_in.any():
            p = pts_int[m_in]
            overlay[p[:, 1], p[:, 0]] = color
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
    xs_line = np.linspace(0, w - 1, 200)
    for (m, c, color) in [(state["m_far"], state["c_far"], (60, 60, 255)),
                            (state["m_near"], state["c_near"], (255, 80, 80))]:
        ys_line = m * xs_line + c
        cv2.polylines(canvas, [np.stack([xs_line, ys_line], axis=1).astype(np.int32)],
                      False, color, 1, cv2.LINE_AA)
    for _, role, x, y in state["keypoints"]:
        col = (60, 60, 255) if role == "far" else (255, 80, 80)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, col, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (0, 0, 0), 1)
    n_far = sum(1 for _, r, _, _ in state["keypoints"] if r == "far")
    n_near = sum(1 for _, r, _, _ in state["keypoints"] if r == "near")
    cv2.putText(canvas,
                f"k1={state['k1']:+.4f}  yl={state['n_yl']}  "
                f"hash px={len(state['hash_u'])}  "
                f"{n_far}far + {n_near}near = {n_far+n_near} keypoints",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    import random
    random.seed(7)
    clip_paths = []
    for game in sorted(os.listdir(os.path.join(PROJECT_ROOT, "videos/clips"))):
        gd = os.path.join(PROJECT_ROOT, "videos/clips", game)
        if not os.path.isdir(gd):
            continue
        plays = [p for p in sorted(os.listdir(gd)) if p.startswith("play_")]
        if not plays:
            continue
        # Pick one random play per game
        play = random.choice(plays)
        cp = os.path.join(gd, play, "sideline.mp4")
        if os.path.exists(cp):
            clip_paths.append(cp)

    print(f"  {len(clip_paths)} clips selected")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    panels = []
    for cp in clip_paths:
        cap = cv2.VideoCapture(cp)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  {cp}: read failed"); continue
        rel = os.path.relpath(cp, PROJECT_ROOT)
        try:
            t0 = time.time()
            state = process_frame(frame, device)
            elapsed = (time.time() - t0) * 1000
        except Exception as e:
            print(f"  {rel}: {type(e).__name__}: {e}")
            continue
        if state is None:
            print(f"  {rel}: no lines"); continue
        canvas = render(state)
        # Header bar
        hdr = np.full((28, canvas.shape[1], 3), 30, dtype=np.uint8)
        msg = f"{rel}  ({elapsed:.0f} ms)"
        cv2.putText(hdr, msg, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        panels.append(np.vstack([hdr, canvas]))
        print(f"  {rel}: ok ({elapsed:.0f} ms)")

    if not panels:
        print("  no panels generated"); return

    # Resize all to common height for stacking
    target_h = 360
    rs = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h))
           for p in panels]
    max_w = max(r.shape[1] for r in rs)
    padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)),
                       mode="constant") for r in rs]
    grid = np.vstack(padded)
    out = os.path.join(PROJECT_ROOT, "output/rebuild/step4_hashes_v2_multi.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, grid)
    print(f"  wrote {out}  shape={grid.shape}")


if __name__ == "__main__":
    main()
