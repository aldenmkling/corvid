#!/usr/bin/env python3
"""Trace where play_023's hashes get dropped through the pipeline."""

import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    run_unet, run_hash_w18, group_yardline_pixels_cc,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from scripts.testing.rebuild_full_clip_viz import (
    group_sideline_pixels as cc_group_sideline,
    fit_yardline_linear, fit_sideline_linear, total_mse,
    otsu_split_1d, fit_row_line, perp_dist,
    YardlineTracker,
    MAX_DIST_PX, HASH_CONF, MIN_CLUSTER_GAP_PX, PCA_S0_S1_MIN, MAD_K,
    G0_NGS_X, YD_PER_GRID,
)

CLIP = os.path.join(PROJECT_ROOT,
                     "videos/clips/2019092204/play_023/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
HASH = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")


def main():
    cap = cv2.VideoCapture(CLIP)
    ok, frame = cap.read(); cap.release()
    h, w = frame.shape[:2]
    focal = float(max(h, w)); cx, cy = w / 2.0, h / 2.0

    yard_mask, side_mask = run_unet(frame, UNET, device="mps")
    yl_g = group_yardline_pixels_cc(yard_mask)
    sl_g = cc_group_sideline(side_mask)
    print(f"  CC: {len(yl_g)} yardlines, {len(sl_g)} sidelines")

    line_pts = [g.pixels for g in yl_g] + [g.pixels for g in sl_g]
    line_kinds = ["yardline"] * len(yl_g) + ["sideline"] * len(sl_g)
    sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    print(f"  k1 = {k1:+.4f} (calibrated only on yardlines, no sidelines)")

    yl_fits = [fit_yardline_linear(g.pixels, intr) for g in yl_g]
    yl_tracker = YardlineTracker(g_min=-8, g_max=12, frame_h=h)
    bs = yl_tracker.init_from(yl_fits, cy)
    if bs is None: print("  yardline init failed"); return
    yl_fits, g_index, x_at_center = bs
    print(f"  {len(yl_fits)} yardlines kept after g-range gate, "
          f"g={list(g_index)}")
    print(f"    x_at_center: {[round(x, 1) for x in x_at_center]}")

    # Now reproduce hashes_with_roles step by step
    pxs, confs = run_hash_w18(frame, HASH, device="mps", conf_thresh=HASH_CONF)
    print(f"\n  HRNet (thresh=HASH_CONF={HASH_CONF}): {len(pxs)} hashes")
    if len(pxs) < 2:
        print("  < 2 hashes after thresh — bail")
        return
    hu = undistort_points(pxs.astype(np.float64), intr)
    print(f"  undistorted hashes:")
    for i in range(len(hu)):
        print(f"    hash[{i}]  raw=({pxs[i,0]:.0f},{pxs[i,1]:.0f})  "
              f"undistorted=({hu[i,0]:.1f},{hu[i,1]:.1f})  conf={confs[i]:.3f}")

    perp = np.full((len(hu), len(yl_fits)), np.inf)
    for j, yf in enumerate(yl_fits):
        d = (hu[:, 0] - (yf["a"] + yf["b"] * hu[:, 1])) / np.sqrt(1 + yf["b"] ** 2)
        perp[:, j] = np.abs(d)
    nearest_yl = np.argmin(perp, axis=1)
    nearest_d = perp[np.arange(len(hu)), nearest_yl]
    matched = nearest_d < MAX_DIST_PX
    print(f"\n  perp-distance to yardlines (MAX={MAX_DIST_PX}):")
    for i in range(len(hu)):
        ok = "✓" if matched[i] else "✗"
        print(f"    hash[{i}]  nearest yl[{nearest_yl[i]}]  "
              f"perp={nearest_d[i]:.1f}px  {ok}")
    work = np.where(matched)[0]
    print(f"  matched: {matched.sum()}/{len(hu)}")
    if len(work) < 2:
        print("  < 2 matched — return empty"); return

    pts_w = hu[work]
    center = pts_w.mean(axis=0)
    _, S, Vt = np.linalg.svd(pts_w - center, full_matrices=False)
    pc2 = Vt[1]; ratio = float(S[0] / max(S[1], 1e-6))
    proj = (pts_w - center) @ pc2
    split_p = otsu_split_1d(proj)
    a_side = proj < split_p; b_side = ~a_side
    if a_side.any() and b_side.any() and \
       pts_w[a_side, 1].mean() < pts_w[b_side, 1].mean():
        far_l, near_l = a_side, b_side
    else:
        far_l, near_l = b_side, a_side
    gap = (abs(proj[far_l].mean() - proj[near_l].mean())
           if far_l.any() and near_l.any() else 0.0)
    print(f"\n  PCA: S0={S[0]:.1f}  S1={S[1]:.1f}  ratio={ratio:.2f}  "
          f"(min req {PCA_S0_S1_MIN})")
    print(f"  Otsu split @ proj={split_p:.2f}  gap={gap:.1f}px  "
          f"(min req {MIN_CLUSTER_GAP_PX})")
    print(f"  far cluster: {int(far_l.sum())} hashes  near cluster: {int(near_l.sum())}")
    print(f"  far proj range: [{float(proj[far_l].min()):.1f}, "
          f"{float(proj[far_l].max()):.1f}]")
    if near_l.any():
        print(f"  near proj range: [{float(proj[near_l].min()):.1f}, "
              f"{float(proj[near_l].max()):.1f}]")
    print(f"  far cluster image-y mean: {float(pts_w[far_l, 1].mean()):.1f}")
    if near_l.any():
        print(f"  near cluster image-y mean: {float(pts_w[near_l, 1].mean()):.1f}")
    print()
    if ratio < PCA_S0_S1_MIN:
        print(f"  ✗ FAIL: ratio={ratio:.2f} < {PCA_S0_S1_MIN} — return empty")
    if gap < MIN_CLUSTER_GAP_PX:
        print(f"  ✗ FAIL: gap={gap:.1f} < {MIN_CLUSTER_GAP_PX} — return empty")
    if ratio >= PCA_S0_S1_MIN and gap >= MIN_CLUSTER_GAP_PX:
        print(f"  ✓ Both gates pass — would emit {int(far_l.sum() + near_l.sum())} hash corrs")


if __name__ == "__main__":
    main()
