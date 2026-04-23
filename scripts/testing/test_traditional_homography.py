#!/usr/bin/env python3
"""
Traditional 8-DOF homography per frame (no camera model).

Pipeline:
  1. HRNet + grid solver → correspondences (pixel ↔ field)
  2. Plumb-line distortion calibration → k1, k2 (from frame itself)
  3. Undistort pixel points
  4. cv2.findHomography with RANSAC → H matrix
  5. Visualize: projected yard grid + reprojection errors

Each frame is independent — no shared parameters. We're just asking
"given these correspondences, can a plain homography map pixel↔field well?"
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines,
)
from src.homography.grid_solver import (
    run_hrnet, extract_peaks,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.apply_homography import pixel_to_field, field_to_pixel
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "homography_tests")
HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30


def compute_homography_for_frame(frame_path, base_ngs_x, label, out_name):
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    print(f"\n[{label}] {os.path.basename(frame_path)}  ({w}x{h})")

    # 1. HRNet + grid solver
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    groups, _ = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, base_ngs_x)
    print(f"  {len(pixel_pts)} correspondences")
    if len(pixel_pts) < 4:
        print(f"  Not enough points for homography")
        return

    # 2. Plumb-line distortion calibration
    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]
    focal_guess = float(max(h, w))
    if line_sets:
        k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w), focal_guess)
    else:
        k1, k2 = 0.0, 0.0
    if abs(k1) > 1.0 or abs(k2) > 1.0:
        k1, k2 = 0.0, 0.0
    print(f"  Distortion: k1={k1:.4f}, k2={k2:.4f}")

    # 3. Undistort pixel points
    intr = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                             cx=w/2.0, cy=h/2.0, k1=k1, k2=k2)
    pixel_pts_u = undistort_points(pixel_pts, intr)

    # 4. findHomography (with RANSAC for robustness)
    H, mask = cv2.findHomography(
        pixel_pts_u.astype(np.float64),
        field_pts.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,  # field-coord threshold (yards)
    )
    if H is None:
        print("  findHomography failed")
        return

    # H maps pixel → field. We also want H_inv for projecting field → pixel.
    H_inv = np.linalg.inv(H)
    n_inliers = int(mask.sum()) if mask is not None else len(pixel_pts_u)
    print(f"  RANSAC inliers: {n_inliers}/{len(pixel_pts_u)}")

    # 5. Reprojection error (field-space)
    projected_field = pixel_to_field(pixel_pts_u, H)
    errs = np.linalg.norm(projected_field - field_pts, axis=1)
    print(f"  Field-space error: mean={errs.mean():.3f}yd, "
          f"median={np.median(errs):.3f}yd, max={errs.max():.3f}yd")

    # Also report back-projection pixel error for visual comparison
    pixel_back = field_to_pixel(field_pts, H_inv)
    pix_errs = np.linalg.norm(pixel_back - pixel_pts_u, axis=1)
    print(f"  Pixel-space error: mean={pix_errs.mean():.2f}px, "
          f"median={np.median(pix_errs):.2f}px, max={pix_errs.max():.2f}px")

    # 6. Overlay — undistort image first, then draw projected field grid
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        K = np.array([[focal_guess, 0, w/2.0],
                      [0, focal_guess, h/2.0],
                      [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        frame_u = cv2.undistort(frame, K, dist_coeffs)
    else:
        frame_u = frame

    vis = frame_u.copy()
    # Yard lines
    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        fp = np.column_stack([np.full_like(ys, x), ys])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            pt1 = tuple(pp[i].astype(int))
            pt2 = tuple(pp[i+1].astype(int))
            # Only draw if both points are reasonably inside the frame
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    # Hash rows
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            pt1 = tuple(pp[i].astype(int))
            pt2 = tuple(pp[i+1].astype(int))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(vis, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

    # Sidelines (y=0 and y=FIELD_WIDTH)
    for y in [0, FIELD_WIDTH]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            pt1 = tuple(pp[i].astype(int))
            pt2 = tuple(pp[i+1].astype(int))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(vis, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    # Correspondence markers
    for i in range(len(pixel_pts_u)):
        det = tuple(pixel_pts_u[i].astype(int))
        proj = tuple(pixel_back[i].astype(int))
        color_det = (0, 0, 255) if mask is None or mask[i][0] else (128, 128, 128)
        cv2.circle(vis, det, 5, color_det, 2)
        cv2.circle(vis, proj, 3, (0, 255, 0), -1)
        cv2.line(vis, det, proj, (255, 255, 0), 1)

    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, vis)
    print(f"  Saved: {out_path}")

    return {"H": H, "H_inv": H_inv, "errors_yards": errs, "errors_pixels": pix_errs,
            "inliers": n_inliers, "n_corr": len(pixel_pts_u), "k1": k1, "k2": k2}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", default="real_kickoff_raw.jpg")
    parser.add_argument("--ngs1", type=float, default=35.0)
    parser.add_argument("--f2", default="play002_first.jpg")
    parser.add_argument("--ngs2", type=float, default=70.0)
    args = parser.parse_args()

    compute_homography_for_frame(
        os.path.join(OUTPUT_DIR, args.f1), args.ngs1, "F1",
        "traditional_homography_f1.jpg",
    )
    compute_homography_for_frame(
        os.path.join(OUTPUT_DIR, args.f2), args.ngs2, "F2",
        "traditional_homography_f2.jpg",
    )


if __name__ == "__main__":
    main()
