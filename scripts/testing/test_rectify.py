#!/usr/bin/env python3
"""Rectify a frame into field (top-down) coordinates using the homography
from the current pipeline. Outputs:
  - <name>_undistorted.jpg: pixel-space undistortion (k1, k2 applied)
  - <name>_rectified.jpg: top-down field view with yard lines drawn
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
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "homography_tests")
HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30

# Field-to-pixel scaling for the rectified image
YD_PER_PX = 0.1   # 10 pixels per yard → field image is 1200 × 534 px


def rectify_frame(frame_path, base_ngs_x, name):
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    print(f"\n[{name}] {frame_path}  ({w}x{h})")

    # 1. Detection + grid solver
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, _ = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, _ = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    sideline_confs = np.ones(len(sideline_pxs))
    groups, _ = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, _ = groups_to_correspondences(groups, base_ngs_x)
    print(f"  {len(pixel_pts)} correspondences")
    if len(pixel_pts) < 4:
        print("  Not enough points for homography — skipping")
        return

    # 2. Distortion
    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]
    focal = float(max(h, w))
    if line_sets:
        k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w), focal)
    else:
        k1, k2 = 0.0, 0.0
    if abs(k1) > 1.0 or abs(k2) > 1.0:
        k1, k2 = 0.0, 0.0
    print(f"  k1={k1:.4f} k2={k2:.4f}")

    # 3. Undistort image + correspondence points
    K = np.array([[focal, 0, w/2.0], [0, focal, h/2.0], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        frame_u = cv2.undistort(frame, K, dist_coeffs)
    else:
        frame_u = frame.copy()
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=w/2.0, cy=h/2.0, k1=k1, k2=k2)
    pixel_pts_u = undistort_points(pixel_pts, intr)

    # 4. Homography: undistorted pixel → field (yd)
    H, mask = cv2.findHomography(pixel_pts_u.astype(np.float64),
                                 field_pts.astype(np.float64),
                                 method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if H is None:
        print("  findHomography failed"); return
    n_inliers = int(mask.sum()) if mask is not None else len(pixel_pts_u)
    print(f"  inliers: {n_inliers}/{len(pixel_pts_u)}")

    # Compose "field → rectified-pixel": scale by 1/YD_PER_PX and flip y so the
    # near sideline (field y=0) is at the IMAGE BOTTOM (matching broadcast
    # orientation). Without the flip, near-side yard numbers appear upside-down.
    field_img_w = int(FIELD_LENGTH / YD_PER_PX)
    field_img_h = int(FIELD_WIDTH / YD_PER_PX)
    S = np.array([[1.0/YD_PER_PX, 0, 0],
                  [0, -1.0/YD_PER_PX, float(field_img_h)],
                  [0, 0, 1]], dtype=np.float64)
    H_pixel_to_rect = S @ H  # undistorted pixel → rectified-pixel

    # Warp
    rectified = cv2.warpPerspective(frame_u, H_pixel_to_rect,
                                    (field_img_w, field_img_h))

    # Annotate rectified with yard lines + hash rows + sidelines (in green)
    def field_to_rect(x_yd, y_yd):
        return (int(x_yd / YD_PER_PX), int(field_img_h - y_yd / YD_PER_PX))

    vis_rect = rectified.copy()
    # yard lines (every 5 yd)
    for x in np.arange(0, FIELD_LENGTH + 1, 5):
        pt1 = field_to_rect(x, 0)
        pt2 = field_to_rect(x, FIELD_WIDTH)
        cv2.line(vis_rect, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        # label every 10 yd
        if int(x) % 10 == 0:
            cv2.putText(vis_rect, f"{int(x)}", field_to_rect(x + 0.3, 2.5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    # hash rows
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        cv2.line(vis_rect, field_to_rect(0, y), field_to_rect(FIELD_LENGTH, y),
                 (0, 200, 200), 1, cv2.LINE_AA)
    # sidelines
    for y in [0, FIELD_WIDTH]:
        cv2.line(vis_rect, field_to_rect(0, y), field_to_rect(FIELD_LENGTH, y),
                 (255, 255, 255), 2)

    # Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    undist_path = os.path.join(OUTPUT_DIR, f"{name}_undistorted.jpg")
    rect_path = os.path.join(OUTPUT_DIR, f"{name}_rectified.jpg")
    raw_rect_path = os.path.join(OUTPUT_DIR, f"{name}_rectified_raw.jpg")
    cv2.imwrite(undist_path, frame_u)
    cv2.imwrite(rect_path, vis_rect)
    cv2.imwrite(raw_rect_path, rectified)
    print(f"  saved: {undist_path}")
    print(f"  saved: {rect_path}")


if __name__ == "__main__":
    rectify_frame(
        os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"),
        base_ngs_x=35.0,
        name="real_kickoff_pipeline_v2",
    )
    rectify_frame(
        os.path.join(OUTPUT_DIR, "play002_first.jpg"),
        base_ngs_x=70.0,
        name="play002_v2",
    )
