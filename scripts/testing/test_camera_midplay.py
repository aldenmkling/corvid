#!/usr/bin/env python3
"""
Test per-frame PTZ solving on a mid-play frame using calibration from kickoff.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.homography.classical.yard_lines import detect_yard_lines
from src.homography.classical.hash_marks import detect_hashes
from src.homography.camera_model import (
    CameraIntrinsics, CameraExtrinsics, CameraCalibration, CameraState,
    calibrate_camera, solve_ptz, camera_state_to_homography,
    project_field_to_pixel,
)
from src.homography.apply_homography import pixel_to_field
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "videos", "clips", "2019102712")


def get_hash_correspondences(yl_result, hash_result, base_x):
    """Extract pixel/field correspondences from hash detections."""
    pixel_pts = []
    field_pts = []
    min_grid = min(yl_result.grid)

    for li, entry in hash_result.hashes.items():
        line = yl_result.lines[li]
        gpos = yl_result.grid[li]
        field_x = base_x + (gpos - min_grid) * 5
        x_top, y_top, x_bot, y_bot = line
        dx, dy = x_bot - x_top, y_bot - y_top

        for ht in ['far', 'near']:
            if ht in entry:
                t = entry[ht]
                pixel_pts.append([x_top + t * dx, y_top + t * dy])
                field_pts.append([field_x, HASH_Y_FAR if ht == 'far' else HASH_Y_NEAR])

    return np.array(pixel_pts), np.array(field_pts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Calibrate from kickoff frame (same as before) ───────────
    print("=== Calibrating from kickoff ===")
    kickoff_frame = cv2.imread(os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"))
    yl_kick = detect_yard_lines(kickoff_frame)
    hr_kick = detect_hashes(kickoff_frame, yl_kick, yl_kick.gray, yl_kick.canny)

    kick_px, kick_field = get_hash_correspondences(yl_kick, hr_kick, base_x=40)
    cal = calibrate_camera(kick_px, kick_field, (720, 1280))

    pos = cal.extrinsics.position
    print(f"Camera: Cx={pos[0]:.1f}, Cy={pos[1]:.1f}, Cz={pos[2]:.1f}")
    print(f"Roll: {np.degrees(cal.initial_state.roll):.2f}°")
    print(f"Kickoff RMSE: {cal.calibration_error:.2f}px")

    # ── Step 2: Load mid-play frame ─────────────────────────────────────
    print("\n=== Mid-play frame ===")
    frame = cv2.imread(os.path.join(OUTPUT_DIR, "real_midplay_raw.jpg"))
    h, w = frame.shape[:2]

    yl = detect_yard_lines(frame)
    if yl is None:
        print("Yard line detection failed")
        return

    print(f"Detected {len(yl.lines)} yard lines, grid: {yl.grid}")

    hr = detect_hashes(frame, yl, yl.gray, yl.canny)
    if hr is None:
        print("Hash detection failed")
        return

    print(f"Hash pairs: {hr.n_confident_pairs}")

    # ── Step 3: ID the yard lines ───────────────────────────────────────
    # From visual inspection: "40" on left (near sideline) = NGS 50
    # "50" on right = NGS 60
    # Need to figure out which grid position corresponds to which yard line.
    min_grid = min(yl.grid)
    max_grid = max(yl.grid)

    # Print grid positions with x_mid so we can figure out mapping
    print("\nYard lines:")
    for i, (line, gpos) in enumerate(zip(yl.lines, yl.grid)):
        x_mid = (line[0] + line[2]) / 2
        print(f"  grid={gpos} x_mid={x_mid:.0f}px")

    # From the frame: "40" is at roughly x=350-400px, "50" is at roughly x=900-950px
    # The "40" near sideline = NGS 50, "50" near sideline = NGS 60
    # If the leftmost detected line (grid=min_grid) is at the 45 yard line (NGS 55)...
    # Actually let me just look at which grid position has x_mid near where "40" is painted
    # "40" bottom-left is around x=350. The nearest grid line to x=350 should be NGS 50.
    # "50" bottom-right is around x=900. That should be NGS 60.
    # Spacing between them: ~550px for 10 yards = 55px/yard
    # So grid spacing of ~148px = ~2.7 yards? No, that's 5 yards per grid unit.

    # Let me find which grid position is closest to x_mid=350 and x_mid=900
    x_mids = [(yl.lines[i][0] + yl.lines[i][2]) / 2 for i in range(len(yl.lines))]

    # The painted "40" is between two yard lines. It's at the 40-yard line (NGS 50).
    # The painted "50" is at the 50-yard line (NGS 60).
    # Find grid pos whose x_mid is closest to where "40" text appears (~380px)
    # and "50" text (~920px)

    # Actually, easier: the numbers are painted AT the 10-yard lines.
    # "40" is AT NGS 50. "50" is AT NGS 60.
    # I need to find which grid_pos has x_mid near 380 (for "40") and near 920 (for "50")

    # Manual ID from visual inspection:
    # grid 0 = "30" yard line = NGS 40
    # grid 1 = "35" = NGS 45
    # grid 2 = "40" = NGS 50
    # grid 3 = "45" = NGS 55
    # grid 4 = "50" = NGS 60 (not detected)
    # grid 5 = other "45" = NGS 65
    base_x = 40
    print(f"\nManual ID: grid {min_grid} → NGS {base_x}")
    for gpos in sorted(set(yl.grid)):
        print(f"  grid {gpos} → NGS {base_x + (gpos - min_grid) * 5}")

    # Get correspondences
    mid_px, mid_field = get_hash_correspondences(yl, hr, base_x=base_x)
    print(f"\nCorrespondences: {len(mid_px)} hash marks")

    if len(mid_px) < 2:
        print("Not enough correspondences")
        return

    # ── Step 4: Solve PTZ ───────────────────────────────────────────────
    print("\n=== PTZ Solve ===")
    state = solve_ptz(mid_px, mid_field, cal)

    if state is None:
        print("PTZ solve failed")
        return

    print(f"pan={state.pan:.4f} ({np.degrees(state.pan):.2f}°)")
    print(f"tilt={state.tilt:.4f} ({np.degrees(state.tilt):.2f}°)")
    print(f"f={state.focal_length:.1f}")

    # ── Step 5: Overlay + Rectify ───────────────────────────────────────
    H_result = camera_state_to_homography(state, cal,
                                           n_inliers=len(mid_px),
                                           n_correspondences=len(mid_px))

    # Overlay
    overlay = frame.copy()
    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        pts_field = np.column_stack([np.full_like(ys, x), ys])
        pts_pixel = project_field_to_pixel(pts_field, state, cal, apply_dist=False)
        for i in range(len(pts_pixel) - 1):
            pt1 = tuple(pts_pixel[i].astype(int))
            pt2 = tuple(pts_pixel[i + 1].astype(int))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        x_min, x_max = H_result.yard_range
        xs = np.linspace(x_min, x_max, 50)
        pts_field = np.column_stack([xs, np.full_like(xs, y)])
        pts_pixel = project_field_to_pixel(pts_field, state, cal, apply_dist=False)
        for i in range(len(pts_pixel) - 1):
            pt1 = tuple(pts_pixel[i].astype(int))
            pt2 = tuple(pts_pixel[i + 1].astype(int))
            cv2.line(overlay, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

    # Mark detected + projected hash points
    for i in range(len(mid_px)):
        pt = tuple(mid_px[i].astype(int))
        cv2.circle(overlay, pt, 5, (0, 0, 255), 2)
    projected = project_field_to_pixel(mid_field, state, cal, apply_dist=False)
    for i in range(len(projected)):
        pt = tuple(projected[i].astype(int))
        cv2.circle(overlay, pt, 3, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_midplay_overlay.jpg"), overlay)

    # Rectify (flip y so near sideline is at bottom)
    scale = 15
    x_min = max(0, H_result.yard_range[0] - 5)
    x_max = min(FIELD_LENGTH, H_result.yard_range[1] + 5)
    out_w = int((x_max - x_min) * scale)
    out_h = int(FIELD_WIDTH * scale)

    S_out = np.array([
        [scale, 0, -x_min * scale],
        [0, -scale, FIELD_WIDTH * scale],  # flip y
        [0, 0, 1],
    ])
    warp_mat = S_out @ H_result.H
    rectified = cv2.warpPerspective(frame, warp_mat, (out_w, out_h))

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_midplay_rectified.jpg"), rectified)

    print(f"\nSaved: real_midplay_overlay.jpg, real_midplay_rectified.jpg")


if __name__ == "__main__":
    main()
