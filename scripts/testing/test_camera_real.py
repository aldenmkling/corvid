#!/usr/bin/env python3
"""
Test the camera model on a real kickoff frame using classical detection.

Extracts a kickoff frame from the Chiefs@Packers game, runs classical
yard line + hash detection, and attempts calibration.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.homography.classical.yard_lines import detect_yard_lines
from src.homography.classical.hash_marks import detect_hashes
from src.homography.camera_model import (
    calibrate_camera, solve_ptz, camera_state_to_homography,
    CameraState,
)
from src.homography.apply_homography import pixel_to_field
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR, GOAL_LINE_LEFT,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

GAME_ID = "2019102712"  # Chiefs @ Packers
CLIP_DIR = os.path.join(PROJECT_ROOT, "videos", "clips", GAME_ID)


def extract_frame(video_path, frame_idx=0):
    """Extract a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_hash_pixel_positions(yl_result, hash_result):
    """Extract pixel positions and t-values of detected hashes.

    Returns list of dicts with:
      - pixel_xy: (x, y) pixel position
      - grid_pos: relative grid position of the yard line
      - hash_type: 'far' or 'near'
      - t: position along the yard line (0=top, 1=bottom)
    """
    detections = []

    for li, entry in hash_result.hashes.items():
        line = yl_result.lines[li]
        gpos = yl_result.grid[li]
        x_top, y_top, x_bot, y_bot = line
        dx = x_bot - x_top
        dy = y_bot - y_top

        for hash_type in ['far', 'near']:
            if hash_type in entry:
                t = entry[hash_type]
                px = x_top + t * dx
                py = y_top + t * dy
                detections.append({
                    'pixel_xy': (px, py),
                    'grid_pos': gpos,
                    'hash_type': hash_type,
                    't': t,
                })

    return detections


def assign_yard_line_identity(yl_result, hash_result, frame):
    """Try to assign absolute yard line identity to detected grid positions.

    For a kickoff frame, the camera is roughly centered. We use the fact
    that yard lines are at known x-coordinates (10, 15, 20, ..., 110)
    and try different offset mappings to find the best one.

    Returns: dict mapping grid_pos -> field x-coordinate, or None
    """
    grid_positions = sorted(set(yl_result.grid))
    n_lines = len(grid_positions)

    if n_lines < 3:
        return None

    # The grid positions are relative (0, 1, 2, ...) with each step = 5 yards
    # We need to find which YARD_LINE_POSITIONS[i] corresponds to grid_pos=0

    # Try all possible offsets
    best_offset = None
    best_score = -1

    min_grid = min(grid_positions)
    max_grid = max(grid_positions)

    for start_idx in range(len(YARD_LINE_POSITIONS)):
        # grid_pos=min_grid maps to YARD_LINE_POSITIONS[start_idx]
        # Check if all grid positions have valid yard lines
        valid = True
        for gp in grid_positions:
            yl_idx = start_idx + (gp - min_grid)
            if yl_idx < 0 or yl_idx >= len(YARD_LINE_POSITIONS):
                valid = False
                break

        if valid:
            # Score by how centered this makes the view
            # (kickoff should be roughly centered)
            center_x = YARD_LINE_POSITIONS[start_idx + (max_grid - min_grid) // 2]
            # Prefer mappings that put center near midfield (x=60)
            score = 1.0 / (1.0 + abs(center_x - 60.0))

            if score > best_score:
                best_score = score
                best_offset = start_idx

    if best_offset is None:
        return None

    mapping = {}
    for gp in grid_positions:
        yl_idx = best_offset + (gp - min_grid)
        if 0 <= yl_idx < len(YARD_LINE_POSITIONS):
            mapping[gp] = YARD_LINE_POSITIONS[yl_idx]

    return mapping


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Extract kickoff frame ───────────────────────────────────────────
    sideline_path = os.path.join(CLIP_DIR, "play_001", "sideline.mp4")
    if not os.path.exists(sideline_path):
        print(f"Video not found: {sideline_path}")
        return

    # Frame 0 = start of kickoff, should be zoomed out
    frame = extract_frame(sideline_path, frame_idx=0)
    if frame is None:
        print("Failed to extract frame")
        return

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # Save raw frame
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"), frame)

    # ── Classical detection ─────────────────────────────────────────────
    print("\n=== Classical Detection ===")

    yl_result = detect_yard_lines(frame)
    if yl_result is None:
        print("Yard line detection failed")
        return

    print(f"Detected {len(yl_result.lines)} yard lines")
    print(f"Grid positions: {yl_result.grid}")
    print(f"Period: {yl_result.period_px:.1f} px")

    hash_result = detect_hashes(frame, yl_result, yl_result.gray, yl_result.canny)
    if hash_result is None:
        print("Hash detection failed")
        # Still try with just sideline intersections from yard lines
        hash_detections = []
    else:
        print(f"Hash marks: {hash_result.n_confident_pairs} confident pairs")
        print(f"Far median t: {hash_result.far_median:.3f}")
        print(f"Near median t: {hash_result.near_median:.3f}")
        hash_detections = get_hash_pixel_positions(yl_result, hash_result)
        print(f"Total hash detections: {len(hash_detections)}")

    # ── Assign absolute identity ────────────────────────────────────────
    print("\n=== Identity Assignment ===")

    # Try auto-identity first, then allow manual override
    id_mapping = assign_yard_line_identity(yl_result, hash_result, frame)

    # For the Chiefs@Packers kickoff: numbers read 30, 40, 50, 40, 30
    # The visible painted numbers (30 on left) = NGS x=40
    # Grid spacing is 5 yards, so grid_pos 0 should map to the leftmost
    # detected yard line. Looking at the frame, the leftmost line visible
    # is around the 25-yard line (NGS x=35).
    # With 8 lines at grid [0,1,2,3,4,6,7,8] and period ~148px:
    #   grid 0 = ~35, grid 2 = ~45 (painted "35" yard), etc.
    # Let's use the painted numbers to anchor:
    #   "30" on left is at NGS x=40, "50" in center is at NGS x=60
    # If grid pos 1 = NGS 40 (the 30-yard line from left goal):
    #   grid 0 = 35, grid 1 = 40, grid 2 = 45, grid 3 = 50,
    #   grid 4 = 55, grid 5 = 60, grid 6 = 65, grid 7 = 70, grid 8 = 75
    # Check: "40" on left = NGS 50 (grid 3) ✓, "50" center = NGS 60 (grid 5) ✓
    # But grid 5 is missing from detections... let's check.

    # Actually, the auto-identity might work if we verify against the frame.
    # Let's print both and use the correct one.
    if id_mapping:
        print("Auto-identity mapping:")
        for gp, field_x in sorted(id_mapping.items()):
            print(f"  grid {gp} → {field_x:.0f} yard line")

    # Manual override based on reading the numbers in the frame:
    # Grid positions: [0, 1, 2, 3, 4, 6, 7, 8]
    # From frame: leftmost line appears to be the 35 yard line (NGS x=35)
    min_grid = min(yl_result.grid)
    manual_mapping = {}
    # grid 0 → NGS 35, grid 1 → NGS 40 ("30" painted), etc.
    base_x = 40  # NGS x for grid position 0
    for gp in set(yl_result.grid):
        manual_mapping[gp] = base_x + (gp - min_grid) * 5

    print("\nManual identity mapping (from reading numbers):")
    for gp, field_x in sorted(manual_mapping.items()):
        print(f"  grid {gp} → {field_x:.0f} yard line")

    id_mapping = manual_mapping

    # ── Build correspondences ───────────────────────────────────────────
    # Only use hash intersections — yard line endpoints DON'T reach the
    # actual sidelines, so mapping them to y=0/53.33 is wrong.
    pixel_pts = []
    field_pts = []

    for det in hash_detections:
        gpos = det['grid_pos']
        if gpos not in id_mapping:
            continue
        field_x = id_mapping[gpos]
        field_y = HASH_Y_FAR if det['hash_type'] == 'far' else HASH_Y_NEAR
        pixel_pts.append(list(det['pixel_xy']))
        field_pts.append([field_x, field_y])

    pixel_pts = np.array(pixel_pts)
    field_pts = np.array(field_pts)
    print(f"\nTotal correspondences: {len(pixel_pts)} (hash marks only)")

    # ── Draw detections on frame ────────────────────────────────────────
    vis = frame.copy()
    for i in range(len(pixel_pts)):
        pt = tuple(pixel_pts[i].astype(int))
        fy = field_pts[i, 1]
        if fy == 0.0 or fy == FIELD_WIDTH:
            color = (0, 0, 255)  # red = sideline
        else:
            color = (255, 0, 0)  # blue = hash
        cv2.circle(vis, pt, 4, color, -1)
        label = f"{field_pts[i, 0]:.0f}"
        cv2.putText(vis, label, (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_kickoff_detections.jpg"), vis)

    # ── Calibrate ───────────────────────────────────────────────────────
    print("\n=== Calibration ===")

    if len(pixel_pts) < 6:
        print(f"Not enough correspondences ({len(pixel_pts)}), need 6+")
        return

    cal = calibrate_camera(pixel_pts, field_pts, (h, w))

    pos = cal.extrinsics.position
    print(f"Camera position: Cx={pos[0]:.1f}, Cy={pos[1]:.1f}, Cz={pos[2]:.1f}")
    print(f"Initial state: pan={cal.initial_state.pan:.4f}, tilt={cal.initial_state.tilt:.4f}, f={cal.initial_state.focal_length:.1f}, roll={np.degrees(cal.initial_state.roll):.3f}°")
    print(f"Reprojection RMSE: {cal.calibration_error:.2f} pixels")

    # ── Rectify ─────────────────────────────────────────────────────────
    print("\n=== Rectification ===")

    H_result = camera_state_to_homography(cal.initial_state, cal)
    print(f"Visible yard range: {H_result.yard_range[0]:.1f} to {H_result.yard_range[1]:.1f}")

    # Rectify
    scale = 15
    x_min = max(0, H_result.yard_range[0] - 5)
    x_max = min(FIELD_LENGTH, H_result.yard_range[1] + 5)
    out_w = int((x_max - x_min) * scale)
    out_h = int(FIELD_WIDTH * scale)

    S_out = np.array([
        [scale, 0, -x_min * scale],
        [0, scale, 0],
        [0, 0, 1],
    ])
    warp_mat = S_out @ H_result.H
    rectified = cv2.warpPerspective(frame, warp_mat, (out_w, out_h))

    # Draw yard line grid
    for x in YARD_LINE_POSITIONS:
        if x_min <= x <= x_max:
            ox = int((x - x_min) * scale)
            cv2.line(rectified, (ox, 0), (ox, out_h), (0, 0, 200), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_kickoff_rectified.jpg"), rectified)

    # ── Overlay projected grid on original ──────────────────────────────
    overlay = frame.copy()
    from src.homography.camera_model import project_field_to_pixel

    # Draw projected yard lines (y=0 to y=53.33)
    for x in YARD_LINE_POSITIONS:
        # Sample many points along the line for accuracy
        ys = np.linspace(0, FIELD_WIDTH, 20)
        pts_field = np.column_stack([np.full_like(ys, x), ys])
        pts_pixel = project_field_to_pixel(pts_field, cal.initial_state, cal, apply_dist=False)
        for i in range(len(pts_pixel) - 1):
            pt1 = tuple(pts_pixel[i].astype(int))
            pt2 = tuple(pts_pixel[i + 1].astype(int))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw projected hash lines
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(x_min, x_max, 50)
        pts_field = np.column_stack([xs, np.full_like(xs, y)])
        pts_pixel = project_field_to_pixel(pts_field, cal.initial_state, cal, apply_dist=False)
        for i in range(len(pts_pixel) - 1):
            pt1 = tuple(pts_pixel[i].astype(int))
            pt2 = tuple(pts_pixel[i + 1].astype(int))
            cv2.line(overlay, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)

    # Mark the detected hash points (red circles)
    for i in range(len(pixel_pts)):
        pt = tuple(pixel_pts[i].astype(int))
        cv2.circle(overlay, pt, 5, (0, 0, 255), 2)

    # Mark the projected hash points (green circles) - should overlap red
    projected_hashes = project_field_to_pixel(field_pts, cal.initial_state, cal, apply_dist=False)
    for i in range(len(projected_hashes)):
        pt = tuple(projected_hashes[i].astype(int))
        cv2.circle(overlay, pt, 3, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_kickoff_overlay.jpg"), overlay)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print("  real_kickoff_raw.jpg — original frame")
    print("  real_kickoff_detections.jpg — with detected keypoints")
    print("  real_kickoff_overlay.jpg — with projected field grid (green)")
    print("  real_kickoff_rectified.jpg — top-down rectified view")


if __name__ == "__main__":
    main()
