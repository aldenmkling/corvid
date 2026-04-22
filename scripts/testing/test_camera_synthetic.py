#!/usr/bin/env python3
"""
Test the camera model on synthetic frames with known ground truth.

Generates synthetic field images from a pinhole camera model with known
parameters, then tests calibration recovery and per-frame PTZ solving.
Produces rectified (warped-to-top-down) images for visual inspection.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.homography.camera_model import (
    CameraIntrinsics, CameraExtrinsics, CameraCalibration, CameraState,
    rotation_matrix, project_field_to_pixel, apply_distortion,
    calibrate_camera, solve_ptz, camera_state_to_homography,
)
from src.homography.apply_homography import pixel_to_field
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR, GOAL_LINE_LEFT, GOAL_LINE_RIGHT,
    NUMBER_Y_NEAR, NUMBER_Y_FAR, TEN_YARD_POSITIONS,
    ngs_x_to_field_number,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

# ── Ground truth camera parameters ──────────────────────────────────────────

TRUE_CX = 58.0      # slightly off midfield
TRUE_CY = -45.0     # 45 yards behind near sideline
TRUE_CZ = 33.0      # ~100 feet up
TRUE_K1 = 0.0       # no distortion
TRUE_K2 = 0.0

FRAME_W, FRAME_H = 1280, 720


def make_true_calibration():
    """Create the ground truth calibration (what we're trying to recover)."""
    intrinsics = CameraIntrinsics(
        fx=800.0, fy=800.0, cx=FRAME_W / 2, cy=FRAME_H / 2,
        k1=TRUE_K1, k2=TRUE_K2,
    )
    extrinsics = CameraExtrinsics(position=np.array([TRUE_CX, TRUE_CY, TRUE_CZ]))
    initial_state = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)
    return CameraCalibration(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        calibration_error=0.0,
        n_points_used=0,
        initial_state=initial_state,
    )


# ── Synthetic frame rendering ──────────────────────────────────────────────

def generate_field_correspondences():
    """Generate all identifiable field points with their coordinates."""
    points = []
    for x in YARD_LINE_POSITIONS:
        is_goal = (x == GOAL_LINE_LEFT or x == GOAL_LINE_RIGHT)
        points.append({"field_xy": (x, 0.0), "type": "near_sideline"})
        points.append({"field_xy": (x, FIELD_WIDTH), "type": "far_sideline"})
        if not is_goal:
            points.append({"field_xy": (x, HASH_Y_NEAR), "type": "near_hash"})
            points.append({"field_xy": (x, HASH_Y_FAR), "type": "far_hash"})
    return points


def render_synthetic_frame(state, cal, label=""):
    """Render a synthetic field image from the given camera state.

    Draws yard lines, hash marks, sidelines, and numbers as they would
    appear through the camera.
    """
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[:] = (40, 100, 40)  # dark green field

    # Helper: project field points to pixel, respecting distortion
    def proj(field_pts):
        pts = np.array(field_pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        return project_field_to_pixel(pts, state, cal, apply_dist=True)

    def in_frame(px):
        return (-200 < px[0] < FRAME_W + 200) and (-200 < px[1] < FRAME_H + 200)

    def draw_line(p1_field, p2_field, color=(255, 255, 255), thickness=1):
        """Draw a field line by sampling points along it."""
        n_samples = 50
        ts = np.linspace(0, 1, n_samples)
        p1 = np.array(p1_field)
        p2 = np.array(p2_field)
        field_pts = np.array([p1 + t * (p2 - p1) for t in ts])
        pixel_pts = proj(field_pts)

        for i in range(len(pixel_pts) - 1):
            pt1 = tuple(pixel_pts[i].astype(int))
            pt2 = tuple(pixel_pts[i + 1].astype(int))
            if in_frame(pt1) and in_frame(pt2):
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw sidelines
    draw_line([0, 0], [FIELD_LENGTH, 0], thickness=2)
    draw_line([0, FIELD_WIDTH], [FIELD_LENGTH, FIELD_WIDTH], thickness=2)

    # Draw end lines
    draw_line([0, 0], [0, FIELD_WIDTH], thickness=2)
    draw_line([FIELD_LENGTH, 0], [FIELD_LENGTH, FIELD_WIDTH], thickness=2)

    # Draw yard lines
    for x in YARD_LINE_POSITIONS:
        draw_line([x, 0], [x, FIELD_WIDTH], thickness=1)

    # Draw hash marks (small ticks)
    for x in YARD_LINE_POSITIONS:
        if x == GOAL_LINE_LEFT or x == GOAL_LINE_RIGHT:
            continue
        # Near hash
        draw_line([x, HASH_Y_NEAR - 0.5], [x, HASH_Y_NEAR + 0.5], thickness=2)
        # Far hash
        draw_line([x, HASH_Y_FAR - 0.5], [x, HASH_Y_FAR + 0.5], thickness=2)

    # Draw numbers
    for x in TEN_YARD_POSITIONS:
        num = ngs_x_to_field_number(x)
        if num <= 0:
            continue
        text = str(num)
        for y_num in [NUMBER_Y_NEAR, NUMBER_Y_FAR]:
            px = proj([[x, y_num]])
            if in_frame(px[0]):
                pt = tuple(px[0].astype(int))
                cv2.putText(img, text, (pt[0] - 10, pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw keypoint markers (small circles at intersections)
    all_pts = generate_field_correspondences()
    for p in all_pts:
        px = proj([p["field_xy"]])
        if in_frame(px[0]):
            pt = tuple(px[0].astype(int))
            color = (0, 0, 255) if "sideline" in p["type"] else (255, 0, 0)
            cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)

    # Label
    if label:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img


def undistort_image(frame, intrinsics):
    """Remove radial distortion from an image using OpenCV."""
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ])
    dist_coeffs = np.array([intrinsics.k1, intrinsics.k2, 0, 0, 0])
    return cv2.undistort(frame, K, dist_coeffs)


def rectify_frame(frame, H_result, intrinsics=None, yard_range=None):
    """Warp a frame to top-down field view using the homography.

    If intrinsics are provided, undistorts the image first (H operates
    in undistorted pixel space). Crops to the visible yard range.
    """
    if intrinsics is not None and (abs(intrinsics.k1) > 1e-8 or abs(intrinsics.k2) > 1e-8):
        frame = undistort_image(frame, intrinsics)

    scale = 15  # pixels per yard

    # Crop to visible range (or full field)
    if yard_range is not None:
        x_min = max(0, yard_range[0] - 5)
        x_max = min(FIELD_LENGTH, yard_range[1] + 5)
    else:
        x_min, x_max = 0, FIELD_LENGTH

    out_w = int((x_max - x_min) * scale)
    out_h = int(FIELD_WIDTH * scale)

    # S maps output pixel to field coords, with offset for x_min
    S = np.array([
        [1.0 / scale, 0, x_min],
        [0, 1.0 / scale, 0],
        [0, 0, 1],
    ])

    # Scale matrix: maps field coords to output pixels
    # S_out: field (x,y) → output pixel (ox, oy) = ((x - x_min)*scale, y*scale)
    S_out = np.array([
        [scale, 0, -x_min * scale],
        [0, scale, 0],
        [0, 0, 1],
    ])

    # warp_mat maps input pixel → output pixel: S_out @ H @ pixel → output
    warp_mat = S_out @ H_result.H

    rectified = cv2.warpPerspective(frame, warp_mat, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

    # Draw yard line labels on the rectified image
    for x in YARD_LINE_POSITIONS:
        if x_min <= x <= x_max:
            ox = int((x - x_min) * scale)
            cv2.line(rectified, (ox, 0), (ox, out_h), (100, 100, 100), 1)

    return rectified


# ── Test scenarios ──────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "kickoff_wide",
        "pan": 0.02,       # almost centered
        "tilt": -0.55,
        "focal_length": 550.0,  # zoomed out
        "description": "Kickoff - wide shot, nearly full field visible",
    },
    {
        "name": "midfield_play",
        "pan": 0.05,
        "tilt": -0.58,
        "focal_length": 800.0,
        "description": "Mid-play around the 50, moderate zoom",
    },
    {
        "name": "redzone_tight",
        "pan": -0.25,
        "tilt": -0.50,
        "focal_length": 1100.0,
        "description": "Red zone, tighter zoom, panned left",
    },
    {
        "name": "zoomed_in",
        "pan": 0.15,
        "tilt": -0.45,
        "focal_length": 1400.0,
        "description": "Very tight zoom, few features visible",
    },
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cal_true = make_true_calibration()

    print("=" * 70)
    print("CAMERA MODEL SYNTHETIC TEST")
    print("=" * 70)
    print(f"\nGround truth camera:")
    print(f"  Position: Cx={TRUE_CX}, Cy={TRUE_CY}, Cz={TRUE_CZ}")
    print()

    # ── Generate synthetic frames ───────────────────────────────────────
    print("Generating synthetic frames...")
    frames = {}
    for sc in SCENARIOS:
        state = CameraState(pan=sc["pan"], tilt=sc["tilt"], focal_length=sc["focal_length"])
        img = render_synthetic_frame(state, cal_true, label=sc["description"])
        path = os.path.join(OUTPUT_DIR, f"{sc['name']}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames[sc["name"]] = {"image": img, "state": state, "scenario": sc}
        print(f"  {sc['name']}: pan={sc['pan']:.2f}, tilt={sc['tilt']:.2f}, f={sc['focal_length']:.0f}")

    # ── Calibration from kickoff frame ──────────────────────────────────
    print("\n" + "=" * 70)
    print("CALIBRATION (from kickoff frame)")
    print("=" * 70)

    kickoff_state = frames["kickoff_wide"]["state"]

    # Generate all correspondences and project them
    all_corr = generate_field_correspondences()
    field_pts_all = np.array([p["field_xy"] for p in all_corr])
    pixel_pts_all = project_field_to_pixel(field_pts_all, kickoff_state, cal_true, apply_dist=True)

    # Filter to visible in frame (with margin)
    visible = (
        (pixel_pts_all[:, 0] >= 10) & (pixel_pts_all[:, 0] < FRAME_W - 10) &
        (pixel_pts_all[:, 1] >= 10) & (pixel_pts_all[:, 1] < FRAME_H - 10)
    )
    field_pts_vis = field_pts_all[visible]
    pixel_pts_vis = pixel_pts_all[visible]
    print(f"\nVisible keypoints in kickoff frame: {len(field_pts_vis)}")

    # Add a tiny bit of noise (sub-pixel)
    np.random.seed(42)
    pixel_pts_noisy = pixel_pts_vis + np.random.randn(*pixel_pts_vis.shape) * 0.5

    # Calibrate
    cal_solved = calibrate_camera(pixel_pts_noisy, field_pts_vis, (FRAME_H, FRAME_W))

    print(f"\nCalibration results vs ground truth:")
    print(f"  {'Parameter':<15} {'True':>10} {'Solved':>10} {'Error':>10}")
    print(f"  {'-'*45}")

    pos = cal_solved.extrinsics.position
    params = [
        ("Cx (yards)", TRUE_CX, pos[0]),
        ("Cy (yards)", TRUE_CY, pos[1]),
        ("Cz (yards)", TRUE_CZ, pos[2]),
        ("pan (rad)", kickoff_state.pan, cal_solved.initial_state.pan),
        ("tilt (rad)", kickoff_state.tilt, cal_solved.initial_state.tilt),
        ("f (pixels)", kickoff_state.focal_length, cal_solved.initial_state.focal_length),
    ]
    for name, true_val, solved_val in params:
        err = solved_val - true_val
        print(f"  {name:<15} {true_val:>10.4f} {solved_val:>10.4f} {err:>+10.4f}")

    print(f"\n  Reprojection RMSE: {cal_solved.calibration_error:.3f} pixels")

    # ── Per-frame PTZ solving ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-FRAME PTZ SOLVING")
    print("=" * 70)

    for sc in SCENARIOS:
        name = sc["name"]
        true_state = frames[name]["state"]

        # Generate correspondences for this frame
        pixel_pts_sc = project_field_to_pixel(field_pts_all, true_state, cal_true, apply_dist=True)
        vis = (
            (pixel_pts_sc[:, 0] >= 10) & (pixel_pts_sc[:, 0] < FRAME_W - 10) &
            (pixel_pts_sc[:, 1] >= 10) & (pixel_pts_sc[:, 1] < FRAME_H - 10)
        )
        fp = field_pts_all[vis]
        pp = pixel_pts_sc[vis]

        print(f"\n  {name} ({len(fp)} visible keypoints):")

        if len(fp) < 2:
            print(f"    SKIP — not enough visible keypoints")
            continue

        # Solve PTZ using the SOLVED calibration (not ground truth)
        solved_state = solve_ptz(pp, fp, cal_solved)

        if solved_state is None:
            print(f"    FAILED — solver returned None")
            continue

        pan_err = abs(solved_state.pan - true_state.pan)
        tilt_err = abs(solved_state.tilt - true_state.tilt)
        f_err = abs(solved_state.focal_length - true_state.focal_length)

        print(f"    pan:  true={true_state.pan:.4f}  solved={solved_state.pan:.4f}  err={pan_err:.4f} rad ({np.degrees(pan_err):.2f}°)")
        print(f"    tilt: true={true_state.tilt:.4f}  solved={solved_state.tilt:.4f}  err={tilt_err:.4f} rad ({np.degrees(tilt_err):.2f}°)")
        print(f"    f:    true={true_state.focal_length:.1f}  solved={solved_state.focal_length:.1f}  err={f_err:.1f} px")

        # Generate rectified image using solved state + solved calibration
        H_result = camera_state_to_homography(solved_state, cal_solved)

        # Compute position error on a grid of test points
        test_field = np.array([[50, 10], [60, 26], [70, 40], [40, 5], [80, 50]])
        test_pixel = project_field_to_pixel(test_field, true_state, cal_true, apply_dist=True)
        recovered_field = pixel_to_field(test_pixel, H_result.H)
        pos_errors = np.sqrt(np.sum((recovered_field - test_field)**2, axis=1))
        print(f"    Position RMSE: {np.sqrt(np.mean(pos_errors**2)):.3f} yards (max: {pos_errors.max():.3f})")

        # Save rectified image
        rectified = rectify_frame(frames[name]["image"], H_result, cal_solved.intrinsics,
                                  yard_range=H_result.yard_range)
        rect_path = os.path.join(OUTPUT_DIR, f"{name}_rectified.jpg")
        cv2.imwrite(rect_path, rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save original with projected grid overlay
        overlay = frames[name]["image"].copy()
        # Draw projected yard lines using solved H
        for x in YARD_LINE_POSITIONS:
            for y_pair in [(0, FIELD_WIDTH)]:
                fp1 = np.array([[x, y_pair[0]]])
                fp2 = np.array([[x, y_pair[1]]])
                pp1 = project_field_to_pixel(fp1, solved_state, cal_solved, apply_dist=True)
                pp2 = project_field_to_pixel(fp2, solved_state, cal_solved, apply_dist=True)
                pt1 = tuple(pp1[0].astype(int))
                pt2 = tuple(pp2[0].astype(int))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        overlay_path = os.path.join(OUTPUT_DIR, f"{name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── Test with only 2 keypoints ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("MINIMAL KEYPOINTS TEST (2 points only)")
    print("=" * 70)

    true_state_min = CameraState(pan=0.1, tilt=-0.55, focal_length=900.0)
    # Pick just 2 points: one hash near, one hash far
    min_field = np.array([[55.0, HASH_Y_NEAR], [65.0, HASH_Y_FAR]])
    min_pixel = project_field_to_pixel(min_field, true_state_min, cal_true, apply_dist=True)

    # Use previous state as prior (close but not exact)
    prev = CameraState(pan=0.08, tilt=-0.53, focal_length=880.0)
    solved_min = solve_ptz(min_pixel, min_field, cal_solved, prev_state=prev)

    if solved_min:
        print(f"  pan:  true={true_state_min.pan:.4f}  solved={solved_min.pan:.4f}")
        print(f"  tilt: true={true_state_min.tilt:.4f}  solved={solved_min.tilt:.4f}")
        print(f"  f:    true={true_state_min.focal_length:.1f}  solved={solved_min.focal_length:.1f}")

        H_min = camera_state_to_homography(solved_min, cal_solved)
        test_pixel_min = project_field_to_pixel(
            np.array([[55, 10], [60, 26], [65, 40]]),
            true_state_min, cal_true, apply_dist=True,
        )
        recovered_min = pixel_to_field(test_pixel_min, H_min.H)
        errs = np.sqrt(np.sum((recovered_min - np.array([[55, 10], [60, 26], [65, 40]]))**2, axis=1))
        print(f"  Position RMSE: {np.sqrt(np.mean(errs**2)):.3f} yards")
    else:
        print("  FAILED — solver returned None")

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  *_original.jpg — synthetic frames")
    print("  *_overlay.jpg — with projected grid from solved calibration (green)")
    print("  *_rectified.jpg — warped to top-down field view")


if __name__ == "__main__":
    main()
