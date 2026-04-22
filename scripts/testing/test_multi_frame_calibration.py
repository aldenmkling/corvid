#!/usr/bin/env python3
"""
Joint multi-frame camera calibration.

Shared parameters (solved once):
  - Camera position (Cx, Cy, Cz)
  - Roll
  - Distortion (k1, k2)

Per-frame parameters (solved for each frame):
  - Pan, tilt, focal length

Shared parameters constrained by BOTH frames simultaneously, which breaks
the single-frame degeneracy between (Cz, tilt, focal_length).
"""

import os
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.testing.test_grid_solver_camera import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines, line_fit_residuals,
)
from scripts.testing.test_yard_line_grouping import (
    run_hrnet, extract_peaks,
)
from src.homography.camera_model import (
    CameraIntrinsics, CameraExtrinsics, CameraCalibration, CameraState,
    rotation_matrix, camera_state_to_homography, project_field_to_pixel,
    undistort_points,
)
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30


def process_frame(frame_path, base_ngs_x, label):
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    groups, angle_deg = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, base_ngs_x)

    # Collect line groups for distortion (raw keypoints that should be collinear)
    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]

    print(f"[{label}] {os.path.basename(frame_path)}")
    print(f"  {len(hash_pxs)} hashes, {len(sideline_pxs)} sidelines, "
          f"tilt {angle_deg:.1f}°, {len(pixel_pts)} correspondences")

    return {
        "frame": frame,
        "shape": (h, w),
        "groups": groups,
        "pixel_pts": pixel_pts,
        "field_pts": field_pts,
        "labels": labels,
        "line_sets": line_sets,
    }


def joint_calibrate(frames, frame_shape, focal_guess, k1, k2,
                    initial_guess=None, bounds=None, verbose=True):
    """Jointly calibrate shared camera params + per-frame pan/tilt/focal.

    Points are pre-undistorted using the given (k1, k2).

    Variables:
      [Cx, Cy, Cz, roll, pan_1, tilt_1, f_1, pan_2, tilt_2, f_2, ...]
    """
    N = len(frames)

    # Pre-undistort all pixel points
    for fr in frames:
        h, w = fr["shape"]
        intr = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                                 cx=w / 2.0, cy=h / 2.0, k1=k1, k2=k2)
        fr["pixel_pts_u"] = undistort_points(fr["pixel_pts"], intr)

    # Initial guess
    if initial_guess is None:
        x0 = np.array([
            60.0,    # Cx
            -80.0,   # Cy
            40.0,    # Cz
            0.0,     # roll
        ] + sum(([0.0, -0.45, 1000.0] for _ in range(N)), []))
    else:
        x0 = np.asarray(initial_guess, dtype=np.float64)

    # Bounds — tighter physical constraints to avoid degenerate solutions.
    # NFL press box heights are typically 75-150 ft ≈ 25-50 yards.
    # Press box distance from near sideline is typically 20-80 yards.
    if bounds is None:
        lo = [20, -120, 20, -np.radians(10)] + sum(
            ([-np.pi / 2, -np.pi / 2, 200] for _ in range(N)), [])
        hi = [100, -15, 60, np.radians(10)] + sum(
            ([np.pi / 2, 0.0, 10000] for _ in range(N)), [])
    else:
        lo, hi = bounds

    def residuals(params):
        Cx, Cy, Cz, roll = params[:4]
        C = np.array([Cx, Cy, Cz])
        all_res = []
        for i, fr in enumerate(frames):
            pan = params[4 + 3 * i]
            tilt = params[4 + 3 * i + 1]
            f = params[4 + 3 * i + 2]
            R = rotation_matrix(pan, tilt, roll)
            t = -R @ C
            h, w = fr["shape"]
            cx, cy = w / 2.0, h / 2.0
            field_pts = fr["field_pts"]
            pix_u = fr["pixel_pts_u"]
            pts_3d = np.column_stack([field_pts, np.zeros(len(field_pts))])
            p_cam = (R @ pts_3d.T).T + t
            behind = p_cam[:, 2] <= 0
            if behind.any():
                all_res.extend([1000.0] * (2 * len(field_pts)))
                continue
            x_norm = p_cam[:, 0] / p_cam[:, 2]
            y_norm = p_cam[:, 1] / p_cam[:, 2]
            u_pred = f * x_norm + cx
            v_pred = f * y_norm + cy
            all_res.extend((u_pred - pix_u[:, 0]).tolist())
            all_res.extend((v_pred - pix_u[:, 1]).tolist())
        return np.array(all_res)

    result = least_squares(
        residuals, x0, bounds=(lo, hi),
        method="trf", loss="soft_l1", f_scale=5.0, max_nfev=10000,
    )

    params = result.x
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))

    Cx, Cy, Cz, roll = params[:4]
    shared = {
        "Cx": float(Cx), "Cy": float(Cy), "Cz": float(Cz),
        "roll": float(roll), "k1": float(k1), "k2": float(k2),
    }
    per_frame = []
    for i in range(N):
        pan = float(params[4 + 3 * i])
        tilt = float(params[4 + 3 * i + 1])
        f = float(params[4 + 3 * i + 2])
        per_frame.append({"pan": pan, "tilt": tilt, "f": f})

    if verbose:
        print(f"\n=== Joint calibration ===")
        print(f"Shared:")
        print(f"  Cx={shared['Cx']:.1f}, Cy={shared['Cy']:.1f}, Cz={shared['Cz']:.1f}")
        print(f"  roll={np.degrees(shared['roll']):.2f}°")
        print(f"  k1={shared['k1']:.5f}, k2={shared['k2']:.5f}")
        for i, pf in enumerate(per_frame):
            print(f"Frame {i}: pan={pf['pan']:.3f}, tilt={pf['tilt']:.3f}, f={pf['f']:.0f}")
        print(f"Overall RMSE: {rmse:.2f}px")

    return shared, per_frame, rmse


def report_per_frame_errors(frames, shared, per_frame):
    for i, fr in enumerate(frames):
        C = np.array([shared["Cx"], shared["Cy"], shared["Cz"]])
        roll = shared["roll"]
        pan = per_frame[i]["pan"]
        tilt = per_frame[i]["tilt"]
        f = per_frame[i]["f"]
        h, w = fr["shape"]
        cx, cy = w / 2.0, h / 2.0
        R = rotation_matrix(pan, tilt, roll)
        t = -R @ C
        pts_3d = np.column_stack([fr["field_pts"], np.zeros(len(fr["field_pts"]))])
        p_cam = (R @ pts_3d.T).T + t
        x_norm = p_cam[:, 0] / p_cam[:, 2]
        y_norm = p_cam[:, 1] / p_cam[:, 2]
        u_pred = f * x_norm + cx
        v_pred = f * y_norm + cy
        projected = np.column_stack([u_pred, v_pred])
        det = fr["pixel_pts_u"]
        errs = np.hypot(det[:, 0] - projected[:, 0], det[:, 1] - projected[:, 1])
        print(f"  Frame {i}: mean={errs.mean():.1f}px, median={np.median(errs):.1f}px, "
              f"max={errs.max():.1f}px")


def draw_overlay(fr, shared, per_frame_state, out_path):
    frame = fr["frame"]
    h, w = fr["shape"]
    focal_guess = float(max(h, w))
    k1, k2 = shared["k1"], shared["k2"]

    # Build fake calibration for projection
    intrinsics = CameraIntrinsics(fx=per_frame_state["f"], fy=per_frame_state["f"],
                                    cx=w / 2.0, cy=h / 2.0, k1=0, k2=0)
    extrinsics = CameraExtrinsics(
        position=np.array([shared["Cx"], shared["Cy"], shared["Cz"]])
    )
    state = CameraState(pan=per_frame_state["pan"], tilt=per_frame_state["tilt"],
                         focal_length=per_frame_state["f"], roll=shared["roll"])
    cal = CameraCalibration(intrinsics=intrinsics, extrinsics=extrinsics,
                              calibration_error=0.0, n_points_used=0,
                              initial_state=state)

    # Undistort frame
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        K = np.array([[focal_guess, 0, w / 2.0],
                      [0, focal_guess, h / 2.0],
                      [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        vis = cv2.undistort(frame, K, dist_coeffs).copy()
    else:
        vis = frame.copy()

    # Compute visible yard range
    H_result = camera_state_to_homography(state, cal)
    x_min = max(0, H_result.yard_range[0] - 5)
    x_max = min(FIELD_LENGTH, H_result.yard_range[1] + 5)

    # Project yard lines
    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        pf = np.column_stack([np.full_like(ys, x), ys])
        pp = project_field_to_pixel(pf, state, cal, apply_dist=False)
        for j in range(len(pp) - 1):
            cv2.line(vis, tuple(pp[j].astype(int)), tuple(pp[j+1].astype(int)),
                     (0, 255, 0), 1, cv2.LINE_AA)

    # Hash lines
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(x_min, x_max, 50)
        pf = np.column_stack([xs, np.full_like(xs, y)])
        pp = project_field_to_pixel(pf, state, cal, apply_dist=False)
        for j in range(len(pp) - 1):
            cv2.line(vis, tuple(pp[j].astype(int)), tuple(pp[j+1].astype(int)),
                     (0, 200, 200), 1, cv2.LINE_AA)

    # Correspondences
    for i in range(len(fr["pixel_pts_u"])):
        det = tuple(np.asarray(fr["pixel_pts_u"][i]).astype(int))
        proj = project_field_to_pixel(np.array([fr["field_pts"][i]]),
                                         state, cal, apply_dist=False)[0]
        proj = tuple(proj.astype(int))
        cv2.circle(vis, det, 5, (0, 0, 255), 2)
        cv2.circle(vis, proj, 3, (0, 255, 0), -1)
        cv2.line(vis, det, proj, (255, 255, 0), 1)

    cv2.imwrite(out_path, vis)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame1",
                        default=os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"))
    parser.add_argument("--ngs1", type=float, default=35.0)
    parser.add_argument("--frame2",
                        default=os.path.join(OUTPUT_DIR, "play002_first.jpg"))
    parser.add_argument("--ngs2", type=float, default=70.0)
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-FRAME CALIBRATION")
    print("=" * 60)
    fr1 = process_frame(args.frame1, args.ngs1, "F1")
    fr2 = process_frame(args.frame2, args.ngs2, "F2")

    if len(fr1["pixel_pts"]) < 4:
        print(f"Frame 1 has too few correspondences ({len(fr1['pixel_pts'])})")
        return
    if len(fr2["pixel_pts"]) < 4:
        print(f"Frame 2 has too few correspondences ({len(fr2['pixel_pts'])})")
        return

    # Shared distortion: use the frame with more correspondences (better line groups).
    # Pooling across frames is dicey when one frame has sparse lines.
    primary = fr1 if len(fr1["pixel_pts"]) >= len(fr2["pixel_pts"]) else fr2
    print(f"\nDistortion calibration from primary frame "
          f"({len(primary['line_sets'])} line groups, "
          f"{sum(len(s) for s in primary['line_sets'])} points)")
    h, w = fr1["shape"]
    focal_guess = float(max(h, w))
    k1, k2 = calibrate_distortion_from_lines(primary["line_sets"], (h, w), focal_guess)
    print(f"  k1={k1:.5f}, k2={k2:.5f}")
    # Sanity check — physical distortion should be small
    if abs(k1) > 1.0 or abs(k2) > 1.0:
        print(f"  WARNING: large distortion values, falling back to 0")
        k1, k2 = 0.0, 0.0

    # Joint calibration
    shared, per_frame, rmse = joint_calibrate(
        [fr1, fr2], (h, w), focal_guess, k1, k2,
    )

    print("\nPer-frame reprojection errors:")
    report_per_frame_errors([fr1, fr2], shared, per_frame)

    # Overlay each frame
    for i, fr in enumerate([fr1, fr2]):
        out_path = os.path.join(OUTPUT_DIR, f"multi_frame_f{i+1}_overlay.jpg")
        draw_overlay(fr, shared, per_frame[i], out_path)
        print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()
