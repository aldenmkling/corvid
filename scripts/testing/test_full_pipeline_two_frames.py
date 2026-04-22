#!/usr/bin/env python3
"""
End-to-end test: calibrate on frame 1, solve PTZ only on frame 2.

Validates that the camera model's fixed-camera assumption holds:
  - Camera position (Cx, Cy, Cz) and roll: fixed (from kickoff calibration)
  - Lens distortion (k1, k2): fixed (from kickoff plumb-line calibration)
  - Pan, tilt, focal length: solved per-frame (3 DOF instead of 7)

This is the whole point of the camera model — once calibrated, later frames
only need 2+ keypoints to solve, not 6+.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.testing.test_grid_solver_camera import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines, line_fit_residuals, draw_overlay,
)
from scripts.testing.test_yard_line_grouping import (
    run_hrnet, extract_peaks,
)
from src.homography.camera_model import (
    CameraIntrinsics, CameraExtrinsics, CameraCalibration, CameraState,
    calibrate_camera, solve_ptz, camera_state_to_homography,
    project_field_to_pixel, undistort_points,
)
from src.homography.field_model import FIELD_WIDTH, FIELD_LENGTH, HASH_Y_NEAR, HASH_Y_FAR

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30


def process_frame(frame_path, base_ngs_x, label):
    """Run HRNet + grid solver on a frame. Returns detections + correspondences."""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(frame_path)
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    groups, angle_deg = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, base_ngs_x)

    print(f"[{label}] {frame_path}")
    print(f"  {len(hash_pxs)} hashes, {len(sideline_pxs)} sidelines, "
          f"tilt {angle_deg:.1f}°, {len(pixel_pts)} correspondences")

    return frame, groups, pixel_pts, field_pts, labels, (h, w)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-frame",
                        default=os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"),
                        help="Frame used for full camera+distortion calibration")
    parser.add_argument("--reference-ngs-x", type=float, default=40.0)
    parser.add_argument("--target-frame",
                        default=os.path.join(OUTPUT_DIR, "real_midplay_raw.jpg"),
                        help="Frame to solve PTZ on using reference calibration")
    parser.add_argument("--target-ngs-x", type=float, default=40.0,
                        help="Manual grid-0 anchor for target frame")
    parser.add_argument("--out-prefix", default="two_frame")
    args = parser.parse_args()

    # ── Reference frame: full calibration ────────────────────────────────
    print("=" * 60)
    print("REFERENCE FRAME — full calibration")
    print("=" * 60)
    ref_frame, ref_groups, ref_px, ref_fp, ref_lbls, (rh, rw) = process_frame(
        args.reference_frame, args.reference_ngs_x, "REF",
    )
    if len(ref_px) < 6:
        print("Reference frame has too few correspondences")
        return

    # Distortion from collinearity
    side_row = [g['sideline'] for g in ref_groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in ref_groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in ref_groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]

    focal_guess = float(max(rh, rw))
    k1, k2 = calibrate_distortion_from_lines(line_sets, (rh, rw), focal_guess)
    print(f"\n  Distortion: k1={k1:.5f}, k2={k2:.5f}")

    # Camera calibration on undistorted points
    intr_dist = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                                  cx=rw / 2.0, cy=rh / 2.0, k1=k1, k2=k2)
    ref_px_u = undistort_points(ref_px, intr_dist)
    ref_cal = calibrate_camera(ref_px_u, ref_fp, (rh, rw))
    pos = ref_cal.extrinsics.position
    print(f"  Camera position: Cx={pos[0]:.1f}, Cy={pos[1]:.1f}, Cz={pos[2]:.1f}")
    print(f"  Roll: {np.degrees(ref_cal.initial_state.roll):.2f}°")
    print(f"  Reference pan={ref_cal.initial_state.pan:.3f}, "
          f"tilt={ref_cal.initial_state.tilt:.3f}, f={ref_cal.initial_state.focal_length:.0f}")
    print(f"  Reference RMSE: {ref_cal.calibration_error:.2f}px")

    # Attach distortion to the calibration's intrinsics so later use is consistent
    ref_cal.intrinsics.k1 = k1
    ref_cal.intrinsics.k2 = k2

    # ── Target frame: PTZ-only solve ─────────────────────────────────────
    print()
    print("=" * 60)
    print("TARGET FRAME — PTZ solve with fixed camera + distortion")
    print("=" * 60)
    tgt_frame, tgt_groups, tgt_px, tgt_fp, tgt_lbls, (th, tw) = process_frame(
        args.target_frame, args.target_ngs_x, "TGT",
    )
    if len(tgt_px) < 2:
        print("Target frame has too few correspondences")
        return

    # Apply the reference distortion to target points.
    # undistort_points internally uses the intrinsics' fx/cx/cy, so construct
    # with the same scale as we used for distortion estimation.
    intr_target_dist = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                                         cx=tw / 2.0, cy=th / 2.0,
                                         k1=k1, k2=k2)
    tgt_px_u = undistort_points(tgt_px, intr_target_dist)

    # Solve for pan, tilt, focal length only
    tgt_state = solve_ptz(tgt_px_u, tgt_fp, ref_cal, prev_state=None,
                           temporal_weight=0.0)
    if tgt_state is None:
        print("PTZ solve failed")
        return
    print(f"\n  Solved PTZ: pan={tgt_state.pan:.3f}, tilt={tgt_state.tilt:.3f}, "
          f"f={tgt_state.focal_length:.0f}")

    # Report per-point reprojection error on target
    tgt_projected = project_field_to_pixel(tgt_fp, tgt_state, ref_cal,
                                             apply_dist=False)
    errs = []
    for i in range(len(tgt_px_u)):
        e = float(np.hypot(tgt_px_u[i, 0] - tgt_projected[i, 0],
                           tgt_px_u[i, 1] - tgt_projected[i, 1]))
        errs.append((e, tgt_lbls[i], tgt_px_u[i], tgt_projected[i], tgt_fp[i]))
    errs.sort(reverse=True)
    print("\n  Worst 10 per-point errors (undistorted space):")
    for e, lab, det, pr, fp in errs[:10]:
        print(f"    {lab:12s} NGS({fp[0]:5.1f},{fp[1]:5.1f})  "
              f"det=({det[0]:6.0f},{det[1]:4.0f})  proj=({pr[0]:6.0f},{pr[1]:4.0f})  "
              f"err={e:.1f}px")
    all_errs = np.array([e[0] for e in errs])
    print(f"\n  Error stats: mean={all_errs.mean():.1f}px, "
          f"median={np.median(all_errs):.1f}px, max={all_errs.max():.1f}px")

    # ── Visualize target overlay ─────────────────────────────────────────
    H_result = camera_state_to_homography(tgt_state, ref_cal,
                                            n_inliers=len(tgt_px_u),
                                            n_correspondences=len(tgt_px))
    # Undistort the target frame for overlay
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        K = np.array([
            [focal_guess, 0, tw / 2.0],
            [0, focal_guess, th / 2.0],
            [0, 0, 1],
        ])
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        tgt_frame_undistorted = cv2.undistort(tgt_frame, K, dist_coeffs)
    else:
        tgt_frame_undistorted = tgt_frame

    vis = draw_overlay(tgt_frame_undistorted, ref_cal, tgt_state,
                        tgt_px_u, tgt_fp, tgt_lbls, H_result.yard_range)
    out_path = os.path.join(OUTPUT_DIR, f"{args.out_prefix}_target_overlay.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
