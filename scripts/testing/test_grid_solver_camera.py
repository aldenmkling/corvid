#!/usr/bin/env python3
"""
End-to-end test: HRNet + Yard Line Grid Solver + Camera Model.

1. Detect hashes and sidelines with HRNet (per-channel thresholds).
2. Run the Grid Solver: pair hashes, estimate yard-line tilt, match sidelines
   along the yard lines, assign relative grid positions via horizontal spacing.
3. Anchor the grid to absolute NGS yard-line positions (manual anchor for now
   — one yard-line identification needed, e.g. from number recognition).
4. Build pixel↔field correspondences and feed to the camera model calibration.
5. Visualize: detected vs projected keypoints, yard-line overlay.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Reuse the grid solver logic from the yard-line grouping test
from scripts.testing.test_yard_line_grouping import (
    run_hrnet, extract_peaks, split_hash_rows, pair_hashes,
    find_sideline_on_yard_line, assign_grid_positions,
    compute_hash_pca, _row_coord, yardline_tilt_slope_from_pairs,
)
from src.homography.camera_model import (
    calibrate_camera, camera_state_to_homography, project_field_to_pixel,
    CameraIntrinsics, undistort_points,
)
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)
from scipy.optimize import minimize

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30


def build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs):
    """Run the grid solver: hash pairing + sideline matching + grid positions.

    Returns list of yard-line dicts with 'grid_pos', 'far_hash', 'near_hash',
    'sideline', 'sideline_perp_dist', 'singleton'.
    """
    far_hashes, near_hashes = split_hash_rows(hash_pxs)
    pairs, unpaired_far, unpaired_near, angle_deg, angle_std = pair_hashes(
        far_hashes, near_hashes,
    )

    groups = []
    used_sideline = set()
    for fh, nh in pairs:
        fh = np.asarray(fh)
        nh = np.asarray(nh)
        sl_idx, sl_perp = find_sideline_on_yard_line(
            nh, fh, sideline_pxs, max_perp_distance=12,
        )
        sideline_pt = None
        sideline_conf = None
        if sl_idx is not None and sl_idx not in used_sideline:
            used_sideline.add(sl_idx)
            sideline_pt = sideline_pxs[sl_idx].tolist()
            sideline_conf = float(sideline_confs[sl_idx])
        groups.append({
            'far_hash': fh.tolist(),
            'near_hash': nh.tolist(),
            'sideline': sideline_pt,
            'sideline_conf': sideline_conf,
            'singleton': False,
        })

    # Singleton hashes
    for fh in unpaired_far:
        groups.append({
            'far_hash': np.asarray(fh).tolist(),
            'near_hash': None,
            'sideline': None,
            'sideline_conf': None,
            'singleton': True,
        })
    for nh in unpaired_near:
        groups.append({
            'far_hash': None,
            'near_hash': np.asarray(nh).tolist(),
            'sideline': None,
            'sideline_conf': None,
            'singleton': True,
        })

    # Note: singleton sidelines intentionally not emitted — HRNet sideline
    # detections are too clustered to reliably assign to yard-line columns
    # without a paired-hash confirmation.
    assign_grid_positions(groups)
    return groups, angle_deg


def line_fit_residuals(points):
    """Fit a line through points via SVD, return signed perpendicular distances."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts))
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return centered @ normal


def calibrate_distortion_from_lines(line_point_sets, frame_shape,
                                     focal_length_guess=None):
    """Estimate radial distortion coefficients (k1, k2) via plumb-line fit.

    Each entry in line_point_sets is a (N, 2) array of pixel points that SHOULD
    be collinear in undistorted space (e.g. a sideline, a hash row). We pick
    (k1, k2) that minimize the total sum of squared perpendicular distances
    of each point to the best-fit line through its group after undistortion.

    focal_length_guess sets the normalization scale. It doesn't have to be the
    true focal length — we just need a consistent scale for the distortion
    model. Default: max(image dimensions).
    """
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    if focal_length_guess is None:
        focal_length_guess = float(max(w, h))

    usable_sets = [np.asarray(pts, dtype=np.float64) for pts in line_point_sets
                   if len(pts) >= 3]
    if not usable_sets:
        return 0.0, 0.0

    def cost(params):
        k1, k2 = params
        intr = CameraIntrinsics(fx=focal_length_guess, fy=focal_length_guess,
                                 cx=cx, cy=cy, k1=k1, k2=k2)
        total = 0.0
        for pts in usable_sets:
            u = undistort_points(pts, intr)
            r = line_fit_residuals(u)
            total += float(np.sum(r ** 2))
        return total

    # Nelder-Mead over (k1, k2). Start at 0 (no distortion).
    result = minimize(
        cost, x0=np.array([0.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 400},
    )
    return float(result.x[0]), float(result.x[1])


def groups_to_correspondences(groups, base_ngs_x, frame_shape=None):
    """Convert yard-line groups to (pixel, field) correspondence pairs.

    base_ngs_x = NGS x-coordinate of grid_pos = 0 (the leftmost detected line).

    Keeps:
      - All paired groups (both hashes + any matched sideline).
      - Singleton hashes that sit on the grid within tolerance (`grid_fit_ok`).
      - Singleton sidelines that sit on the grid within tolerance AND align
        with the row of a paired sideline (established in build_yard_line_groups).
        Near vs far sideline classified by image half (frame_shape required).

    Drops singletons whose position doesn't fit the established grid spacing.
    """
    pixel_pts = []
    field_pts = []
    labels = []

    img_mid_y = frame_shape[0] / 2.0 if frame_shape is not None else None

    for yl in groups:
        gp = yl.get('grid_pos')
        if gp is None:
            continue
        field_x = base_ngs_x + gp * 5

        if yl.get('singleton'):
            if not yl.get('grid_fit_ok', False):
                continue
            if yl['far_hash'] is not None:
                pixel_pts.append(yl['far_hash'])
                field_pts.append([field_x, HASH_Y_FAR])
                labels.append(f'g{gp}_far_s')
            elif yl['near_hash'] is not None:
                pixel_pts.append(yl['near_hash'])
                field_pts.append([field_x, HASH_Y_NEAR])
                labels.append(f'g{gp}_near_s')
            elif yl['sideline'] is not None and img_mid_y is not None:
                sl = yl['sideline']
                # Top of frame = far sideline; bottom = near sideline
                field_y = FIELD_WIDTH if sl[1] < img_mid_y else 0.0
                pixel_pts.append(sl)
                field_pts.append([field_x, field_y])
                tag = 'side_s_far' if field_y > 0 else 'side_s_near'
                labels.append(f'g{gp}_{tag}')
            continue

        # Paired group: both hashes, plus any matched sideline
        if yl['near_hash'] is not None:
            pixel_pts.append(yl['near_hash'])
            field_pts.append([field_x, HASH_Y_NEAR])
            labels.append(f'g{gp}_near')
        if yl['far_hash'] is not None:
            pixel_pts.append(yl['far_hash'])
            field_pts.append([field_x, HASH_Y_FAR])
            labels.append(f'g{gp}_far')
        if yl['sideline'] is not None:
            pixel_pts.append(yl['sideline'])
            field_pts.append([field_x, FIELD_WIDTH])
            labels.append(f'g{gp}_side')

    return np.array(pixel_pts), np.array(field_pts), labels


def draw_overlay(frame, cal, state, pixel_pts, field_pts, labels, yard_range):
    vis = frame.copy()
    h, w = frame.shape[:2]
    x_min = max(0, yard_range[0] - 5)
    x_max = min(FIELD_LENGTH, yard_range[1] + 5)

    # Projected yard lines (green)
    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        pf = np.column_stack([np.full_like(ys, x), ys])
        pp = project_field_to_pixel(pf, state, cal, apply_dist=False)
        for i in range(len(pp) - 1):
            cv2.line(vis, tuple(pp[i].astype(int)), tuple(pp[i+1].astype(int)),
                     (0, 255, 0), 1, cv2.LINE_AA)
    # Projected hash rows (yellow-ish)
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(x_min, x_max, 50)
        pf = np.column_stack([xs, np.full_like(xs, y)])
        pp = project_field_to_pixel(pf, state, cal, apply_dist=False)
        for i in range(len(pp) - 1):
            cv2.line(vis, tuple(pp[i].astype(int)), tuple(pp[i+1].astype(int)),
                     (0, 200, 200), 1, cv2.LINE_AA)

    # Detected (red) vs projected (green circle) keypoints
    projected = project_field_to_pixel(field_pts, state, cal, apply_dist=False)
    for i in range(len(pixel_pts)):
        det = tuple(np.asarray(pixel_pts[i]).astype(int))
        proj = tuple(projected[i].astype(int))
        cv2.circle(vis, det, 5, (0, 0, 255), 2)
        cv2.circle(vis, proj, 3, (0, 255, 0), -1)
        cv2.line(vis, det, proj, (255, 255, 0), 1)
    return vis


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame",
                        default=os.path.join(OUTPUT_DIR, "real_kickoff_raw.jpg"))
    parser.add_argument("--base-ngs-x", type=float, default=40.0,
                        help="NGS x-coordinate of grid_pos=0 (leftmost yard line). "
                             "For the Chiefs@Packers kickoff this is 40.")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"Failed to read {args.frame}")
        return
    h, w = frame.shape[:2]
    print(f"Frame: {args.frame}  ({w}x{h})")

    # 1. HRNet
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    print(f"Detections: {len(hash_pxs)} hashes (>={HASH_THRESH}), "
          f"{len(sideline_pxs)} sidelines (>={SIDELINE_THRESH})")

    # 2. Grid solver
    groups, angle_deg = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    print(f"Yard-line tilt estimate: {angle_deg:.2f}° from vertical")
    n_paired = sum(1 for g in groups if not g.get('singleton'))
    n_with_side = sum(1 for g in groups if g.get('sideline') is not None)
    print(f"Grid groups: {len(groups)} total, {n_paired} paired, "
          f"{n_with_side} with attached sideline")

    # Report grid positions
    for g in sorted(groups, key=lambda x: x.get('grid_pos', 999)):
        if g.get('singleton'):
            continue
        gp = g.get('grid_pos', '?')
        ngs = args.base_ngs_x + (gp or 0) * 5
        has_side = '+side' if g.get('sideline') is not None else ''
        print(f"  g{gp} → NGS {ngs:.0f} {has_side}")

    # 3. Build correspondences
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, args.base_ngs_x)
    n_side = sum(1 for l in labels if 'side' in l)
    n_near = sum(1 for l in labels if 'near' in l)
    n_far = sum(1 for l in labels if 'far' in l)
    print(f"\nCorrespondences: {len(pixel_pts)} total "
          f"({n_far} far-hash, {n_near} near-hash, {n_side} sideline)")

    if len(pixel_pts) < 6:
        print("Not enough correspondences")
        return

    # 3b. Estimate distortion from collinearity of sideline + hash rows.
    # Build the line groups BEFORE selecting only grouped keypoints.
    line_sets = []
    side_row = [yl['sideline'] for yl in groups
                if yl.get('sideline') is not None]
    far_row = [yl['far_hash'] for yl in groups
               if yl.get('far_hash') is not None]
    near_row = [yl['near_hash'] for yl in groups
                if yl.get('near_hash') is not None]
    if len(side_row) >= 3:
        line_sets.append(np.array(side_row))
    if len(far_row) >= 3:
        line_sets.append(np.array(far_row))
    if len(near_row) >= 3:
        line_sets.append(np.array(near_row))

    print(f"\n=== Distortion calibration ===")
    print(f"Line groups: {len(line_sets)} "
          f"(sideline={len(side_row)}, far_hash={len(far_row)}, near_hash={len(near_row)})")

    # Measure pre-correction straightness
    pre_residual = 0.0
    for pts in line_sets:
        pre_residual += float(np.sum(line_fit_residuals(pts) ** 2))
    print(f"Pre-correction total squared residual: {pre_residual:.2f}")

    focal_guess = float(max(h, w))  # scale for distortion model
    k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w),
                                              focal_length_guess=focal_guess)
    print(f"Solved distortion: k1={k1:.5f}, k2={k2:.5f}")

    # Measure post-correction straightness
    intr_dist = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                                  cx=w / 2.0, cy=h / 2.0, k1=k1, k2=k2)
    post_residual = 0.0
    for pts in line_sets:
        u = undistort_points(pts, intr_dist)
        post_residual += float(np.sum(line_fit_residuals(u) ** 2))
    print(f"Post-correction total squared residual: {post_residual:.2f} "
          f"({post_residual / max(pre_residual, 1e-9) * 100:.1f}% of pre)")

    # Undistort all pixel points before camera calibration
    pixel_pts_undistorted = undistort_points(pixel_pts, intr_dist)

    # 4. Calibrate camera on undistorted points (camera model assumes no
    # distortion; we've already removed it)
    cal = calibrate_camera(pixel_pts_undistorted, field_pts, (h, w))
    pos = cal.extrinsics.position
    print(f"\n=== Calibration ===")
    print(f"Position: Cx={pos[0]:.1f}, Cy={pos[1]:.1f}, Cz={pos[2]:.1f}")
    print(f"State: pan={cal.initial_state.pan:.4f}, tilt={cal.initial_state.tilt:.4f}, "
          f"f={cal.initial_state.focal_length:.0f}")
    print(f"Roll: {np.degrees(cal.initial_state.roll):.2f}°")
    print(f"Reprojection RMSE: {cal.calibration_error:.2f} pixels")

    # 5. Per-point errors (in undistorted space, since that's what the camera solves)
    projected = project_field_to_pixel(field_pts, cal.initial_state, cal,
                                         apply_dist=False)
    errs = []
    for i in range(len(pixel_pts_undistorted)):
        det = pixel_pts_undistorted[i]
        pr = projected[i]
        e = float(np.hypot(det[0] - pr[0], det[1] - pr[1]))
        errs.append((e, labels[i], det, pr, field_pts[i]))
    errs.sort(reverse=True)
    print("\nWorst 10 per-point errors (undistorted space):")
    for e, lab, det, pr, fp in errs[:10]:
        print(f"  {lab:12s} NGS({fp[0]:5.1f},{fp[1]:5.1f})  "
              f"det=({det[0]:6.0f},{det[1]:4.0f})  proj=({pr[0]:6.0f},{pr[1]:4.0f})  "
              f"err={e:.1f}px")
    all_errs = np.array([e[0] for e in errs])
    print(f"\nError stats: mean={all_errs.mean():.1f}px, "
          f"median={np.median(all_errs):.1f}px, max={all_errs.max():.1f}px")

    # 6. Overlay on UNDISTORTED frame (so projections match 1:1)
    H_result = camera_state_to_homography(
        cal.initial_state, cal,
        n_inliers=len(pixel_pts), n_correspondences=len(pixel_pts),
    )
    if abs(k1) > 1e-6 or abs(k2) > 1e-6:
        K = np.array([
            [focal_guess, 0, w / 2.0],
            [0, focal_guess, h / 2.0],
            [0, 0, 1],
        ])
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        frame_undistorted = cv2.undistort(frame, K, dist_coeffs)
    else:
        frame_undistorted = frame
    vis = draw_overlay(frame_undistorted, cal, cal.initial_state,
                        pixel_pts_undistorted, field_pts,
                        labels, H_result.yard_range)

    out_name = args.out_name or (
        os.path.splitext(os.path.basename(args.frame))[0] + "_grid_camera.jpg"
    )
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, vis)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
