#!/usr/bin/env python3
"""
Similarity-delta homography tracking.

When a frame has fewer than 4 identified keypoints, we can still update the
homography from the previous frame using an approximate scale+rotation+
translation (similarity) transform, which is solvable from 2 correspondences.

    H_cur = H_prev @ inv(S)

    where S maps previous-frame pixels → current-frame pixels (4 DOF).

This works well when the camera change between frames is dominated by zoom
and pan (which is normally the case during a play) and poorly when the
camera changes position (which tripod-mounted broadcast cameras don't).

Test: take two frames with full detections, compute H on frame 1, simulate
losing all but 2 correspondences on frame 2, recover H_2 via similarity delta,
compare to frame 2's own full homography (ground truth).
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.testing.test_grid_solver_camera import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines,
)
from scripts.testing.test_yard_line_grouping import (
    run_hrnet, extract_peaks,
)
from src.homography.camera_model import CameraIntrinsics, undistort_points
from src.homography.apply_homography import pixel_to_field, field_to_pixel
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")


def compute_full_homography(frame, base_ngs_x):
    """Returns (H, H_inv, pixel_pts_undistorted, field_pts, k1, k2) or None."""
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sl_confs = extract_peaks(heatmaps[0], 0.30, (h, w))
    hash_pxs, h_confs = extract_peaks(heatmaps[1], 0.40, (h, w))
    groups, _ = build_yard_line_groups(hash_pxs, sideline_pxs, sl_confs)
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, base_ngs_x)
    if len(pixel_pts) < 4:
        return None

    # Distortion
    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]
    focal_guess = float(max(h, w))
    if line_sets:
        k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w), focal_guess)
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            k1, k2 = 0.0, 0.0
    else:
        k1, k2 = 0.0, 0.0

    intr = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                             cx=w/2.0, cy=h/2.0, k1=k1, k2=k2)
    pixel_pts_u = undistort_points(pixel_pts, intr)

    H, _ = cv2.findHomography(pixel_pts_u.astype(np.float64),
                                field_pts.astype(np.float64),
                                method=cv2.RANSAC,
                                ransacReprojThreshold=1.5)
    if H is None:
        return None
    return {
        "H": H,
        "H_inv": np.linalg.inv(H),
        "pixel_pts_u": pixel_pts_u,
        "field_pts": field_pts,
        "labels": labels,
        "k1": k1, "k2": k2,
        "focal_guess": focal_guess,
        "frame_shape": (h, w),
    }


def similarity_from_two_points(src_pts, dst_pts):
    """Solve 4-DOF similarity (scale, rotation, translation) mapping src → dst.

    OpenCV's estimateAffinePartial2D does this directly (requires 2+ points).
    Returns 3x3 similarity matrix.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return None
    S = np.vstack([M, [0, 0, 1]])
    return S


def similarity_update(H_prev, prev_pixel_pts, cur_pixel_pts):
    """Update H given new pixel correspondences.

    prev_pixel_pts and cur_pixel_pts are pixel positions of the SAME physical
    field points in the previous and current frames (all in undistorted space).

    Returns H_cur = H_prev @ inv(S), where S maps previous pixels to current.
    """
    S = similarity_from_two_points(prev_pixel_pts, cur_pixel_pts)
    if S is None:
        return None
    H_cur = H_prev @ np.linalg.inv(S)
    return H_cur, S


def draw_overlay(frame_undistorted, H_inv, pixel_pts_u, field_pts, out_path,
                  title=""):
    """Draw yard-line grid via H_inv on the (undistorted) frame."""
    vis = frame_undistorted.copy()
    h, w = vis.shape[:2]

    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        fp = np.column_stack([np.full_like(ys, x), ys])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1 = tuple(pp[i].astype(int))
            p2 = tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1 = tuple(pp[i].astype(int))
            p2 = tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (0, 200, 200), 1, cv2.LINE_AA)

    for y in [0, FIELD_WIDTH]:
        xs = np.linspace(0, FIELD_LENGTH, 100)
        fp = np.column_stack([xs, np.full_like(xs, y)])
        pp = field_to_pixel(fp, H_inv)
        for i in range(len(pp) - 1):
            p1 = tuple(pp[i].astype(int))
            p2 = tuple(pp[i + 1].astype(int))
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(vis, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

    # Markers
    for i in range(len(pixel_pts_u)):
        det = tuple(np.asarray(pixel_pts_u[i]).astype(int))
        pb = field_to_pixel(np.array([field_pts[i]]), H_inv)[0]
        proj = tuple(pb.astype(int))
        cv2.circle(vis, det, 5, (0, 0, 255), 2)
        cv2.circle(vis, proj, 3, (0, 255, 0), -1)
        cv2.line(vis, det, proj, (255, 255, 0), 1)

    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)


def undistort_frame(frame, k1, k2, focal_guess):
    h, w = frame.shape[:2]
    if abs(k1) < 1e-6 and abs(k2) < 1e-6:
        return frame
    K = np.array([[focal_guess, 0, w/2.0], [0, focal_guess, h/2.0], [0, 0, 1]])
    return cv2.undistort(frame, K, np.array([k1, k2, 0, 0, 0]))


def report_errors(H_inv, pixel_pts_u, field_pts, label=""):
    projected = field_to_pixel(field_pts, H_inv)
    errs_px = np.linalg.norm(projected - pixel_pts_u, axis=1)
    field_projected = pixel_to_field(pixel_pts_u, np.linalg.inv(H_inv))
    errs_fd = np.linalg.norm(field_projected - field_pts, axis=1)
    print(f"  {label} Pixel err: mean={errs_px.mean():.2f}, max={errs_px.max():.2f}")
    print(f"  {label} Field err: mean={errs_fd.mean():.3f}yd, max={errs_fd.max():.3f}yd")
    return errs_px, errs_fd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", default="real_kickoff_raw.jpg")
    parser.add_argument("--ngs1", type=float, default=35.0)
    parser.add_argument("--f2", default="play002_first.jpg")
    parser.add_argument("--ngs2", type=float, default=70.0)
    args = parser.parse_args()

    print("=" * 60)
    print("SIMILARITY-DELTA HOMOGRAPHY TEST")
    print("=" * 60)

    # 1. Full homography on both frames (ground truth)
    frame1 = cv2.imread(os.path.join(OUTPUT_DIR, args.f1))
    r1 = compute_full_homography(frame1, args.ngs1)
    print(f"\n[F1] {args.f1}: {len(r1['pixel_pts_u'])} correspondences")
    report_errors(r1['H_inv'], r1['pixel_pts_u'], r1['field_pts'], "F1 full:")

    frame2 = cv2.imread(os.path.join(OUTPUT_DIR, args.f2))
    r2 = compute_full_homography(frame2, args.ngs2)
    print(f"\n[F2] {args.f2}: {len(r2['pixel_pts_u'])} correspondences")
    report_errors(r2['H_inv'], r2['pixel_pts_u'], r2['field_pts'], "F2 full:")

    # 2. Simulate: frame 2 only has 2 correspondences (pick 2 sidelines)
    side_mask = np.array(['side' in lbl for lbl in r2['labels']])
    side_idx = np.where(side_mask)[0]
    if len(side_idx) < 2:
        # Fall back to any 2 correspondences
        print("Few sidelines — using first 2 of any type")
        kept_idx = np.arange(min(2, len(r2['pixel_pts_u'])))
    else:
        # Prefer the leftmost and rightmost sideline detections for good baseline
        side_field_x = r2['field_pts'][side_idx, 0]
        left_idx = side_idx[np.argmin(side_field_x)]
        right_idx = side_idx[np.argmax(side_field_x)]
        kept_idx = np.array([left_idx, right_idx])

    kept_cur_pixels = r2['pixel_pts_u'][kept_idx]
    kept_field = r2['field_pts'][kept_idx]
    print(f"\nSimulating: kept only {len(kept_idx)} correspondences on F2:")
    for i, idx in enumerate(kept_idx):
        print(f"  {r2['labels'][idx]:12s} field=({kept_field[i][0]:.0f}, {kept_field[i][1]:.1f}) "
              f"pixel_u=({kept_cur_pixels[i][0]:.0f}, {kept_cur_pixels[i][1]:.0f})")

    # 3. Project those field points to F1's pixel space using H_prev (F1's H)
    # We need F1-undistorted pixel positions where these SAME field points lie.
    kept_prev_pixels = field_to_pixel(kept_field, r1['H_inv'])
    print(f"\nF1 pixel positions of same field points (via F1's H):")
    for i, px in enumerate(kept_prev_pixels):
        print(f"  ({px[0]:.0f}, {px[1]:.0f})")

    # 4. Solve similarity F1_pixels → F2_pixels
    # IMPORTANT: both sets must be in the SAME pixel space (undistorted).
    # F1's pixel_pts_u and F2's pixel_pts_u might have different distortion
    # coefficients; for a clean test we'll assume they're both undistorted
    # with their own coefficients (good enough for demonstration).
    result = similarity_update(r1['H'], kept_prev_pixels, kept_cur_pixels)
    if result is None:
        print("Failed to solve similarity")
        return
    H_cur_from_delta, S = result
    H_cur_from_delta_inv = np.linalg.inv(H_cur_from_delta)

    # Decompose S
    det = S[0, 0] ** 2 + S[1, 0] ** 2
    scale = float(np.sqrt(det))
    rotation_deg = float(np.degrees(np.arctan2(S[1, 0], S[0, 0])))
    tx, ty = float(S[0, 2]), float(S[1, 2])
    print(f"\nSimilarity transform F1 → F2:")
    print(f"  Scale: {scale:.3f} (F2 is {scale:.2f}x zoomed relative to F1)")
    print(f"  Rotation: {rotation_deg:.2f}°")
    print(f"  Translation: ({tx:.1f}, {ty:.1f}) pixels")

    # 5. Test F2's delta-derived homography against all F2's correspondences
    print("\nDelta-derived H on F2 (validated against ALL F2 correspondences):")
    errs_px_delta, errs_fd_delta = report_errors(
        H_cur_from_delta_inv, r2['pixel_pts_u'], r2['field_pts'], "F2 delta:",
    )

    # 6. Compare to F2's own full homography
    errs_px_full, errs_fd_full = report_errors(
        r2['H_inv'], r2['pixel_pts_u'], r2['field_pts'], "F2 full:",
    )

    print(f"\nCOMPARISON — F2 error (field yards):")
    print(f"  Full H from F2 itself: mean={errs_fd_full.mean():.3f}, max={errs_fd_full.max():.3f}")
    print(f"  Delta H from F1 + 2 pts: mean={errs_fd_delta.mean():.3f}, max={errs_fd_delta.max():.3f}")
    print(f"  Ratio (delta/full): {errs_fd_delta.mean() / max(errs_fd_full.mean(), 1e-6):.1f}x")

    # 7. Overlays
    f1_u = undistort_frame(frame1, r1['k1'], r1['k2'], r1['focal_guess'])
    f2_u = undistort_frame(frame2, r2['k1'], r2['k2'], r2['focal_guess'])

    draw_overlay(f1_u, r1['H_inv'], r1['pixel_pts_u'], r1['field_pts'],
                  os.path.join(OUTPUT_DIR, "delta_F1_full.jpg"),
                  title=f"F1 full H ({len(r1['pixel_pts_u'])} pts)")
    draw_overlay(f2_u, r2['H_inv'], r2['pixel_pts_u'], r2['field_pts'],
                  os.path.join(OUTPUT_DIR, "delta_F2_full.jpg"),
                  title=f"F2 full H ({len(r2['pixel_pts_u'])} pts)")
    draw_overlay(f2_u, H_cur_from_delta_inv, kept_cur_pixels, kept_field,
                  os.path.join(OUTPUT_DIR, "delta_F2_from_delta.jpg"),
                  title=f"F2 delta H (2 pts from F1)")

    print("\nOutputs:")
    print(f"  {OUTPUT_DIR}/delta_F1_full.jpg")
    print(f"  {OUTPUT_DIR}/delta_F2_full.jpg (ground truth)")
    print(f"  {OUTPUT_DIR}/delta_F2_from_delta.jpg (from similarity update)")


if __name__ == "__main__":
    main()
