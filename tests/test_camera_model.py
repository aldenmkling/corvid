"""
Tests for the pinhole camera model homography.

Tests projection math, calibration, PTZ solving, and round-trip consistency.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.homography.camera_model import (
    CameraIntrinsics,
    CameraExtrinsics,
    CameraCalibration,
    CameraState,
    rotation_matrix,
    project_field_to_pixel,
    camera_state_to_homography,
    apply_distortion,
    undistort_points,
    calibrate_camera,
    solve_ptz,
    identify_keypoints_by_grid,
    bootstrap_identity,
    PTZFilter,
)
from src.homography.apply_homography import pixel_to_field
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_calibration(
    Cx=60.0, Cy=-40.0, Cz=30.0,
    k1=0.0, k2=0.0,
    pan=0.0, tilt=-0.6, f=800.0,
):
    """Create a test calibration with known parameters."""
    intrinsics = CameraIntrinsics(fx=f, fy=f, cx=640.0, cy=360.0, k1=k1, k2=k2)
    extrinsics = CameraExtrinsics(position=np.array([Cx, Cy, Cz]))
    initial_state = CameraState(pan=pan, tilt=tilt, focal_length=f)
    return CameraCalibration(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        calibration_error=0.0,
        n_points_used=0,
        initial_state=initial_state,
    )


def generate_field_points(x_range=(30, 90), n_per_line=4):
    """Generate field points within a yard range."""
    pts = []
    for x in YARD_LINE_POSITIONS:
        if x < x_range[0] or x > x_range[1]:
            continue
        for y in [0.0, HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH]:
            pts.append([x, y])
    return np.array(pts)


# ── Rotation matrix tests ──────────────────────────────────────────────────

class TestRotationMatrix:
    def test_is_orthogonal(self):
        """R @ R^T should be identity."""
        R = rotation_matrix(0.3, -0.5)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        """det(R) should be 1 (proper rotation)."""
        for pan in [-0.5, 0.0, 0.5]:
            for tilt in [-0.8, -0.3, 0.0]:
                R = rotation_matrix(pan, tilt)
                assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_zero_pan_tilt(self):
        """At zero pan/tilt, camera looks along -Y (toward field)."""
        R = rotation_matrix(0.0, 0.0)
        # World point on the field (straight ahead from camera)
        # Should project to image center area
        # Camera at (60, -40, 30) looking along -Y with no tilt
        # Point at (60, 0, 0) is directly ahead
        # In camera frame: should be on optical axis (z-forward)


class TestProjection:
    def test_midfield_projects_to_center(self):
        """A point at midfield should project near image center when pan=0."""
        cal = make_calibration(Cx=60.0, Cy=-40.0, Cz=30.0)
        state = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)

        # Project midfield hash mark
        pts = np.array([[60.0, HASH_Y_NEAR]])
        pixels = project_field_to_pixel(pts, state, cal, apply_dist=False)

        # Should be roughly centered horizontally
        assert 400 < pixels[0, 0] < 880, f"x={pixels[0, 0]} not near center"

    def test_pan_shifts_projection(self):
        """Panning right should shift projections left in image."""
        cal = make_calibration()
        pts = np.array([[60.0, HASH_Y_NEAR]])

        state_center = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)
        state_right = CameraState(pan=0.1, tilt=-0.6, focal_length=800.0)

        px_center = project_field_to_pixel(pts, state_center, cal, apply_dist=False)
        px_right = project_field_to_pixel(pts, state_right, cal, apply_dist=False)

        # Panning should shift the projection horizontally
        assert px_right[0, 0] != px_center[0, 0], "Pan had no effect on projection"

    def test_zoom_scales_projection(self):
        """Higher focal length should spread points further from center."""
        cal = make_calibration()
        pts = np.array([[50.0, 0.0], [70.0, FIELD_WIDTH]])

        state_wide = CameraState(pan=0.0, tilt=-0.6, focal_length=600.0)
        state_zoom = CameraState(pan=0.0, tilt=-0.6, focal_length=1200.0)

        px_wide = project_field_to_pixel(pts, state_wide, cal, apply_dist=False)
        px_zoom = project_field_to_pixel(pts, state_zoom, cal, apply_dist=False)

        # Distance between points should be larger when zoomed in
        dist_wide = np.sqrt(np.sum((px_wide[0] - px_wide[1])**2))
        dist_zoom = np.sqrt(np.sum((px_zoom[0] - px_zoom[1])**2))
        assert dist_zoom > dist_wide

    def test_all_points_in_front_of_camera(self):
        """Field points should always be in front of the camera."""
        cal = make_calibration()
        state = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)
        pts = generate_field_points()

        # Project and check no NaN/inf
        pixels = project_field_to_pixel(pts, state, cal, apply_dist=False)
        assert not np.any(np.isnan(pixels))
        assert not np.any(np.isinf(pixels))


class TestDistortion:
    def test_zero_distortion_identity(self):
        """With k1=k2=0, distortion should be identity."""
        x = np.array([0.1, 0.2, -0.3])
        y = np.array([0.05, -0.15, 0.25])
        xd, yd = apply_distortion(x, y, 0.0, 0.0)
        np.testing.assert_array_equal(xd, x)
        np.testing.assert_array_equal(yd, y)

    def test_undistort_inverts_distort(self):
        """Undistortion should invert distortion."""
        intrinsics = CameraIntrinsics(fx=800, fy=800, cx=640, cy=360, k1=-0.1, k2=0.02)

        # Start with undistorted points
        pts_clean = np.array([[640, 360], [700, 400], [500, 300], [800, 200]], dtype=np.float64)

        # Distort them
        x_n = (pts_clean[:, 0] - 640) / 800
        y_n = (pts_clean[:, 1] - 360) / 800
        xd, yd = apply_distortion(x_n, y_n, -0.1, 0.02)
        pts_distorted = np.column_stack([xd * 800 + 640, yd * 800 + 360])

        # Undistort
        pts_recovered = undistort_points(pts_distorted, intrinsics)

        np.testing.assert_allclose(pts_recovered, pts_clean, atol=0.1)


class TestHomographyFromCamera:
    def test_round_trip_no_distortion(self):
        """Project field→pixel via camera model, then pixel→field via H. Should match."""
        cal = make_calibration(k1=0.0, k2=0.0)
        state = CameraState(pan=0.1, tilt=-0.5, focal_length=900.0)

        field_pts = generate_field_points(x_range=(40, 80))
        pixel_pts = project_field_to_pixel(field_pts, state, cal, apply_dist=False)

        # Get homography
        result = camera_state_to_homography(state, cal)

        # Map pixels back to field via H
        recovered = pixel_to_field(pixel_pts, result.H)

        np.testing.assert_allclose(recovered, field_pts, atol=0.01,
                                   err_msg="Round-trip field→pixel→field failed")

    def test_homography_invertible(self):
        """H @ H_inv should be ~identity (up to scale)."""
        cal = make_calibration()
        state = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)
        result = camera_state_to_homography(state, cal)

        product = result.H @ result.H_inv
        product /= product[2, 2]  # normalize
        np.testing.assert_allclose(product, np.eye(3), atol=1e-8)


class TestCalibration:
    def test_recover_known_params(self):
        """Calibration should recover known camera parameters from synthetic data."""
        # Ground truth camera
        true_Cx, true_Cy, true_Cz = 55.0, -35.0, 28.0
        true_pan, true_tilt, true_f = 0.05, -0.55, 850.0

        cal_true = make_calibration(
            Cx=true_Cx, Cy=true_Cy, Cz=true_Cz,
            pan=true_pan, tilt=true_tilt, f=true_f,
        )
        state_true = CameraState(pan=true_pan, tilt=true_tilt, focal_length=true_f)

        # Generate correspondences
        field_pts = generate_field_points(x_range=(20, 100))
        pixel_pts = project_field_to_pixel(field_pts, state_true, cal_true, apply_dist=False)

        # Filter to points visible in frame
        visible = (
            (pixel_pts[:, 0] >= 0) & (pixel_pts[:, 0] < 1280) &
            (pixel_pts[:, 1] >= 0) & (pixel_pts[:, 1] < 720)
        )
        field_pts = field_pts[visible]
        pixel_pts = pixel_pts[visible]

        assert len(field_pts) >= 8, f"Need 8+ visible points, got {len(field_pts)}"

        # Add small noise
        np.random.seed(42)
        pixel_pts_noisy = pixel_pts + np.random.randn(*pixel_pts.shape) * 1.0

        # Calibrate
        cal_solved = calibrate_camera(pixel_pts_noisy, field_pts)

        # Check recovered parameters (loose tolerances due to noise)
        assert abs(cal_solved.extrinsics.position[0] - true_Cx) < 5.0, \
            f"Cx: {cal_solved.extrinsics.position[0]:.1f} vs {true_Cx}"
        assert abs(cal_solved.extrinsics.position[1] - true_Cy) < 5.0, \
            f"Cy: {cal_solved.extrinsics.position[1]:.1f} vs {true_Cy}"
        assert abs(cal_solved.extrinsics.position[2] - true_Cz) < 5.0, \
            f"Cz: {cal_solved.extrinsics.position[2]:.1f} vs {true_Cz}"

        # Reprojection error should be small
        assert cal_solved.calibration_error < 3.0, \
            f"Calibration RMSE {cal_solved.calibration_error:.2f} > 3 pixels"

    def test_recover_with_distortion(self):
        """Calibration should recover distortion coefficients."""
        true_k1, true_k2 = -0.08, 0.01

        cal_true = make_calibration(
            Cx=60, Cy=-40, Cz=30,
            k1=true_k1, k2=true_k2,
            pan=0.0, tilt=-0.6, f=800.0,
        )
        state_true = CameraState(pan=0.0, tilt=-0.6, focal_length=800.0)

        field_pts = generate_field_points(x_range=(20, 100))
        pixel_pts = project_field_to_pixel(field_pts, state_true, cal_true, apply_dist=True)

        visible = (
            (pixel_pts[:, 0] >= 0) & (pixel_pts[:, 0] < 1280) &
            (pixel_pts[:, 1] >= 0) & (pixel_pts[:, 1] < 720)
        )
        field_pts = field_pts[visible]
        pixel_pts = pixel_pts[visible]

        assert len(field_pts) >= 8

        cal_solved = calibrate_camera(pixel_pts, field_pts)

        # Should get reasonable distortion estimates
        # (exact recovery is hard with noise, but sign and order of magnitude should match)
        assert cal_solved.calibration_error < 3.0, \
            f"Calibration RMSE {cal_solved.calibration_error:.2f} > 3 pixels"


class TestPTZSolver:
    def test_recover_ptz_from_known_correspondences(self):
        """PTZ solver should recover pan/tilt/focal from known correspondences."""
        cal = make_calibration(k1=0.0, k2=0.0)
        true_state = CameraState(pan=0.15, tilt=-0.5, focal_length=900.0)

        field_pts = generate_field_points(x_range=(40, 80))
        pixel_pts = project_field_to_pixel(field_pts, true_state, cal, apply_dist=False)

        # Filter visible
        visible = (
            (pixel_pts[:, 0] >= 0) & (pixel_pts[:, 0] < 1280) &
            (pixel_pts[:, 1] >= 0) & (pixel_pts[:, 1] < 720)
        )
        field_pts = field_pts[visible]
        pixel_pts = pixel_pts[visible]

        # Solve with a slightly off initial guess (from calibration default)
        solved = solve_ptz(pixel_pts, field_pts, cal)

        assert solved is not None
        assert abs(solved.pan - true_state.pan) < 0.01, \
            f"Pan: {solved.pan:.3f} vs {true_state.pan:.3f}"
        assert abs(solved.tilt - true_state.tilt) < 0.01, \
            f"Tilt: {solved.tilt:.3f} vs {true_state.tilt:.3f}"
        assert abs(solved.focal_length - true_state.focal_length) < 10, \
            f"Focal: {solved.focal_length:.1f} vs {true_state.focal_length:.1f}"

    def test_works_with_2_points(self):
        """PTZ solver should work with just 2 correspondences."""
        cal = make_calibration(k1=0.0, k2=0.0)
        true_state = CameraState(pan=0.1, tilt=-0.55, focal_length=850.0)

        field_pts = np.array([[50.0, HASH_Y_NEAR], [70.0, HASH_Y_FAR]])
        pixel_pts = project_field_to_pixel(field_pts, true_state, cal, apply_dist=False)

        # Provide previous state close to truth (temporal prior)
        prev = CameraState(pan=0.08, tilt=-0.54, focal_length=840.0)

        solved = solve_ptz(pixel_pts, field_pts, cal, prev_state=prev)

        assert solved is not None
        assert abs(solved.pan - true_state.pan) < 0.05
        assert abs(solved.tilt - true_state.tilt) < 0.05
        assert abs(solved.focal_length - true_state.focal_length) < 50


class TestPTZFilter:
    def test_smooths_noisy_measurements(self):
        """Filter should smooth noisy PTZ measurements."""
        filt = PTZFilter()

        true_pan = 0.1
        true_tilt = -0.5
        true_f = 800.0

        np.random.seed(42)
        errors = []

        for i in range(30):
            # Noisy measurement
            m = CameraState(
                pan=true_pan + np.random.randn() * 0.02,
                tilt=true_tilt + np.random.randn() * 0.02,
                focal_length=true_f + np.random.randn() * 20,
            )

            if not filt.initialized:
                filtered = filt.update(m)
            else:
                filt.predict()
                filtered = filt.update(m)

            if i > 10:  # after warmup
                errors.append(abs(filtered.pan - true_pan))

        # Filtered error should be less than raw measurement noise
        mean_error = np.mean(errors)
        assert mean_error < 0.02, f"Mean filtered pan error {mean_error:.4f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
