"""
Pinhole camera model for broadcast All-22 homography.

Instead of solving an 8-DOF homography per frame, models the physical camera:
  - Fixed position (press box tripod, constant for a game)
  - Variable pan, tilt, focal length (operator tracks the play)

Calibrate once per game from a wide kickoff frame, then solve only 3 parameters
per frame. Works with as few as 2 detected keypoints.

Sideline view only for now.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from .compute_homography import HomographyResult
from .field_model import (
    FIELD_LENGTH, FIELD_WIDTH, YARD_LINE_POSITIONS, HASH_Y_NEAR, HASH_Y_FAR,
)
from .keypoint_schema import FIELD_POINTS, FIELD_COORDS, POINT_CHANNELS


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float              # focal length in pixels (horizontal)
    fy: float              # focal length in pixels (vertical), typically == fx
    cx: float              # principal point x (pixels)
    cy: float              # principal point y (pixels)
    k1: float = 0.0        # radial distortion coefficient 1
    k2: float = 0.0        # radial distortion coefficient 2


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (fixed for a game)."""
    position: np.ndarray   # (3,) camera position in field coords [Cx, Cy, Cz] yards
    # Cx: along-field position (roughly midfield ~60)
    # Cy: negative (behind near sideline y=0)
    # Cz: positive (above field)


@dataclass
class CameraCalibration:
    """Full camera calibration result (one per game)."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    calibration_error: float       # reprojection RMSE in pixels
    n_points_used: int
    initial_state: "CameraState"   # PTZ state at calibration frame


@dataclass
class CameraState:
    """Per-frame camera state (what changes frame to frame)."""
    pan: float              # radians — rotation around world Z
    tilt: float             # radians — rotation around camera X (negative = looking down)
    focal_length: float     # pixels
    roll: float = 0.0       # radians — rotation around optical axis (small, ~tripod misalignment)


# ── Rotation matrix ─────────────────────────────────────────────────────────

def rotation_matrix(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """Build 3x3 rotation matrix from pan, tilt, and roll angles.

    Convention:
      - Pan: rotation around world Z-axis (vertical).
        0 = camera optical axis points along -Y (from behind near sideline toward field).
        Positive = camera rotates to look toward +X (right along field).
      - Tilt: rotation around camera's local X-axis.
        0 = looking horizontal. Negative = looking down.
      - Roll: rotation around camera's optical axis (Z).
        0 = level. Small values expected (~0.5° for tripod misalignment).
      - R transforms world coordinates to camera coordinates.

    Camera frame: x-right, y-down, z-forward (standard CV convention).
    """
    cp, sp = np.cos(pan), np.sin(pan)
    ct, st = np.cos(tilt), np.sin(tilt)

    # Pan around Z (world vertical)
    R_pan = np.array([
        [ cp, sp, 0],
        [-sp, cp, 0],
        [  0,  0, 1],
    ])

    # Tilt around X (camera horizontal)
    R_tilt = np.array([
        [1,  0,   0],
        [0,  ct, st],
        [0, -st, ct],
    ])

    # World-to-camera: first align the world so that -Y becomes the camera's
    # forward direction, then apply pan and tilt.
    # Base rotation: world (x-right, y-forward, z-up) → camera (x-right, y-down, z-forward)
    # This rotates so that world -Y maps to camera +Z (forward), world Z maps to camera -Y (up→down flip)
    R_base = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ])

    R = R_tilt @ R_pan @ R_base

    # Roll around optical axis (camera Z)
    if abs(roll) > 1e-10:
        cr, sr = np.cos(roll), np.sin(roll)
        R_roll = np.array([
            [ cr, sr, 0],
            [-sr, cr, 0],
            [  0,  0, 1],
        ])
        R = R_roll @ R

    return R


# ── Distortion ──────────────────────────────────────────────────────────────

def apply_distortion(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply radial distortion to normalized image coordinates.

    Args:
        x_norm, y_norm: coordinates relative to principal point, divided by focal length
        k1, k2: radial distortion coefficients

    Returns:
        Distorted (x_norm, y_norm)
    """
    r2 = x_norm**2 + y_norm**2
    radial = 1.0 + k1 * r2 + k2 * r2**2
    return x_norm * radial, y_norm * radial


def undistort_points(
    pixel_pts: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_iters: int = 10,
) -> np.ndarray:
    """Remove radial distortion from pixel coordinates.

    Iterative method: given distorted point, find the undistorted point
    that would map to it under the distortion model.

    Args:
        pixel_pts: (N, 2) distorted pixel coordinates
        intrinsics: camera intrinsics with distortion coefficients
        n_iters: number of iterations for convergence

    Returns:
        (N, 2) undistorted pixel coordinates
    """
    if abs(intrinsics.k1) < 1e-12 and abs(intrinsics.k2) < 1e-12:
        return pixel_pts.copy()

    # Normalize to principal-point-centered, focal-length-scaled coords
    x_dist = (pixel_pts[:, 0] - intrinsics.cx) / intrinsics.fx
    y_dist = (pixel_pts[:, 1] - intrinsics.cy) / intrinsics.fy

    # Iterative undistortion
    x_u = x_dist.copy()
    y_u = y_dist.copy()

    for _ in range(n_iters):
        r2 = x_u**2 + y_u**2
        radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r2**2
        x_u = x_dist / radial
        y_u = y_dist / radial

    # Back to pixel coordinates
    result = np.empty_like(pixel_pts)
    result[:, 0] = x_u * intrinsics.fx + intrinsics.cx
    result[:, 1] = y_u * intrinsics.fy + intrinsics.cy
    return result


# ── Projection ──────────────────────────────────────────────────────────────

def project_field_to_pixel(
    field_pts: np.ndarray,
    state: CameraState,
    calibration: CameraCalibration,
    apply_dist: bool = True,
) -> np.ndarray:
    """Project 3D field points (z=0) to pixel coordinates.

    Args:
        field_pts: (N, 2) field coordinates [x, y] in yards
        state: current camera PTZ state
        calibration: camera calibration (position + distortion)
        apply_dist: whether to apply radial distortion

    Returns:
        (N, 2) pixel coordinates
    """
    N = len(field_pts)
    intr = calibration.intrinsics
    C = calibration.extrinsics.position

    # Build 3D points on field plane (z=0)
    pts_3d = np.column_stack([field_pts, np.zeros(N)])  # (N, 3)

    # Camera transform
    R = rotation_matrix(state.pan, state.tilt, state.roll)
    t = -R @ C  # translation vector

    # Project: p_cam = R @ (P - C) = R @ P + t
    p_cam = (R @ pts_3d.T).T + t  # (N, 3)

    # Perspective division → normalized image coordinates
    x_norm = p_cam[:, 0] / p_cam[:, 2]
    y_norm = p_cam[:, 1] / p_cam[:, 2]

    # Apply distortion in normalized space
    if apply_dist and (abs(intr.k1) > 1e-12 or abs(intr.k2) > 1e-12):
        x_norm, y_norm = apply_distortion(x_norm, y_norm, intr.k1, intr.k2)

    # Apply intrinsics → pixel coordinates
    # Use per-axis focal length from the state (zoom changes f)
    u = state.focal_length * x_norm + intr.cx
    v = state.focal_length * y_norm + intr.cy

    return np.column_stack([u, v])


# ── Homography from camera state ────────────────────────────────────────────

def camera_state_to_homography(
    state: CameraState,
    calibration: CameraCalibration,
    n_inliers: int = 0,
    n_correspondences: int = 0,
) -> HomographyResult:
    """Convert camera PTZ state to a HomographyResult.

    Builds the 3x3 homography matrix for the z=0 field plane.
    This H operates in undistorted pixel space — callers should
    undistort detected points before using pixel_to_field(pts, H).

    Args:
        state: current pan/tilt/focal_length
        calibration: fixed camera calibration
        n_inliers: number of keypoints used in the solve
        n_correspondences: number of keypoints detected

    Returns:
        HomographyResult with H (pixel→field) and H_inv (field→pixel)
    """
    C = calibration.extrinsics.position
    f = state.focal_length
    cx = calibration.intrinsics.cx
    cy = calibration.intrinsics.cy

    K = np.array([
        [f,  0, cx],
        [0,  f, cy],
        [0,  0,  1],
    ])

    R = rotation_matrix(state.pan, state.tilt, state.roll)
    t = -R @ C

    # For z=0 plane: drop the 3rd column of R, keep columns 0,1 and t
    # H_inv maps (X, Y, 1) → (u*w, v*w, w) in undistorted pixel space
    H_inv = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = np.linalg.inv(H_inv)

    # Estimate yard range from what's visible
    yard_range = _estimate_visible_yard_range(state, calibration)

    # Compute reprojection error on a grid of field points
    reproj_error = _compute_reproj_error(H, H_inv, yard_range)

    return HomographyResult(
        H=H,
        H_inv=H_inv,
        reprojection_error=reproj_error,
        n_inliers=n_inliers,
        n_correspondences=n_correspondences,
        yard_range=yard_range,
    )


def _estimate_visible_yard_range(
    state: CameraState,
    calibration: CameraCalibration,
) -> tuple[float, float]:
    """Estimate which yard lines are visible given camera state.

    Projects image corners to field coordinates and returns the x-range,
    clamped to the actual field boundaries.
    """
    cx = calibration.intrinsics.cx
    cy = calibration.intrinsics.cy
    f = state.focal_length
    C = calibration.extrinsics.position
    R = rotation_matrix(state.pan, state.tilt, state.roll)
    t = -R @ C

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    H_inv = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = np.linalg.inv(H_inv)

    # Project all 4 corners + edge midpoints
    w, h = 2 * cx, 2 * cy
    test_pixels = np.array([
        [0, 0, 1], [w, 0, 1],           # top corners
        [0, h, 1], [w, h, 1],           # bottom corners
        [0, h / 2, 1], [w, h / 2, 1],   # left/right midpoints
    ])

    x_vals = []
    for px in test_pixels:
        field_h = H @ px
        if abs(field_h[2]) < 1e-10:
            continue
        field_x = field_h[0] / field_h[2]
        field_y = field_h[1] / field_h[2]
        # Only include if the field point is roughly on the field
        if -20 < field_y < FIELD_WIDTH + 20:
            x_vals.append(field_x)

    if not x_vals:
        return (0.0, FIELD_LENGTH)

    x_min = max(0.0, min(x_vals))
    x_max = min(FIELD_LENGTH, max(x_vals))
    return (float(x_min), float(x_max))


def _compute_reproj_error(
    H: np.ndarray,
    H_inv: np.ndarray,
    yard_range: tuple[float, float],
) -> float:
    """Compute self-consistency reprojection error on field grid points."""
    # Generate test points within visible range
    x_min, x_max = yard_range
    test_pts = []
    for x in YARD_LINE_POSITIONS:
        if x_min - 5 <= x <= x_max + 5:
            for y in [0.0, HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH]:
                test_pts.append([x, y])

    if not test_pts:
        return 0.0

    test_pts = np.array(test_pts)

    # Field → pixel → field round trip
    ones = np.ones((len(test_pts), 1))
    field_h = np.hstack([test_pts, ones])
    pixel_h = (H_inv @ field_h.T).T
    pixel_h /= pixel_h[:, 2:3]

    back_h = (H @ pixel_h.T).T
    back_h /= back_h[:, 2:3]

    errors = np.sqrt(np.sum((back_h[:, :2] - test_pts) ** 2, axis=1))
    return float(np.mean(errors))


# ── Calibration ─────────────────────────────────────────────────────────────

def calibrate_camera(
    pixel_pts: np.ndarray,
    field_pts: np.ndarray,
    frame_shape: tuple[int, int] = (720, 1280),
) -> CameraCalibration:
    """Calibrate camera from a wide frame with many known correspondences.

    Solves for camera position (Cx, Cy, Cz), pan, tilt, and focal length.

    Args:
        pixel_pts: (N, 2) detected keypoint pixel positions
        field_pts: (N, 2) corresponding field coordinates (must be identified)
        frame_shape: (height, width) of the frame

    Returns:
        CameraCalibration with solved parameters
    """
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    N = len(pixel_pts)

    assert N >= 6, f"Need at least 6 correspondences for calibration, got {N}"

    # Initial guess
    # Camera roughly at midfield, behind near sideline, elevated
    x0 = np.array([
        60.0,    # Cx — roughly midfield
        -40.0,   # Cy — behind near sideline (negative y)
        30.0,    # Cz — ~90 feet up
        0.0,     # pan — looking at midfield
        -0.6,    # tilt — looking down (~35 degrees)
        800.0,   # focal length — moderate zoom
        0.0,     # roll — should be near zero
    ])

    def residuals(params):
        Cx, Cy, Cz, pan, tilt, f, roll = params
        C = np.array([Cx, Cy, Cz])

        R = rotation_matrix(pan, tilt, roll)
        t = -R @ C

        # Project field points to 3D (z=0)
        pts_3d = np.column_stack([field_pts, np.zeros(N)])
        p_cam = (R @ pts_3d.T).T + t

        # Check for points behind camera
        behind = p_cam[:, 2] <= 0
        if behind.any():
            return np.full(2 * N, 1000.0)

        x_norm = p_cam[:, 0] / p_cam[:, 2]
        y_norm = p_cam[:, 1] / p_cam[:, 2]

        u_pred = f * x_norm + cx
        v_pred = f * y_norm + cy

        res_x = u_pred - pixel_pts[:, 0]
        res_y = v_pred - pixel_pts[:, 1]

        return np.concatenate([res_x, res_y])

    # Bounds — roll limited to ±5 degrees
    bounds_lo = [10, -200, 10, -np.pi / 2, -np.pi / 2, 200, -np.radians(10)]
    bounds_hi = [110, -5, 120, np.pi / 2, 0.0, 10000, np.radians(10)]

    result = least_squares(
        residuals, x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=5000,
    )

    Cx, Cy, Cz, pan, tilt, f, roll = result.x

    # Compute RMSE
    res = result.fun
    rmse = np.sqrt(np.mean(res**2))

    intrinsics = CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy)
    extrinsics = CameraExtrinsics(position=np.array([Cx, Cy, Cz]))
    initial_state = CameraState(pan=pan, tilt=tilt, focal_length=f, roll=roll)

    return CameraCalibration(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        calibration_error=rmse,
        n_points_used=N,
        initial_state=initial_state,
    )


# ── Per-frame PTZ solver ───────────────────────────────────────────────────

def solve_ptz(
    pixel_pts: np.ndarray,
    field_pts: np.ndarray,
    calibration: CameraCalibration,
    prev_state: CameraState | None = None,
    temporal_weight: float = 0.1,
) -> CameraState | None:
    """Solve for pan, tilt, focal length given known correspondences.

    Args:
        pixel_pts: (N, 2) undistorted pixel positions of detected keypoints
        field_pts: (N, 2) identified field coordinates
        calibration: fixed camera calibration
        prev_state: previous frame's state (for temporal prior)
        temporal_weight: strength of temporal smoothness prior (pixels)

    Returns:
        CameraState or None if solve fails
    """
    N = len(pixel_pts)
    if N < 1:
        return None

    C = calibration.extrinsics.position
    cx = calibration.intrinsics.cx
    cy = calibration.intrinsics.cy

    # Initial guess from previous state or calibration
    if prev_state is not None:
        x0 = np.array([prev_state.pan, prev_state.tilt, prev_state.focal_length])
    else:
        x0 = np.array([
            calibration.initial_state.pan,
            calibration.initial_state.tilt,
            calibration.initial_state.focal_length,
        ])

    # Undistort the pixel points
    pixel_pts_u = undistort_points(pixel_pts, calibration.intrinsics)

    def residuals(params):
        pan, tilt, f = params
        # Roll is fixed from calibration, not solved per-frame
        roll = calibration.initial_state.roll
        R = rotation_matrix(pan, tilt, roll)
        t = -R @ C

        pts_3d = np.column_stack([field_pts, np.zeros(N)])
        p_cam = (R @ pts_3d.T).T + t

        behind = p_cam[:, 2] <= 0
        if behind.any():
            return np.full(2 * N + (3 if prev_state else 0), 1000.0)

        x_norm = p_cam[:, 0] / p_cam[:, 2]
        y_norm = p_cam[:, 1] / p_cam[:, 2]

        u_pred = f * x_norm + cx
        v_pred = f * y_norm + cy

        res_x = u_pred - pixel_pts_u[:, 0]
        res_y = v_pred - pixel_pts_u[:, 1]

        res = np.concatenate([res_x, res_y])

        # Temporal prior
        if prev_state is not None and temporal_weight > 0:
            temporal = np.array([
                temporal_weight * (pan - prev_state.pan),
                temporal_weight * (tilt - prev_state.tilt),
                temporal_weight * 0.01 * (f - prev_state.focal_length),
            ])
            res = np.concatenate([res, temporal])

        return res

    # Bounds
    bounds_lo = [-np.pi / 2, -np.pi / 2, 200]
    bounds_hi = [np.pi / 2, 0.0, 10000]

    result = least_squares(
        residuals, x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        max_nfev=500,
    )

    pan, tilt, f = result.x
    return CameraState(pan=pan, tilt=tilt, focal_length=f)


# ── Keypoint identity from grid spacing ─────────────────────────────────────

def identify_keypoints_by_grid(
    pixel_pts: np.ndarray,
    channel_ids: np.ndarray,
    calibration: CameraCalibration,
    state: CameraState,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign field coordinates to detected keypoints using ICP-style matching.

    Projects all known field points into the image using the current camera
    state, then matches each detection to its nearest projected field point
    (with channel consistency).

    Args:
        pixel_pts: (K, 2) detected keypoint pixel positions
        channel_ids: (K,) channel per keypoint (0=sideline, 1=hash)
        calibration: camera calibration
        state: current PTZ estimate

    Returns:
        (matched_pixel_pts, matched_field_pts) — only the successfully matched pairs
    """
    if len(pixel_pts) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    # Project all known field points
    all_field = FIELD_COORDS  # (M, 2)
    all_channels = POINT_CHANNELS  # (M,)
    all_projected = project_field_to_pixel(all_field, state, calibration)

    matched_pixel = []
    matched_field = []

    for i in range(len(pixel_pts)):
        ch = channel_ids[i]
        px = pixel_pts[i]

        # Only match against same channel
        ch_mask = all_channels == ch
        if not ch_mask.any():
            continue

        candidates = all_projected[ch_mask]
        candidate_field = all_field[ch_mask]

        # Find nearest
        dists = np.sqrt(np.sum((candidates - px) ** 2, axis=1))
        best = np.argmin(dists)

        # Reject if too far (more than 50 pixels)
        if dists[best] < 50.0:
            matched_pixel.append(px)
            matched_field.append(candidate_field[best])

    if not matched_pixel:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    return np.array(matched_pixel), np.array(matched_field)


def bootstrap_identity(
    pixel_pts: np.ndarray,
    channel_ids: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, CameraState]:
    """Establish keypoint identity on the first frame of a clip.

    Tries multiple pan hypotheses (sweeping across the field) and picks
    the one that produces the best reprojection error after PTZ solve.

    Args:
        pixel_pts: (K, 2) detected keypoint pixel positions
        channel_ids: (K,) channel per keypoint
        calibration: camera calibration

    Returns:
        (matched_pixel_pts, matched_field_pts, solved_state)
    """
    best_error = float("inf")
    best_result = None

    # Try pan values corresponding to looking at different parts of the field
    # Pan=0 looks at midfield; sweep ±30 degrees
    for pan_deg in range(-25, 26, 5):
        pan = np.radians(pan_deg)

        # Try a few focal lengths too
        for f in [600, 800, 1000, 1200, 1500]:
            trial_state = CameraState(
                pan=pan,
                tilt=calibration.initial_state.tilt,
                focal_length=f,
            )

            matched_px, matched_field = identify_keypoints_by_grid(
                pixel_pts, channel_ids, calibration, trial_state,
            )

            if len(matched_px) < 2:
                continue

            # Solve PTZ from these matches
            solved = solve_ptz(matched_px, matched_field, calibration)
            if solved is None:
                continue

            # Re-match with solved state (tighter fit)
            matched_px2, matched_field2 = identify_keypoints_by_grid(
                pixel_pts, channel_ids, calibration, solved,
            )

            if len(matched_px2) < 2:
                continue

            # Compute reprojection error
            projected = project_field_to_pixel(matched_field2, solved, calibration)
            error = np.sqrt(np.mean(np.sum((projected - matched_px2) ** 2, axis=1)))

            if error < best_error:
                best_error = error
                best_result = (matched_px2, matched_field2, solved)

    return best_result if best_result is not None else (
        np.array([]).reshape(0, 2),
        np.array([]).reshape(0, 2),
        calibration.initial_state,
    )


# ── PTZ Kalman Filter ──────────────────────────────────────────────────────

class PTZFilter:
    """Kalman filter for smooth pan/tilt/zoom tracking.

    State: [pan, tilt, f, d_pan, d_tilt, d_f]
    Constant-velocity motion model.
    """

    def __init__(
        self,
        process_noise_ptz: float = 0.001,
        process_noise_vel: float = 0.0001,
        measurement_noise: float = 0.01,
    ):
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1.0  # initial uncertainty
        self.initialized = False

        # Process noise
        self.Q = np.diag([
            process_noise_ptz,    # pan
            process_noise_ptz,    # tilt
            process_noise_ptz * 100,  # focal length (larger scale)
            process_noise_vel,    # pan velocity
            process_noise_vel,    # tilt velocity
            process_noise_vel * 100,  # focal length velocity
        ])

        # Measurement noise
        self.R = np.diag([
            measurement_noise,
            measurement_noise,
            measurement_noise * 100,
        ])

        # Measurement matrix: observe [pan, tilt, f]
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1  # pan
        self.H[1, 1] = 1  # tilt
        self.H[2, 2] = 1  # f

    def reset(self, state: CameraState):
        """Initialize or reset the filter from a known state."""
        self.state = np.array([
            state.pan, state.tilt, state.focal_length,
            0.0, 0.0, 0.0,  # zero initial velocity
        ])
        self.P = np.eye(6) * 0.1
        self.initialized = True

    def predict(self, dt: float = 1.0 / 30.0) -> CameraState:
        """Predict next state using constant-velocity model."""
        if not self.initialized:
            raise RuntimeError("PTZFilter not initialized. Call reset() first.")

        # State transition: x_new = F @ x
        F = np.eye(6)
        F[0, 3] = dt  # pan += d_pan * dt
        F[1, 4] = dt  # tilt += d_tilt * dt
        F[2, 5] = dt  # f += d_f * dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        return CameraState(
            pan=self.state[0],
            tilt=self.state[1],
            focal_length=self.state[2],
        )

    def update(self, measurement: CameraState) -> CameraState:
        """Update filter with a new PTZ measurement."""
        if not self.initialized:
            self.reset(measurement)
            return measurement

        z = np.array([measurement.pan, measurement.tilt, measurement.focal_length])

        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return CameraState(
            pan=self.state[0],
            tilt=self.state[1],
            focal_length=self.state[2],
        )

    def get_state(self) -> CameraState:
        """Get current filtered state."""
        return CameraState(
            pan=self.state[0],
            tilt=self.state[1],
            focal_length=self.state[2],
        )


# ── Integration ─────────────────────────────────────────────────────────────

def create_camera_homography_fn(
    calibration: CameraCalibration,
    detector=None,
    conf_thresh: float = 0.3,
    use_temporal_filter: bool = True,
):
    """Create a per-frame homography function using the camera model.

    Args:
        calibration: pre-computed camera calibration (from kickoff frame)
        detector: FieldKeypointDetector instance (if None, caller must
                  provide detections externally)
        conf_thresh: detection confidence threshold
        use_temporal_filter: whether to apply Kalman smoothing

    Returns:
        callable: frame -> HomographyResult | None
    """
    ptz_filter = PTZFilter() if use_temporal_filter else None
    prev_state = [None]  # mutable container for closure
    frame_idx = [0]

    def homography_fn(frame: np.ndarray) -> HomographyResult | None:
        if detector is None:
            return None

        # Detect keypoints
        detection = detector.detect(frame)

        if len(detection.pixel_xy) == 0 and prev_state[0] is None:
            return None

        # First frame: bootstrap identity
        if prev_state[0] is None:
            matched_px, matched_field, state = bootstrap_identity(
                detection.pixel_xy,
                detection.channel_ids,
                calibration,
            )
            if len(matched_px) < 2:
                return None

            if ptz_filter is not None:
                ptz_filter.reset(state)
            prev_state[0] = state

        else:
            # Predict
            if ptz_filter is not None:
                predicted = ptz_filter.predict()
            else:
                predicted = prev_state[0]

            # Match detections to field points using predicted state
            matched_px, matched_field = identify_keypoints_by_grid(
                detection.pixel_xy,
                detection.channel_ids,
                calibration,
                predicted,
            )

            if len(matched_px) >= 2:
                # Solve PTZ
                state = solve_ptz(
                    matched_px, matched_field,
                    calibration, prev_state[0],
                )
                if state is None:
                    state = predicted
            elif len(matched_px) == 1:
                # Single point: fix focal length, solve pan+tilt
                state = solve_ptz(
                    matched_px, matched_field,
                    calibration, prev_state[0],
                    temporal_weight=1.0,  # stronger prior
                )
                if state is None:
                    state = predicted
            else:
                # No matches: use prediction
                state = predicted

            # Update filter
            if ptz_filter is not None:
                state = ptz_filter.update(state)

            prev_state[0] = state

        frame_idx[0] += 1

        return camera_state_to_homography(
            prev_state[0],
            calibration,
            n_inliers=len(matched_px) if 'matched_px' in dir() else 0,
            n_correspondences=len(detection.pixel_xy),
        )

    return homography_fn
