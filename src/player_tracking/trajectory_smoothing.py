"""
Trajectory smoothing and downsampling.

Takes raw 30fps tracked positions and produces clean 10fps output:
  1. Savitzky-Golay filter on 30fps pixel positions (removes box jitter)
  2. Downsample to 10fps (pick every 3rd frame)
  3. Optionally compute velocity and acceleration from smoothed positions

The key insight: bounding box positions jitter frame-to-frame even when the
player moves smoothly, because the detector independently estimates the box
on each frame. A low-pass filter removes this high-frequency noise before
downsampling, preventing aliased artifacts in the 10fps output.

We use Savitzky-Golay over Kalman because:
  - SG is non-causal (uses future frames) → better for offline batch processing
  - SG preserves sharp changes (route breaks) better than a Kalman smoother
  - No need for a motion model — the filter is purely data-driven
"""

import numpy as np
from scipy.signal import savgol_filter
from dataclasses import dataclass

from .tracker import PlayerTrajectory, TrajectoryPoint


@dataclass
class SmoothedTrajectory:
    """Smoothed and downsampled player trajectory."""
    track_id: int
    frame_indices: np.ndarray      # (T,) frame indices at 10fps
    times: np.ndarray              # (T,) time in seconds
    pixel_xy: np.ndarray           # (T, 2) smoothed pixel positions
    field_xy: np.ndarray           # (T, 2) field coordinates (NaN if unavailable)
    confidence: np.ndarray         # (T,) detection confidence
    interrupted: np.ndarray        # (T,) bool — True where data is unreliable
    velocity: np.ndarray | None = None      # (T, 2) yards/sec [vx, vy]
    speed: np.ndarray | None = None         # (T,) yards/sec magnitude
    acceleration: np.ndarray | None = None  # (T, 2) yards/sec^2 [ax, ay]
    accel_magnitude: np.ndarray | None = None  # (T,) yards/sec^2 magnitude


def smooth_trajectory(
    traj: PlayerTrajectory,
    source_fps: float = 30.0,
    target_fps: float = 10.0,
    window_ms: int = 200,
    poly_order: int = 2,
) -> SmoothedTrajectory | None:
    """Smooth a single player's trajectory and downsample to target fps.

    Args:
        traj: Raw trajectory from the tracker (at source_fps).
        source_fps: Frame rate of the source video.
        target_fps: Desired output frame rate (10fps to match NGS).
        window_ms: Savitzky-Golay window size in milliseconds. Controls
                   how much smoothing is applied. 200ms = 7 frames at 30fps.
        poly_order: Polynomial order for SG filter. 2 = quadratic, preserves
                    acceleration-level changes while smoothing jitter.

    Returns:
        SmoothedTrajectory at target_fps, or None if trajectory is too short.
    """
    if len(traj.points) < 5:
        return None

    # Extract raw arrays from trajectory
    frames = np.array([p.frame_idx for p in traj.points])
    pixel_xy = np.array([p.pixel_xy for p in traj.points], dtype=np.float64)
    confidences = np.array([p.confidence for p in traj.points])
    interrupted = np.array([p.interrupted for p in traj.points])

    field_xy_raw = np.array([
        p.field_xy if p.field_xy is not None else [np.nan, np.nan]
        for p in traj.points
    ], dtype=np.float64)

    # --- Savitzky-Golay smoothing on pixel positions ---
    # Window must be odd and > poly_order
    window_frames = int(window_ms / 1000.0 * source_fps)
    window_frames = max(window_frames, poly_order + 2)
    if window_frames % 2 == 0:
        window_frames += 1
    # Don't exceed trajectory length
    window_frames = min(window_frames, len(frames))
    if window_frames % 2 == 0:
        window_frames -= 1
    if window_frames <= poly_order:
        return None

    smooth_px = np.column_stack([
        savgol_filter(pixel_xy[:, 0], window_frames, poly_order),
        savgol_filter(pixel_xy[:, 1], window_frames, poly_order),
    ])

    # Smooth field coordinates where available (interpolate gaps first)
    smooth_field = _smooth_field_coords(field_xy_raw, window_frames, poly_order)

    # --- Downsample to target fps ---
    step = int(round(source_fps / target_fps))  # 30/10 = 3
    ds_indices = np.arange(0, len(frames), step)

    result = SmoothedTrajectory(
        track_id=traj.track_id,
        frame_indices=frames[ds_indices],
        times=frames[ds_indices] / source_fps,
        pixel_xy=smooth_px[ds_indices],
        field_xy=smooth_field[ds_indices],
        confidence=confidences[ds_indices],
        interrupted=interrupted[ds_indices],
    )

    # --- Compute derivatives from smoothed field coordinates ---
    _compute_derivatives(result, target_fps)

    return result


def _smooth_field_coords(
    field_xy: np.ndarray,
    window: int,
    poly_order: int,
) -> np.ndarray:
    """Smooth field coordinates, handling NaN gaps.

    For segments with valid field coordinates, applies SG smoothing.
    NaN regions are left as NaN.
    """
    result = field_xy.copy()
    valid = ~np.isnan(field_xy[:, 0])

    if valid.sum() < window:
        return result

    # Find contiguous valid segments
    segments = _contiguous_segments(valid)

    for start, end in segments:
        length = end - start
        if length < max(5, poly_order + 2):
            continue
        w = min(window, length)
        if w % 2 == 0:
            w -= 1
        if w <= poly_order:
            continue
        result[start:end, 0] = savgol_filter(field_xy[start:end, 0], w, poly_order)
        result[start:end, 1] = savgol_filter(field_xy[start:end, 1], w, poly_order)

    return result


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True segments in a boolean array.

    Returns list of (start, end) pairs (end is exclusive).
    """
    segments = []
    in_segment = False
    start = 0

    for i in range(len(mask)):
        if mask[i] and not in_segment:
            start = i
            in_segment = True
        elif not mask[i] and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(mask)))

    return segments


def _compute_derivatives(traj: SmoothedTrajectory, fps: float):
    """Compute velocity, speed, and acceleration from smoothed field positions.

    Uses central differences for interior points and forward/backward
    differences at the boundaries. Only computes where field coords are valid.
    """
    dt = 1.0 / fps
    n = len(traj.field_xy)

    if n < 2:
        return

    xy = traj.field_xy
    valid = ~np.isnan(xy[:, 0])

    # Velocity via central differences
    vel = np.full((n, 2), np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        if i > 0 and i < n - 1 and valid[i - 1] and valid[i + 1]:
            vel[i] = (xy[i + 1] - xy[i - 1]) / (2 * dt)
        elif i < n - 1 and valid[i + 1]:
            vel[i] = (xy[i + 1] - xy[i]) / dt
        elif i > 0 and valid[i - 1]:
            vel[i] = (xy[i] - xy[i - 1]) / dt

    speed = np.sqrt(np.nansum(vel ** 2, axis=1))
    speed[~valid] = np.nan

    # Acceleration via central differences on velocity
    accel = np.full((n, 2), np.nan)
    vel_valid = ~np.isnan(vel[:, 0])
    for i in range(n):
        if not vel_valid[i]:
            continue
        if i > 0 and i < n - 1 and vel_valid[i - 1] and vel_valid[i + 1]:
            accel[i] = (vel[i + 1] - vel[i - 1]) / (2 * dt)
        elif i < n - 1 and vel_valid[i + 1]:
            accel[i] = (vel[i + 1] - vel[i]) / dt
        elif i > 0 and vel_valid[i - 1]:
            accel[i] = (vel[i] - vel[i - 1]) / dt

    accel_mag = np.sqrt(np.nansum(accel ** 2, axis=1))
    accel_mag[~valid] = np.nan

    traj.velocity = vel
    traj.speed = speed
    traj.acceleration = accel
    traj.accel_magnitude = accel_mag


def smooth_all_trajectories(
    trajectories: dict[int, PlayerTrajectory],
    source_fps: float = 30.0,
    target_fps: float = 10.0,
    window_ms: int = 200,
    min_frames: int = 10,
) -> dict[int, SmoothedTrajectory]:
    """Smooth and downsample all player trajectories.

    Args:
        trajectories: dict of track_id -> PlayerTrajectory from the tracker.
        source_fps: Input frame rate.
        target_fps: Output frame rate (10fps for NGS comparison).
        window_ms: Smoothing window in milliseconds.
        min_frames: Skip trajectories shorter than this.

    Returns:
        dict of track_id -> SmoothedTrajectory.
    """
    smoothed = {}
    for track_id, traj in trajectories.items():
        if len(traj.points) < min_frames:
            continue
        result = smooth_trajectory(traj, source_fps, target_fps, window_ms)
        if result is not None:
            smoothed[track_id] = result
    return smoothed
