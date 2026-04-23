"""Radial lens distortion: intrinsics + undistort helper.

Extracted from the old `camera_model.py` (the full PTZ camera-model approach
was abandoned in favor of per-frame 8-DOF homography; only distortion
correction survived).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for radial distortion correction."""
    fx: float              # focal length in pixels (horizontal)
    fy: float              # focal length in pixels (vertical), usually == fx
    cx: float              # principal point x (pixels)
    cy: float              # principal point y (pixels)
    k1: float = 0.0        # radial distortion coefficient 1
    k2: float = 0.0        # radial distortion coefficient 2


def apply_distortion(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    k1: float,
    k2: float,
) -> tuple:
    """Apply radial distortion to normalized (focal-length-scaled) coords."""
    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2
    return x_norm * radial, y_norm * radial


def undistort_points(
    pixel_pts: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_iters: int = 10,
) -> np.ndarray:
    """Remove radial distortion from pixel coordinates.

    Iterative method: given distorted point, find the undistorted point
    that would map to it under the distortion model.
    """
    if abs(intrinsics.k1) < 1e-12 and abs(intrinsics.k2) < 1e-12:
        return pixel_pts.copy()

    x_dist = (pixel_pts[:, 0] - intrinsics.cx) / intrinsics.fx
    y_dist = (pixel_pts[:, 1] - intrinsics.cy) / intrinsics.fy

    x_u = x_dist.copy()
    y_u = y_dist.copy()
    for _ in range(n_iters):
        r2 = x_u ** 2 + y_u ** 2
        radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r2 ** 2
        x_u = x_dist / radial
        y_u = y_dist / radial

    result = np.empty_like(pixel_pts)
    result[:, 0] = x_u * intrinsics.fx + intrinsics.cx
    result[:, 1] = y_u * intrinsics.fy + intrinsics.cy
    return result
