"""
Transform coordinates between pixel space and field space using homography.
"""

import numpy as np

from . import field_model


def pixel_to_field(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform pixel coordinates to field coordinates.

    Args:
        pixel_xy: (N, 2) array of pixel coordinates
        H: 3x3 homography matrix (pixel → field)

    Returns: (N, 2) array of field coordinates (x=NGS yard, y=width)
    """
    if pixel_xy.ndim == 1:
        pixel_xy = pixel_xy.reshape(1, 2)

    pts_h = np.hstack([pixel_xy, np.ones((len(pixel_xy), 1))])
    result = (H @ pts_h.T).T
    w = result[:, 2:3]
    w[np.abs(w) < 1e-10] = 1e-10  # avoid division by zero
    return result[:, :2] / w


def field_to_pixel(field_xy: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Transform field coordinates to pixel coordinates.

    Args:
        field_xy: (N, 2) array of field coordinates
        H_inv: 3x3 inverse homography matrix (field → pixel)

    Returns: (N, 2) array of pixel coordinates
    """
    if field_xy.ndim == 1:
        field_xy = field_xy.reshape(1, 2)

    pts_h = np.hstack([field_xy, np.ones((len(field_xy), 1))])
    result = (H_inv @ pts_h.T).T
    w = result[:, 2:3]
    w[np.abs(w) < 1e-10] = 1e-10
    return result[:, :2] / w


def is_on_field(field_xy: np.ndarray, margin: float = 2.0) -> np.ndarray:
    """Check if field coordinates are within the field boundaries.

    Args:
        field_xy: (N, 2) array of field coordinates
        margin: extra yards outside the field to still accept (for near-sideline players)

    Returns: (N,) boolean array
    """
    if field_xy.ndim == 1:
        field_xy = field_xy.reshape(1, 2)

    x_ok = (field_xy[:, 0] >= field_model.GOAL_LINE_LEFT - margin) & \
           (field_xy[:, 0] <= field_model.GOAL_LINE_RIGHT + margin)
    y_ok = (field_xy[:, 1] >= -margin) & \
           (field_xy[:, 1] <= field_model.FIELD_WIDTH + margin)

    return x_ok & y_ok
