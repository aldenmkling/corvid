"""
Compute homography matrix from detected field features and yard line identifications.

Key insight: yard lines are our strongest constraints. Every point along a
detected yard line has a known field x-coordinate. The homography MUST map
detected yard line positions exactly to their field x values.

Correspondences come from:
  - Multiple points sampled along each identified yard line (strong x-constraints)
  - Yard line × sideline intersections (gives y=0 and y=53.33)
  - Yard line × hash mark positions (gives y=23.58 and y=29.75)
"""

import numpy as np
import cv2
from dataclasses import dataclass

from .field_features import FieldFeatures
from . import field_model


@dataclass
class HomographyResult:
    H: np.ndarray
    H_inv: np.ndarray
    reprojection_error: float  # RMSE in pixels
    n_inliers: int
    n_correspondences: int
    yard_range: tuple[float, float]
    inlier_mask: np.ndarray | None = None


def _line_intersection(
    line1: tuple[float, float, float, float],
    line2: tuple[float, float, float, float],
) -> np.ndarray | None:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def _estimate_y_at_fraction(
    frac: float,
    sideline_near_y: float | None,
    sideline_far_y: float | None,
    field_mask_top: float,
    field_mask_bottom: float,
    frame_height: int,
) -> float:
    """Estimate the field y-coordinate for a point at a given fraction along a yard line.

    frac=0 is the top of the frame (far sideline), frac=1 is the bottom (near sideline).
    Uses sideline positions if available, otherwise interpolates from field mask.
    """
    # If both sidelines are known, linear interpolation
    if sideline_near_y is not None and sideline_far_y is not None:
        return sideline_far_y + frac * (sideline_near_y - sideline_far_y)

    # If one sideline is known, use field mask for the other
    # Map the field mask extent to [0, FIELD_WIDTH]
    mask_range = field_mask_bottom - field_mask_top
    if mask_range < 10:
        return field_model.FIELD_WIDTH * frac

    # Estimate: field mask top ≈ far sideline, bottom ≈ near sideline
    if sideline_near_y is not None:
        # We know the bottom, estimate top from mask
        return field_model.FIELD_WIDTH * (1 - frac) + sideline_near_y * frac
    elif sideline_far_y is not None:
        return sideline_far_y * (1 - frac)
    else:
        # No sidelines — pure interpolation
        return field_model.FIELD_WIDTH * frac


def generate_correspondences(
    features: FieldFeatures,
    yard_ids: dict[int, float],
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pixel ↔ field coordinate correspondences.

    Samples multiple points along each identified yard line to create
    strong x-constraints. Cross-field y-coordinates come from sidelines,
    hash marks, or field mask interpolation.
    """
    h, w = frame_shape
    pixel_pts = []
    field_pts = []

    # Get sideline y-positions in pixel space
    sl_near_pixel_y = None
    sl_far_pixel_y = None
    if features.sideline_near:
        sl = features.sideline_near
        sl_near_pixel_y = (sl.y1 + sl.y2) / 2  # approximate
    if features.sideline_far:
        sl = features.sideline_far
        sl_far_pixel_y = (sl.y1 + sl.y2) / 2

    # Field mask vertical extent
    if features.field_mask is not None:
        row_coverage = np.sum(features.field_mask > 0, axis=1) / w
        field_rows = np.where(row_coverage > 0.15)[0]
        if len(field_rows) > 0:
            mask_top = float(field_rows[0])
            mask_bottom = float(field_rows[-1])
        else:
            mask_top, mask_bottom = 0.0, float(h)
    else:
        mask_top, mask_bottom = 0.0, float(h)

    for cluster_idx, ngs_x in yard_ids.items():
        if cluster_idx >= len(features.yard_line_clusters):
            continue

        cluster = features.yard_line_clusters[cluster_idx]
        if cluster.fitted_line is None:
            if len(cluster.segments) >= 1:
                cluster.fit_line(h)
            if cluster.fitted_line is None:
                continue

        x1, y1, x2, y2 = cluster.fitted_line

        # A. Sideline intersection points (exact y-coordinates)
        if features.sideline_near:
            sl = features.sideline_near
            pt = _line_intersection(cluster.fitted_line, (sl.x1, sl.y1, sl.x2, sl.y2))
            if pt is not None and -w * 0.3 < pt[0] < w * 1.3 and -h * 0.3 < pt[1] < h * 1.3:
                pixel_pts.append(pt)
                field_pts.append(np.array([ngs_x, 0.0]))

        if features.sideline_far:
            sl = features.sideline_far
            pt = _line_intersection(cluster.fitted_line, (sl.x1, sl.y1, sl.x2, sl.y2))
            if pt is not None and -w * 0.3 < pt[0] < w * 1.3 and -h * 0.3 < pt[1] < h * 1.3:
                pixel_pts.append(pt)
                field_pts.append(np.array([ngs_x, field_model.FIELD_WIDTH]))

        # B. Hash mark intersection points
        if cluster.hash_points and len(cluster.hash_points) >= 2:
            hash_ys_sorted = sorted(cluster.hash_points, key=lambda p: p[1])
            mid_hash_y = (hash_ys_sorted[0][1] + hash_ys_sorted[-1][1]) / 2

            upper_hashes = [p for p in hash_ys_sorted if p[1] < mid_hash_y]
            lower_hashes = [p for p in hash_ys_sorted if p[1] >= mid_hash_y]

            if upper_hashes:
                pt = np.mean(upper_hashes, axis=0)
                pixel_pts.append(pt)
                field_pts.append(np.array([ngs_x, field_model.HASH_Y_FAR]))
            if lower_hashes:
                pt = np.mean(lower_hashes, axis=0)
                pixel_pts.append(pt)
                field_pts.append(np.array([ngs_x, field_model.HASH_Y_NEAR]))

        # No interpolated points — only use hard constraints (sideline/hash intersections)
        # to avoid pulling the homography off the exact yard line positions.

    if not pixel_pts:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    return np.array(pixel_pts), np.array(field_pts)


def compute_homography(
    pixel_pts: np.ndarray,
    field_pts: np.ndarray,
    ransac_threshold: float = 5.0,
) -> HomographyResult | None:
    if len(pixel_pts) < 4:
        return None

    H, mask = cv2.findHomography(
        pixel_pts.astype(np.float64),
        field_pts.astype(np.float64),
        cv2.RANSAC,
        ransac_threshold,
    )

    if H is None:
        return None

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    n = len(pixel_pts)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(n, dtype=bool)
    n_inliers = int(inlier_mask.sum())

    # Pixel-space reprojection error
    field_h = np.hstack([field_pts, np.ones((n, 1))])
    reprojected_px = (H_inv @ field_h.T).T
    reprojected_px_2d = reprojected_px[:, :2] / reprojected_px[:, 2:3]
    px_errors = np.sqrt(np.sum((reprojected_px_2d - pixel_pts) ** 2, axis=1))
    reproj_error_px = float(np.mean(px_errors[inlier_mask]))

    yard_range = (float(field_pts[:, 0].min()), float(field_pts[:, 0].max()))

    return HomographyResult(
        H=H,
        H_inv=H_inv,
        reprojection_error=reproj_error_px,
        n_inliers=n_inliers,
        n_correspondences=n,
        yard_range=yard_range,
        inlier_mask=inlier_mask,
    )


def compute_homography_from_features(
    frame: np.ndarray,
    features: FieldFeatures,
    yard_ids: dict[int, float],
) -> HomographyResult | None:
    pixel_pts, field_pts = generate_correspondences(
        features, yard_ids, frame.shape[:2]
    )
    return compute_homography(pixel_pts, field_pts)
