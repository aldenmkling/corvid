"""Line-fitting primitives for the homography pipeline.

Provides:
  - `total_mse`: residual MSE for joint k1 calibration over yardline +
    sideline pixel groups (uses undistortion).
  - `ransac_line`: sequential RANSAC `y = m·x + c` line fit.
  - `fit_yardline_undistorted`, `fit_sideline_undistorted`: per-CC linear
    fits in undistorted space (yardline as `x = a + b·y`, sideline as
    `y = a + b·x`).
  - Mask thresholds + hash subsample cap consumed by the rectify pipeline.

No I/O, no rendering, no `main()`. Pure numerics.
"""

import numpy as np

from src.homography.distortion import CameraIntrinsics, undistort_points


# UNet mask thresholds (per-channel sigmoid).
YARD_THRESH = 0.5
SIDE_THRESH = 0.5
HASH_THRESH = 0.5

# Subsample hash mask pixels to keep undistortion + line fitting fast.
MAX_HASH_PIXELS = 8000


def total_mse(line_pts, line_kinds, intr: CameraIntrinsics):
    """Joint residual MSE across yardline + sideline pixel groups after
    undistortion. Used by `scipy.optimize.minimize_scalar` to calibrate k1.

    Yardlines are fit as `x = a + b·y`; sidelines as `y = a + b·x`. The
    perpendicular residual to the fitted line is normalized by
    `sqrt(1 + b²)` so yardline + sideline residuals share the same scale.
    """
    total_sq = 0.0; n = 0
    for p, kind in zip(line_pts, line_kinds):
        p_u = undistort_points(p.astype(np.float64), intr)
        if kind == "yardline":
            ys, xs = p_u[:, 1], p_u[:, 0]
            b, a = np.polyfit(ys, xs, 1)
            resid = (xs - (a + b * ys)) / np.sqrt(1 + b * b)
        else:
            xs, ys = p_u[:, 0], p_u[:, 1]
            b, a = np.polyfit(xs, ys, 1)
            resid = (ys - (a + b * xs)) / np.sqrt(1 + b * b)
        total_sq += float((resid ** 2).sum())
        n += len(p)
    return total_sq / max(n, 1)


def ransac_line(pts: np.ndarray, n_iters: int = 800,
                inlier_dist: float = 2.0,
                min_inliers: int = 30,
                seed: int = 0) -> tuple[float | None, float | None, np.ndarray | None]:
    """Sequential RANSAC line fit. Returns (m, c, inlier_mask) for
    `y = m·x + c`, or (None, None, None) if no consensus.
    """
    n = len(pts)
    if n < 2:
        return None, None, None
    rng = np.random.RandomState(seed)
    best_count = 0
    best_in = None
    for _ in range(n_iters):
        i, j = rng.choice(n, 2, replace=False)
        dx = pts[j, 0] - pts[i, 0]
        if abs(dx) < 1e-3:
            continue
        m = (pts[j, 1] - pts[i, 1]) / dx
        c = pts[i, 1] - m * pts[i, 0]
        d = np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(1.0 + m * m)
        in_mask = d < inlier_dist
        cnt = int(in_mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_in = in_mask
    if best_in is None or best_count < min_inliers:
        return None, None, None
    in_pts = pts[best_in]
    m, c = np.polyfit(in_pts[:, 0], in_pts[:, 1], 1)
    return float(m), float(c), best_in


def fit_yardline_undistorted(pixels: np.ndarray, intr: CameraIntrinsics):
    """Fit `x = a + b·y` to a yardline CC's pixels in undistorted space."""
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    ys, xs = pts_u[:, 1], pts_u[:, 0]
    b, a = np.polyfit(ys, xs, 1)
    return {'a': float(a), 'b': float(b),
            'ymin': float(ys.min()), 'ymax': float(ys.max())}


def fit_sideline_undistorted(pixels: np.ndarray, intr: CameraIntrinsics):
    """Fit `y = a + b·x` to a sideline CC's pixels in undistorted space."""
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    xs, ys = pts_u[:, 0], pts_u[:, 1]
    b, a = np.polyfit(xs, ys, 1)
    return {'a': float(a), 'b': float(b),
            'xmin': float(xs.min()), 'xmax': float(xs.max())}
