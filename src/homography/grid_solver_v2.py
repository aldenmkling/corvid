"""Grid solver v2 — polynomial-fit pipeline from UNet line masks + W18 hashes.

Rewrite of `grid_solver.py` for the new input regime:
  - UNet produces dense per-pixel line masks (2 channels: yard, side).
  - HRNet-W18 produces sparse hash-intersection point detections.

Pipeline:
  1. `group_yardline_pixels(yard_mask)` — peak-pick on cross-line projection to
     split the yardline mask into one object per yard line.
  2. `group_sideline_pixels(side_mask)` — top/bottom split to give up to 2
     sideline objects (far / near).
  3. `fit_line_polynomial(line_obj)` — degree-2 polynomial through each line.
     Yardlines fit x = f(y); sidelines fit y = g(x).
  4. `assign_hashes_to_yardlines(hash_pxs, yardlines)` — each hash snaps to its
     nearest yardline polynomial (cross-distance in px). Far/near classified
     by y vs. paired-hash midline.
  5. `intersect_yardline_sideline(yl, sl)` — polynomial intersection gives sub-
     pixel sideline-intersection keypoints.
  6. `assign_grid_positions(yardlines)` — integer slot index per yardline
     (leftmost = 0), via min-diff + weighted LS refit for bimodal 5yd/10yd.
  7. `solve_grid(...)` — top-level orchestrator, emits correspondences.
  8. `calibrate_distortion` — plumb-line (k1, k2) fit from polynomial residuals.

Kept API-compatible with `grid_solver.py` where practical:
  - `groups_to_correspondences(yardlines, base_ngs_x, frame_shape)` — same sig.
  - `GridSolverResult` is the canonical return type (new).

The old solver in `grid_solver.py` is left untouched so we can A/B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from scipy import ndimage, signal as sp_signal

from .field_model import (
    HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH,
)


# ── Constants ────────────────────────────────────────────────────────────────

UNET_INPUT_H, UNET_INPUT_W = 512, 896
HASH_INPUT_H, HASH_INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Operating thresholds for inference-time peak extraction
UNET_YARD_THRESH = 0.5
UNET_SIDE_THRESH = 0.5
HASH_THRESH = 0.40

# Polynomial degree defaults. Start at 4 because the radial distortion model is
# k1·r² + k2·r⁴ (quartic in r), so a straight field line traces a curve whose
# Taylor expansion needs up to deg 4 to capture k2·r⁴ effects off-axis. Can
# drop to 2 later if residuals show deg 4 is overfitting.
POLY_DEG_YARDLINE = 4
POLY_DEG_SIDELINE = 4

# Yardline grouping via tilt-corrected column-coord peak-pick on UNet mask
# pixels. Robust to occluded/broken lines (unlike connected components) because
# grouping is by column-coord similarity, not spatial connectivity.
#   column_coord = (pt - mean) @ cross_corrected
# where cross_corrected is perpendicular to the global along-line tilt.
YARDLINE_MIN_PEAK_SEP_PX = 25.0     # min px between adjacent yardline peaks.
                                     # Empirically: real yardlines are ≥30 px
                                     # apart even in the most perspective-
                                     # compressed regions. Below this gets
                                     # double-peaks from single-line mask
                                     # thickness variation.
YARDLINE_PEAK_BANDWIDTH_FRAC = 0.45  # half-width (frac of peak spacing) for
                                       # pixel-to-peak assignment
YARDLINE_MIN_PEAK_PROM = 0.10        # peak prominence as frac of hist max
YARDLINE_MIN_PIXELS_PER_LINE = 500   # absolute floor on group pixel count
YARDLINE_MIN_PIXEL_FRAC_OF_MEDIAN = 0.10   # also drop groups < 10% of the
                                            # median real-line group size
                                            # (adaptive noise rejection)
YARDLINE_POLY_REFINE_ITERS = 0       # iterative pixel→poly reassignment passes.
                                       # Disabled by default — with the angle-
                                       # sweep tilt estimate, initial peak-pick
                                       # is already clean, and refinement with
                                       # deg-4 polys can pull pixels from
                                       # entirely different lines into a group
                                       # by bending the fit. Re-enable only
                                       # with a tight cross_tol.
YARDLINE_POLY_CROSS_TOL_PX = 6.0     # drop pixels > this from any line's poly
                                       # during refinement

# Sideline grouping: simple peak-pick on y works well because sidelines are
# mostly horizontal in frame and there are only ever 0-2 of them.
SIDELINE_MIN_PEAK_SEP_PX = 60.0
SIDELINE_MIN_PEAK_PROM = 0.10

# Hash-to-yardline assignment tolerance in px. Must be big enough to allow for
# yardline polynomial fit residuals + HRNet peak jitter. Revisit.
HASH_TO_YARDLINE_MAX_DIST_PX = 25.0


# ── Data types ───────────────────────────────────────────────────────────────

# Module-level slot for handing the VP from grouping to grid assignment.
# A clean dependency-injection version would thread vp through every call;
# this is simpler and keeps the public API unchanged.
_LAST_VP: list = [None]


@dataclass
class LineObject:
    """One grouped line (yardline or sideline) — UNet pixel cluster + fit.

    Polynomial is fit on centered+scaled input:  q_norm = (q - q_offset) / q_scale
    Stored coeffs apply to q_norm. `eval_poly()` handles the inverse transform.
    Centering avoids ill-conditioning at deg 4+ over a ~720-px y range.
    """
    pixels: np.ndarray                    # (N, 2) pixel (x, y) coords
    kind: str                             # "yardline" or "sideline"
    poly: Optional[np.ndarray] = None     # polyfit coeffs (highest-order first)
    poly_axis: str = ""                   # "x_of_y" (yardlines) or "y_of_x"
    poly_q_offset: float = 0.0            # mean of the fit input axis
    poly_q_scale: float = 1.0             # std of the fit input axis
    residual_rmse: Optional[float] = None
    # For grouping diagnostics:
    peak_coord: Optional[float] = None    # cross-line-axis coordinate of peak


@dataclass
class Yardline:
    """A yardline polynomial with all keypoints attached to it."""
    line: LineObject                      # the yardline polynomial
    near_hash: Optional[np.ndarray] = None     # (2,) pixel coord
    far_hash: Optional[np.ndarray] = None      # (2,) pixel coord
    near_sideline: Optional[np.ndarray] = None # (2,) pixel, poly ∩ near SL
    far_sideline: Optional[np.ndarray] = None  # (2,) pixel, poly ∩ far SL
    grid_pos: Optional[int] = None             # integer slot (0 = leftmost)
    grid_fit_residual: Optional[float] = None  # column-coord error from grid
    grid_fit_ok: bool = False


@dataclass
class GridSolverResult:
    """Complete output of the grid solver for one frame."""
    yardlines: list[Yardline] = field(default_factory=list)
    near_sideline: Optional[LineObject] = None
    far_sideline: Optional[LineObject] = None
    # Vanishing point (vx, vy) used for yardline rectification.  Pixels on the
    # same yardline share the same atan2(y-vy, x-vx) angle, so this is the
    # natural perspective-invariant column-coord.
    vp: Optional[tuple[float, float]] = None
    # Diagnostic info:
    frame_shape: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)
    # If solve_grid linearized the result (default), the lines+keypoints are
    # in UNDISTORTED pixel space and the polynomial fits are deg-1. The
    # intrinsics that were applied are stored here so downstream callers can
    # use them directly without re-doing distortion calibration.
    is_linearized: bool = False
    intrinsics: Optional[object] = None       # CameraIntrinsics if linearized


# ── 1. Pixel grouping ────────────────────────────────────────────────────────

def _mask_pixels(mask: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (x, y) pixel coords where mask is truthy."""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _pca_axes(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, along_dir, cross_dir) from 2D PCA.

    For N parallel yardlines each running y=0..H and spread across x=0..W with
    W > H (standard 1280x720), the pixel cloud has more horizontal variance
    (across lines) than vertical (along lines per-line averaged), so vt[0]
    points horizontally — i.e. ACROSS lines. cross_dir = vt[0], along_dir
    perpendicular.
    """
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    cross_dir = vt[0]                              # largest variance = across lines
    along_dir = np.array([-cross_dir[1], cross_dir[0]])
    return mean, along_dir, cross_dir


def _peak_pick_1d(
    coords: np.ndarray,
    bin_width: float = 1.0,
    min_sep: float = 20.0,
    min_prom_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram `coords` then find peaks.

    Returns (peak_coords, peak_heights), each length K.
    """
    c_min, c_max = float(coords.min()), float(coords.max())
    if c_max - c_min < 1e-6:
        return np.array([c_min]), np.array([len(coords)], dtype=float)

    n_bins = max(8, int(np.ceil((c_max - c_min) / bin_width)))
    hist, edges = np.histogram(coords, bins=n_bins, range=(c_min, c_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth the histogram with a small box filter so rough edges don't create
    # spurious micro-peaks. Kernel width ~ min_sep / 4 bins, capped so it's
    # always <= n_bins (otherwise np.convolve mode="same" returns an array
    # longer than hist, misaligning downstream indexing).
    bins_per_sep = (c_max - c_min) / n_bins
    box_width = max(3, int(round(min_sep / max(bins_per_sep, 1e-6) / 4.0)))
    if box_width % 2 == 0:
        box_width += 1
    box_width = min(box_width, n_bins - 1 if n_bins > 3 else 3)
    if box_width % 2 == 0:
        box_width -= 1
    box_width = max(box_width, 1)
    kernel = np.ones(box_width) / box_width
    smoothed = np.convolve(hist.astype(float), kernel, mode="same")

    distance_bins = max(1, int(round(min_sep / max(bins_per_sep, 1e-6))))
    prominence = float(smoothed.max()) * min_prom_frac

    peaks, props = sp_signal.find_peaks(
        smoothed, distance=distance_bins, prominence=prominence,
    )
    return centers[peaks], smoothed[peaks]


def _score_projection(
    coords: np.ndarray,
    min_peak_sep: float,
    min_prom_frac: float,
) -> float:
    """Score a 1D projection by within-peak / between-peak variance ratio.

    Classical Fisher-discriminant framing: we want TIGHT peaks that are WELL-
    SEPARATED. A global scale that shrinks all coords (e.g. VP at infinity
    bunching all atan2 values near ±π/2) shrinks both within- and between-
    variance proportionally, so the ratio is invariant and doesn't reward
    trivial bunching.

    Returns `within_var / between_var`, lower is better. +inf if <2 peaks.
    """
    c_min, c_max = float(coords.min()), float(coords.max())
    if c_max - c_min < 1e-9:
        return float("inf")
    bin_width = (c_max - c_min) / max(300, int((c_max - c_min) / (min_peak_sep * 0.25)))
    n_bins = max(32, int(np.ceil((c_max - c_min) / bin_width)))
    hist, _ = np.histogram(coords, bins=n_bins, range=(c_min, c_max))
    box_width = max(3, int(round(min_peak_sep / bin_width / 4.0)))
    if box_width % 2 == 0:
        box_width += 1
    smoothed = np.convolve(hist.astype(float),
                            np.ones(box_width) / box_width, mode="same")
    distance_bins = max(1, int(round(min_peak_sep / bin_width)))
    prom = float(smoothed.max()) * min_prom_frac
    peaks, _ = sp_signal.find_peaks(
        smoothed, distance=distance_bins, prominence=prom,
    )
    if len(peaks) < 2:
        return float("inf")

    peak_coords_c = c_min + (peaks + 0.5) * bin_width
    dist = np.abs(coords[:, None] - peak_coords_c[None, :])
    nearest = dist.argmin(axis=1)

    within_var = 0.0
    assigned = 0
    peak_means: list[float] = []
    peak_counts: list[int] = []
    for k in range(len(peak_coords_c)):
        sel = nearest == k
        n_k = int(sel.sum())
        if n_k < 20:
            continue
        m_k = float(coords[sel].mean())
        resid = coords[sel] - m_k
        within_var += float((resid ** 2).sum())
        assigned += n_k
        peak_means.append(m_k)
        peak_counts.append(n_k)
    if assigned == 0 or len(peak_means) < 2:
        return float("inf")

    within_var /= assigned

    # Between-peak variance: weighted variance of peak means (weight = pixel count)
    total_mean = sum(c * m for c, m in zip(peak_counts, peak_means)) / assigned
    between_var = sum(c * (m - total_mean) ** 2 for c, m in zip(peak_counts, peak_means)) / assigned

    if between_var < 1e-12:
        return float("inf")
    return within_var / between_var


def _find_best_vanishing_point(
    pts: np.ndarray,
    frame_shape: tuple[int, int],
    coarse_angle_deg: tuple[float, float] = (-45.0, 45.0),
    coarse_n_angle: int = 46,
    coarse_log_dist_range: tuple[float, float] = (2.5, 5.0),  # log10(dist_px): 316 → 100000
    coarse_n_dist: int = 20,
    refine_angle_halfwidth_deg: float = 2.0,
    refine_log_dist_halfwidth: float = 0.3,
    refine_steps: int = 15,
    min_peak_sep_px: float = YARDLINE_MIN_PEAK_SEP_PX,
    min_prom_frac: float = YARDLINE_MIN_PEAK_PROM,
) -> tuple[float, float]:
    """Find the vanishing point (vx, vy) whose polar-around-VP projection
    minimizes within-peak variance.

    Parameterization: VP sits at (image_center_x + d·sin(θ), image_center_y - d·cos(θ))
    where θ is angle from vertical (positive → VP up-right) and d is the
    distance from image center. We search in log10(d) so an infinite-VP
    (parallel yardlines) case is approached smoothly at large d.

    Projection: u(pixel) = atan2(y - vy, x - vx). All pixels on one yardline
    through VP share the same u (up to lens-distortion residuals).

    Two-stage: coarse grid, then local refine around the winner.
    """
    if len(pts) < 100:
        return (frame_shape[1] / 2.0, -1e9)

    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0

    def eval_vp(vx: float, vy: float) -> float:
        dx = pts[:, 0] - vx
        dy = pts[:, 1] - vy
        if (dx == 0).any() and (dy == 0).any():
            return float("inf")  # VP sits on a pixel — degenerate
        u = np.arctan2(dy, dx)
        # atan2 discontinuity at ±π: unwrap onto a contiguous range for the
        # case where pixels straddle the branch cut (VP inside/near pixel cloud).
        u_min, u_max = float(u.min()), float(u.max())
        if u_max - u_min > np.pi:
            u = np.where(u < 0, u + 2 * np.pi, u)
        # min_peak_sep in angular terms: ~ YARDLINE_MIN_PEAK_SEP_PX / distance_to_cloud
        mean_dist = float(np.hypot(dx, dy).mean())
        ang_min_sep = min_peak_sep_px / max(mean_dist, 1.0)
        return _score_projection(u, ang_min_sep, min_prom_frac)

    # ── Coarse grid ────────────────────────────────────────────────────
    angles = np.linspace(coarse_angle_deg[0], coarse_angle_deg[1], coarse_n_angle)
    log_dists = np.linspace(coarse_log_dist_range[0], coarse_log_dist_range[1], coarse_n_dist)

    best_score = float("inf")
    best_angle, best_log_dist = 0.0, coarse_log_dist_range[1]
    best_vp = (cx, -(10 ** coarse_log_dist_range[1]))

    for angle_deg in angles:
        ang_rad = np.radians(angle_deg)
        sin_a, cos_a = np.sin(ang_rad), np.cos(ang_rad)
        for log_d in log_dists:
            d = 10 ** log_d
            vx = cx + d * sin_a
            vy = cy - d * cos_a
            score = eval_vp(vx, vy)
            if score < best_score:
                best_score = score
                best_angle, best_log_dist = float(angle_deg), float(log_d)
                best_vp = (vx, vy)

    # ── Local refine ───────────────────────────────────────────────────
    ang_lo = best_angle - refine_angle_halfwidth_deg
    ang_hi = best_angle + refine_angle_halfwidth_deg
    ld_lo = best_log_dist - refine_log_dist_halfwidth
    ld_hi = best_log_dist + refine_log_dist_halfwidth
    for angle_deg in np.linspace(ang_lo, ang_hi, refine_steps):
        ang_rad = np.radians(angle_deg)
        sin_a, cos_a = np.sin(ang_rad), np.cos(ang_rad)
        for log_d in np.linspace(ld_lo, ld_hi, refine_steps):
            d = 10 ** log_d
            vx = cx + d * sin_a
            vy = cy - d * cos_a
            score = eval_vp(vx, vy)
            if score < best_score:
                best_score = score
                best_vp = (vx, vy)

    return best_vp


# ── Connected-component VP search (fast, gap-resistant via RANSAC) ──────────

def _find_best_vp_cc(
    yard_mask: np.ndarray,
    frame_shape: tuple[int, int],
    min_pixels: int = 100,
    min_aspect: float = 4.0,
    ransac_iters: int = 100,
    ransac_inlier_rho_px: float = 8.0,
    rng_seed: int = 0,
) -> tuple[float, float] | None:
    """Direct VP estimation: connected components → PCA per component → RANSAC.

    For each yardline-shaped component we run PCA: largest singular vector =
    line direction. A component must have ≥`min_pixels` pixels and a
    length/width aspect ratio of ≥`min_aspect` (so we don't trust circular
    blobs that have no clear direction).

    Each surviving component contributes one infinite line in (ρ, θ) form.
    Yardlines from the same world line cast collinear constraints (good —
    they reinforce). Other components (sidelines, sparse hash artifacts, etc.)
    contribute outliers, so we use RANSAC: sample 2 lines, compute their
    intersection, count how many other lines pass within `ransac_inlier_rho_px`.
    Best-supported intersection = VP. Refine with least-squares on inliers.

    Returns None if <3 usable components — caller should fall back to grid search.
    """
    h, w = frame_shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        yard_mask.astype(np.uint8), connectivity=8,
    )

    lines: list[tuple[float, float]] = []  # (rho, theta) per usable component
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < min_pixels:
            continue
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = pts.mean(axis=0)
        centered = pts - center
        # SVD: largest singular vector is the principal (line) direction.
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6:
            continue
        aspect = float(S[0] / S[1])
        if aspect < min_aspect:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        # Canonicalize: theta in [0, π), flip rho sign if needed.
        if theta < 0:
            theta += np.pi
            rho = -rho
        lines.append((rho, theta))

    if len(lines) < 3:
        return None

    rhos = np.array([l[0] for l in lines])
    thetas = np.array([l[1] for l in lines])
    A = np.column_stack([np.cos(thetas), np.sin(thetas)])  # (K, 2)
    K = len(lines)

    # RANSAC: pair-wise line intersections, vote for best VP.
    rng = np.random.default_rng(rng_seed)
    best_inliers = 0
    best_vp = None
    for _ in range(ransac_iters):
        i, j = rng.choice(K, size=2, replace=False)
        sub_A = A[[i, j]]
        sub_b = rhos[[i, j]]
        # Skip near-parallel pairs (singular system).
        if abs(np.linalg.det(sub_A)) < 1e-6:
            continue
        vp_cand = np.linalg.solve(sub_A, sub_b)
        # Residual = signed perpendicular distance from each line to vp_cand.
        residuals = np.abs(A @ vp_cand - rhos)
        inliers = residuals < ransac_inlier_rho_px
        n_in = int(inliers.sum())
        if n_in > best_inliers:
            best_inliers = n_in
            best_vp = vp_cand

    if best_vp is None or best_inliers < 3:
        return None

    # Refine on inliers via least squares.
    residuals = np.abs(A @ best_vp - rhos)
    inlier_mask = residuals < ransac_inlier_rho_px
    if inlier_mask.sum() >= 2:
        vp_refined, *_ = np.linalg.lstsq(A[inlier_mask], rhos[inlier_mask], rcond=None)
        best_vp = vp_refined

    return (float(best_vp[0]), float(best_vp[1]))


# ── GPU-batched VP search (MPS/CUDA) ────────────────────────────────────────

def _vp_to_polar(vp: tuple[float, float], cx: float, cy: float) -> tuple[float, float]:
    """Convert (vx, vy) → (angle_deg from vertical, log10 distance from center).

    Inverse of `vx = cx + d*sin(θ), vy = cy - d*cos(θ)`.
    """
    vx, vy = vp
    dx = vx - cx
    dy = cy - vy
    d = float(np.hypot(dx, dy))
    log_d = float(np.log10(max(d, 1e-3)))
    ang_deg = float(np.degrees(np.arctan2(dx, dy))) if d > 1e-9 else 0.0
    return ang_deg, log_d


def _build_vp_candidates(
    angles_deg: np.ndarray, log_dists: np.ndarray, cx: float, cy: float,
) -> np.ndarray:
    """Return (M, 2) array of candidate VPs from polar grid."""
    A, D = np.meshgrid(angles_deg, log_dists, indexing="ij")
    a = np.radians(A.ravel())
    d = 10 ** D.ravel()
    vx = cx + d * np.sin(a)
    vy = cy - d * np.cos(a)
    return np.stack([vx, vy], axis=1)


def _gpu_score_candidates_all(
    pts_t,                  # (N, 2) torch tensor on `dev`
    candidates: np.ndarray, # (M, 2) numpy
    n_bins: int,
    dev,
    top_k: int = 12,
    nms_radius: int = 5,
):
    """Score all M candidate VPs against a fixed pixel set, fully on GPU.

    Score = entropy of top-K peak heights × peak-mass concentration. Returns
    (M,) numpy scores (higher=better) plus a "wide net" — used as a coarse
    filter before CPU Fisher rerank picks the actual best.

    See `_score_candidates_gpu` (the wrapper) for the design rationale.
    """
    import torch

    cands_t = torch.from_numpy(candidates).float().to(dev)        # (M, 2)
    M = cands_t.shape[0]

    # (M, N) pairwise differences and angles.
    dx = pts_t[None, :, 0] - cands_t[:, None, 0]
    dy = pts_t[None, :, 1] - cands_t[:, None, 1]
    u = torch.atan2(dy, dx)                                       # (-π, π]

    # Bin into [-π, π] → [0, n_bins-1].
    bin_idx = ((u + np.pi) / (2 * np.pi) * n_bins).long().clamp_(0, n_bins - 1)

    # scatter_add: hist[m, bin_idx[m, n]] += 1
    hist = torch.zeros((M, n_bins), device=dev, dtype=torch.float32)
    src = torch.ones_like(bin_idx, dtype=torch.float32)
    hist.scatter_add_(1, bin_idx, src)                            # (M, B)

    # Narrow smoothing — preserves peaks, kills 1-bin noise.
    narrow = torch.ones(1, 1, 3, device=dev) / 3.0
    h_sm = torch.nn.functional.conv1d(
        hist.unsqueeze(1), narrow, padding=1,
    ).squeeze(1)                                                   # (M, B)

    # NMS: a bin is a local maximum iff it equals max in its window.
    pooled = torch.nn.functional.max_pool1d(
        h_sm.unsqueeze(1),
        kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius,
    ).squeeze(1)
    is_peak = (h_sm >= pooled) & (h_sm > 0)                       # (M, B) bool
    peaks_only = h_sm * is_peak.float()                            # zero where not peak

    # Top-K peak heights and bin positions.
    top_vals, top_idx = peaks_only.topk(top_k, dim=1)             # both (M, K)
    top_angles = (top_idx.float() / n_bins) * (2 * np.pi) - np.pi # (M, K) in [-π, π]

    eps = 1e-9
    # Normalize top-K heights into a discrete probability distribution.
    top_sum = top_vals.sum(dim=1) + eps                           # (M,)
    p = top_vals / top_sum.unsqueeze(1)                           # (M, K)
    # Entropy of top-K probabilities.
    #   Single tall peak: p=[1,0,…,0] → entropy=0 (bad — concentrated VP).
    #   K equal peaks:    p=[1/K]×K   → entropy=log(K) (good — multi-peak).
    entropy = -(p * (p + eps).log()).sum(dim=1)                   # (M,)

    # Concentration: fraction of pixel mass that landed in topK peaks vs total.
    # High when peaks are sharp+strong, low when histogram is broadly noisy
    # (uniform-junk inside-image VP).
    N = float(pts_t.shape[0])
    concentration = top_sum / N                                   # (M,)

    score = entropy * concentration                               # (M,)
    return score.cpu().numpy()


def _cpu_fisher_rerank(
    pts: np.ndarray,
    candidates: np.ndarray,
    frame_shape: tuple[int, int],
    min_peak_sep_px: float = YARDLINE_MIN_PEAK_SEP_PX,
    min_prom_frac: float = YARDLINE_MIN_PEAK_PROM,
) -> tuple[float, float]:
    """Run the original CPU Fisher score on a small set of candidate VPs.

    Used after a GPU coarse filter narrows 1145 → 20 candidates. Lower score
    is better (within-var / between-var); returns the candidate with min score.
    """
    h, w = frame_shape
    best_score = float("inf")
    best_vp = (float(candidates[0, 0]), float(candidates[0, 1]))
    for vp in candidates:
        vx, vy = float(vp[0]), float(vp[1])
        dx = pts[:, 0] - vx
        dy = pts[:, 1] - vy
        u = np.arctan2(dy, dx)
        u_min, u_max = float(u.min()), float(u.max())
        if u_max - u_min > np.pi:
            u = np.where(u < 0, u + 2 * np.pi, u)
        mean_dist = float(np.hypot(dx, dy).mean())
        ang_min_sep = min_peak_sep_px / max(mean_dist, 1.0)
        score = _score_projection(u, ang_min_sep, min_prom_frac)
        if score < best_score:
            best_score = score
            best_vp = (vx, vy)
    return best_vp


def _find_best_vp_gpu(
    pts: np.ndarray,
    frame_shape: tuple[int, int],
    vp_init: tuple[float, float] | None = None,
    n_pixel_subsample: int = 5000,
    device: str = "mps",
    n_bins: int = 512,
    coarse_angle_deg: tuple[float, float] = (-45.0, 45.0),
    coarse_n_angle: int = 46,
    coarse_log_dist_range: tuple[float, float] = (2.5, 5.0),
    coarse_n_dist: int = 20,
    refine_angle_halfwidth_deg: float = 2.0,
    refine_log_dist_halfwidth: float = 0.3,
    refine_steps: int = 15,
    rerank_top_n: int = 250,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Hybrid GPU+CPU VP search. Drop-in for `_find_best_vanishing_point`.

    Pipeline:
      1. Subsample mask pixels to `n_pixel_subsample` (~5K) to bound tensor sizes.
      2. (Coarse pass, skipped if `vp_init` is given): GPU scores all 1145
         candidates with the entropy×concentration proxy in a single batched
         pass. Top-N (default 20) by GPU score are passed to the CPU Fisher
         re-ranker, which picks the actual best — this preserves the EXACT
         same VP the pure-CPU search would have found, since CPU Fisher is
         the "ground truth" score.
      3. (Refine pass): GPU evaluates a 15×15 grid around `best_vp`, top-N
         CPU-reranked again. Cheap (~225 candidates).

    Empirically: GPU coarse score correctly puts the true VP in top-20 even
    when its argmax is wrong, so the CPU rerank recovers the right answer.

    `vp_init`: warm-start. If given, skip coarse pass entirely.
    """
    import torch

    if len(pts) < 100:
        return (frame_shape[1] / 2.0, -1e9)

    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0

    # Subsample pixels (deterministic for repeatability).
    if len(pts) > n_pixel_subsample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), n_pixel_subsample, replace=False)
        pts = pts[idx]

    dev = torch.device(device)
    pts_t = torch.from_numpy(pts.astype(np.float32)).to(dev)

    def gpu_pick_topN_then_cpu_rerank(cands: np.ndarray) -> tuple[float, float]:
        gpu_scores = _gpu_score_candidates_all(pts_t, cands, n_bins, dev)
        n_keep = min(rerank_top_n, len(cands))
        # argpartition is faster than argsort and we only care about top-N
        top_idx = np.argpartition(gpu_scores, -n_keep)[-n_keep:]
        return _cpu_fisher_rerank(pts, cands[top_idx], frame_shape=(h, w))

    if vp_init is None:
        # ── Coarse pass ──
        coarse_angles = np.linspace(coarse_angle_deg[0], coarse_angle_deg[1], coarse_n_angle)
        coarse_logd = np.linspace(coarse_log_dist_range[0], coarse_log_dist_range[1], coarse_n_dist)
        cands = _build_vp_candidates(coarse_angles, coarse_logd, cx, cy)
        best_vp = gpu_pick_topN_then_cpu_rerank(cands)
    else:
        best_vp = vp_init

    # ── Refine pass (always done) ──
    ang_init, ld_init = _vp_to_polar(best_vp, cx, cy)
    refine_angles = np.linspace(ang_init - refine_angle_halfwidth_deg,
                                 ang_init + refine_angle_halfwidth_deg, refine_steps)
    refine_logd = np.linspace(ld_init - refine_log_dist_halfwidth,
                               ld_init + refine_log_dist_halfwidth, refine_steps)
    cands = _build_vp_candidates(refine_angles, refine_logd, cx, cy)
    best_vp = gpu_pick_topN_then_cpu_rerank(cands)
    return best_vp


def _column_coords_vp(pts: np.ndarray, vp: tuple[float, float]) -> np.ndarray:
    """Project pixels to angle-from-VP (their polar-around-VP angle).

    All pixels on one yardline (which passes through VP) get the same u value.
    """
    dx = pts[:, 0] - vp[0]
    dy = pts[:, 1] - vp[1]
    u = np.arctan2(dy, dx)
    # Handle atan2 branch cut when VP is near/inside the pixel cloud
    if float(u.max() - u.min()) > np.pi:
        u = np.where(u < 0, u + 2 * np.pi, u)
    return u


# ── Legacy tilt-slope interface (kept for callers; delegates to VP) ──────────

def _estimate_tilt_slope(pts: np.ndarray) -> float:
    """Deprecated shim — approximates tilt from the best VP's direction."""
    # Find VP then convert to an equivalent tilt slope at the image center.
    # tilt_slope = dx/dy = sin(angle_to_vp) / cos(angle_to_vp) = tan(angle)
    h_w = (720, 1280)  # fallback if called without frame_shape
    vp = _find_best_vanishing_point(pts, h_w)
    vx, vy = vp
    cx, cy = h_w[1] / 2.0, h_w[0] / 2.0
    dy = cy - vy    # positive = VP above center
    dx = vx - cx
    if abs(dy) < 1e-6:
        return 0.0
    return float(-dx / dy)  # tilt from vertical


def _column_coords(pts: np.ndarray, tilt_slope: float, y_ref: float) -> np.ndarray:
    """Project pts to column coord with explicit tilt correction."""
    return pts[:, 0] - tilt_slope * (pts[:, 1] - y_ref)


def _assign_pixels_to_peaks(
    pts: np.ndarray,
    col_coords: np.ndarray,
    peak_coords: np.ndarray,
    half_bw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each pixel to the nearest peak within `half_bw` in col-coord.

    Returns (nearest_idx, keep_mask).
    """
    dist = np.abs(col_coords[:, None] - peak_coords[None, :])
    nearest_idx = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(pts)), nearest_idx]
    keep = nearest_dist <= half_bw
    return nearest_idx, keep


def _poly_cross_distance(pts: np.ndarray, line_obj: "LineObject") -> np.ndarray:
    """Signed cross-distance from each pt to the line's polynomial.

    For yardlines (x = f(y)), this is |x - f(y)|. Cheap and exact for our
    nearly-vertical lines; a true perpendicular distance would add arc-length
    bookkeeping without improving the assignment meaningfully.
    """
    ys = pts[:, 1]
    xs_pred = eval_poly(line_obj, ys)
    return np.abs(pts[:, 0] - xs_pred)


def group_yardline_pixels_cc(
    yard_mask: np.ndarray,
    min_pixels_per_component: int = 40,
    min_aspect_ratio: float = 3.0,
    rho_tol_px: float = 25.0,
    theta_tol_rad: float = 0.08,
    min_fragments_per_line: int = 2,
    min_pixels_for_singleton: int = 500,
    dedup_peak_x_px: float = 50.0,
    min_pixels_per_line: int = YARDLINE_MIN_PIXELS_PER_LINE,
) -> list[LineObject]:
    """Group yardline pixels via connected components + collinearity merge.

    Skips VP search entirely. Each CC fragment is parameterized by PCA as
    (ρ, θ); fragments with matching (ρ, θ) belong to the same yardline and
    get their pixels merged before polynomial fitting.

    Why this works without VP: parallel-in-world yardlines have *different*
    (ρ, θ) in the image (perspective makes them converge), but FRAGMENTS of
    the SAME yardline share (ρ, θ). So clustering by (ρ, θ) similarity
    reunites split pieces without needing to know where they all converge.

    Pipeline:
      1. Connected components on the binary mask.
      2. Filter components by size (≥ min_pixels_per_component) and shape
         (PCA aspect ratio ≥ min_aspect_ratio — drop blob-like noise).
      3. Each surviving component → (ρ, θ) via PCA.
      4. Cluster by (ρ, θ) similarity (within rho_tol_px and theta_tol_rad).
      5. **Drop unmerged singletons** unless they're substantial (≥
         `min_pixels_for_singleton`). A real yardline under occlusion almost
         always shows up as multiple fragments; a lone small fragment is more
         likely noise than a real yardline.
      6. **Dedup near-duplicate yardlines** — if two clusters end up with
         peak_x within `dedup_peak_x_px` (i.e., they sit on the same image
         column), drop the one with fewer pixels. Catches collinear fragments
         that slipped past the (ρ, θ) merge due to PCA direction noise.
      7. For each surviving cluster, concatenate pixels, fit polynomial.

    Uses LINEAR (not deg-4) fit to compute peak_coord at a shared reference
    y, avoiding extrapolation noise from high-order polynomials when a
    yardline's pixels span a different y range than the shared reference.

    Returns list of LineObjects sorted left-to-right by their x-at-shared-y.
    """
    yard_mask_u8 = yard_mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        yard_mask_u8, connectivity=8,
    )

    # ── 1-3. Per-component PCA → (ρ, θ) ──
    components: list[dict] = []
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_pixels_per_component:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < min_pixels_per_component:
            continue
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = pts.mean(axis=0)
        centered = pts - center
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6:
            continue
        if S[0] / S[1] < min_aspect_ratio:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        # Canonicalize to theta ∈ [0, π).
        if theta < 0:
            theta += np.pi
            rho = -rho
        components.append({
            "pixels": pts, "rho": rho, "theta": theta, "center": center,
            "n": len(pts), "direction": direction,
        })

    if not components:
        return []

    # ── 4. Cluster by (ρ, θ) collinearity ──
    # Greedy single-pass: each unassigned component starts a new group; absorbs
    # later components within (rho_tol, theta_tol). The wrap at θ=π means a
    # near-vertical line could have θ near 0 OR near π — handle both.
    used = [False] * len(components)
    clusters: list[list[dict]] = []
    for i, c in enumerate(components):
        if used[i]:
            continue
        used[i] = True
        cluster = [c]
        for j in range(i + 1, len(components)):
            if used[j]:
                continue
            d = components[j]
            theta_diff = abs(c["theta"] - d["theta"])
            theta_diff = min(theta_diff, np.pi - theta_diff)
            if theta_diff > theta_tol_rad:
                continue
            # ρ comparison must account for the θ wrap-flip (if d's θ wrapped,
            # its ρ sign flipped too).
            wrap_flip = (abs(c["theta"] - d["theta"]) > np.pi / 2)
            d_rho_eff = -d["rho"] if wrap_flip else d["rho"]
            if abs(c["rho"] - d_rho_eff) > rho_tol_px:
                continue
            used[j] = True
            cluster.append(d)
        clusters.append(cluster)

    # ── 5-7. Drop singletons, build LineObjects, dedup near-duplicate columns ──
    # Shared reference y for peak_coord — comparable across yardlines.
    all_pts_combined = np.vstack([
        np.vstack([c["pixels"] for c in cl]) for cl in clusters
    ]) if clusters else np.zeros((0, 2))
    shared_ref_y = float(np.median(all_pts_combined[:, 1])) if len(all_pts_combined) > 0 else 0.0

    candidates: list[LineObject] = []
    for cluster in clusters:
        all_pts = np.vstack([c["pixels"] for c in cluster])
        # Singleton guard.
        if len(cluster) < min_fragments_per_line and len(all_pts) < min_pixels_for_singleton:
            continue
        if len(all_pts) < min_pixels_per_line:
            continue
        line_obj = LineObject(pixels=all_pts, kind="yardline")
        fit_line_polynomial(line_obj)
        if line_obj.poly is None:
            continue
        # Linear x(y) for peak_coord — robust to extrapolation when a yardline's
        # pixels are far from shared_ref_y. Deg-4 poly diverges in extrapolation.
        ys = all_pts[:, 1]
        xs = all_pts[:, 0]
        if ys.std() > 1e-6:
            slope, intercept = np.polyfit(ys, xs, 1)
            line_obj.peak_coord = float(slope * shared_ref_y + intercept)
        else:
            line_obj.peak_coord = float(xs.mean())
        candidates.append(line_obj)

    # Dedup near-duplicate columns: collinear fragments that slipped past the
    # (ρ, θ) merge will land at nearly the same peak_x. Keep the one with
    # more pixels (more reliable polynomial fit).
    candidates.sort(key=lambda lo: -len(lo.pixels))
    out: list[LineObject] = []
    for c in candidates:
        if any(abs(c.peak_coord - kept.peak_coord) < dedup_peak_x_px for kept in out):
            continue
        out.append(c)

    out.sort(key=lambda lo: (lo.peak_coord if lo.peak_coord is not None else 0.0))
    return out


def group_yardline_pixels(
    yard_mask: np.ndarray,
    min_peak_sep: float = YARDLINE_MIN_PEAK_SEP_PX,
    bandwidth_frac: float = YARDLINE_PEAK_BANDWIDTH_FRAC,
    min_prom_frac: float = YARDLINE_MIN_PEAK_PROM,
    min_pixels_per_line: int = YARDLINE_MIN_PIXELS_PER_LINE,
    refine_iters: int = YARDLINE_POLY_REFINE_ITERS,
    poly_cross_tol_px: float = YARDLINE_POLY_CROSS_TOL_PX,
    use_gpu_vp: bool = False,
    vp_init: tuple[float, float] | None = None,
    vp_device: str = "mps",
    vp_rerank_top_n: int = 250,
) -> list[LineObject]:
    """Split a yardline UNet mask into one LineObject per yard line.

    Approach (robust to occlusions/breaks because grouping is by column-coord
    similarity, NOT spatial connectivity — CC would fragment broken lines):

      1. Global PCA → tilt_slope = along_dir.x / along_dir.y.
      2. Column coord for every pixel: col = x - tilt_slope * (y - y_ref).
         All pixels on one yardline collapse to the same col regardless of
         where along the line they are.
      3. Peak-pick the col-coord histogram. One peak per yardline.
      4. Assign each pixel to nearest peak within bandwidth.
      5. Poly-refine: fit deg-4 poly per group, then reassign each pixel to
         the group whose polynomial is closest (in |x - f(y)|). This sharpens
         groupings for lines whose individual tilt differs from global tilt
         (perspective convergence). Repeat `refine_iters` times.
      6. Drop groups with < min_pixels_per_line after final assignment.

    Returns list of LineObjects sorted left-to-right by col-coord peak.
    """
    pts = _mask_pixels(yard_mask)
    if len(pts) < min_pixels_per_line:
        return []

    h, w = yard_mask.shape[:2]

    # ── Vanishing-point rectification ────────────────────────────────────
    # Yardlines are parallel in world coords so they converge to a VP in the
    # image. Polar coords around VP collapse each yardline's pixels to a
    # single angle (column coord), enabling clean 1D peak separation even
    # under perspective convergence.
    if use_gpu_vp:
        vp = _find_best_vp_gpu(pts, frame_shape=(h, w),
                                vp_init=vp_init, device=vp_device,
                                rerank_top_n=vp_rerank_top_n)
    else:
        vp = _find_best_vanishing_point(pts, frame_shape=(h, w))
    # Stash on a module-level slot so solve_grid can pick it up for grid
    # assignment (cleaner than threading vp through every call).
    _LAST_VP[0] = vp
    col_coords = _column_coords_vp(pts, vp)

    # Convert min_peak_sep from pixel units to angular units at mean distance.
    # (Grouping is in radians now, not pixels.)
    mean_dist_to_vp = float(np.hypot(
        pts[:, 0] - vp[0], pts[:, 1] - vp[1],
    ).mean())
    ang_min_sep = min_peak_sep / max(mean_dist_to_vp, 1.0)

    peak_coords, _ = _peak_pick_1d(
        col_coords,
        bin_width=ang_min_sep * 0.1,
        min_sep=ang_min_sep,
        min_prom_frac=min_prom_frac,
    )
    if len(peak_coords) == 0:
        return []

    if len(peak_coords) >= 2:
        spacing = float(np.diff(np.sort(peak_coords)).min())
    else:
        spacing = ang_min_sep
    half_bw = max(ang_min_sep, bandwidth_frac * spacing)

    nearest_idx, keep = _assign_pixels_to_peaks(pts, col_coords, peak_coords, half_bw)

    # Initial groups: collect ALL above the absolute pixel floor, then apply
    # an adaptive secondary filter (drop anything < frac × median group size).
    # The two-step approach lets short stub yardlines pass while still
    # rejecting tiny noise blobs.
    order = np.argsort(peak_coords)
    candidates = []
    for k in order:
        sel = keep & (nearest_idx == k)
        n_px = int(sel.sum())
        if n_px < min_pixels_per_line:
            continue
        candidates.append((sel, k, n_px))

    if not candidates:
        return []

    sizes = [n for _, _, n in candidates]
    median_sz = float(np.median(sizes))
    adaptive_floor = YARDLINE_MIN_PIXEL_FRAC_OF_MEDIAN * median_sz

    initial_groups: list[LineObject] = []
    for sel, k, n_px in candidates:
        if n_px < adaptive_floor:
            continue
        initial_groups.append(LineObject(
            pixels=pts[sel].copy(),
            kind="yardline",
            peak_coord=float(peak_coords[k]),
        ))

    if not initial_groups or refine_iters <= 0:
        return initial_groups

    # Iterative poly-refine: fit each group's poly, then reassign pixels to
    # nearest poly. Handles per-line tilt variation (perspective convergence)
    # that a single global tilt can't capture.
    groups = initial_groups
    for _ in range(refine_iters):
        for g in groups:
            fit_line_polynomial(g)

        # For each pixel (from the FULL set, not just kept), compute distance
        # to every group's poly; assign to nearest.
        cross_dists = np.stack([_poly_cross_distance(pts, g.line if isinstance(g, Yardline) else g)
                                for g in groups], axis=1)   # (N, G)
        nearest_group = cross_dists.argmin(axis=1)
        nearest_poly_dist = cross_dists[np.arange(len(pts)), nearest_group]
        keep = nearest_poly_dist <= poly_cross_tol_px

        new_groups = []
        for gi, g in enumerate(groups):
            sel = keep & (nearest_group == gi)
            if int(sel.sum()) < min_pixels_per_line:
                continue
            new_groups.append(LineObject(
                pixels=pts[sel].copy(),
                kind="yardline",
                peak_coord=g.peak_coord,
            ))
        if not new_groups:
            break
        groups = new_groups

    # Final sort by peak_coord just to be safe.
    groups.sort(key=lambda g: g.peak_coord or 0.0)
    return groups


def group_sideline_pixels(
    side_mask: np.ndarray,
    min_peak_sep: float = SIDELINE_MIN_PEAK_SEP_PX,
    min_prom_frac: float = SIDELINE_MIN_PEAK_PROM,
    min_pixels_per_line: int = 50,
) -> list[LineObject]:
    """Split a sideline UNet mask into up to 2 LineObjects (near / far).

    Unlike yardlines (which converge to a VP), the two sidelines are parallel
    and effectively share a single direction in image space, so PCA on their
    union gives a reliable along-axis. Projection onto the perpendicular
    (cross-axis) yields 1 or 2 clean peaks.

    Returns list sorted by peak cross-coord (smaller = far sideline first,
    because "far" is above "near" in image y).
    """
    pts = _mask_pixels(side_mask)
    if len(pts) < min_pixels_per_line:
        return []

    # PCA: vt[0] = along-sideline direction (largest variance, since each
    # sideline spans most of the frame width), vt[1] = perpendicular.
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    along_dir = vt[0]
    cross_dir = np.array([-along_dir[1], along_dir[0]])
    # Orient cross_dir so + = downward in image (consistent "near-sideline-
    # is-larger-cross" convention).
    if cross_dir[1] < 0:
        cross_dir = -cross_dir

    cross = centered @ cross_dir

    peak_coords, _ = _peak_pick_1d(
        cross,
        bin_width=2.0,
        min_sep=min_peak_sep,
        min_prom_frac=min_prom_frac,
    )
    if len(peak_coords) == 0:
        return [LineObject(pixels=pts.copy(), kind="sideline",
                           peak_coord=float(np.median(cross)))]

    if len(peak_coords) >= 2:
        spacing = float(np.diff(np.sort(peak_coords)).min())
    else:
        spacing = min_peak_sep
    half_bw = 0.45 * spacing if len(peak_coords) >= 2 else min_peak_sep

    dist = np.abs(cross[:, None] - peak_coords[None, :])
    nearest_idx = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(pts)), nearest_idx]
    keep = nearest_dist <= half_bw

    groups: list[LineObject] = []
    order = np.argsort(peak_coords)            # smallest cross → far sideline
    for k in order:
        sel = keep & (nearest_idx == k)
        if int(sel.sum()) < min_pixels_per_line:
            continue
        groups.append(LineObject(
            pixels=pts[sel].copy(),
            kind="sideline",
            peak_coord=float(peak_coords[k]),
        ))
    return groups


# ── 2. Polynomial fitting ────────────────────────────────────────────────────

def fit_line_polynomial(line_obj: LineObject, degree: int | None = None) -> LineObject:
    """Fit a polynomial through a LineObject's pixels (mutates + returns it).

    Yardlines: x = f(y), degree=POLY_DEG_YARDLINE.
    Sidelines: y = g(x), degree=POLY_DEG_SIDELINE.

    Stores coeffs in `line_obj.poly`, axis in `line_obj.poly_axis`, and
    perpendicular-residual RMSE in `line_obj.residual_rmse`.
    """
    pts = line_obj.pixels
    if line_obj.kind == "yardline":
        q, p = pts[:, 1], pts[:, 0]   # q = y (independent), p = x (dependent)
        axis = "x_of_y"
        deg = degree if degree is not None else POLY_DEG_YARDLINE
    elif line_obj.kind == "sideline":
        q, p = pts[:, 0], pts[:, 1]
        axis = "y_of_x"
        deg = degree if degree is not None else POLY_DEG_SIDELINE
    else:
        raise ValueError(f"unknown line kind: {line_obj.kind}")

    if len(pts) < deg + 1:
        return line_obj  # not enough for fit — leave poly=None

    q_offset = float(np.mean(q))
    q_scale = float(np.std(q)) or 1.0
    q_norm = (q - q_offset) / q_scale

    coeffs = np.polyfit(q_norm, p, deg)
    p_fit = np.polyval(coeffs, q_norm)
    rmse = float(np.sqrt(np.mean((p_fit - p) ** 2)))

    line_obj.poly = coeffs
    line_obj.poly_axis = axis
    line_obj.poly_q_offset = q_offset
    line_obj.poly_q_scale = q_scale
    line_obj.residual_rmse = rmse
    return line_obj


def eval_poly(line_obj: LineObject, q):
    """Evaluate the line's polynomial at q. Returns the other-axis value.

    For yardlines (x = f(y)): q is y, returns x.
    For sidelines (y = g(x)): q is x, returns y.
    Handles the fit-time centering/scaling transparently.
    """
    if line_obj.poly is None:
        raise ValueError("line has no polynomial fit")
    q_norm = (np.asarray(q, dtype=np.float64) - line_obj.poly_q_offset) / line_obj.poly_q_scale
    return np.polyval(line_obj.poly, q_norm)


def yardline_point_at_y(line_obj: LineObject, y: float) -> tuple[float, float]:
    """Return (x, y) pixel on a yardline polynomial at row y."""
    return float(eval_poly(line_obj, float(y))), float(y)


def sideline_point_at_x(line_obj: LineObject, x: float) -> tuple[float, float]:
    """Return (x, y) pixel on a sideline polynomial at column x."""
    return float(x), float(eval_poly(line_obj, float(x)))


# ── 3. Polynomial intersection (yardline × sideline) ─────────────────────────

def _eval_with_tangent(line_obj: LineObject, q: float) -> float:
    """Evaluate a line's polynomial at q, using a LINEAR TANGENT extrapolation
    when q is outside the observed pixel range.

    Inside the observed range: standard `eval_poly` (deg-4 interpolation —
    accurate, captures real lens-distortion curvature).
    Outside: tangent line at the nearest observed endpoint (matches the
    polynomial's value AND slope at the boundary, then continues linearly).
    Avoids the deg-4 extrapolation explosion when a yardline doesn't reach
    far enough up to meet the sideline.
    """
    if line_obj.poly is None or line_obj.pixels is None or len(line_obj.pixels) == 0:
        return float("nan")
    # Pick the observed-range axis (x for sideline since y=g(x); y for yardline since x=f(y)).
    obs_axis = 0 if line_obj.kind == "sideline" else 1
    q_min = float(line_obj.pixels[:, obs_axis].min())
    q_max = float(line_obj.pixels[:, obs_axis].max())
    if q_min <= q <= q_max:
        return float(eval_poly(line_obj, q))
    # Outside observed range — tangent at the nearest endpoint.
    q_anchor = q_min if q < q_min else q_max
    p_at_anchor = float(eval_poly(line_obj, q_anchor))
    # df/dq at q_anchor: polyder gives df/dq_norm; chain-rule with q_scale.
    deriv_norm_coeffs = np.polyder(line_obj.poly)
    q_anchor_norm = (q_anchor - line_obj.poly_q_offset) / line_obj.poly_q_scale
    slope_norm = float(np.polyval(deriv_norm_coeffs, q_anchor_norm))
    slope = slope_norm / line_obj.poly_q_scale
    return p_at_anchor + slope * (q - q_anchor)


def intersect_yardline_sideline(
    yl: LineObject,
    sl: LineObject,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    extrapolate_with_tangent: bool = False,
) -> tuple[float, float] | None:
    """Find the pixel where a yardline poly meets a sideline poly.

    Yardline:  x = f(y)  (polyval(yl.poly, y) = x)
    Sideline:  y = g(x)  (polyval(sl.poly, x) = y)

    Solve by 1D root-finding on F(y) = g(f(y)) - y. For deg-2 polynomials this
    is deg-4 in y but still cheap. Use Brent's method over a bracket derived
    from the yardline's pixel y-range (which must span the sideline row if the
    intersection is real).

    Returns (x, y) or None if no bracket or root.
    """
    if yl.poly is None or sl.poly is None:
        return None

    ymin = float(yl.pixels[:, 1].min()) if y_range is None else y_range[0]
    ymax = float(yl.pixels[:, 1].max()) if y_range is None else y_range[1]
    if ymax - ymin < 1.0:
        return None

    # Extend the search bracket. With tangent extrapolation, we can extend a
    # lot (the linear tangent is well-behaved arbitrarily far) — needed when
    # the yardline doesn't reach the sideline. Without tangent (legacy mode),
    # cap at 20% to avoid deg-4 explosion.
    span = ymax - ymin
    if extrapolate_with_tangent:
        ymin -= 1.0 * span
        ymax += 1.0 * span
    else:
        ymin -= 0.2 * span
        ymax += 0.2 * span

    eval_yl = (lambda y: _eval_with_tangent(yl, float(y))) if extrapolate_with_tangent \
              else (lambda y: float(eval_poly(yl, float(y))))
    eval_sl = (lambda x: _eval_with_tangent(sl, float(x))) if extrapolate_with_tangent \
              else (lambda x: float(eval_poly(sl, float(x))))

    def F(y: float) -> float:
        x = eval_yl(y)
        y_sl = eval_sl(x)
        return y_sl - y

    f_lo, f_hi = F(ymin), F(ymax)
    if f_lo * f_hi > 0:
        # Sample densely and look for any sign change (poly can dip).
        ys = np.linspace(ymin, ymax, 64)
        fs = np.array([F(y) for y in ys])
        sign_change = np.where(np.diff(np.sign(fs)) != 0)[0]
        if len(sign_change) == 0:
            return None
        k = int(sign_change[0])
        ymin, ymax = float(ys[k]), float(ys[k + 1])
        f_lo, f_hi = fs[k], fs[k + 1]

    # Bisection (fine for a few evals; scipy brentq would need an import).
    for _ in range(60):
        ymid = 0.5 * (ymin + ymax)
        f_mid = F(ymid)
        if abs(f_mid) < 1e-3:
            break
        if f_mid * f_lo < 0:
            ymax, f_hi = ymid, f_mid
        else:
            ymin, f_lo = ymid, f_mid

    y_root = 0.5 * (ymin + ymax)
    x_root = eval_yl(y_root)

    if x_range is not None:
        if not (x_range[0] <= x_root <= x_range[1]):
            return None

    return x_root, float(y_root)


# ── 4. Hash → yardline assignment ────────────────────────────────────────────

def assign_hashes_to_yardlines(
    hash_pxs: np.ndarray,
    yardlines: list[Yardline],
    max_dist_px: float = HASH_TO_YARDLINE_MAX_DIST_PX,
) -> None:
    """Attach each hash detection to its nearest yardline polynomial.

    Distance = |x_on_yardline(hash.y) - hash.x|. A hash matches only if the
    distance is within `max_dist_px`. Within each yardline, if 2+ hashes match,
    the smaller-y (upper in image) = far_hash, the larger-y = near_hash. If
    only 1 hash matches, classification is deferred to `classify_singleton_hashes`.

    Mutates `yardlines` in place, populating `near_hash` / `far_hash`.
    """
    if len(hash_pxs) == 0 or not yardlines:
        return

    hashes = np.asarray(hash_pxs, dtype=np.float64)
    # For each (hash, yardline) pair, compute cross-distance.
    # hash_x_on_yl[i, j] = polyval(yardlines[j].poly, hashes[i, 1])
    per_yl_hashes: list[list[np.ndarray]] = [[] for _ in yardlines]

    for i, hx_hy in enumerate(hashes):
        hx, hy = float(hx_hy[0]), float(hx_hy[1])
        best_j, best_d = -1, float("inf")
        for j, yl in enumerate(yardlines):
            if yl.line.poly is None:
                continue
            yl_x_at_hy = float(eval_poly(yl.line, hy))
            d = abs(yl_x_at_hy - hx)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= max_dist_px:
            per_yl_hashes[best_j].append(hashes[i])

    for yl, h_list in zip(yardlines, per_yl_hashes):
        if len(h_list) == 0:
            continue
        if len(h_list) == 1:
            # Singleton — park it on far_hash for now; classification happens
            # in classify_singleton_hashes after paired assignments are known.
            yl.far_hash = h_list[0]     # placeholder side; overwritten later
            yl.near_hash = None
            continue
        # 2+ hashes on same yardline: keep only min-y (far) and max-y (near).
        hs = np.vstack(h_list)
        order = np.argsort(hs[:, 1])     # ascending y
        yl.far_hash = hs[order[0]]       # smallest y = far (top of image)
        yl.near_hash = hs[order[-1]]     # largest y = near


def classify_singleton_hashes(yardlines: list[Yardline]) -> None:
    """For yardlines with exactly one matched hash, decide far vs near.

    Uses paired yardlines (both far_hash + near_hash set) as reference:
    estimate the typical far-hash y and near-hash y, then classify each
    singleton by whichever is closer (in y) at that yardline's x-position.

    Falls back to the image-relative midline (mean of all hash y's) if
    no paired yardline exists in this frame.
    """
    paired = [yl for yl in yardlines
              if yl.far_hash is not None and yl.near_hash is not None
              and not np.array_equal(yl.far_hash, yl.near_hash)]

    if not paired:
        # No reference — use image-wise hash y midline as the cutoff.
        all_hash_ys = []
        for yl in yardlines:
            if yl.far_hash is not None:
                all_hash_ys.append(float(yl.far_hash[1]))
            if yl.near_hash is not None and yl.near_hash is not yl.far_hash:
                all_hash_ys.append(float(yl.near_hash[1]))
        if not all_hash_ys:
            return
        midline = float(np.median(all_hash_ys))
        for yl in yardlines:
            # Singleton heuristic: if only far_hash was set as placeholder
            # but not from a pair, check its y vs midline.
            if yl.far_hash is not None and yl.near_hash is None:
                if float(yl.far_hash[1]) > midline:
                    yl.near_hash = yl.far_hash
                    yl.far_hash = None
        return

    # With paired references: fit far-y(x) and near-y(x) lines for robustness
    # across perspective.
    pxs = np.array([[float(yl.far_hash[0]), float(yl.far_hash[1])] for yl in paired])
    nxs = np.array([[float(yl.near_hash[0]), float(yl.near_hash[1])] for yl in paired])
    if len(paired) >= 2:
        far_slope, far_int = np.polyfit(pxs[:, 0], pxs[:, 1], 1)
        near_slope, near_int = np.polyfit(nxs[:, 0], nxs[:, 1], 1)
    else:
        far_slope, far_int = 0.0, float(pxs[0, 1])
        near_slope, near_int = 0.0, float(nxs[0, 1])

    for yl in yardlines:
        if yl.far_hash is None or yl.near_hash is not None:
            continue  # not a singleton
        hx, hy = float(yl.far_hash[0]), float(yl.far_hash[1])
        far_ref = far_slope * hx + far_int
        near_ref = near_slope * hx + near_int
        if abs(hy - near_ref) < abs(hy - far_ref):
            yl.near_hash = yl.far_hash
            yl.far_hash = None


# ── 5. Grid position assignment ─────────────────────────────────────────────

def _yardline_vp_angle(yl: Yardline, vp: tuple[float, float]) -> float | None:
    """Angle from VP for one yardline = atan2 averaged over its pixels.

    Every pixel on a yardline-through-VP shares the same atan2 angle (mod
    distortion / mask noise). Mean over all pixels gives a tight estimate
    that's defined identically for stubs and full-length lines.
    """
    if yl.line.pixels is None or len(yl.line.pixels) == 0:
        return None
    dx = yl.line.pixels[:, 0] - vp[0]
    dy = yl.line.pixels[:, 1] - vp[1]
    u = np.arctan2(dy, dx)
    if float(u.max() - u.min()) > np.pi:
        u = np.where(u < 0, u + 2 * np.pi, u)
    return float(np.median(u))


def assign_grid_positions(yardlines: list[Yardline],
                           vp: tuple[float, float] | None = None) -> None:
    """Assign integer grid slots to each yardline (leftmost = 0).

    Uses **angle from the vanishing point** as the column coord: invariant
    under perspective, defined for all yardlines (including stubs that only
    appear at one end of the frame), and naturally unit-spaced.

    Falls back to poly-eval-at-ref_y if vp is None.
    """
    if not yardlines:
        return

    coords = []
    if vp is not None:
        for yl in yardlines:
            ang = _yardline_vp_angle(yl, vp)
            coords.append(ang)
    elif all(yl.line.peak_coord is not None for yl in yardlines):
        # CC mode: use the precomputed linear-fit peak_coord. This is x-at-
        # shared-ref-y from a deg-1 fit per yardline, robust to extrapolation
        # noise that bites the deg-4 poly when a yardline's pixels span a
        # different y range than the shared reference.
        for yl in yardlines:
            coords.append(float(yl.line.peak_coord))
    else:
        # Legacy fallback: poly-evaluation at hash row mean
        paired = [yl for yl in yardlines
                  if yl.far_hash is not None and yl.near_hash is not None]
        if paired:
            ref_y = float(np.mean([
                0.5 * (float(yl.far_hash[1]) + float(yl.near_hash[1])) for yl in paired
            ]))
        else:
            all_ys = np.concatenate([yl.line.pixels[:, 1] for yl in yardlines
                                      if yl.line.pixels is not None and len(yl.line.pixels) > 0])
            ref_y = float(np.median(all_ys)) if len(all_ys) > 0 else 0.0
        for yl in yardlines:
            if yl.line.poly is None:
                coords.append(None)
                continue
            coords.append(float(eval_poly(yl.line, ref_y)))

    valid_idx = [i for i, c in enumerate(coords) if c is not None]
    if not valid_idx:
        return
    if len(valid_idx) == 1:
        yardlines[valid_idx[0]].grid_pos = 0
        yardlines[valid_idx[0]].grid_fit_residual = 0.0
        yardlines[valid_idx[0]].grid_fit_ok = True
        return

    sorted_idx = sorted(valid_idx, key=lambda i: coords[i])
    xs = np.array([coords[i] for i in sorted_idx])
    diffs = np.diff(xs)

    # Robust unit estimate: in angle-from-VP space, each 5-yd field step is
    # NOT exactly equal in angle (perspective makes far-field steps smaller),
    # but adjacent yardlines are still close to a single unit. Use the median
    # diff for unit0, then iterate counts and refit weighted-LS.
    unit0 = float(np.median(diffs))
    # Reject diffs > 3× median as indicating a missing yardline
    counts = [max(1, int(round(d / unit0))) for d in diffs]
    denom = float(sum(c * c for c in counts))
    unit = (float(sum(d * c for d, c in zip(diffs, counts))) / denom
            if denom > 0 else unit0)

    grid_slots = [0]
    for c in counts:
        grid_slots.append(grid_slots[-1] + c)

    tol = 0.30 * unit
    x0 = xs[0]
    for rank, i in enumerate(sorted_idx):
        gp = grid_slots[rank]
        ideal = x0 + gp * unit
        residual = abs(xs[rank] - ideal)
        yardlines[i].grid_pos = gp
        yardlines[i].grid_fit_residual = float(residual)
        yardlines[i].grid_fit_ok = bool(residual <= tol)


# ── 6. Top-level solver ──────────────────────────────────────────────────────

def _linearize_line_in_undistorted_space(
    line_obj: LineObject, intrinsics,
) -> LineObject:
    """Undistort a LineObject's pixels and refit as deg-1 (linear).

    Yardlines: x = a + b·y  (linear in y, slope=b, intercept=a)
    Sidelines: y = a + b·x  (linear in x)

    Mutates the line_obj in place: pixels become undistorted, poly becomes
    a 2-element [slope, intercept]-style array (top-down via np.polyfit
    convention: highest order first → [slope, intercept]). poly_axis stays
    the same. residual_rmse is updated.

    Returns the same line_obj for chaining.
    """
    from .distortion import undistort_points
    pts = line_obj.pixels
    if pts is None or len(pts) < 2:
        return line_obj
    pts_u = undistort_points(pts.astype(np.float64), intrinsics)
    if line_obj.kind == "yardline":
        q, p = pts_u[:, 1], pts_u[:, 0]   # x = f(y)
    else:
        q, p = pts_u[:, 0], pts_u[:, 1]   # y = g(x)
    # Centered fit for numerical stability (matches fit_line_polynomial).
    q_offset = float(np.mean(q))
    q_scale = float(np.std(q)) or 1.0
    q_norm = (q - q_offset) / q_scale
    coeffs = np.polyfit(q_norm, p, 1)         # [slope, intercept]
    p_fit = np.polyval(coeffs, q_norm)
    rmse = float(np.sqrt(np.mean((p_fit - p) ** 2)))
    line_obj.pixels = pts_u
    line_obj.poly = coeffs
    line_obj.poly_q_offset = q_offset
    line_obj.poly_q_scale = q_scale
    line_obj.residual_rmse = rmse
    return line_obj


def _intersect_linear_yardline_sideline(
    yl: LineObject, sl: LineObject,
) -> tuple[float, float] | None:
    """Closed-form intersection of two linear lines in undistorted pixel space.

    yl is a yardline (x = a + b·y, deg-1 in y).  sl is a sideline (y = c + d·x).
    Solve the 2x2 system:
        x − b·y = a  ↔  yardline rearranged
        −d·x + y = c  ↔  sideline rearranged
    With deg-1 fits stored as `poly = [slope, intercept]` (because
    `np.polyfit(q_norm, p, 1)` returns `[slope, intercept]`), we still need to
    account for the fit centering (q_offset/q_scale).

    Returns (x, y) or None if lines are parallel / fit missing.
    """
    if yl.poly is None or sl.poly is None:
        return None
    if len(yl.poly) != 2 or len(sl.poly) != 2:
        return None
    # Yardline: x = poly_yl[0] · ((y − q_off_yl)/q_sc_yl) + poly_yl[1]
    #         = (poly_yl[0]/q_sc_yl)·y + (poly_yl[1] − poly_yl[0]·q_off_yl/q_sc_yl)
    b = float(yl.poly[0]) / yl.poly_q_scale
    a = float(yl.poly[1]) - float(yl.poly[0]) * yl.poly_q_offset / yl.poly_q_scale
    # Sideline: y = poly_sl[0]·((x − q_off_sl)/q_sc_sl) + poly_sl[1]
    #         = (poly_sl[0]/q_sc_sl)·x + (poly_sl[1] − poly_sl[0]·q_off_sl/q_sc_sl)
    d = float(sl.poly[0]) / sl.poly_q_scale
    c = float(sl.poly[1]) - float(sl.poly[0]) * sl.poly_q_offset / sl.poly_q_scale
    # Solve x = a + b·y, y = c + d·x  →  x = a + b·(c + d·x)  →  x(1 − b·d) = a + b·c
    denom = 1.0 - b * d
    if abs(denom) < 1e-9:
        return None  # parallel / degenerate
    x = (a + b * c) / denom
    y = c + d * x
    return float(x), float(y)


def solve_grid(
    yard_mask: np.ndarray,
    side_mask: np.ndarray,
    hash_pxs: np.ndarray,
    hash_confs: np.ndarray | None = None,
    frame_shape: tuple[int, int] | None = None,
    use_gpu_vp: bool = False,
    vp_init: tuple[float, float] | None = None,
    vp_device: str = "mps",
    vp_rerank_top_n: int = 250,
    grouping_mode: str = "vp",  # "vp" | "cc"
    linearize: bool = False,    # default OFF historically; for clip-level
                                 # tracker use, pair True with `intrinsics_override`
                                 # so distortion is calibrated once per clip.
                                 # Per-frame recalibration causes H jitter.
    intrinsics_override=None,    # If provided (CameraIntrinsics), skip the
                                 # internal distortion calibration step in
                                 # the linearize path and use these instead.
                                 # Used by HomographyTracker to cache
                                 # intrinsics across frames.
    focal_length_guess: float | None = None,
) -> GridSolverResult:
    """Run the v2 pipeline end-to-end on pre-computed masks + hash detections.

    Inputs:
      yard_mask:  (H, W) binary mask of yard-line pixels (at frame resolution)
      side_mask:  (H, W) binary mask of sideline pixels
      hash_pxs:   (K, 2) hash detections in pixel coords (W18 model)
      hash_confs: (K,)   hash confidences (optional, unused for now)
      frame_shape: (H, W) tuple, inferred from yard_mask if None

    Returns a GridSolverResult with yardlines, sidelines, and diagnostics.
    """
    if frame_shape is None:
        frame_shape = yard_mask.shape[:2]
    result = GridSolverResult(frame_shape=tuple(frame_shape))

    # 1. Group UNet pixels into per-line objects.
    if grouping_mode == "cc":
        yl_groups = group_yardline_pixels_cc(yard_mask)
    else:
        yl_groups = group_yardline_pixels(
            yard_mask, use_gpu_vp=use_gpu_vp, vp_init=vp_init,
            vp_device=vp_device, vp_rerank_top_n=vp_rerank_top_n,
        )
    sl_groups = group_sideline_pixels(side_mask)
    result.notes.append(f"yardline groups: {len(yl_groups)}, sideline groups: {len(sl_groups)}")

    # 2. Fit polynomials.
    for g in yl_groups:
        fit_line_polynomial(g)
    for g in sl_groups:
        fit_line_polynomial(g)

    # 3. Assign sidelines by peak y (smaller y = far, larger = near).
    sl_sorted = sorted([g for g in sl_groups if g.poly is not None],
                        key=lambda g: g.peak_coord or 0.0)
    if len(sl_sorted) >= 1:
        result.far_sideline = sl_sorted[0]
    if len(sl_sorted) >= 2:
        result.near_sideline = sl_sorted[-1]

    # 4. Wrap yardline groups as Yardline objects.
    result.yardlines = [Yardline(line=g) for g in yl_groups if g.poly is not None]

    # ── 4b. Linearize step (default ON): undistort all line pixels, refit
    #        as deg-1, and replace deg-4 polys. Yardlines should be straight
    #        in undistorted space — fitting them as such avoids the deg-4
    #        extrapolation explosion when intersecting with sidelines that
    #        sit outside the yardline's observed pixel range. Also drops the
    #        cv2.undistortPoints step out of fit_homography_from_result. ──
    if linearize:
        from .distortion import CameraIntrinsics, undistort_points
        h_, w_ = result.frame_shape
        f_ = (focal_length_guess if focal_length_guess is not None
              else float(max(w_, h_)))
        if intrinsics_override is not None:
            # Caller (e.g. HomographyTracker on non-bootstrap frames) passed
            # cached intrinsics — skip the per-frame calibration so undistortion
            # is consistent across the whole clip.
            intrinsics = intrinsics_override
        else:
            # Calibrate from line pixel sets (deg-4 polys still on hand from step 2,
            # used by the plumb-line residual; same call we'd make later).
            k1, k2 = calibrate_distortion_from_result(
                result, frame_shape=(h_, w_), focal_length_guess=f_,
            )
            intrinsics = CameraIntrinsics(
                fx=f_, fy=f_, cx=w_ / 2.0, cy=h_ / 2.0, k1=k1, k2=k2,
            )
        # Undistort + linear-refit each yardline + sideline.
        for yl_obj in result.yardlines:
            _linearize_line_in_undistorted_space(yl_obj.line, intrinsics)
        if result.far_sideline is not None:
            _linearize_line_in_undistorted_space(result.far_sideline, intrinsics)
        if result.near_sideline is not None:
            _linearize_line_in_undistorted_space(result.near_sideline, intrinsics)
        # Undistort hash pixels for downstream assignment + correspondence use.
        if hash_pxs is not None and len(hash_pxs) > 0:
            hash_pxs = undistort_points(np.asarray(hash_pxs, dtype=np.float64),
                                          intrinsics)
        result.is_linearized = True
        result.intrinsics = intrinsics

    # 5. Attach hashes to yardlines (in whichever space we're in — distorted
    #    if linearize=False, undistorted if linearize=True).
    assign_hashes_to_yardlines(np.asarray(hash_pxs), result.yardlines)
    classify_singleton_hashes(result.yardlines)

    # 6. Sideline keypoints: intersect yardline × sideline. With linearize=True
    #    we use the closed-form linear-linear solver (handles arbitrary
    #    extrapolation distance because both lines are exactly linear in
    #    undistorted space). Otherwise fall back to the deg-4 Brent root-find.
    for yl in result.yardlines:
        for sl_attr, sl_obj in (("far_sideline", result.far_sideline),
                                  ("near_sideline", result.near_sideline)):
            if sl_obj is None:
                continue
            if linearize:
                pt = _intersect_linear_yardline_sideline(yl.line, sl_obj)
            else:
                pt = intersect_yardline_sideline(yl.line, sl_obj)
            if pt is not None:
                setattr(yl, sl_attr, np.array(pt))

    # Capture the VP that was used for grouping (None in CC mode — assign_grid_positions
    # falls back to poly-eval-at-ref-y, which is fine since CC grouping doesn't need VP).
    result.vp = None if grouping_mode == "cc" else _LAST_VP[0]

    # 7. Assign grid slots using angle-from-VP (or poly-eval fallback if vp is None).
    assign_grid_positions(result.yardlines, vp=result.vp)

    return result


# ── 7. Correspondences / export ──────────────────────────────────────────────

def yardlines_to_correspondences(
    result: GridSolverResult,
    base_ngs_x: float,
    field_x_range: tuple[float, float] = (10.0, 110.0),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Flatten the solved yardlines into (pixel, field) correspondence arrays.

    `base_ngs_x` is the NGS field x-coord assigned to grid_pos == 0 (the
    leftmost detected yardline). Each yardline contributes up to 4 keypoints:
    near_hash, far_hash, near_sideline, far_sideline.

    Only yardlines with `grid_fit_ok` are emitted, and only those whose
    computed field_x falls inside `field_x_range` (default [10, 110] — the
    playable field; back-of-endzone lines at NGS x≈0 or x≈120 look like
    yardlines to UNet but aren't, so we drop them).
    """
    pixel_pts, field_pts, labels = [], [], []
    fx_lo, fx_hi = field_x_range
    for yl in result.yardlines:
        if not yl.grid_fit_ok or yl.grid_pos is None:
            continue
        field_x = base_ngs_x + yl.grid_pos * 5
        if field_x < fx_lo or field_x > fx_hi:
            continue

        if yl.near_hash is not None:
            pixel_pts.append(yl.near_hash.tolist())
            field_pts.append([field_x, HASH_Y_NEAR])
            labels.append(f"g{yl.grid_pos}_near_hash")
        if yl.far_hash is not None:
            pixel_pts.append(yl.far_hash.tolist())
            field_pts.append([field_x, HASH_Y_FAR])
            labels.append(f"g{yl.grid_pos}_far_hash")
        if yl.near_sideline is not None:
            pixel_pts.append(yl.near_sideline.tolist())
            field_pts.append([field_x, 0.0])
            labels.append(f"g{yl.grid_pos}_near_side")
        if yl.far_sideline is not None:
            pixel_pts.append(yl.far_sideline.tolist())
            field_pts.append([field_x, FIELD_WIDTH])
            labels.append(f"g{yl.grid_pos}_far_side")

    return (np.array(pixel_pts, dtype=np.float64) if pixel_pts else np.zeros((0, 2)),
            np.array(field_pts, dtype=np.float64) if field_pts else np.zeros((0, 2)),
            labels)


# ── 8. UNet + W18 hash inference wrappers ────────────────────────────────────

_UNET_CACHE: dict = {}
_HASH_CACHE: dict = {}


def _load_unet(weights_path: str, device: torch.device):
    import segmentation_models_pytorch as smp

    key = (weights_path, str(device))
    if key in _UNET_CACHE:
        return _UNET_CACHE[key]
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=2, activation=None)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    _UNET_CACHE[key] = model
    return model


def run_unet(
    frame: np.ndarray,
    weights_path: str,
    device: str = "mps",
    yard_thresh: float = UNET_YARD_THRESH,
    side_thresh: float = UNET_SIDE_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UNet on a BGR frame; return (yard_mask, side_mask) at frame resolution."""
    dev = torch.device(device)
    model = _load_unet(weights_path, dev)

    h0, w0 = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (UNET_INPUT_W, UNET_INPUT_H))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normed, (2, 0, 1))).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()    # (2, H_in, W_in)

    # Mutually exclusive class assignment: a pixel where both classes exceed
    # threshold goes only to the more confident class. Without this, sideline
    # paint sometimes appears in both masks (or worse, only in yard_mask), so
    # CC grouping picks up a horizontal "yardline" outlier.
    yard_p, side_p = probs[0], probs[1]
    yard = ((yard_p > yard_thresh) & (yard_p >= side_p)).astype(np.uint8)
    side = ((side_p > side_thresh) & (side_p > yard_p)).astype(np.uint8)
    yard = cv2.resize(yard, (w0, h0), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return yard, side


def _load_hash_w18(weights_path: str, device: torch.device):
    from .keypoint_detector import HRNetKeypointModel
    import timm

    key = (weights_path, str(device))
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]

    # W18 hash-only: 1-channel head, timm backbone "hrnet_w18".
    model = HRNetKeypointModel(num_channels=1)
    # Swap the W48 backbone for W18 in the module. The training script allows
    # swapping via args; here we hot-patch the created model to match.
    model.backbone = timm.create_model(
        "hrnet_w18", pretrained=False, features_only=True, out_indices=(0,),
    )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _HASH_CACHE[key] = model
    return model


def run_hash_w18(
    frame: np.ndarray,
    weights_path: str,
    device: str = "mps",
    conf_thresh: float = HASH_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Run W18 hash-only detector. Returns (hash_pxs (K,2), hash_confs (K,))."""
    from .keypoint_detector import _extract_peaks

    dev = torch.device(device)
    model = _load_hash_w18(weights_path, dev)

    h0, w0 = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (HASH_INPUT_W, HASH_INPUT_H))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normed, (2, 0, 1))).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(tensor)
        heatmap = torch.sigmoid(logits[0, 0]).cpu().numpy()   # (H_hm, W_hm)

    hm_h, hm_w = heatmap.shape
    peaks = _extract_peaks(heatmap, conf_thresh)
    if not peaks:
        return np.zeros((0, 2)), np.zeros(0)
    pxs = np.array([[px / hm_w * w0, py / hm_h * h0] for px, py, _ in peaks])
    confs = np.array([c for _, _, c in peaks], dtype=np.float32)
    return pxs, confs


# ── 9. Distortion calibration + homography ───────────────────────────────────

def calibrate_distortion_from_result(
    result: GridSolverResult,
    frame_shape: tuple[int, int] | None = None,
    focal_length_guess: float | None = None,
    subsample: int = 1,
    n_poly_samples: int | None = 15,
) -> tuple[float, float]:
    """Plumb-line radial-distortion fit from the grid solver's line pixels.

    Each yardline + sideline polynomial group is a "plumb line" — straight in
    the real world, curved in image due to distortion. Pick (k1, k2) that
    minimize the perpendicular-residual² of each group's pixels after
    undistortion. Reuses the old module's solver.

    Two thinning modes:
      - `n_poly_samples` (default 15): sample N points uniformly along each
        line's polynomial fit (yardlines: along y; sidelines: along x). The
        polynomial summarizes the line shape — sampling fewer points along it
        captures the same geometry with way less compute. ~5-15ms total.
      - `subsample` (default 1, no thinning): legacy stride-subsample of raw
        pixels. Slower, used if `n_poly_samples` is set to None.

    Set `n_poly_samples=None` to disable poly-sampling and use raw pixels
    (with `subsample` stride if >1).
    """
    from .grid_solver import calibrate_distortion_from_lines

    shape = frame_shape if frame_shape is not None else result.frame_shape
    line_point_sets: list[np.ndarray] = []

    if n_poly_samples is not None and n_poly_samples >= 3:
        # Sample N points uniformly along each polynomial in its observed range.
        for yl in result.yardlines:
            line_obj = yl.line
            if line_obj.poly is None or line_obj.pixels is None:
                continue
            ymin = float(line_obj.pixels[:, 1].min())
            ymax = float(line_obj.pixels[:, 1].max())
            if ymax - ymin < 1.0:
                continue
            ys = np.linspace(ymin, ymax, n_poly_samples)
            xs = eval_poly(line_obj, ys)
            line_point_sets.append(np.column_stack([xs, ys]))
        for sl in (result.far_sideline, result.near_sideline):
            if sl is None or sl.poly is None or sl.pixels is None:
                continue
            xmin = float(sl.pixels[:, 0].min())
            xmax = float(sl.pixels[:, 0].max())
            if xmax - xmin < 1.0:
                continue
            xs = np.linspace(xmin, xmax, n_poly_samples)
            ys = eval_poly(sl, xs)
            line_point_sets.append(np.column_stack([xs, ys]))
    else:
        for yl in result.yardlines:
            pts = yl.line.pixels
            if subsample > 1:
                pts = pts[::subsample]
            if len(pts) >= 3:
                line_point_sets.append(pts)
        for sl in (result.far_sideline, result.near_sideline):
            if sl is None:
                continue
            pts = sl.pixels
            if subsample > 1:
                pts = pts[::subsample]
            if len(pts) >= 3:
                line_point_sets.append(pts)

    if not line_point_sets:
        return 0.0, 0.0
    return calibrate_distortion_from_lines(
        line_point_sets, shape, focal_length_guess=focal_length_guess,
    )


def fit_homography_from_result(
    result: GridSolverResult,
    base_ngs_x: float,
    distortion: tuple[float, float] = (0.0, 0.0),
    frame_shape: tuple[int, int] | None = None,
    focal_length_guess: float | None = None,
    ransac_thresh_px: float = 3.0,
) -> dict:
    """Undistort the keypoint correspondences and fit a homography.

    Returns a dict with:
      H, H_inv: 3x3 homography (pixel→field) and its inverse
      intrinsics: CameraIntrinsics used for undistortion
      n_correspondences, n_inliers
      pixel_pts, field_pts, labels: the input correspondences
      mean_err_yd, median_err_yd, max_err_yd: reprojection error in yards
    """
    from .distortion import CameraIntrinsics, undistort_points

    shape = frame_shape if frame_shape is not None else result.frame_shape
    h, w = shape
    cx, cy = w / 2.0, h / 2.0
    f = focal_length_guess if focal_length_guess is not None else float(max(w, h))

    # If solve_grid already linearized the result, the keypoints in
    # `yardlines_to_correspondences()` are ALREADY in undistorted space and
    # the calibrated (k1, k2) live on result.intrinsics. Skip the redundant
    # undistort_points() and use the cached intrinsics for reporting.
    if result.is_linearized and result.intrinsics is not None:
        intrinsics = result.intrinsics
    else:
        intrinsics = CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy,
                                       k1=distortion[0], k2=distortion[1])

    pixel_pts, field_pts, labels = yardlines_to_correspondences(result, base_ngs_x)
    if len(pixel_pts) < 4:
        return {
            "H": None, "H_inv": None, "intrinsics": intrinsics,
            "n_correspondences": int(len(pixel_pts)), "n_inliers": 0,
            "pixel_pts": pixel_pts, "field_pts": field_pts, "labels": labels,
            "mean_err_yd": float("nan"), "median_err_yd": float("nan"),
            "max_err_yd": float("nan"),
        }

    if result.is_linearized:
        undist_pxs = pixel_pts.astype(np.float64)
    else:
        undist_pxs = undistort_points(pixel_pts, intrinsics)
    H, mask = cv2.findHomography(
        undist_pxs, field_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px,
    )
    if H is None:
        return {
            "H": None, "H_inv": None, "intrinsics": intrinsics,
            "n_correspondences": int(len(pixel_pts)), "n_inliers": 0,
            "pixel_pts": pixel_pts, "field_pts": field_pts, "labels": labels,
            "mean_err_yd": float("nan"), "median_err_yd": float("nan"),
            "max_err_yd": float("nan"),
        }

    # Reprojection error in FIELD yards
    ones = np.ones((len(undist_pxs), 1))
    hom = np.hstack([undist_pxs, ones])
    proj = (H @ hom.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - field_pts, axis=1)

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(err), dtype=bool)
    err_inlier = err[inliers] if inliers.any() else err

    return {
        "H": H, "H_inv": np.linalg.inv(H), "intrinsics": intrinsics,
        "n_correspondences": int(len(pixel_pts)), "n_inliers": int(inliers.sum()),
        "pixel_pts": pixel_pts, "field_pts": field_pts, "labels": labels,
        "mean_err_yd": float(err_inlier.mean()),
        "median_err_yd": float(np.median(err_inlier)),
        "max_err_yd": float(err_inlier.max()),
    }


def solve_frame_full(
    frame: np.ndarray,
    unet_weights: str,
    hash_weights: str,
    base_ngs_x: float,
    device: str = "mps",
    calibrate_k: bool = True,
) -> tuple[GridSolverResult, dict]:
    """End-to-end: UNet + W18 + grid solve + distortion + homography.

    Returns (GridSolverResult, homography_result_dict).
    """
    yard_mask, side_mask = run_unet(frame, unet_weights, device=device)
    hash_pxs, hash_confs = run_hash_w18(frame, hash_weights, device=device)

    result = solve_grid(yard_mask, side_mask, hash_pxs,
                        hash_confs=hash_confs, frame_shape=frame.shape[:2],
                        linearize=calibrate_k)

    if result.is_linearized and result.intrinsics is not None:
        # solve_grid already calibrated and applied (k1, k2) to the keypoints.
        k1, k2 = float(result.intrinsics.k1), float(result.intrinsics.k2)
    elif calibrate_k:
        k1, k2 = calibrate_distortion_from_result(result, frame_shape=frame.shape[:2])
    else:
        k1, k2 = 0.0, 0.0

    homo = fit_homography_from_result(result, base_ngs_x, distortion=(k1, k2),
                                       frame_shape=frame.shape[:2])
    homo["k1"], homo["k2"] = float(k1), float(k2)
    return result, homo
