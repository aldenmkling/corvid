"""
Yard line detection and rectification for sideline-view All-22 frames.

Pipeline:
  1. Canny edge detection
  2. HoughLinesP for long segments, angle filter (30-150°)
  3. Vectorized pairing: find parallel edge pairs (2-15px apart, same angle)
  4. Cluster by x-intercept at mid-frame, reject by angle spread, fit line
  5. Period filter: assign lines to a regular grid using x-intercept spacing
  6. Rectify: map detected line endpoints to evenly-spaced vertical targets

No vanishing point needed — the x-intercept period filter rejects false
positives, and the homography maps directly from detected positions to
target positions.
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, field as dataclass_field


@dataclass
class YardLineResult:
    """Result of yard line detection."""
    lines: list[tuple[float, float, float, float]]  # (x_top, 0, x_bot, h) per line
    grid: list[int]              # grid position for each line (0-based, gaps allowed)
    period_px: float             # detected period in pixels at mid-frame
    elapsed_ms: float            # detection time
    gray: np.ndarray | None = None   # grayscale image (for reuse by hash detection)
    canny: np.ndarray | None = None  # Canny edges (for reuse by hash detection)


@dataclass
class RectificationResult:
    """Result of image rectification."""
    warped: np.ndarray           # rectified image
    H: np.ndarray                # 3x3 homography (includes translation)
    target_cols: list[float]     # x-pixel of each yard line in rectified image
    px_per_unit: float           # pixels per grid unit in rectified image
    margin: float                # left margin in pixels
    tx: float                    # x translation applied
    ty: float                    # y translation applied


def detect_yard_lines(frame: np.ndarray) -> YardLineResult | None:
    """Detect yard lines as paired parallel edges with periodic spacing.

    Uses Canny edges with wide angle tolerance (30-150°) to handle both
    mid-field and goal-line views. Clusters are rejected if their segment
    angles span more than 20°, which filters player noise.

    Returns None if fewer than 3 yard lines are found.
    """
    h, w = frame.shape[:2]
    t0 = time.time()

    # --- Step 1: Canny edge detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # --- Step 2: Detect long segments, filter by angle ---
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                          minLineLength=80, maxLineGap=15)
    if raw is None:
        return None
    segs = raw[:, 0, :]

    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.arctan2(dy, dx)
    steep = (np.abs(angles) > np.radians(30)) & (np.abs(angles) < np.radians(150))
    segs = segs[steep]
    if len(segs) < 4:
        return None

    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.arctan2(dy, dx)
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    N = len(segs)
    mid_y = h / 2

    # --- Step 3: Vectorized pairing ---
    i_idx, j_idx = np.triu_indices(N, k=1)

    a_diff = np.abs(angles[i_idx] - angles[j_idx])
    a_diff = np.minimum(a_diff, np.pi - a_diff)
    angle_ok = a_diff < np.radians(10)

    mid_x_i = (segs[i_idx, 0] + segs[i_idx, 2]) / 2
    mid_y_i = (segs[i_idx, 1] + segs[i_idx, 3]) / 2
    mid_x_j = (segs[j_idx, 0] + segs[j_idx, 2]) / 2
    mid_y_j = (segs[j_idx, 1] + segs[j_idx, 3]) / 2
    nx = dy[i_idx] / lengths[i_idx]
    ny = -dx[i_idx] / lengths[i_idx]
    perp_dist = np.abs((mid_x_j - mid_x_i) * nx + (mid_y_j - mid_y_i) * ny)
    dist_ok = (perp_dist >= 2) & (perp_dist <= 15)

    y_min_i = np.minimum(segs[i_idx, 1], segs[i_idx, 3])
    y_max_i = np.maximum(segs[i_idx, 1], segs[i_idx, 3])
    y_min_j = np.minimum(segs[j_idx, 1], segs[j_idx, 3])
    y_max_j = np.maximum(segs[j_idx, 1], segs[j_idx, 3])
    overlap = np.maximum(0, np.minimum(y_max_i, y_max_j) -
                         np.maximum(y_min_i, y_min_j))
    max_span = np.maximum(y_max_i - y_min_i, y_max_j - y_min_j)
    overlap_ok = overlap >= 0.3 * max_span

    valid = angle_ok & dist_ok & overlap_ok
    paired_indices = set(i_idx[valid].tolist() + j_idx[valid].tolist())
    if len(paired_indices) < 4:
        return None

    # --- Step 4: Cluster and fit ---
    paired_segs = segs[list(paired_indices)]
    paired_dx = paired_segs[:, 2] - paired_segs[:, 0]
    paired_dy = paired_segs[:, 3] - paired_segs[:, 1]
    paired_lengths = np.sqrt(paired_dx ** 2 + paired_dy ** 2)

    t_mid_p = np.where(np.abs(paired_dy) > 1e-6,
                       (mid_y - paired_segs[:, 1]) / paired_dy, 0.0)
    x_mid_p = paired_segs[:, 0] + paired_dx * t_mid_p

    order = np.argsort(x_mid_p)
    clusters: list[list[int]] = []
    cur_idxs = [order[0]]
    for k in range(1, len(order)):
        if x_mid_p[order[k]] - x_mid_p[order[k - 1]] < 20:
            cur_idxs.append(order[k])
        else:
            clusters.append(cur_idxs)
            cur_idxs = [order[k]]
    clusters.append(cur_idxs)

    fitted_lines: list[tuple[float, float, float, float]] = []
    line_x_mids: list[float] = []

    paired_angles = np.arctan2(paired_dy, paired_dx)

    for cidxs in clusters:
        c_segs = paired_segs[cidxs]
        c_lens = paired_lengths[list(cidxs)]
        pts = c_segs.reshape(-1, 2)
        y_span = pts[:, 1].max() - pts[:, 1].min()
        total_len = float(c_lens.sum())
        if y_span < 0.25 * h or total_len < 150:
            continue

        # Reject clusters with inconsistent segment angles
        c_angles = paired_angles[cidxs]
        angle_spread = np.ptp(c_angles)
        if angle_spread > np.radians(20):
            continue

        vx, vy, cx, cy = cv2.fitLine(
            pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).flatten()
        if abs(vy) < 1e-6:
            continue

        t_top = -cy / vy
        t_bot = (h - cy) / vy
        x_top = float(cx + vx * t_top)
        x_bot = float(cx + vx * t_bot)
        fitted_lines.append((x_top, 0.0, x_bot, float(h)))
        line_x_mids.append(x_top + 0.5 * (x_bot - x_top))

    if len(fitted_lines) < 3:
        return None

    sort_order = np.argsort(line_x_mids)
    sorted_x_mids = np.array([line_x_mids[i] for i in sort_order])
    sorted_lines = [fitted_lines[i] for i in sort_order]

    # --- Step 5: Period filter (x-intercept spacing) ---
    best_period: float | None = None
    best_grid: dict[int, tuple[int, float]] | None = None
    best_score = (0, float('inf'), 0.0)

    n_cands = len(sorted_x_mids)
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            gap = sorted_x_mids[j] - sorted_x_mids[i]
            for div in range(1, 5):
                period = gap / div
                if period < 30 or period > w / 2:
                    continue

                offsets = (sorted_x_mids - sorted_x_mids[0]) / period
                grid_pos = np.round(offsets).astype(int)
                residuals = np.abs(offsets - grid_pos)

                tol = 0.15
                inlier_mask = residuals < tol
                if inlier_mask.sum() < 3:
                    continue

                slot_best: dict[int, tuple[int, float]] = {}
                for k in range(n_cands):
                    if not inlier_mask[k]:
                        continue
                    slot = int(grid_pos[k])
                    res = float(residuals[k])
                    if slot not in slot_best or res < slot_best[slot][1]:
                        slot_best[slot] = (k, res)

                n_unique = len(slot_best)
                grid_range = max(slot_best.keys()) - min(slot_best.keys())
                score = (n_unique, -grid_range, period)

                if score > best_score:
                    best_score = score
                    best_period = period
                    best_grid = slot_best

    if best_grid is None or best_period is None or len(best_grid) < 3:
        return None

    final_indices = [v[0] for v in best_grid.values()]
    final_grid = list(best_grid.keys())
    final_lines = [sorted_lines[i] for i in final_indices]

    min_g = min(final_grid)
    final_grid = [g - min_g for g in final_grid]

    elapsed = time.time() - t0
    return YardLineResult(
        lines=final_lines,
        grid=final_grid,
        period_px=best_period,
        elapsed_ms=elapsed * 1000,
        gray=gray,
        canny=edges,
    )


def rectify(frame: np.ndarray, result: YardLineResult) -> RectificationResult | None:
    """Rectify a frame so detected yard lines become vertical and evenly spaced.

    Maps detected line endpoints directly to target positions. Computes the
    full bounding box so no content is cropped.
    """
    h, w = frame.shape[:2]
    lines = result.lines
    grid = result.grid

    max_grid = max(grid)
    margin = 100.0
    usable_w = w - 2 * margin
    px_per_unit = usable_w / max(max_grid, 1)

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []

    for line, gpos in zip(lines, grid):
        x_top, y_top, x_bot, y_bot = line
        target_x = margin + gpos * px_per_unit
        src_pts.extend([[x_top, y_top], [x_bot, y_bot]])
        dst_pts.extend([[target_x, 0.0], [target_x, float(h)]])

    H, mask = cv2.findHomography(
        np.array(src_pts, np.float64),
        np.array(dst_pts, np.float64),
        cv2.RANSAC, 3.0,
    )
    if H is None:
        return None

    # Compute bounding box of all four warped corners
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    wc = (H @ corners_h.T).T
    wc = wc[:, :2] / wc[:, 2:3]

    x_min = int(np.floor(wc[:, 0].min()))
    x_max = int(np.ceil(wc[:, 0].max()))
    y_min = int(np.floor(wc[:, 1].min()))
    y_max = int(np.ceil(wc[:, 1].max()))

    # Translation to keep everything in positive coords
    tx = float(-x_min)
    ty = float(-y_min)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_full = T @ H

    out_w = x_max - x_min
    out_h = y_max - y_min
    warped = cv2.warpPerspective(frame, H_full, (out_w, out_h))

    target_cols = [margin + g * px_per_unit + tx for g in grid]

    return RectificationResult(
        warped=warped,
        H=H_full,
        target_cols=target_cols,
        px_per_unit=px_per_unit,
        margin=margin,
        tx=tx,
        ty=ty,
    )
