"""Grid solver — yard-line identification from HRNet keypoints.

Production module (used by HomographyTracker). Pipeline:

  1. `run_hrnet` — HRNet-W48 inference on a BGR frame, returns 2-ch heatmaps.
  2. `extract_peaks` — connected-component peak extraction per channel.
  3. `pair_hashes` — row-clustering + hybrid column/angle hash pairing
     (handles both wide and heavily tilted red-zone shots).
  4. `find_sideline_on_yard_line` — attach a sideline detection if collinear
     with a paired hash group.
  5. `assign_grid_positions` — integer slot indices per yard line; handles
     bimodal spacings (5yd + 10yd) via min-diff + weighted LS refit.
  6. `build_yard_line_groups` — full-stack wrapper returning grouped yard lines.
  7. `groups_to_correspondences` — flatten groups to pixel↔field pairs for
     homography fitting.
  8. `calibrate_distortion_from_lines` — plumb-line fit for (k1, k2).

Moved here from `scripts/testing/test_yard_line_grouping.py` and
`scripts/testing/test_grid_solver_camera.py` when the grid solver was
promoted to production.
"""

import os
import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import minimize

from .keypoint_detector import HRNetKeypointModel, _refine_peak
from .distortion import CameraIntrinsics, undistort_points
from .field_model import (
    HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH, FIELD_LENGTH,
)


HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30

INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Real-world field geometry (yards)
HASH_ROW_SPAN_YARDS = HASH_Y_FAR - HASH_Y_NEAR          # 6.167 yd
HASH_TO_FAR_SIDELINE_YARDS = FIELD_WIDTH - HASH_Y_FAR    # 23.58 yd
HASH_TO_NEAR_SIDELINE_YARDS = HASH_Y_NEAR               # 23.58 yd


_HRNET_CACHE = {}  # (weights_path, device_str) -> (model, torch.device)


def _get_hrnet(weights_path, device="cpu"):
    """Load HRNet once per (weights, device) and cache. Avoids reloading the
    ~100MB .pth for every frame when running batches."""
    key = (weights_path, str(device))
    if key in _HRNET_CACHE:
        return _HRNET_CACHE[key]
    dev = torch.device(device)
    model = HRNetKeypointModel(num_channels=2)
    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(dev).eval()
    _HRNET_CACHE[key] = (model, dev)
    return model, dev


def run_hrnet(frame, weights_path, device="cpu"):
    """Run HRNet and return raw heatmaps."""
    model, dev = _get_hrnet(weights_path, device)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(tensor)
        heatmaps = torch.sigmoid(logits[0]).cpu().numpy()
    return heatmaps


def extract_peaks(heatmap, threshold, orig_shape):
    """Return (N, 2) peak pixel positions in original frame coords + confidences."""
    orig_h, orig_w = orig_shape
    hm_h, hm_w = heatmap.shape
    mask = heatmap >= threshold
    if not mask.any():
        return np.zeros((0, 2)), np.zeros(0)
    labels, n = ndimage.label(mask)
    out_pxs = []
    out_confs = []
    for comp_id in range(1, n + 1):
        comp_mask = labels == comp_id
        vals = heatmap * comp_mask
        peak_idx = vals.argmax()
        py, px = peak_idx // hm_w, peak_idx % hm_w
        peak_val = float(heatmap[py, px])
        ref_x, ref_y = _refine_peak(heatmap, py, px)
        out_pxs.append([ref_x / hm_w * orig_w, ref_y / hm_h * orig_h])
        out_confs.append(peak_val)
    return np.array(out_pxs), np.array(out_confs)


def cluster_hashes_by_column(hash_pxs, max_same_column_gap=0.30):
    """Group hash detections into "columns" (same yard line).

    Sorts hashes by column coord (projection onto the PCA dominant axis, which
    runs across yard lines). Adjacent hashes in that order whose column-coord
    gap is small (< max_same_column_gap × typical yard-line spacing) are
    considered to share a yard line.

    Returns a list of columns, where each column is a list of (hash_idx, column_coord, y).
    Columns are ordered left-to-right.
    """
    n = len(hash_pxs)
    if n == 0:
        return []

    # PCA on all hashes to find the cross-field axis
    mean = hash_pxs.mean(axis=0)
    centered = hash_pxs - mean
    if n == 1:
        return [[(0, 0.0, float(hash_pxs[0, 1]))]]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    row_dir = vt[0]  # dominant direction (along hash rows, across yard lines)

    # Column coord per hash
    col_coords = centered @ row_dir

    # Sort by column coord and group adjacent entries
    order = np.argsort(col_coords)
    sorted_coords = col_coords[order]
    sorted_ys = hash_pxs[order, 1]

    # Estimate typical yard-line spacing from the largest gaps
    diffs = np.diff(sorted_coords)
    if len(diffs) == 0:
        return [[(int(order[0]), float(sorted_coords[0]), float(sorted_ys[0]))]]
    typical_spacing = float(np.median(diffs)) if len(diffs) > 0 else 140.0
    # If every consecutive gap is roughly the same, that IS our spacing.
    # But if there's bimodal distribution (tight clusters + big gaps between),
    # the median may fall in the wrong bucket. Use the larger modes: gaps
    # bigger than 1/3 of max diff are probably between yard lines.
    big_gap_threshold = max(
        max_same_column_gap * typical_spacing,
        0.30 * float(np.max(diffs)),
    )

    # Group by walking through sorted order; start new column when gap too big
    columns = []
    current = [(int(order[0]), float(sorted_coords[0]), float(sorted_ys[0]))]
    for k in range(1, len(order)):
        gap = sorted_coords[k] - sorted_coords[k - 1]
        if gap > big_gap_threshold:
            columns.append(current)
            current = []
        current.append((int(order[k]), float(sorted_coords[k]),
                        float(sorted_ys[k])))
    if current:
        columns.append(current)

    return columns


def split_hash_rows(hash_pxs):
    """Deprecated alias for backward compatibility.

    Kept so legacy callers that still split first then pair continue working.
    Returns (far_hashes, near_hashes) by PCA perpendicular-distance split.
    """
    if len(hash_pxs) < 2:
        return hash_pxs.copy(), np.zeros((0, 2))

    mean = hash_pxs.mean(axis=0)
    centered = hash_pxs - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    perp = centered @ normal

    group_a = hash_pxs[perp < 0]
    group_b = hash_pxs[perp >= 0]

    if len(group_a) == 0:
        return group_b, group_a
    if len(group_b) == 0:
        return group_a, group_b
    if group_a[:, 1].mean() < group_b[:, 1].mean():
        return group_a, group_b
    return group_b, group_a


def _greedy_pair(far_hashes, near_hashes, is_valid_pair, cost_fn):
    """Generic greedy pairing given a validity predicate and cost function."""
    pairs = []
    used_near = set()
    sort_order = np.argsort(far_hashes[:, 0]) if len(far_hashes) > 0 else []
    for i in sort_order:
        fh = far_hashes[i]
        best_j = None
        best_cost = float('inf')
        for j in range(len(near_hashes)):
            if j in used_near:
                continue
            nh = near_hashes[j]
            if not is_valid_pair(fh, nh):
                continue
            c = cost_fn(fh, nh)
            if c < best_cost:
                best_cost = c
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            used_near.add(best_j)
    return pairs, used_near


def estimate_yard_line_angle(pairs, far_hashes, near_hashes):
    """Return (median_angle_deg, std_angle_deg) of pair vectors (near -> far).

    Angle is measured from vertical (upward = 0°). Positive = leaning right.
    """
    if not pairs:
        return None, None
    angles = []
    for far_idx, near_idx in pairs:
        fh = far_hashes[far_idx]
        nh = near_hashes[near_idx]
        dx = fh[0] - nh[0]
        dy = fh[1] - nh[1]  # negative (far is above near)
        # Angle from vertical, positive = leaning right (top-right)
        angle_rad = np.arctan2(dx, -dy)   # -dy because "up" is negative in image coords
        angles.append(np.degrees(angle_rad))
    angles = np.array(angles)
    return float(np.median(angles)), float(np.std(angles))


def _row_coord(pt, mean_point, yardline_dir):
    """Project a point onto the yardline_dir axis (distance perpendicular to
    the hash rows). All detections in the same row should share row_coord."""
    centered = np.array([pt[0] - mean_point[0], pt[1] - mean_point[1]])
    return float(np.dot(centered, yardline_dir))


def yardline_tilt_slope_from_pairs(pairs, far_hashes=None, near_hashes=None):
    """Average dx/dy of near→far hash pair vectors — i.e. the yard-line
    tilt in image coords. Returns None if no pairs.

    Accepts either:
      - new-style pairs: list of (far_pt, near_pt) point tuples
      - legacy pairs: list of (far_idx, near_idx) with far_hashes/near_hashes

    dx/dy is the horizontal shift per unit of vertical rise along a yard
    line. A sideline at the top of the frame can be projected down to the
    hash row via `x_at_ref = x - slope * (y - ref_y)`.
    """
    if not pairs:
        return None
    dxdy = []
    for pair in pairs:
        a, b = pair
        # Point tuple form (np.ndarray or list) vs index form (int)
        if np.ndim(a) >= 1:
            fh = np.asarray(a)
            nh = np.asarray(b)
        else:
            fh = far_hashes[a]
            nh = near_hashes[b]
        dy = float(fh[1]) - float(nh[1])
        if abs(dy) < 1e-6:
            continue
        dx = float(fh[0]) - float(nh[0])
        dxdy.append(dx / dy)
    if not dxdy:
        return None
    return float(np.median(dxdy))


def compute_hash_pca(hash_pxs):
    """Return (mean_point, row_dir, yardline_dir) from PCA on the hashes.

    - `row_dir` is the dominant axis — points ACROSS yard lines, along a row.
    - `yardline_dir` is perpendicular — points ALONG a yard line, from one
      row to the other (direction may be up or down in image coords).
    - `mean_point` is the hash centroid (needed to compute column coords
      for any point, e.g. a standalone sideline).

    Returns (None, None, None) if fewer than 2 hashes.
    """
    if len(hash_pxs) < 2:
        return None, None, None
    mean = hash_pxs.mean(axis=0)
    centered = hash_pxs - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    row_dir = vt[0]
    yardline_dir = np.array([-row_dir[1], row_dir[0]])
    return mean, row_dir, yardline_dir


def pair_hashes(far_hashes, near_hashes, **kwargs):
    """Row-clustering hash pairing.

    Combines far+near hashes, column-clusters by PCA (PCA is safe for column
    coord since x-spread dominates), then does the row classification by
    DIRECTLY FITTING two parallel-ish lines (one per row) rather than relying
    on a single global slope. Global PCA slope gets biased by outliers.

    Algorithm:
      1. Column-cluster via cluster_hashes_by_column (uses PCA row_dir).
      2. Cap each column at max 2 hashes (keep extremes by y).
      3. Seed row clustering by raw-y: find the largest y-gap that produces
         a balanced split (both sides >= min_per_cluster). This ignores
         outlier-created gaps which are almost always imbalanced.
      4. Fit a line y = slope*x + intercept through each seed cluster.
      5. Reassign every hash to the closer line if residual <= tol, else drop.
         Refit lines once with new assignments.
      6. If the gap between the two row lines is too small relative to within-
         line spread (single-row case), emit all hashes as singletons by
         caller's original classification.
      7. Emit pair when a column has one kept hash from each row.

    Ignores caller's far/near split for classification. Return values use
    POINTS (np.ndarray shape (2,)) rather than indices.

    Returns:
      pairs: list of (far_pt, near_pt) tuples
      unpaired_far: list of far-singleton points
      unpaired_near: list of near-singleton points
      angle_deg: yard-line tilt from vertical, None if <2 hashes
      std_deg: placeholder, always 0.0
    """
    n_far = len(far_hashes)
    n_near = len(near_hashes)
    if n_far == 0 and n_near == 0:
        return [], [], [], None, None

    single_row_ratio = float(kwargs.get('single_row_ratio', 1.5))
    outlier_min_px = float(kwargs.get('outlier_min_px', 15.0))
    outlier_gap_frac = float(kwargs.get('outlier_gap_frac', 0.3))

    # Combine. Indices < n_far are caller's "far"; >= n_far are caller's "near".
    pts_list = []
    if n_far > 0:
        pts_list.append(np.asarray(far_hashes, dtype=np.float64))
    if n_near > 0:
        pts_list.append(np.asarray(near_hashes, dtype=np.float64))
    all_pts = np.vstack(pts_list)
    n = len(all_pts)

    if n == 1:
        if n_far == 1:
            return [], [all_pts[0]], [], None, 0.0
        return [], [], [all_pts[0]], None, 0.0

    # Column clustering (PCA row_dir is robust enough for column coord because
    # x-spread dominates; outliers in y barely shift it).
    columns = cluster_hashes_by_column(all_pts)

    # Cap each column at max 2 hashes (pick extremes by y → most likely pair).
    trimmed_columns = []
    for col in columns:
        if len(col) <= 2:
            trimmed_columns.append(list(col))
        else:
            by_y = sorted(col, key=lambda e: e[2])
            trimmed_columns.append([by_y[0], by_y[-1]])

    kept_idxs = [ci for col in trimmed_columns for (ci, _, _) in col]
    if len(kept_idxs) < 2:
        ci = kept_idxs[0]
        if ci < n_far:
            return [], [all_pts[ci]], [], None, 0.0
        return [], [], [all_pts[ci]], None, 0.0

    # Seed row clustering by raw-y with balanced-split constraint.
    # We want a gap that cleanly separates the two rows, not an outlier gap.
    kept_ys = np.array([all_pts[ci, 1] for ci in kept_idxs])
    kept_xs = np.array([all_pts[ci, 0] for ci in kept_idxs])
    order = np.argsort(kept_ys)
    sorted_ys = kept_ys[order]
    nk = len(kept_idxs)
    min_per_cluster = max(2, int(round(0.25 * nk)))

    best_gap = -1.0
    best_split = None
    for k in range(min_per_cluster - 1, nk - min_per_cluster):
        gap = float(sorted_ys[k + 1] - sorted_ys[k])
        if gap > best_gap:
            best_gap = gap
            best_split = k

    def _single_row_return(angle_guess):
        u_far, u_near = [], []
        for ci in kept_idxs:
            if ci < n_far:
                u_far.append(all_pts[ci])
            else:
                u_near.append(all_pts[ci])
        return [], u_far, u_near, angle_guess, 0.0

    if best_split is None:
        # Not enough points for balanced split — fall back to single-row.
        return _single_row_return(None)

    # Balanced split seeds.
    seed_a_ci = [kept_idxs[order[i]] for i in range(best_split + 1)]
    seed_b_ci = [kept_idxs[order[i]] for i in range(best_split + 1, nk)]

    def _fit_line(cis):
        xs = np.array([all_pts[c, 0] for c in cis], dtype=np.float64)
        ys = np.array([all_pts[c, 1] for c in cis], dtype=np.float64)
        if len(cis) == 1:
            return 0.0, float(ys[0])
        # Guard against degenerate x-spread (vertical line)
        if xs.max() - xs.min() < 1e-6:
            return 0.0, float(np.median(ys))
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)

    slope_a, intercept_a = _fit_line(seed_a_ci)
    slope_b, intercept_b = _fit_line(seed_b_ci)

    x_ref = float(np.median(kept_xs))

    def _row_gap_and_tol():
        ya = slope_a * x_ref + intercept_a
        yb = slope_b * x_ref + intercept_b
        gap = abs(ya - yb)
        tol = max(outlier_min_px, outlier_gap_frac * gap)
        return gap, tol

    # Reclassify + refit once using the seed lines.
    def _reclassify(slope_a, intercept_a, slope_b, intercept_b, tol):
        labels = {}
        keep_a, keep_b = [], []
        for ci in kept_idxs:
            x, y = all_pts[ci, 0], all_pts[ci, 1]
            ra = abs(y - (slope_a * x + intercept_a))
            rb = abs(y - (slope_b * x + intercept_b))
            if ra <= rb and ra <= tol:
                labels[ci] = 'a'
                keep_a.append(ci)
            elif rb < ra and rb <= tol:
                labels[ci] = 'b'
                keep_b.append(ci)
            else:
                labels[ci] = 'drop'
        return labels, keep_a, keep_b

    _, tol_seed = _row_gap_and_tol()
    labels, keep_a, keep_b = _reclassify(
        slope_a, intercept_a, slope_b, intercept_b, tol_seed,
    )
    if len(keep_a) >= 1:
        slope_a, intercept_a = _fit_line(keep_a)
    if len(keep_b) >= 1:
        slope_b, intercept_b = _fit_line(keep_b)

    # Second pass with refined lines.
    row_gap, tol = _row_gap_and_tol()
    labels, keep_a, keep_b = _reclassify(
        slope_a, intercept_a, slope_b, intercept_b, tol,
    )

    # Single-row sanity check: gap between rows should substantially exceed
    # within-row spread. Use MAD of y-residuals to the fitted lines.
    def _mad_residuals(cis, slope, intercept):
        if not cis:
            return 0.0
        rs = np.array([
            all_pts[c, 1] - (slope * all_pts[c, 0] + intercept) for c in cis
        ])
        return float(np.median(np.abs(rs - np.median(rs)))) if len(rs) > 1 else 0.0

    mad_a = _mad_residuals(keep_a, slope_a, intercept_a)
    mad_b = _mad_residuals(keep_b, slope_b, intercept_b)
    within_spread = max(mad_a, mad_b, 1.0)
    angle_est = float(np.degrees(np.arctan((slope_a + slope_b) * 0.5)))

    if row_gap / within_spread < single_row_ratio:
        return _single_row_return(angle_est)

    # Determine which line is far (upper in image = smaller y).
    y_a_mid = slope_a * x_ref + intercept_a
    y_b_mid = slope_b * x_ref + intercept_b
    if y_a_mid < y_b_mid:
        slope_far, intercept_far = slope_a, intercept_a
        slope_near, intercept_near = slope_b, intercept_b
        far_label, near_label = 'a', 'b'
    else:
        slope_far, intercept_far = slope_b, intercept_b
        slope_near, intercept_near = slope_a, intercept_a
        far_label, near_label = 'b', 'a'

    # === Pairing: hybrid strategy ===
    # STRATEGY 1: column-based (works for wide shots where yard lines are
    # nearly perpendicular to hash rows in image).
    # STRATEGY 2: yard-line-angle search (works for red-zone / steep-perspective
    # shots where yard lines tilt 30-45° so column clustering misassigns).
    #
    # Column-based is more robust to spurious global-angle matches in
    # regular-grid frames, so we prefer it when it yields any pairs. Angle
    # search is used only when column-based returns 0 pairs.
    keep_far = [ci for ci in kept_idxs if labels.get(ci) == far_label]
    keep_near = [ci for ci in kept_idxs if labels.get(ci) == near_label]

    def _resid_far(ci):
        return abs(all_pts[ci, 1] - (slope_far * all_pts[ci, 0] + intercept_far))

    def _resid_near(ci):
        return abs(all_pts[ci, 1] - (slope_near * all_pts[ci, 0] + intercept_near))

    # --- Strategy 1: column-based ---
    col_pairs = []
    col_used_far = set()
    col_used_near = set()
    for col in trimmed_columns:
        f_in = [ci for (ci, _, _) in col if labels.get(ci) == far_label]
        n_in = [ci for (ci, _, _) in col if labels.get(ci) == near_label]
        if f_in and n_in:
            bf = min(f_in, key=_resid_far)
            bn = min(n_in, key=_resid_near)
            col_pairs.append((bf, bn))
            col_used_far.add(bf)
            col_used_near.add(bn)

    # --- Strategy 2: yard-line angle search ---
    pair_tol = max(outlier_min_px, 0.15 * row_gap)

    def _score_angle(angle_deg):
        a = np.radians(angle_deg)
        dx_dir = float(np.sin(a))
        dy_dir = float(np.cos(a))
        denom = dy_dir - slope_near * dx_dir
        if abs(denom) < 1e-6:
            return 0, float('inf'), []
        matches = []
        for fi in keep_far:
            fx, fy = float(all_pts[fi, 0]), float(all_pts[fi, 1])
            t = (slope_near * fx + intercept_near - fy) / denom
            if t <= 0.0:
                continue
            px = fx + t * dx_dir
            py = fy + t * dy_dir
            best_ni, best_d = None, pair_tol
            for ni in keep_near:
                nx, ny = float(all_pts[ni, 0]), float(all_pts[ni, 1])
                d = float(np.hypot(px - nx, py - ny))
                if d < best_d:
                    best_d = d
                    best_ni = ni
            if best_ni is not None:
                matches.append((fi, best_ni, best_d))
        matches.sort(key=lambda m: m[2])
        used_f, used_n, final = set(), set(), []
        for fi, ni, d in matches:
            if fi in used_f or ni in used_n:
                continue
            used_f.add(fi)
            used_n.add(ni)
            final.append((fi, ni, d))
        return len(final), sum(d for _, _, d in final), final

    ang_best_n, ang_best_total, ang_best_raw = 0, float('inf'), []
    ang_best_angle = 0.0
    for angle_deg in np.arange(-55.0, 55.5, 1.0):
        n_p, tot, pr = _score_angle(float(angle_deg))
        if n_p > ang_best_n or (n_p == ang_best_n and tot < ang_best_total):
            ang_best_n, ang_best_total, ang_best_raw = n_p, tot, pr
            ang_best_angle = float(angle_deg)

    # --- Choose between strategies ---
    if len(col_pairs) > 0:
        chosen_pairs_ci = col_pairs
        report_angle = angle_est
    else:
        chosen_pairs_ci = [(fi, ni) for fi, ni, _ in ang_best_raw]
        report_angle = ang_best_angle

    pairs = []
    used_far_set = set()
    used_near_set = set()
    for fi, ni in chosen_pairs_ci:
        pairs.append((all_pts[fi], all_pts[ni]))
        used_far_set.add(fi)
        used_near_set.add(ni)

    unpaired_far = [all_pts[fi] for fi in keep_far if fi not in used_far_set]
    unpaired_near = [all_pts[ni] for ni in keep_near if ni not in used_near_set]

    return pairs, unpaired_far, unpaired_near, report_angle, 0.0


def point_line_perpendicular_distance(point, line_pt1, line_pt2):
    """Perpendicular distance from a point to the line through line_pt1 and line_pt2."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(line_pt1, dtype=float)
    b = np.asarray(line_pt2, dtype=float)
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-6:
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))
    # Perpendicular distance via cross-product magnitude
    cross = abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
    return float(cross / np.sqrt(L2))


def find_sideline_on_yard_line(near_hash, far_hash, sideline_pxs,
                                max_perp_distance=12, min_above_px=50,
                                max_above_px=400):
    """Find a sideline detection that lies on the line extended from near→far hash.

    Searches only above the far hash (smaller y), within min/max range.
    Returns (idx, perp_dist) of closest-on-line detection, or (None, inf).
    """
    if len(sideline_pxs) == 0:
        return None, float('inf')

    # Direction vector (near → far). We go further in the same direction.
    dx = far_hash[0] - near_hash[0]
    dy = far_hash[1] - near_hash[1]   # negative (far is above near)
    if abs(dy) < 1e-6:
        return None, float('inf')

    best_idx = None
    best_perp = float('inf')
    for i, s in enumerate(sideline_pxs):
        # Only consider detections above the far hash
        if s[1] >= far_hash[1] - min_above_px:
            continue
        if s[1] < far_hash[1] - max_above_px:
            continue
        perp = point_line_perpendicular_distance(s, near_hash, far_hash)
        if perp > max_perp_distance:
            continue
        if perp < best_perp:
            best_perp = perp
            best_idx = i
    return best_idx, best_perp


def _group_reference_point(yl):
    """The keypoint that defines this group's position for grid fitting.

    Hashes preferred over sidelines (more reliable localization).
    """
    if yl.get('far_hash') is not None:
        return yl['far_hash']
    if yl.get('near_hash') is not None:
        return yl['near_hash']
    if yl.get('sideline') is not None:
        return yl['sideline']
    return None


def assign_grid_positions(yard_lines, paired_groups_for_slope=None, **kwargs):
    """Assign grid positions based on column-coordinate ordering.

    Each keypoint gets a "column coord": its x-position projected along a
    local yard-line direction to the hash-row y-level. For paired groups,
    that's just their own x (they ARE at the hash-row level). For unpaired
    keypoints (singletons at different y), we use the slope of the NEAREST
    paired group to project them down to the hash row.

    This handles perspective tilt that varies across the frame — each
    edge of the frame can have a different yard-line slope.

    If `paired_groups_for_slope` is None or empty, falls back to raw x.

    Unit spacing is the median consecutive column-coord diff between PAIRED
    groups. Each group gets tagged with:
      - 'grid_pos': integer grid index (0 = leftmost)
      - 'grid_fit_residual': column-coord distance to its ideal grid position
      - 'grid_fit_ok': True if residual is within 10% of unit spacing
    """
    if not yard_lines:
        return

    # Build per-pair (x_at_hash_row, slope_dx_dy) anchors for local tilt correction.
    # For a paired group, x_at_hash_row is its average (far+near)/2; slope from pair vector.
    anchors = []  # list of (x_anchor, y_anchor, slope)
    for yl in yard_lines:
        if yl.get('singleton'):
            continue
        fh, nh = yl.get('far_hash'), yl.get('near_hash')
        if fh is None or nh is None:
            continue
        dy = fh[1] - nh[1]
        if abs(dy) < 1e-6:
            continue
        dx = fh[0] - nh[0]
        slope = dx / dy
        x_anchor = (fh[0] + nh[0]) / 2.0
        y_anchor = (fh[1] + nh[1]) / 2.0
        anchors.append((x_anchor, y_anchor, slope))

    ref_y = float(np.mean([a[1] for a in anchors])) if anchors else None

    def column_coord(pt):
        """Project pt to the reference row using the slope of the nearest
        paired anchor (by x). If no anchors, just use raw x."""
        if not anchors or ref_y is None:
            return float(pt[0])
        # Use the anchor whose x_anchor is closest to pt[0]
        best = min(anchors, key=lambda a: abs(a[0] - pt[0]))
        slope = best[2]
        return float(pt[0] - slope * (pt[1] - ref_y))

    ref_coords = []
    for yl in yard_lines:
        pt = _group_reference_point(yl)
        if pt is None:
            ref_coords.append(None)
            continue
        ref_coords.append(column_coord(pt))

    # Separate paired and singleton indices (by original yard_lines index)
    paired_idx = [i for i, yl in enumerate(yard_lines)
                  if not yl.get('singleton') and ref_coords[i] is not None]
    singleton_idx = [i for i, yl in enumerate(yard_lines)
                     if yl.get('singleton') and ref_coords[i] is not None]

    # Degenerate cases
    if not paired_idx:
        if not singleton_idx:
            return
        # No paired anchors — can't establish a grid, just sort singletons
        sort_order = sorted(singleton_idx, key=lambda i: ref_coords[i])
        for k, i in enumerate(sort_order):
            yard_lines[i]['grid_pos'] = k
            yard_lines[i]['grid_fit_residual'] = 0.0
            yard_lines[i]['grid_fit_ok'] = False
        return

    # Sort paired groups and compute unit spacing
    paired_sorted = sorted(paired_idx, key=lambda i: ref_coords[i])
    paired_xs = [ref_coords[i] for i in paired_sorted]

    if len(paired_xs) == 1:
        # Single paired group: gives grid_pos 0 but no unit; can't validate singletons
        yard_lines[paired_sorted[0]]['grid_pos'] = 0
        yard_lines[paired_sorted[0]]['grid_fit_residual'] = 0.0
        yard_lines[paired_sorted[0]]['grid_fit_ok'] = True
        for i in singleton_idx:
            yard_lines[i]['grid_pos'] = None
            yard_lines[i]['grid_fit_residual'] = float('inf')
            yard_lines[i]['grid_fit_ok'] = False
        return

    paired_diffs = [paired_xs[k+1] - paired_xs[k]
                    for k in range(len(paired_xs) - 1)]

    # Estimate the 5-yd unit robustly for bimodal spacings like [10yd, 10yd,
    # 5yd, 5yd]: use the minimum paired diff as the initial unit, derive integer
    # slot-counts per pair gap, then refit unit via weighted least-squares
    # through the origin (minimize sum (d_i - unit * count_i)^2 → unit =
    # sum(d_i * count_i) / sum(count_i^2)). Median-of-diffs fails here because
    # it lands halfway between the 5yd and 10yd modes.
    unit0 = float(min(paired_diffs))
    counts = [max(1, int(round(d / unit0))) for d in paired_diffs]
    denom = float(sum(c * c for c in counts))
    if denom > 0:
        unit = float(sum(d * c for d, c in zip(paired_diffs, counts)) / denom)
    else:
        unit = unit0
    tolerance = 0.10 * unit
    x0 = paired_xs[0]

    # Assign paired grid positions using the integer slot counts derived above.
    # Paired groups ARE the grid anchors, so they're always grid_fit_ok=True.
    paired_grids = [0]
    for c in counts:
        paired_grids.append(paired_grids[-1] + c)
    for k, idx in enumerate(paired_sorted):
        gp = paired_grids[k]
        yard_lines[idx]['grid_pos'] = gp
        ideal = x0 + gp * unit
        residual = abs(paired_xs[k] - ideal)
        yard_lines[idx]['grid_fit_residual'] = float(residual)
        yard_lines[idx]['grid_fit_ok'] = True  # paired groups are anchors

    # Assign singleton grid positions: snap each to nearest grid slot.
    # If multiple singletons of the same TYPE land on the same (grid_pos, row)
    # slot, keep only the best fit — the others are probably HRNet firing
    # multiple times on the same painted feature.
    # A "slot" is (grid_pos, row_key), where row_key distinguishes far-hash,
    # near-hash, and sideline (sideline row inferred by image-half elsewhere).
    singleton_tolerance = 0.15 * unit  # slightly looser than paired tolerance

    def row_key_for_singleton(yl):
        if yl.get('far_hash') is not None:
            return 'far_hash'
        if yl.get('near_hash') is not None:
            return 'near_hash'
        if yl.get('sideline') is not None:
            return 'sideline'
        return 'none'

    # First pass: compute proposed (grid_pos, residual) for each singleton
    proposals = {}  # (grid_pos, row_key) -> list of (residual, idx)
    for idx in singleton_idx:
        c = ref_coords[idx]
        gp = int(round((c - x0) / unit))
        ideal = x0 + gp * unit
        residual = abs(c - ideal)
        yard_lines[idx]['grid_pos'] = gp
        yard_lines[idx]['grid_fit_residual'] = float(residual)
        # Default: rejected unless it wins its slot
        yard_lines[idx]['grid_fit_ok'] = False
        rk = row_key_for_singleton(yard_lines[idx])
        proposals.setdefault((gp, rk), []).append((residual, idx))

    # Second pass: for each slot, pick the best-fit candidate within tolerance
    for (gp, rk), entries in proposals.items():
        # Reject if this grid position is already occupied by a paired group
        # using the same row (hashes do this because paired groups have both
        # far and near hashes; sidelines are paired only if attached).
        paired_occupied = False
        for pidx in paired_sorted:
            if yard_lines[pidx]['grid_pos'] != gp:
                continue
            if rk == 'far_hash' and yard_lines[pidx].get('far_hash') is not None:
                paired_occupied = True
                break
            if rk == 'near_hash' and yard_lines[pidx].get('near_hash') is not None:
                paired_occupied = True
                break
            if rk == 'sideline' and yard_lines[pidx].get('sideline') is not None:
                paired_occupied = True
                break
        if paired_occupied:
            continue
        entries.sort()
        best_res, best_idx = entries[0]
        if best_res <= singleton_tolerance:
            yard_lines[best_idx]['grid_fit_ok'] = True


def build_yard_lines(hash_pxs, hash_confs, sideline_pxs, sideline_confs,
                     sideline_search_radius=35):
    """Run the full grouping pipeline."""
    # Split hashes into rows
    far_hashes, near_hashes = split_hash_rows(hash_pxs)
    print(f"Split hashes: {len(far_hashes)} far, {len(near_hashes)} near")

    # Pair up via row-clustering (returns points, not indices)
    pairs, unpaired_far, unpaired_near, angle_deg, angle_std = pair_hashes(
        far_hashes, near_hashes,
    )
    if angle_deg is not None:
        print(f"Estimated yard-line tilt: {angle_deg:.1f}° (std {angle_std:.1f}°)")
    print(f"Pairs: {len(pairs)}, unpaired far: {len(unpaired_far)}, unpaired near: {len(unpaired_near)}")

    yard_lines = []
    used_sideline = set()

    # Paired yard lines
    for fh, nh in pairs:
        fh = np.asarray(fh)
        nh = np.asarray(nh)
        sl_idx, sl_perp = find_sideline_on_yard_line(
            nh, fh, sideline_pxs, max_perp_distance=12,
        )
        sideline_pt = None
        sideline_conf = None
        sideline_perp = None
        if sl_idx is not None and sl_idx not in used_sideline:
            used_sideline.add(sl_idx)
            sideline_pt = sideline_pxs[sl_idx].tolist()
            sideline_conf = float(sideline_confs[sl_idx])
            sideline_perp = sl_perp

        yard_lines.append({
            'far_hash': fh.tolist(),
            'near_hash': nh.tolist(),
            'sideline': sideline_pt,
            'sideline_conf': sideline_conf,
            'sideline_perp_dist': sideline_perp,
            'singleton': False,
        })

    # Singleton far hashes
    for fh in unpaired_far:
        yard_lines.append({
            'far_hash': np.asarray(fh).tolist(),
            'near_hash': None,
            'sideline': None,
            'sideline_conf': None,
            'sideline_perp_dist': None,
            'singleton': True,
        })

    # Singleton near hashes
    for nh in unpaired_near:
        yard_lines.append({
            'far_hash': None,
            'near_hash': np.asarray(nh).tolist(),
            'sideline': None,
            'sideline_conf': None,
            'sideline_perp_dist': None,
            'singleton': True,
        })

    # PCA on all hashes — used for row-coord (sideline row matching).
    mean_point, row_dir, yardline_dir = compute_hash_pca(hash_pxs)

    # Yard-line tilt from actual near→far pair vectors (NOT from PCA on all
    # hashes, which can't see the tilt since all hashes live in a thin band).
    tilt_slope = yardline_tilt_slope_from_pairs(pairs, far_hashes, near_hashes)

    # Note: singleton sidelines (detections not attached to a paired hash
    # group) are intentionally NOT emitted. HRNet tends to fire multiple
    # detections along the sideline row, making reliable yard-line assignment
    # impossible without a confirming hash pair.

    assign_grid_positions(yard_lines)
    return yard_lines, used_sideline



# ─── From test_grid_solver_camera.py ─────────────────────────────────

def build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs):
    """Run the grid solver: hash pairing + sideline matching + grid positions.

    Returns list of yard-line dicts with 'grid_pos', 'far_hash', 'near_hash',
    'sideline', 'sideline_perp_dist', 'singleton'.
    """
    far_hashes, near_hashes = split_hash_rows(hash_pxs)
    pairs, unpaired_far, unpaired_near, angle_deg, angle_std = pair_hashes(
        far_hashes, near_hashes,
    )

    groups = []
    used_sideline = set()
    for fh, nh in pairs:
        fh = np.asarray(fh)
        nh = np.asarray(nh)
        sl_idx, sl_perp = find_sideline_on_yard_line(
            nh, fh, sideline_pxs, max_perp_distance=12,
        )
        sideline_pt = None
        sideline_conf = None
        if sl_idx is not None and sl_idx not in used_sideline:
            used_sideline.add(sl_idx)
            sideline_pt = sideline_pxs[sl_idx].tolist()
            sideline_conf = float(sideline_confs[sl_idx])
        groups.append({
            'far_hash': fh.tolist(),
            'near_hash': nh.tolist(),
            'sideline': sideline_pt,
            'sideline_conf': sideline_conf,
            'singleton': False,
        })

    # Singleton hashes
    for fh in unpaired_far:
        groups.append({
            'far_hash': np.asarray(fh).tolist(),
            'near_hash': None,
            'sideline': None,
            'sideline_conf': None,
            'singleton': True,
        })
    for nh in unpaired_near:
        groups.append({
            'far_hash': None,
            'near_hash': np.asarray(nh).tolist(),
            'sideline': None,
            'sideline_conf': None,
            'singleton': True,
        })

    # Note: singleton sidelines intentionally not emitted — HRNet sideline
    # detections are too clustered to reliably assign to yard-line columns
    # without a paired-hash confirmation.
    assign_grid_positions(groups)
    return groups, angle_deg


def line_fit_residuals(points):
    """Fit a line through points via SVD, return signed perpendicular distances."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts))
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return centered @ normal


def calibrate_distortion_from_lines(line_point_sets, frame_shape,
                                     focal_length_guess=None):
    """Estimate radial distortion coefficients (k1, k2) via plumb-line fit.

    Each entry in line_point_sets is a (N, 2) array of pixel points that SHOULD
    be collinear in undistorted space (e.g. a sideline, a hash row). We pick
    (k1, k2) that minimize the total sum of squared perpendicular distances
    of each point to the best-fit line through its group after undistortion.

    focal_length_guess sets the normalization scale. It doesn't have to be the
    true focal length — we just need a consistent scale for the distortion
    model. Default: max(image dimensions).
    """
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    if focal_length_guess is None:
        focal_length_guess = float(max(w, h))

    usable_sets = [np.asarray(pts, dtype=np.float64) for pts in line_point_sets
                   if len(pts) >= 3]
    if not usable_sets:
        return 0.0, 0.0

    def cost(params):
        k1, k2 = params
        intr = CameraIntrinsics(fx=focal_length_guess, fy=focal_length_guess,
                                 cx=cx, cy=cy, k1=k1, k2=k2)
        total = 0.0
        for pts in usable_sets:
            u = undistort_points(pts, intr)
            r = line_fit_residuals(u)
            total += float(np.sum(r ** 2))
        return total

    # Nelder-Mead over (k1, k2). Start at 0 (no distortion).
    result = minimize(
        cost, x0=np.array([0.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 400},
    )
    return float(result.x[0]), float(result.x[1])


def groups_to_correspondences(groups, base_ngs_x, frame_shape=None):
    """Convert yard-line groups to (pixel, field) correspondence pairs.

    base_ngs_x = NGS x-coordinate of grid_pos = 0 (the leftmost detected line).

    Keeps:
      - All paired groups (both hashes + any matched sideline).
      - Singleton hashes that sit on the grid within tolerance (`grid_fit_ok`).
      - Singleton sidelines that sit on the grid within tolerance AND align
        with the row of a paired sideline (established in build_yard_line_groups).
        Near vs far sideline classified by image half (frame_shape required).

    Drops singletons whose position doesn't fit the established grid spacing.
    """
    pixel_pts = []
    field_pts = []
    labels = []

    img_mid_y = frame_shape[0] / 2.0 if frame_shape is not None else None

    for yl in groups:
        gp = yl.get('grid_pos')
        if gp is None:
            continue
        field_x = base_ngs_x + gp * 5

        if yl.get('singleton'):
            if not yl.get('grid_fit_ok', False):
                continue
            if yl['far_hash'] is not None:
                pixel_pts.append(yl['far_hash'])
                field_pts.append([field_x, HASH_Y_FAR])
                labels.append(f'g{gp}_far_s')
            elif yl['near_hash'] is not None:
                pixel_pts.append(yl['near_hash'])
                field_pts.append([field_x, HASH_Y_NEAR])
                labels.append(f'g{gp}_near_s')
            elif yl['sideline'] is not None and img_mid_y is not None:
                sl = yl['sideline']
                # Top of frame = far sideline; bottom = near sideline
                field_y = FIELD_WIDTH if sl[1] < img_mid_y else 0.0
                pixel_pts.append(sl)
                field_pts.append([field_x, field_y])
                tag = 'side_s_far' if field_y > 0 else 'side_s_near'
                labels.append(f'g{gp}_{tag}')
            continue

        # Paired group: both hashes, plus any matched sideline
        if yl['near_hash'] is not None:
            pixel_pts.append(yl['near_hash'])
            field_pts.append([field_x, HASH_Y_NEAR])
            labels.append(f'g{gp}_near')
        if yl['far_hash'] is not None:
            pixel_pts.append(yl['far_hash'])
            field_pts.append([field_x, HASH_Y_FAR])
            labels.append(f'g{gp}_far')
        if yl['sideline'] is not None:
            pixel_pts.append(yl['sideline'])
            field_pts.append([field_x, FIELD_WIDTH])
            labels.append(f'g{gp}_side')

    return np.array(pixel_pts), np.array(field_pts), labels
