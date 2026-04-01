"""
Hash mark detection and hash-based rectification for sideline-view All-22 frames.

Pipeline:
  1. Angle-first edge removal (parallel gradient filter within band)
  2. Density filter on cleaned edges
  3. Band mask around detected yard lines
  4. Paired-dot detection with perspective-corrected offset for tilted lines
  5. Hash classification (far/near) using confident pairs
  6. Homography from hash intersection points
"""

import numpy as np
import cv2
from dataclasses import dataclass
from .yard_lines import YardLineResult


# --- Tuning constants ---
DENSITY_THRESH = 0.06       # max edge density to keep (lower = more aggressive)
BAND_WIDTH = 12             # half-width of band around each yard line (px)
ANGLE_TOL = np.radians(25)  # tolerance for gradient-vs-line angle match
CLUSTER_GAP = 4             # max step gap within a cluster
CLUSTER_MIN = 2             # min cluster size to consider
CLUSTER_MATCH = 5           # max adjusted distance between L/R cluster centers
INNER_DIST = 6              # inner flank distance from line center (px)
OUTER_DIST = 14             # outer flank distance from line center (px)
OFFSET_SCALE = 0.64         # perspective correction for expected L/R offset
T_TOL = 0.15                # max deviation from median t for classification
T_MIN = 0.15                # min t-value (reject frame-edge noise)
T_MAX = 0.85                # max t-value (reject frame-edge noise)
HASH_SPACING_YD = 18.5 / 3  # hash mark spacing in yards (18'6" = 6.167 yd)


@dataclass
class HashResult:
    """Result of hash mark detection and classification."""
    hashes: dict[int, dict]  # line_index -> {'far': t, 'near': t}
    far_median: float        # median t-value for far hash row
    near_median: float       # median t-value for near hash row
    pair_gap: float          # near_median - far_median
    n_confident_pairs: int   # number of lines with both hashes detected as pairs
    hash_canny: np.ndarray   # filtered edge mask (for debug)


@dataclass
class HashRectificationResult:
    """Result of hash-based rectification."""
    warped: np.ndarray       # rectified image
    H: np.ndarray            # 3x3 homography (includes translation)
    src_pts: np.ndarray      # source points used
    dst_pts: np.ndarray      # destination points used
    n_far: int               # number of far hash correspondences
    n_near: int              # number of near hash correspondences


def _cluster_steps(steps: set[int]) -> list[list[int]]:
    """Group step indices into clusters separated by gaps > CLUSTER_GAP."""
    if not steps:
        return []
    sorted_s = sorted(steps)
    groups = [[sorted_s[0]]]
    for k in range(1, len(sorted_s)):
        if sorted_s[k] - sorted_s[k - 1] <= CLUSTER_GAP:
            groups[-1].append(sorted_s[k])
        else:
            groups.append([sorted_s[k]])
    return groups


def _build_band_and_removal(
    lines: list, h: int, w: int, edge_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build band mask and angle removal mask (vectorized).

    Uses int() truncation (via .astype(int)) for pixel coordinates to
    match the original pixel-walk implementation exactly.
    """
    band_mask = np.zeros((h, w), dtype=np.uint8)
    angle_removal = np.zeros((h, w), dtype=bool)

    # Precompute offset grid: signs × distances along the normal
    signs = np.array([-1, 1], dtype=np.float64)
    dists = np.arange(0, BAND_WIDTH + 1, dtype=np.float64)
    # shape: (2 * (BAND_WIDTH+1),) — all sign*dist combos
    sd = (signs[:, None] * dists[None, :]).ravel()

    for line in lines:
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        length = np.sqrt(ldx**2 + ldy**2)
        nx = -ldy / length
        ny = ldx / length
        line_angle = np.arctan2(ldy, ldx)

        n_steps = int(length * 2)
        t_arr = np.arange(n_steps + 1, dtype=np.float64) / n_steps
        cx = x_top + t_arr * ldx  # (n_steps+1,)
        cy = y_top + t_arr * ldy

        # Broadcast: center points × offsets → all band pixels
        # px[i, j] = int(cx[i] + sd[j] * nx)
        px = (cx[:, None] + sd[None, :] * nx).astype(int)
        py = (cy[:, None] + sd[None, :] * ny).astype(int)

        # Mask valid pixels
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px_v = px[valid]
        py_v = py[valid]

        band_mask[py_v, px_v] = 255

        # Angle removal within this line's band
        angle_diff = edge_angle[py_v, px_v] - line_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        match = (
            (np.abs(angle_diff) < ANGLE_TOL)
            | (np.abs(angle_diff - np.pi) < ANGLE_TOL)
            | (np.abs(angle_diff + np.pi) < ANGLE_TOL)
        )
        angle_removal[py_v[match], px_v[match]] = True

    return band_mask, angle_removal


def detect_hashes(
    frame: np.ndarray,
    result: YardLineResult,
    gray: np.ndarray | None = None,
    canny: np.ndarray | None = None,
) -> HashResult | None:
    """Detect hash marks along detected yard lines.

    Pipeline: YL edge removal -> density filter -> band mask -> paired-dot search.

    Uses angle-first edge removal, density filtering, and cluster-based
    paired-dot detection with perspective-corrected offset for tilted lines.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    result : YardLineResult
        Output from detect_yard_lines().
    gray : np.ndarray, optional
        Precomputed grayscale image (avoids redundant cvtColor).
    canny : np.ndarray, optional
        Precomputed Canny edges (avoids redundant Canny).

    Returns None if no hash marks are found.
    """
    h, w = frame.shape[:2]

    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if canny is None:
        canny = cv2.Canny(gray, 100, 200)

    # Sobel gradient, rotated by 90° to get edge direction (not gradient
    # direction).  This way the parallel check below removes edges whose
    # orientation matches the yard line — i.e. the yard line edges themselves.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_angle = np.arctan2(sobely, sobelx) + np.pi / 2

    # --- Step 1: Band mask + angle removal (pixel-walk) ---
    band_mask, angle_removal = _build_band_and_removal(
        result.lines, h, w, edge_angle,
    )

    # --- Step 2: Remove yard-line-parallel edges ---
    canny_no_yl = canny.copy()
    canny_no_yl[angle_removal] = 0

    # --- Step 3: Density filter ---
    dens = cv2.boxFilter(
        canny_no_yl.astype(np.float32) / 255.0, -1, (31, 31),
    )
    sparse = (dens < DENSITY_THRESH).astype(np.uint8) * 255
    clean = cv2.bitwise_and(canny_no_yl, sparse)

    # --- Step 4: Restrict to band ---
    hash_canny = cv2.bitwise_and(clean, band_mask)

    # --- Step 5: Hash walk along each yard line ---
    d_mid = (INNER_DIST + OUTER_DIST) / 2
    all_line_hashes: dict[int, list[float]] = {}

    for i, line in enumerate(result.lines):
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        length = np.sqrt(ldx**2 + ldy**2)
        nx_l = -ldy / length
        ny_l = ldx / length
        n_steps = int(length)
        if n_steps < 10:
            continue

        # Expected step offset for tilted lines (perspective-corrected)
        expected_offset = (
            OFFSET_SCALE * 2 * d_mid * ny_l * n_steps / ldy
            if abs(ldy) > 1e-6 else 0.0
        )

        # Vectorized hash walk: check all steps × distances at once
        steps = np.arange(n_steps + 1)
        t_arr = steps / n_steps
        cx = x_top + t_arr * ldx  # (n_steps+1,)
        cy = y_top + t_arr * ldy
        flank_dists = np.arange(INNER_DIST, OUTER_DIST + 1, dtype=np.float64)

        # Left side: center - dist * normal, shape (n_steps+1, n_dists)
        lpx = (cx[:, None] - flank_dists[None, :] * nx_l).astype(int)
        lpy = (cy[:, None] - flank_dists[None, :] * ny_l).astype(int)
        # Right side
        rpx = (cx[:, None] + flank_dists[None, :] * nx_l).astype(int)
        rpy = (cy[:, None] + flank_dists[None, :] * ny_l).astype(int)

        # Clamp to valid range, look up hash_canny, check any hit per step
        lpx_c = np.clip(lpx, 0, w - 1)
        lpy_c = np.clip(lpy, 0, h - 1)
        l_valid = (lpx >= 0) & (lpx < w) & (lpy >= 0) & (lpy < h)
        l_hit = (hash_canny[lpy_c, lpx_c] > 0) & l_valid
        left_steps = set(steps[l_hit.any(axis=1)].tolist())

        rpx_c = np.clip(rpx, 0, w - 1)
        rpy_c = np.clip(rpy, 0, h - 1)
        r_valid = (rpx >= 0) & (rpx < w) & (rpy >= 0) & (rpy < h)
        r_hit = (hash_canny[rpy_c, rpx_c] > 0) & r_valid
        right_steps = set(steps[r_hit.any(axis=1)].tolist())

        # Cluster left and right hits independently
        l_clusters = [
            g for g in _cluster_steps(left_steps)
            if len(g) >= CLUSTER_MIN
        ]
        r_clusters = [
            g for g in _cluster_steps(right_steps)
            if len(g) >= CLUSTER_MIN
        ]
        # Use midrange for cluster center (unbiased by asymmetric dot sizes)
        l_centers = [(min(g) + max(g)) / 2.0 for g in l_clusters]
        r_centers = [(min(g) + max(g)) / 2.0 for g in r_clusters]

        # Match left/right clusters (subtract expected offset, check tolerance)
        hash_ts: list[float] = []
        used_r: set[int] = set()
        for lc in l_centers:
            adjusted_lc = lc - expected_offset
            best_ri = None
            best_dist = CLUSTER_MATCH + 1
            for ri, rc in enumerate(r_centers):
                if ri in used_r:
                    continue
                d = abs(adjusted_lc - rc)
                if d < best_dist:
                    best_dist = d
                    best_ri = ri
            if best_ri is not None and best_dist <= CLUSTER_MATCH:
                used_r.add(best_ri)
                hash_ts.append((lc + r_centers[best_ri]) / 2 / n_steps)

        # Filter out edge-of-frame noise
        hash_ts = [t for t in hash_ts if T_MIN < t < T_MAX]
        if hash_ts:
            all_line_hashes[i] = hash_ts

    if not all_line_hashes:
        return None

    # --- Classification: far vs near ---
    confident_pairs: dict[int, tuple[float, float]] = {}
    far_ts: list[float] = []
    near_ts: list[float] = []

    for li, ts in all_line_hashes.items():
        if len(ts) == 2:
            ts_sorted = sorted(ts)
            confident_pairs[li] = (ts_sorted[0], ts_sorted[1])
            far_ts.append(ts_sorted[0])
            near_ts.append(ts_sorted[1])

    if far_ts and near_ts:
        far_median = float(np.median(far_ts))
        near_median = float(np.median(near_ts))
    else:
        # Fallback: cluster all t-values to find the two hash rows.
        all_ts = sorted(
            t for ts in all_line_hashes.values() for t in ts
            if T_MIN < t < T_MAX
        )
        if len(all_ts) < 2:
            return None

        # Cluster t-values (gap > 0.05 in t-space)
        t_clusters: list[list[float]] = [[all_ts[0]]]
        for t_val in all_ts[1:]:
            if t_val - t_clusters[-1][-1] < 0.05:
                t_clusters[-1].append(t_val)
            else:
                t_clusters.append([t_val])

        if len(t_clusters) < 2:
            return None
        t_clusters.sort(key=len, reverse=True)
        top2 = sorted(t_clusters[:2], key=lambda c: np.mean(c))
        far_median = float(np.median(top2[0]))
        near_median = float(np.median(top2[1]))

    mid_t = (far_median + near_median) / 2
    pair_gap = near_median - far_median

    # Classify each hash as far or near, reject outliers
    final_hashes: dict[int, dict] = {}
    for li, ts in all_line_hashes.items():
        entry: dict[str, float] = {}
        far_candidates = [t for t in ts if t < mid_t]
        near_candidates = [t for t in ts if t >= mid_t]

        if far_candidates:
            best_far = min(far_candidates, key=lambda t: abs(t - far_median))
            if abs(best_far - far_median) < T_TOL:
                entry["far"] = best_far
        if near_candidates:
            best_near = min(
                near_candidates, key=lambda t: abs(t - near_median),
            )
            if abs(best_near - near_median) < T_TOL:
                entry["near"] = best_near
        if entry:
            final_hashes[li] = entry

    # Interpolate missing using pair_gap
    for entry in final_hashes.values():
        if "far" not in entry and "near" in entry:
            entry["far"] = entry["near"] - pair_gap
        elif "near" not in entry and "far" in entry:
            entry["near"] = entry["far"] + pair_gap

    if not final_hashes:
        return None

    return HashResult(
        hashes=final_hashes,
        far_median=far_median,
        near_median=near_median,
        pair_gap=pair_gap,
        n_confident_pairs=len(confident_pairs),
        hash_canny=hash_canny,
    )


def rectify_with_hashes(
    frame: np.ndarray,
    yl_result: YardLineResult,
    hash_result: HashResult,
    scale: float = 20.0,
    margin_x: float = 100.0,
    margin_y: float = 100.0,
) -> HashRectificationResult | None:
    """Rectify using hash mark intersection points.

    Maps each detected hash-yard line intersection to its target position
    in a coordinate system where:
      - x is determined by yard line grid position (5-yard spacing)
      - y is determined by far/near hash row (HASH_SPACING_YD apart)
    """
    h, w = frame.shape[:2]
    hashes = hash_result.hashes

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []

    for li, entry in hashes.items():
        line = yl_result.lines[li]
        gpos = yl_result.grid[li]
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top
        target_x = margin_x + gpos * 5 * scale

        if "far" in entry:
            t = entry["far"]
            src_pts.append([x_top + t * ldx, y_top + t * ldy])
            dst_pts.append([target_x, margin_y])
        if "near" in entry:
            t = entry["near"]
            src_pts.append([x_top + t * ldx, y_top + t * ldy])
            dst_pts.append([target_x, margin_y + HASH_SPACING_YD * scale])

    if len(src_pts) < 4:
        return None

    src = np.array(src_pts, np.float64)
    dst = np.array(dst_pts, np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None

    # Compute bounding box of warped corners
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    wc = (H @ corners_h.T).T
    wc = wc[:, :2] / wc[:, 2:3]

    x_min = int(np.floor(wc[:, 0].min()))
    x_max = int(np.ceil(wc[:, 0].max()))
    y_min = int(np.floor(wc[:, 1].min()))
    y_max = int(np.ceil(wc[:, 1].max()))

    tx = float(-x_min)
    ty = float(-y_min)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_full = T @ H

    out_w = x_max - x_min
    out_h = y_max - y_min
    warped = cv2.warpPerspective(frame, H_full, (out_w, out_h))

    n_far = sum(1 for e in hashes.values() if "far" in e)
    n_near = sum(1 for e in hashes.values() if "near" in e)

    return HashRectificationResult(
        warped=warped,
        H=H_full,
        src_pts=src,
        dst_pts=dst,
        n_far=n_far,
        n_near=n_near,
    )
