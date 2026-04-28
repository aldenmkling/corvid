#!/usr/bin/env python3
"""Rebuild pipeline — full-clip diagnostic videos for play_046 (no H yet).

Generates two side-by-side videos that should make every choice visible:

  1. step_full_masks.mp4
       Original (distorted) frame + CC-grouped UNet masks. Each yardline
       group gets its tracked g-color; sideline groups in yellow. Lets us
       eyeball UNet + CC + cross-frame tracker behavior.

  2. step_full_fits.mp4
       Undistorted frame + linear fits (yardlines + sideline) + hash
       detections classified near/far + sideline×yardline intersections.
       Every yardline has a `g=+X` label; every keypoint on that yardline
       wears the yardline's color and an individual label
       (`near_hash@g+1`, `sl×g+2`, etc.).

Frame-0 k1 cached for the whole clip; cross-frame yardline tracker
maintains g identity even as the camera pans.
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from src.homography.grid_solver_v2 import (
    run_unet, group_yardline_pixels_cc, run_hash_w18,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
HASH = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
OUT_MASKS = os.path.join(PROJECT_ROOT, "output/rebuild/step_full_masks.mp4")
OUT_FITS = os.path.join(PROJECT_ROOT, "output/rebuild/step_full_fits.mp4")


def group_sideline_pixels_cc(
    side_mask: np.ndarray,
    min_pixels_per_component: int = 40,
    min_aspect_ratio: float = 3.0,
    rho_tol_px: float = 25.0,
    theta_tol_rad: float = 0.08,
    max_lines: int = 2,
    min_pixels_per_line: int = 100,
):
    """CC + collinearity merge for sidelines, mirroring the yardline path.
    Returns up to `max_lines` strongest clusters by pixel count."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        side_mask.astype(np.uint8), connectivity=8,
    )
    comps = []
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_pixels_per_component:
            continue
        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = pts.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(pts - center, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6 or S[0] / S[1] < min_aspect_ratio:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        if theta < 0:
            theta += np.pi; rho = -rho
        comps.append({"pixels": pts, "rho": rho, "theta": theta, "n": len(pts)})

    if not comps:
        return []

    clusters = []
    for c in comps:
        placed = False
        for cl in clusters:
            d_rho = abs(c["rho"] - cl["rho"])
            d_theta = abs(c["theta"] - cl["theta"])
            d_theta = min(d_theta, np.pi - d_theta)
            if d_rho <= rho_tol_px and d_theta <= theta_tol_rad:
                cl["pixels"].append(c["pixels"])
                w_old = cl["n"]; w_new = c["n"]
                cl["n"] += c["n"]
                cl["rho"] = (cl["rho"] * w_old + c["rho"] * w_new) / cl["n"]
                cl["theta"] = (cl["theta"] * w_old + c["theta"] * w_new) / cl["n"]
                placed = True; break
        if not placed:
            clusters.append({"pixels": [c["pixels"]],
                              "rho": c["rho"], "theta": c["theta"],
                              "n": c["n"]})

    clusters = [cl for cl in clusters if cl["n"] >= min_pixels_per_line]
    clusters.sort(key=lambda cl: cl["n"], reverse=True)
    clusters = clusters[:max_lines]
    return [SimpleNamespace(pixels=np.concatenate(cl["pixels"], axis=0))
            for cl in clusters]


# Override the import — rest of script uses group_sideline_pixels.
group_sideline_pixels = group_sideline_pixels_cc

YD_PER_GRID = 5.0
MAX_DIST_PX = 12.0
HASH_CONF = 0.45
EXTRAP_FRAC = 0.20
SL_MATCH_THRESH_PX = 80.0      # near/far sideline frame-to-frame line dist
OOB_TOL_PX = 5.0                # tolerance when checking yardline ∈ field
HASH_CLUSTER_PC2_MAX = 12.0     # max within-cluster PC2 std (px); bigger
                                 # than this means an outlier corrupted the
                                 # cluster, reject the split.
HASH_MEMORY_MAX_GAP = 3         # frames of absence before clearing memory
HASH_ROW_TOL_YD = 1.5            # for H_prev-based classification: max
                                 # distance from a hash's H_prev-projected
                                 # field y to the nearest hash row (yd) before
                                 # it's rejected as a false detection.

# Clip-prelude calibration: g=0 of frame 0's leftmost yardline corresponds
# to NGS x=20 for play_046. Yard lines exist between NGS x=10 (left goal)
# and x=110 (right goal), every 5 yards. Anything outside is the back-of-
# endzone painted line or noise → reject.
G0_NGS_X = 20.0
NGS_X_LEFT_GOAL = 10.0
NGS_X_RIGHT_GOAL = 110.0
G_MIN = int((NGS_X_LEFT_GOAL - G0_NGS_X) / YD_PER_GRID)        # = -2
G_MAX = int((NGS_X_RIGHT_GOAL - G0_NGS_X) / YD_PER_GRID)        # = +18
MAD_K = 3.0
MIN_CLUSTER_GAP_PX = 30.0
PCA_S0_S1_MIN = 1.5
TRACK_MATCH_FRAC = 0.40

# Stable per-g palette so colors persist across frames AND across the two
# videos. Indexed as palette[(g + OFFSET) % len].
PALETTE = [
    (60, 60, 255),    # red
    (0, 165, 255),    # orange
    (0, 255, 255),    # yellow
    (100, 255, 100),  # light green
    (0, 200, 255),    # amber
    (255, 100, 0),    # blue
    (255, 0, 200),    # magenta
    (255, 200, 100),  # cyan-blue
    (180, 105, 255),  # pink
    (200, 200, 0),    # teal
    (50, 200, 200),
    (255, 60, 60),
]
PALETTE_OFF = 8

SIDELINE_COLOR = (220, 220, 0)


def color_for_g(g: int):
    return PALETTE[(g + PALETTE_OFF) % len(PALETTE)]


# ── Helpers reused from earlier steps ────────────────────────────────────

def total_mse(line_pts, line_kinds, intr):
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


def fit_yardline_linear(pixels, intr):
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    ys, xs = pts_u[:, 1], pts_u[:, 0]
    b, a = np.polyfit(ys, xs, 1)
    return {"a": float(a), "b": float(b),
            "ymin": float(ys.min()), "ymax": float(ys.max())}


def fit_sideline_linear(pixels, intr):
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    xs, ys = pts_u[:, 0], pts_u[:, 1]
    b, a = np.polyfit(xs, ys, 1)
    return {"a": float(a), "b": float(b),
            "xmin": float(xs.min()), "xmax": float(xs.max())}


def otsu_split_1d(values):
    s = np.sort(values); N = len(s)
    best = -np.inf; bt = float(s.mean())
    for j in range(1, N):
        score = (j * (N - j) / N) * (s[:j].mean() - s[j:].mean()) ** 2
        if score > best:
            best = score; bt = 0.5 * (s[j-1] + s[j])
    return bt


def fit_row_line(pts):
    if len(pts) == 1:
        return 0.0, float(pts[0, 1])
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return float(m), float(c)


def perp_dist(pts, m, c):
    return np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(1.0 + m * m)


# ── Yardline tracker ─────────────────────────────────────────────────────

class YardlineTracker:
    """Tracks yardlines across frames by direct line-parameter similarity.

    Each tracked yardline has (a, b) where x = a + b·y in undistorted space.
    Frame-to-frame at 30fps the same yardline's (a, b) shifts only a few
    pixels' worth, while adjacent yardlines are 200+ px apart — so matching
    by line distance with a tight threshold is unambiguous.

    Distance metric: max(|Δx(y=0)|, |Δx(y=h)|) = the bigger of the two
    image-edge x displacements. Captures both intercept and slope change.

    For yardlines that fail to match (genuinely new ones entering the frame,
    or post-cut detections), grid-snap to integer g using anchor estimated
    from successfully-matched yardlines.
    """

    def __init__(self, g_min: int = G_MIN, g_max: int = G_MAX,
                 match_thresh_px: float = 50.0, frame_h: int = 720):
        self.last_fit = {}     # {g: (a, b)}
        self.unit_px = None
        self.anchor_x_g0 = None
        self.g_min = g_min
        self.g_max = g_max
        self.match_thresh_px = match_thresh_px
        self.frame_h = frame_h

    def _in_range(self, g: int) -> bool:
        return self.g_min <= g <= self.g_max

    def _line_distance(self, a1, b1, a2, b2):
        """Max image-edge displacement between two lines (x = a + b·y)."""
        d_top = abs(a1 - a2)
        d_bot = abs((a1 + (self.frame_h - 1) * b1) - (a2 + (self.frame_h - 1) * b2))
        return max(d_top, d_bot)

    def init_from(self, fits, cy):
        if len(fits) < 2:
            return None
        x_at_center = np.array([f["a"] + f["b"] * cy for f in fits])
        order = np.argsort(x_at_center)
        sorted_x = x_at_center[order]
        unit_px = float(np.median(np.diff(sorted_x)))
        anchor_x = float(sorted_x[0])
        raw = (sorted_x - anchor_x) / unit_px
        g_sorted = np.round(raw).astype(int)
        g_index = np.zeros(len(fits), dtype=int)
        for k_, orig in enumerate(order):
            g_index[orig] = int(g_sorted[k_])

        keep = np.array([self._in_range(int(g)) for g in g_index])
        fits_kept = [fits[i] for i in range(len(fits)) if keep[i]]
        g_index = g_index[keep]
        x_at_center = x_at_center[keep]
        if len(fits_kept) < 2:
            return None

        self.unit_px = unit_px
        self.anchor_x_g0 = anchor_x
        self.last_fit = {int(g): (float(fits_kept[i]["a"]), float(fits_kept[i]["b"]))
                         for i, g in enumerate(g_index)}
        return fits_kept, g_index, x_at_center

    def update(self, fits, cy):
        """Line-similarity matcher with grid-snap fallback.

        1. For each new fit, find closest tracked yardline by line distance
           = max(|Δx@y=0|, |Δx@y=h|). Greedy assign best-distance pairs
           first, accept only if distance ≤ match_thresh_px.
        2. Unmatched detections → grid-snap using unit_px and anchor
           re-estimated from successfully-matched fits.
        3. g-range gate. Reject anything outside [g_min, g_max].
        4. Update state. If NO yardlines matched (camera cut), the prior
           state is preserved untouched — the unmatched detections simply
           don't get assigned, rather than corrupting the tracker.
        """
        SENTINEL = self.g_min - 1000
        g_index = np.full(len(fits), SENTINEL, dtype=int)
        used_g = set()

        # Step 1: line-similarity matching.
        if self.last_fit and len(fits) > 0:
            pairs = []
            for i, f in enumerate(fits):
                for g, (a_prev, b_prev) in self.last_fit.items():
                    d = self._line_distance(f["a"], f["b"], a_prev, b_prev)
                    pairs.append((d, i, g))
            pairs.sort()
            for d, i, g in pairs:
                if d > self.match_thresh_px:
                    break
                if g_index[i] != SENTINEL or g in used_g:
                    continue
                g_index[i] = g
                used_g.add(g)

        n_matched = int((g_index != SENTINEL).sum())

        # Step 2: estimate unit_px + anchor from matched fits, snap unmatched.
        if n_matched >= 2:
            matched_idx = np.where(g_index != SENTINEL)[0]
            xs = np.array([fits[i]["a"] + fits[i]["b"] * cy for i in matched_idx])
            gs = np.array([g_index[i] for i in matched_idx])
            order = np.argsort(gs)
            xs_s = xs[order]; gs_s = gs[order]
            # unit_px from sorted-by-g differences (unambiguous because
            # we KNOW the integer indices).
            g_diffs = np.diff(gs_s)
            x_diffs = np.diff(xs_s)
            valid = g_diffs > 0
            if valid.any():
                unit_px = float(np.median(x_diffs[valid] / g_diffs[valid]))
            else:
                unit_px = self.unit_px or 220.0
            # anchor estimate: median of (x - g·unit_px) across matched.
            anchor_now = float(np.median(xs_s - gs_s * unit_px))

            for i in range(len(fits)):
                if g_index[i] != SENTINEL:
                    continue
                x_c = fits[i]["a"] + fits[i]["b"] * cy
                target = int(round((x_c - anchor_now) / unit_px))
                resid = abs(x_c - (anchor_now + target * unit_px))
                if not self._in_range(target) or target in used_g:
                    continue
                if resid > 0.5 * unit_px:
                    continue
                g_index[i] = target
                used_g.add(target)

            self.unit_px = unit_px
            self.anchor_x_g0 = anchor_now
        elif n_matched == 1:
            # Only one match → use stored unit_px, derive anchor from this match.
            i_m = int(np.where(g_index != SENTINEL)[0][0])
            x_m = fits[i_m]["a"] + fits[i_m]["b"] * cy
            unit_px = self.unit_px or 220.0
            anchor_now = x_m - g_index[i_m] * unit_px
            for i in range(len(fits)):
                if g_index[i] != SENTINEL:
                    continue
                x_c = fits[i]["a"] + fits[i]["b"] * cy
                target = int(round((x_c - anchor_now) / unit_px))
                resid = abs(x_c - (anchor_now + target * unit_px))
                if not self._in_range(target) or target in used_g:
                    continue
                if resid > 0.5 * unit_px:
                    continue
                g_index[i] = target
                used_g.add(target)
            self.anchor_x_g0 = anchor_now

        # Step 3+4: filter & update state.
        keep = np.array([g != SENTINEL for g in g_index], dtype=bool)
        n_rejected = int((~keep).sum()) if keep.size else 0
        fits_kept = [fits[i] for i in range(len(fits)) if keep[i]]
        g_index_kept = g_index[keep]
        x_at_center_kept = np.array(
            [f["a"] + f["b"] * cy for f in fits_kept]
        )

        # Only update last_fit for matched/snapped yardlines. If nothing
        # matched at all (camera cut), state is preserved so we can resume
        # tracking when the camera returns to a similar pose.
        for i, g in enumerate(g_index_kept):
            self.last_fit[int(g)] = (float(fits_kept[i]["a"]),
                                       float(fits_kept[i]["b"]))

        return fits_kept, g_index_kept, x_at_center_kept, n_rejected


# ── Sideline tracker ─────────────────────────────────────────────────────

class SidelineTracker:
    """Tracks at most 2 sidelines across frames as 'near' or 'far'.

    Sidelines fit as y = a + b·x in undistorted space. Frame-to-frame the
    same sideline's (a, b) shifts only a few px; the two sidelines are
    separated by ~half the field height in image space. Matching by line
    distance with a tight threshold + image-y classification handles the
    cross-frame identity.

    Convention: smaller mean image-y = 'far' (top of image, away from camera).
    """

    def __init__(self, match_thresh_px: float = SL_MATCH_THRESH_PX,
                 frame_w: int = 1280):
        self.last_fit = {}     # {"near": (a, b), "far": (a, b)}
        self.match_thresh_px = match_thresh_px
        self.frame_w = frame_w

    def _line_distance(self, a1, b1, a2, b2):
        d_left = abs(a1 - a2)
        d_right = abs((a1 + (self.frame_w - 1) * b1) -
                      (a2 + (self.frame_w - 1) * b2))
        return max(d_left, d_right)

    def _classify_by_y(self, fits):
        """Sort fits by mean y at x=center: smallest y → far, largest → near."""
        if not fits: return {}
        cx = self.frame_w / 2.0
        sorted_fits = sorted(fits,
                             key=lambda f: f["a"] + f["b"] * cx)
        out = {}
        out["far"] = sorted_fits[0]
        if len(sorted_fits) >= 2:
            out["near"] = sorted_fits[1]
        return out

    def init_from(self, fits):
        labels = self._classify_by_y(fits)
        for k, f in labels.items():
            self.last_fit[k] = (float(f["a"]), float(f["b"]))
        return labels

    def update(self, fits):
        if not fits:
            return {}
        labels = {}
        used_keys = set()
        used_idx = set()

        # Step 1: match by line similarity to tracked.
        if self.last_fit:
            pairs = []
            for i, f in enumerate(fits):
                for k, (a_p, b_p) in self.last_fit.items():
                    d = self._line_distance(f["a"], f["b"], a_p, b_p)
                    pairs.append((d, i, k))
            pairs.sort()
            for d, i, k in pairs:
                if d > self.match_thresh_px:
                    break
                if k in used_keys or i in used_idx:
                    continue
                labels[k] = fits[i]
                used_keys.add(k); used_idx.add(i)

        # Step 2: classify any leftover by y position. Avoid keys already used.
        leftover = [(i, fits[i]) for i in range(len(fits)) if i not in used_idx]
        if leftover:
            cx = self.frame_w / 2.0
            leftover.sort(key=lambda iv: iv[1]["a"] + iv[1]["b"] * cx)
            for i, f in leftover:
                # Try far first (smaller y), then near.
                for k in ("far", "near"):
                    if k in used_keys:
                        continue
                    labels[k] = f
                    used_keys.add(k)
                    break

        # Step 3: update state.
        for k, f in labels.items():
            self.last_fit[k] = (float(f["a"]), float(f["b"]))
        return labels


# ── Out-of-bounds filter (yardlines must lie between sidelines) ──────────

def yardline_in_bounds(yl_pixels_undistorted: np.ndarray,
                        sl_labels: dict, tol: float = OOB_TOL_PX) -> bool:
    """Check that the yardline's pixel centroid lies between the visible
    sidelines (in undistorted space).
    """
    if not sl_labels:
        return True
    cx = float(yl_pixels_undistorted[:, 0].mean())
    cy = float(yl_pixels_undistorted[:, 1].mean())
    if "far" in sl_labels:
        f = sl_labels["far"]
        far_y = f["a"] + f["b"] * cx
        if cy < far_y - tol:
            return False
    if "near" in sl_labels:
        n = sl_labels["near"]
        near_y = n["a"] + n["b"] * cx
        if cy > near_y + tol:
            return False
    return True


# ── Per-frame full pipeline ──────────────────────────────────────────────

def _classify_hashes_pca_otsu(frame, intr, yl_fits, w):
    pxs, confs = run_hash_w18(frame, HASH, device="mps",
                               conf_thresh=HASH_CONF)
    if len(pxs) < 2 or len(yl_fits) == 0:
        return None
    hu = undistort_points(pxs.astype(np.float64), intr)

    perp = np.full((len(hu), len(yl_fits)), np.inf)
    for j, yf in enumerate(yl_fits):
        d = (hu[:, 0] - (yf["a"] + yf["b"] * hu[:, 1])) / np.sqrt(1 + yf["b"] ** 2)
        perp[:, j] = np.abs(d)
    nearest_yl = np.argmin(perp, axis=1)
    matched = perp[np.arange(len(hu)), nearest_yl] < MAX_DIST_PX
    work = np.where(matched)[0]
    if len(work) < 2:
        return None
    pts_w = hu[work]
    center = pts_w.mean(axis=0)
    _, S, Vt = np.linalg.svd(pts_w - center, full_matrices=False)
    pc2 = Vt[1]; ratio = float(S[0] / max(S[1], 1e-6))
    proj = (pts_w - center) @ pc2
    split_p = otsu_split_1d(proj)
    a_side = proj < split_p; b_side = ~a_side
    if a_side.any() and b_side.any() and \
       pts_w[a_side, 1].mean() < pts_w[b_side, 1].mean():
        far_l, near_l = a_side, b_side
    else:
        far_l, near_l = b_side, a_side
    gap = (abs(proj[far_l].mean() - proj[near_l].mean())
           if far_l.any() and near_l.any() else 0.0)
    if ratio < PCA_S0_S1_MIN or gap < MIN_CLUSTER_GAP_PX:
        return None
    return hu, work, nearest_yl, confs, far_l, near_l, pts_w


def _classify_hashes_via_h_prev(frame, intr, yl_fits, w, H_prev,
                                  tol_yd: float = HASH_ROW_TOL_YD):
    """For non-bootstrap frames: classify each detected hash by projecting
    its pixel via H_prev to field, snapping to nearest hash row. Outlier
    rejection is automatic — hashes whose projected y is > tol_yd from
    BOTH rows get dropped. No PCA / Otsu / cluster gates.
    """
    from src.homography.apply_homography import pixel_to_field
    pxs, confs = run_hash_w18(frame, HASH, device="mps",
                               conf_thresh=HASH_CONF)
    if len(pxs) == 0 or len(yl_fits) == 0:
        return []
    hu = undistort_points(pxs.astype(np.float64), intr)

    # Snap each hash to nearest yardline by perp distance.
    perp = np.full((len(hu), len(yl_fits)), np.inf)
    for j, yf in enumerate(yl_fits):
        d = (hu[:, 0] - (yf["a"] + yf["b"] * hu[:, 1])) / np.sqrt(1 + yf["b"] ** 2)
        perp[:, j] = np.abs(d)
    nearest_yl = np.argmin(perp, axis=1)
    matched = perp[np.arange(len(hu)), nearest_yl] < MAX_DIST_PX
    work = np.where(matched)[0]
    if len(work) == 0:
        return []

    # Project each matched hash via H_prev → field; classify by hash row.
    pts_field = pixel_to_field(hu[work], H_prev)
    kept = {}     # (yli, role) → (hx, hy, role, cf, yli, d_to_row)
    for i_local in range(len(work)):
        gi = int(work[i_local])
        f_y = float(pts_field[i_local, 1])
        d_far = abs(f_y - HASH_Y_FAR)
        d_near = abs(f_y - HASH_Y_NEAR)
        d_min = min(d_far, d_near)
        if d_min > tol_yd:
            continue   # outlier — projects too far from any hash row
        role = "far" if d_far < d_near else "near"
        hx, hy = float(hu[gi, 0]), float(hu[gi, 1])
        cf = float(confs[gi])
        yli = int(nearest_yl[gi])
        key = (yli, role)
        if key not in kept or d_min < kept[key][5]:
            kept[key] = (hx, hy, role, cf, yli, d_min)

    return [(hx, hy, role, cf, yli)
            for (hx, hy, role, cf, yli, _) in kept.values()]


class HashClassifier:
    """Stateless wrapper. classify() picks H_prev-based classification when
    H_prev is given (more robust, automatic outlier rejection); falls back
    to PCA-Otsu for bootstrap (no H_prev)."""

    def classify(self, frame, intr, yl_fits, w, frame_idx, H_prev=None):
        if H_prev is not None:
            return _classify_hashes_via_h_prev(frame, intr, yl_fits, w, H_prev)
        # Bootstrap path: PCA-Otsu.
        out = _classify_hashes_pca_otsu(frame, intr, yl_fits, w)
        if out is None:
            return []
        hu, work, nearest_yl, confs, far_l, near_l, pts_w = out

        m_far, c_far = fit_row_line(pts_w[far_l])
        m_near, c_near = fit_row_line(pts_w[near_l])
        d_far = perp_dist(pts_w, m_far, c_far)
        d_near = perp_dist(pts_w, m_near, c_near)

        def thr(d, mask):
            if mask.sum() < 2: return 5.0
            v = d[mask]; med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) + 1e-6
            return max(5.0, med + MAD_K * mad)

        far_thr = thr(d_far, far_l); near_thr = thr(d_near, near_l)

        kept = {}
        for i_local in range(len(work)):
            yli = int(nearest_yl[work[i_local]])
            if far_l[i_local] and d_far[i_local] <= far_thr:
                role = "far"; dist_ = float(d_far[i_local])
            elif near_l[i_local] and d_near[i_local] <= near_thr:
                role = "near"; dist_ = float(d_near[i_local])
            else:
                continue
            hx, hy = float(hu[work[i_local], 0]), float(hu[work[i_local], 1])
            cf = float(confs[work[i_local]])
            key = (yli, role)
            if key not in kept or dist_ < kept[key][5]:
                kept[key] = (hx, hy, role, cf, yli, dist_)

        return [(hx, hy, role, cf, yli)
                for (hx, hy, role, cf, yli, _) in kept.values()]


_default_hash_classifier = HashClassifier()


def hashes_with_roles(frame, intr, yl_fits, w, frame_idx: int = 0,
                      classifier: HashClassifier | None = None,
                      H_prev=None):
    cl = classifier if classifier is not None else _default_hash_classifier
    return cl.classify(frame, intr, yl_fits, w, frame_idx, H_prev=H_prev)


def sideline_yardline_intersections(yl_fits, sl_labels, w, h):
    """sl_labels: dict {"near": fit, "far": fit} (either or both).
    Returns: list of (x, y, yi, sideline_label)."""
    out = []
    for yi, yf in enumerate(yl_fits):
        yspan = yf["ymax"] - yf["ymin"]
        for label, sf in sl_labels.items():
            xspan = sf["xmax"] - sf["xmin"]
            denom = 1.0 - yf["b"] * sf["b"]
            if abs(denom) < 1e-9: continue
            x = (yf["a"] + yf["b"] * sf["a"]) / denom
            y = sf["a"] + sf["b"] * x
            if not (0 <= x <= w - 1 and 0 <= y <= h - 1): continue
            if not (yf["ymin"] - EXTRAP_FRAC * yspan <= y
                    <= yf["ymax"] + EXTRAP_FRAC * yspan): continue
            if not (sf["xmin"] - EXTRAP_FRAC * xspan <= x
                    <= sf["xmax"] + EXTRAP_FRAC * xspan): continue
            out.append((float(x), float(y), yi, label))
    return out


# ── Renderers ────────────────────────────────────────────────────────────

def render_masks(frame, yl_groups, sl_groups, g_index, frame_idx, fps, w, h):
    """Video A: original frame + CC mask overlays colored by g."""
    canvas = frame.copy().astype(np.float32)
    for j, lo in enumerate(yl_groups):
        c = np.array(color_for_g(int(g_index[j])), dtype=np.float32)
        xs = lo.pixels[:, 0].astype(np.int32)
        ys = lo.pixels[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        canvas[ys[valid], xs[valid]] = (
            0.35 * canvas[ys[valid], xs[valid]] + 0.65 * c
        )
    for lo in sl_groups:
        c = np.array(SIDELINE_COLOR, dtype=np.float32)
        xs = lo.pixels[:, 0].astype(np.int32)
        ys = lo.pixels[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        canvas[ys[valid], xs[valid]] = (
            0.35 * canvas[ys[valid], xs[valid]] + 0.65 * c
        )
    canvas = canvas.clip(0, 255).astype(np.uint8)
    cv2.putText(canvas, f"frame {frame_idx}  t={frame_idx/fps:.2f}s  "
                f"yl={len(yl_groups)}  sl={len(sl_groups)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def render_fits(frame_u, yl_fits, sl_fits, g_index, hashes, intersections,
                frame_idx, fps, w, h):
    """Video B: undistorted frame + linear fits + matched/labeled keypoints."""
    canvas = frame_u.copy()
    # Yardlines colored by g.
    for j, yf in enumerate(yl_fits):
        g = int(g_index[j])
        c = color_for_g(g)
        ys = np.linspace(yf["ymin"], yf["ymax"], 200)
        xs = yf["a"] + yf["b"] * ys
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, c, 2, cv2.LINE_AA)
        y_lab = max(yf["ymin"] + 30, 40)
        x_lab = yf["a"] + yf["b"] * y_lab
        cv2.putText(canvas, f"g={g:+d}",
                    (int(x_lab) - 22, int(y_lab)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    # Sidelines.
    for sf in sl_fits:
        xs = np.linspace(sf["xmin"], sf["xmax"], 200)
        ys = sf["a"] + sf["b"] * xs
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1).astype(np.int32)],
                      False, SIDELINE_COLOR, 1, cv2.LINE_AA)

    # Hash points: color = parent yardline's g-color.
    for (hx, hy, role, _conf, yli) in hashes:
        g = int(g_index[yli])
        c = color_for_g(g)
        cv2.circle(canvas, (int(round(hx)), int(round(hy))), 6, c, -1)
        cv2.circle(canvas, (int(round(hx)), int(round(hy))), 8, (0, 0, 0), 1)
        cv2.putText(canvas, f"{role}_hash@g{g:+d}",
                    (int(hx) + 9, int(hy) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

    # Sideline×yardline intersections: color = parent yardline's g-color.
    for (x, y, yi, sl_label) in intersections:
        g = int(g_index[yi])
        c = color_for_g(g)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, c, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (0, 0, 0), 1)
        cv2.putText(canvas, f"{sl_label[:1]}sl×g{g:+d}",
                    (int(x) + 9, int(y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

    cv2.putText(canvas,
                f"frame {frame_idx}  t={frame_idx/fps:.2f}s  "
                f"yl={len(yl_fits)}  hashes={len(hashes)}  "
                f"intersections={len(intersections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  clip: {n_total} frames @ {fps:.1f}fps")

    ok, frame0 = cap.read()
    if not ok: print("read failed"); return
    h, w = frame0.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    # Frame 0: calibrate k1.
    yard_mask, side_mask = run_unet(frame0, UNET, device="mps")
    yl_g0 = group_yardline_pixels_cc(yard_mask)
    sl_g0 = group_sideline_pixels(side_mask)
    line_pts = [g.pixels for g in yl_g0] + [g.pixels for g in sl_g0]
    line_kinds = ["yardline"] * len(yl_g0) + ["sideline"] * len(sl_g0)
    sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    print(f"  k1 = {k1:+.4f} (cached for whole clip)")

    yl_fits0_all = [fit_yardline_linear(g.pixels, intr) for g in yl_g0]
    sl_fits0_all = [fit_sideline_linear(g.pixels, intr) for g in sl_g0]
    yl_tracker = YardlineTracker(g_min=G_MIN, g_max=G_MAX, frame_h=h)
    sl_tracker = SidelineTracker(frame_w=w)
    hash_clf = HashClassifier()

    # Sideline tracker init (no OOB filter — disabled for this run).
    sl_labels0 = sl_tracker.init_from(sl_fits0_all)

    res0 = yl_tracker.init_from(yl_fits0_all, cy)
    if res0 is None:
        print("frame 0: <2 valid yardlines after g-range gate"); return
    yl_fits0, g_index0, _ = res0
    keep0 = [yf in yl_fits0 for yf in yl_fits0_all]
    yl_g0_kept = [yl_g0[i] for i in range(len(yl_g0)) if keep0[i]]
    print(f"  g-range gate: g_min={G_MIN}  g_max={G_MAX}  "
          f"(g0 → NGS x={G0_NGS_X:.0f})  OOB filter: DISABLED")
    print(f"  sideline labels at frame 0: {list(sl_labels0.keys())}")

    os.makedirs(os.path.dirname(OUT_MASKS), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_masks = cv2.VideoWriter(OUT_MASKS, fourcc, fps, (w, h))
    w_fits = cv2.VideoWriter(OUT_FITS, fourcc, fps, (w, h))

    # Frame 0.
    frame0_u = cv2.undistort(frame0, K, dist) if abs(k1) > 1e-6 else frame0.copy()
    hashes0 = hashes_with_roles(frame0, intr, yl_fits0, w, 0, hash_clf)
    inters0 = sideline_yardline_intersections(yl_fits0, sl_labels0, w, h)
    w_masks.write(render_masks(frame0, yl_g0_kept, sl_g0, g_index0, 0, fps, w, h))
    w_fits.write(render_fits(frame0_u, yl_fits0, list(sl_labels0.values()),
                              g_index0, hashes0, inters0, 0, fps, w, h))
    print(f"  frame 0: yl={len(yl_fits0)} (raw {len(yl_g0)})  "
          f"sl={len(sl_labels0)}  hashes={len(hashes0)}  sl×yl={len(inters0)}  "
          f"g={sorted(int(x) for x in g_index0)}")

    # Subsequent frames.
    for fi in range(1, n_total):
        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()
        yard_mask, side_mask = run_unet(frame, UNET, device="mps")
        yl_g_raw = group_yardline_pixels_cc(yard_mask)
        sl_g = group_sideline_pixels(side_mask)
        if len(yl_g_raw) == 0:
            blank = frame.copy()
            cv2.putText(blank, f"frame {fi}: no yardlines", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            w_masks.write(blank); w_fits.write(blank)
            continue
        sl_fits_all = [fit_sideline_linear(g.pixels, intr) for g in sl_g]
        sl_labels = sl_tracker.update(sl_fits_all)

        # OOB filter disabled — pass all yardline candidates straight to
        # the tracker and let it / the g-range gate decide.
        yl_fits_all = [fit_yardline_linear(g.pixels, intr) for g in yl_g_raw]
        yl_fits, g_index, _, n_rej = yl_tracker.update(yl_fits_all, cy)
        keep_mask = [yf in yl_fits for yf in yl_fits_all]
        yl_g_kept = [yl_g_raw[i] for i in range(len(yl_g_raw)) if keep_mask[i]]

        hashes = hashes_with_roles(frame, intr, yl_fits, w, fi, hash_clf)
        inters = sideline_yardline_intersections(yl_fits, sl_labels, w, h)
        frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
        w_masks.write(render_masks(frame, yl_g_kept, sl_g, g_index, fi, fps, w, h))
        w_fits.write(render_fits(frame_u, yl_fits, list(sl_labels.values()),
                                  g_index, hashes, inters, fi, fps, w, h))

        if fi % 10 == 0 or fi < 5:
            t_ms = (time.time() - t0) * 1000
            print(f"  frame {fi:>3}  yl={len(yl_fits)} (raw {len(yl_g_raw)},"
                  f" rej={n_rej})  sl={list(sl_labels.keys())}  "
                  f"hashes={len(hashes)}  sl×yl={len(inters)}  "
                  f"g={sorted(int(x) for x in g_index)}  ({t_ms:.0f}ms)")

    cap.release(); w_masks.release(); w_fits.release()
    print(f"\n  wrote {OUT_MASKS}")
    print(f"  wrote {OUT_FITS}")


if __name__ == "__main__":
    main()
