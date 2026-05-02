"""Yardline g-index tracking + sideline pixel grouping.

Provides:
  - `YardlineTracker`: maintains stable g-index identity across frames by
    line-parameter similarity, with grid-snap fallback for newly entering
    yardlines.
  - `group_sideline_pixels_cc`: CC + collinearity merge for sideline
    pixels, mirroring the yardline grouping path. Returns up to two
    strongest clusters as `SimpleNamespace(pixels=...)`.

The defaults `g_min=-2`, `g_max=+18` derive from the pre-rectify v1
clip-prelude calibration where g0 corresponded to NGS x=20. Callers should
override with per-clip values computed from the resolved g0:
    g_min = (NGS_X_LEFT_GOAL  - g0_ngs_x) / 5.0   # = -2 for g0=20
    g_max = (NGS_X_RIGHT_GOAL - g0_ngs_x) / 5.0   # = +18 for g0=20
"""

from types import SimpleNamespace

import cv2
import numpy as np


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
    Returns up to `max_lines` strongest clusters by pixel count, each as a
    `SimpleNamespace` with a `.pixels` (N, 2) array."""
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

    `g_min` / `g_max` defaults (-2 / +18) come from the original
    play_046 calibration where g=0 corresponded to NGS x=20. Production
    callers should override per-clip with values derived from the resolved
    g0_ngs_x.
    """

    def __init__(self, g_min: int = -2, g_max: int = 18,
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
