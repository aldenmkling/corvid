"""Per-frame homography solver + full / delta / carry tracker.

Provides:
  - `solve_h(corrs)`: cv2.findHomography RANSAC over a list of
    correspondence dicts (`pixel_u`, `field`, `kind`, `label`). Returns
    `(H, inlier_mask, rmse_field)` or `(None, None, None)`.
  - `is_degenerate_for_h(corrs)`: rejects configurations where too many
    correspondences share a single NGS-y row (target-plane collinearity
    that `cv2.findHomography` swallows silently).
  - `HomographyTrackerLite`: full / delta / carry fallback chain plus a
    `KeypointTrackBank` that validates candidate H matrices.
  - `smooth_hs(Hs_raw, window, poly)`: Savitzky-Golay smoothing of a list
    of 3×3 H matrices (h[2,2] normalized before smoothing, scale re-applied).
  - `detect_lost(methods, min_sustained_loss)`: returns the index of the
    first frame in the FIRST sustained-carry run, or None.

No I/O, no rendering, no `main()`.
"""

from collections import Counter

import cv2
import numpy as np

from src.homography.apply_homography import pixel_to_field, field_to_pixel
from src.homography.keypoint_track_bank import KeypointTrackBank


# ── RANSAC + sanity / fallback thresholds ─────────────────────────────────
RANSAC_REPROJ_PX = 4.0
MIN_CORRS_FOR_H = 4    # cv2.findHomography minimum; 4 non-collinear pts → H

FULL_H_SANITY_MAX_YD = 0.5

DELTA_MAX_SCALE = 1.8
DELTA_MAX_ROT_DEG = 3.0
DELTA_MAX_TRANS_PX = 400

# H_prev-based pre-filter: drop correspondences whose pixel projects (via
# H_prev) to a field position more than this far from their claimed
# canonical field position. Catches false hashes + mis-classified hashes
# before they get into RANSAC. Generous enough to absorb 1 frame of camera
# motion at 30 fps.
PREFILTER_TOL_YD = 0.5

# Degenerate-config check: collapse correspondences onto NGS-y rows OR
# NGS-x columns (round to nearest 0.5yd to bucket discrete row positions
# like 0, 14, 23.58, 29.75, 39.33, 53.33 and column positions which are
# integer yardlines). cv2.findHomography needs 4 points with no 3
# collinear in either source or target plane; if 3+ points all share a
# single field-plane row OR column, RANSAC's 4-subsets include collinear
# triples and the H is rank-deficient even though findHomography returns
# a value. We require at least 2 points OFF the most-populated row AND
# at least 2 points OFF the most-populated column.
DEGEN_BUCKET_YD = 0.5
DEGEN_MIN_OFF_DOMINANT = 2


def is_degenerate_for_h(corrs):
    """True if correspondences are degenerate for `cv2.findHomography`:
    too many points share a single NGS-y row OR a single NGS-x column
    (collinear in target plane).

    cv2.findHomography returns a numerically-unstable H rather than failing
    cleanly when 3+ points are collinear among the 4-subsets RANSAC samples.
    We require at least DEGEN_MIN_OFF_DOMINANT points off the most-populated
    NGS-y row AND at least that many off the most-populated NGS-x column,
    so RANSAC has at least one non-degenerate 4-subset along each axis."""
    if len(corrs) < MIN_CORRS_FOR_H:
        return True
    bucket = lambda v: round(v / DEGEN_BUCKET_YD) * DEGEN_BUCKET_YD
    rows = Counter(bucket(c["field"][1]) for c in corrs)
    cols = Counter(bucket(c["field"][0]) for c in corrs)
    if (len(corrs) - max(rows.values())) < DEGEN_MIN_OFF_DOMINANT:
        return True
    if (len(corrs) - max(cols.values())) < DEGEN_MIN_OFF_DOMINANT:
        return True
    return False


def solve_h(corrs):
    """RANSAC homography from undistorted-pixel → NGS field coords.
    Returns (H, inlier_mask, rmse_yd) or (None, None, None) on failure."""
    if len(corrs) < MIN_CORRS_FOR_H:
        return None, None, None
    if is_degenerate_for_h(corrs):
        return None, None, None
    src = np.array([c["pixel_u"] for c in corrs], dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array([c["field"] for c in corrs], dtype=np.float64).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=RANSAC_REPROJ_PX)
    if H is None:
        return None, None, None
    inlier = mask.ravel().astype(bool)
    src_h = np.column_stack([src.reshape(-1, 2), np.ones(len(corrs))])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj_xy - dst.reshape(-1, 2), axis=1)
    rmse_field = float(np.sqrt(np.mean(err[inlier] ** 2))) if inlier.any() else None
    return H, inlier, rmse_field


def smooth_hs(Hs_raw, window: int = 7, poly: int = 2):
    """Savitzky-Golay smooth a list of 3×3 H matrices. Normalizes h[2,2]=1
    before smoothing, re-applies original scale after."""
    from scipy.signal import savgol_filter
    if not Hs_raw or window < 3 or len(Hs_raw) < window:
        return list(Hs_raw)
    H_flat = np.stack([h.flatten() for h in Hs_raw], axis=0)   # (N, 9)
    scales = H_flat[:, 8:9].copy()
    scales[np.abs(scales) < 1e-9] = 1.0
    H_flat_n = H_flat / scales
    sm = savgol_filter(H_flat_n, window_length=window,
                       polyorder=min(poly, window - 1),
                       axis=0, mode="nearest")
    sm = sm * scales
    return [sm[i].reshape(3, 3) for i in range(len(Hs_raw))]


def detect_lost(methods, min_sustained_loss: int = 3):
    """Return the index of the first frame in the FIRST sustained-carry run,
    or None if no such run exists. Once `min_sustained_loss` consecutive
    'carry' methods occur, the clip is treated as LOST from the run's start."""
    consec = 0
    for i, m in enumerate(methods):
        if m == "carry":
            consec += 1
            if consec >= min_sustained_loss:
                return i - consec + 1
        else:
            consec = 0
    return None


class HomographyTrackerLite:
    """Per-frame homography with full / delta / carry fallback + a track
    bank that validates candidate H matrices.

    full   = RANSAC on this frame's correspondences (preferred).
    delta  = similarity transform from prev-frame pixel positions of the
              same field points to this-frame pixel positions; used when
              full is unavailable or rejected.
    carry  = use H_prev unchanged when nothing better is available.

    No coasting: the bank only validates; it doesn't synthesize fake
    correspondences (per the comment in src/homography/tracker.py, coasting
    introduces a feedback loop).
    """

    def __init__(self):
        # Tighter bank thresholds than the defaults: catch single-frame
        # identity flips (was tol=1.0, bad_frac=0.34, min_obs=2).
        self.bank = KeypointTrackBank(
            h_validate_tol_yd=0.7,
            h_validate_bad_frac=0.20,
            min_obs_for_trust=3,
        )
        self.H_prev = None
        self.H_inv_prev = None

    def _solve_delta(self, corrs):
        """Similarity transform: prev pixel positions of these field points
        → current pixel positions. H_cur = H_prev @ inv(S)."""
        if self.H_prev is None or len(corrs) < 2:
            return None
        cur_px = np.array([c["pixel_u"] for c in corrs], dtype=np.float64)
        field = np.array([c["field"] for c in corrs], dtype=np.float64)
        prev_px = field_to_pixel(field, self.H_inv_prev)
        M, _ = cv2.estimateAffinePartial2D(
            prev_px, cur_px, method=cv2.LMEDS,
        )
        if M is None:
            return None
        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        tx, ty = float(M[0, 2]), float(M[1, 2])
        if (scale > DELTA_MAX_SCALE or scale < 1.0 / DELTA_MAX_SCALE):
            return None
        if abs(rot_deg) > DELTA_MAX_ROT_DEG:
            return None
        if np.hypot(tx, ty) > DELTA_MAX_TRANS_PX:
            return None
        S = np.vstack([M, [0.0, 0.0, 1.0]])
        H = self.H_prev @ np.linalg.inv(S)
        H_inv = np.linalg.inv(H)
        return H, H_inv, {"scale": scale, "rot_deg": rot_deg,
                          "trans_px": (tx, ty)}

    def update(self, corrs, frame_idx):
        """Returns dict with H, method, n_corrs, n_inliers, rmse_yd, info."""
        n_in = len(corrs)
        # Pre-filter against H_prev: drop corrs whose pixel doesn't project
        # to its claimed canonical field position. Catches false hashes
        # and mis-classified hashes before they pollute the H solve.
        if self.H_prev is not None and corrs:
            kept = []
            n_dropped = 0
            for c in corrs:
                px = c["pixel_u"]
                f_via_prev = pixel_to_field(
                    px.reshape(1, 2), self.H_prev,
                )[0]
                if np.linalg.norm(f_via_prev - c["field"]) <= PREFILTER_TOL_YD:
                    kept.append(c)
                else:
                    n_dropped += 1
            corrs = kept
            if n_dropped:
                # store for diagnostics
                pass
        H, inl, rmse = solve_h(corrs) if corrs else (None, None, None)
        method = None
        info = {"n_input_corrs": n_in, "n_after_prefilter": len(corrs)}

        # Full path (with sanity + bank validation).
        if H is not None:
            ok = True
            if self.H_prev is not None:
                cur_px = np.array([c["pixel_u"] for c in corrs], dtype=np.float64)
                f_new = pixel_to_field(cur_px, H)
                f_prev = pixel_to_field(cur_px, self.H_prev)
                divergence = float(np.mean(np.linalg.norm(f_new - f_prev, axis=1)))
                info["divergence_yd"] = divergence
                if divergence > FULL_H_SANITY_MAX_YD:
                    ok = False
            if ok and self.H_prev is not None:
                valid, diag = self.bank.validate_h(H, corrs, frame_idx=frame_idx)
                info["bank"] = diag
                if not valid:
                    ok = False
            if ok:
                method = "full"

        # Delta fallback.
        if method is None and self.H_prev is not None and corrs:
            d = self._solve_delta(corrs)
            if d is not None:
                H, H_inv, dinfo = d
                method = "delta"
                info["delta"] = dinfo
                inl = None; rmse = None

        # Carry fallback.
        if method is None and self.H_prev is not None:
            H = self.H_prev; H_inv = self.H_inv_prev
            method = "carry"; inl = None; rmse = None

        # First frame, nothing to fall back to.
        if method is None:
            return {"H": None, "method": "none", "n_corrs": len(corrs),
                    "n_inliers": 0, "rmse_yd": None, "info": info}

        if method == "full":
            H_inv = np.linalg.inv(H)

        # Update state. Only observe on real-H frames (full or delta).
        if method in ("full", "delta"):
            self.bank.observe(corrs, frame_idx=frame_idx)
            self.bank.prune(frame_idx=frame_idx)
        self.H_prev = H
        self.H_inv_prev = H_inv

        return {
            "H": H, "H_inv": H_inv, "method": method,
            "n_corrs": len(corrs),
            "n_inliers": int(inl.sum()) if inl is not None else None,
            "rmse_yd": rmse,
            "info": info,
        }
