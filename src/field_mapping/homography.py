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

from .apply_homography import pixel_to_field, field_to_pixel
from .keypoint_bank import KeypointTrackBank


# ── RANSAC + sanity / fallback thresholds ─────────────────────────────────
RANSAC_REPROJ_PX = 4.0
MIN_CORRS_FOR_H = 4    # cv2.findHomography minimum; 4 non-collinear pts → H

FULL_H_SANITY_MAX_YD = 2.0

DELTA_MAX_SCALE = 1.8
DELTA_MAX_ROT_DEG = 3.0
DELTA_MAX_TRANS_PX = 400

# H_prev-based pre-filter: drop correspondences whose pixel projects (via
# H_prev) to a field position more than this far from their claimed
# canonical field position. Catches false hashes + mis-classified hashes
# before they get into RANSAC. Generous enough to absorb 1 frame of camera
# motion at 30 fps.
#
# Effectively disabled (large tolerance) as of 2026-05-02 after diagnosing
# play_046 sideline-drift: the 0.5yd cap was silently culling far-sideline
# corrs once H_prev wobbled (e.g. when near-sideline corrs panned out of
# frame), creating a feedback loop where each frame's H drifted from a
# biased corr subset. Re-enable only with a much larger threshold OR after
# replacing with a per-corr-kind tolerance.
PREFILTER_TOL_YD = 999.0

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


def _loo_poly_at(Hs, i, half=3, poly=2):
    """Degree-`poly` poly per H coefficient through {i-half..i+half}\\{i}.

    Hs is a list of 3×3 matrices (no Nones). Returns the LOO-extrapolated
    3×3 H at index i, or None if too few neighbors to fit.
    """
    n = len(Hs)
    lo = max(0, i - half); hi = min(n, i + half + 1)
    xs = [j for j in range(lo, hi) if j != i]
    if len(xs) < poly + 1: return None
    ys = np.stack([Hs[j].flatten() for j in xs], axis=0)
    xs_a = np.array(xs, dtype=np.float64)
    out = np.zeros(9, dtype=np.float64)
    for c in range(9):
        coef = np.polyfit(xs_a, ys[:, c], poly)
        out[c] = np.polyval(coef, i)
    return out.reshape(3, 3)


def _corner_residual_yd(H_a, H_b, img_w=1280, img_h=720, margin=100):
    """Max corner reprojection distance (NGS yards) between two H matrices,
    using 4 interior image corners as probes."""
    if H_a is None or H_b is None: return None
    pts = np.array([[margin, margin], [img_w - margin, margin],
                    [img_w - margin, img_h - margin], [margin, img_h - margin]],
                   dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(4)])
    pa = (H_a @ pts_h.T).T; za = pa[:, 2:3]; za[np.abs(za) < 1e-9] = 1e-9
    pb = (H_b @ pts_h.T).T; zb = pb[:, 2:3]; zb[np.abs(zb) < 1e-9] = 1e-9
    return float(np.max(np.linalg.norm(pa[:, :2]/za - pb[:, :2]/zb, axis=1)))


def loo_filter_and_replace(Hs_raw, rmses=None,
                              thr_loo_yd: float = 0.20,
                              thr_rmse_yd: float = 0.30,
                              half: int = 3, poly: int = 2,
                              img_w: int = 1280, img_h: int = 720):
    """Detect bad H frames via leave-one-out polynomial residual + rmse
    threshold, and replace them with the LOO polynomial extrapolation.

    For each frame, fits a degree-`poly` polynomial per H coefficient
    through the `half` neighbors on each side (excluding the frame itself),
    measures the max-corner reprojection distance between the raw H and the
    LOO H in NGS yards, and flags the frame if:
      • loo_resid > thr_loo_yd, OR
      • rmses[i] is provided and rmses[i] > thr_rmse_yd
    Flagged frames are replaced with their LOO polynomial H.

    Args:
        Hs_raw: list of 3×3 H matrices (Nones allowed; they're treated as
            already-bad and replaced with LOO H if possible).
        rmses: optional parallel list of per-frame solver rmse_yd (or None
            per frame). When provided, contributes to the red mask via the
            rmse threshold.
        thr_loo_yd: LOO residual threshold in NGS yards.
        thr_rmse_yd: rmse threshold in NGS yards.
        half: number of neighbors on each side for the polynomial fit.
        poly: polynomial degree.
        img_w / img_h: image dimensions used for corner reprojection.

    Returns:
        (Hs_out, red_mask, loo_resids):
            Hs_out: list of 3×3 H matrices with red frames replaced by LOO.
                Frames where LOO is unavailable (clip edges) keep their
                raw value (or None if it was None).
            red_mask: list[bool] same length as Hs_raw.
            loo_resids: list[float | None] per-frame LOO residual in yards.
    """
    n = len(Hs_raw)
    Hs_out = list(Hs_raw)
    red_mask = [False] * n
    loo_resids = [None] * n

    # Build a contiguous sequence of frames whose raw H is not None, plus a
    # mapping back to the original index space.
    seq_idx = [i for i in range(n) if Hs_raw[i] is not None]
    seq_H = [Hs_raw[i] for i in seq_idx]

    # Per-frame LOO residual against the raw H.
    for local_i, gi in enumerate(seq_idx):
        H_loo = _loo_poly_at(seq_H, local_i, half=half, poly=poly)
        if H_loo is None: continue
        loo_resids[gi] = _corner_residual_yd(Hs_raw[gi], H_loo,
                                               img_w=img_w, img_h=img_h)

    # Red mask: missing H OR loo > thr OR rmse > thr.
    for i in range(n):
        if Hs_raw[i] is None:
            red_mask[i] = True; continue
        if loo_resids[i] is not None and loo_resids[i] > thr_loo_yd:
            red_mask[i] = True
        if rmses is not None and rmses[i] is not None and rmses[i] > thr_rmse_yd:
            red_mask[i] = True

    # Build the GOOD-only sequence and fit LOO polys through it for the
    # bad frames. This is the "drop the bad and re-fit through clean
    # neighbors" step the user asked for.
    good_idx = [i for i in range(n) if not red_mask[i] and Hs_raw[i] is not None]
    good_H = [Hs_raw[i] for i in good_idx]
    if not good_idx:
        return Hs_out, red_mask, loo_resids

    # For each red frame, fit polynomial through good neighbors (3 on each
    # side of the frame's position in the global-frame indexing) and
    # evaluate at the frame.
    good_arr = np.array(good_idx, dtype=np.float64)
    for i in range(n):
        if not red_mask[i]: continue
        # Find indices of good frames bracketing i.
        right_pos = int(np.searchsorted(good_arr, i, side="right"))
        left_pos = right_pos - 1
        left_neighbors = good_idx[max(0, left_pos - half + 1):right_pos]
        right_neighbors = good_idx[right_pos:right_pos + half]
        nb = left_neighbors + right_neighbors
        if len(nb) < poly + 1:
            # Not enough good neighbors to fit. Keep raw (or None).
            continue
        Hs_nb = [Hs_raw[j] for j in nb]
        xs = np.array(nb, dtype=np.float64)
        ys = np.stack([h.flatten() for h in Hs_nb], axis=0)
        H_fill = np.zeros(9, dtype=np.float64)
        for c in range(9):
            coef = np.polyfit(xs, ys[:, c], poly)
            H_fill[c] = np.polyval(coef, i)
        Hs_out[i] = H_fill.reshape(3, 3)

    return Hs_out, red_mask, loo_resids


def detect_bad_runs(red_mask, min_length: int = 5):
    """Return list of (start, end_exclusive) for runs of ≥min_length
    consecutive red frames in `red_mask`. Useful for flagging stretches
    where the H trajectory was bridged across a long gap — the bridge may
    be inaccurate, and the clip is a candidate for manual review or
    rejection at data-collection time.
    """
    runs = []
    n = len(red_mask)
    i = 0
    while i < n:
        if red_mask[i]:
            j = i
            while j < n and red_mask[j]:
                j += 1
            if j - i >= min_length:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


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
        # Per-corr prefilter via the keypoint bank. Three groups returned:
        #   - kept: passes all checks → goes to RANSAC
        #   - probationary: brand-new identity, no conflicts → observed in
        #     the bank (so it can be promoted later) but NOT used in this
        #     frame's H
        #   - rejected (cross-track or same-track failure): not observed,
        #     not used
        # See KeypointTrackBank.prefilter_corrs for the three checks.
        if corrs:
            corrs, probationary, prefilter_diag = self.bank.prefilter_corrs(
                corrs, frame_idx=frame_idx)
        else:
            probationary = []
            prefilter_diag = {"n_input": 0, "n_kept": 0, "n_probationary": 0,
                              "n_dropped_self": 0, "n_dropped_cross": 0}
        H, inl, rmse = solve_h(corrs) if corrs else (None, None, None)
        method = None
        info = {"n_input_corrs": n_in, "n_after_prefilter": len(corrs),
                "prefilter": prefilter_diag}

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

        # Update state. Observe on real-H frames (full or delta). Both
        # kept corrs (used in this frame's H) and probationary corrs
        # (held back from H but accumulating evidence so they can be
        # promoted in future frames) are recorded.
        if method in ("full", "delta"):
            self.bank.observe(corrs + probationary, frame_idx=frame_idx)
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
