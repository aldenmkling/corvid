"""Connected-component tokenizer — v3.

Same I/O contract as cc_tokenizer_v2 (TOKEN_FEATURE_DIM=16), but:

  - YARDLINE channel: CC fragments of the SAME physical yardline are merged
    via PCA (ρ, θ) collinearity clustering before tokenization. One token
    per physical yardline (instead of one per CC fragment, which v2 did).
  - SIDELINE channel: CC fragments of the same sideline are merged via
    perpendicular-axis projection peak picking. One token per sideline.
  - HASH channel: unchanged from v2 (one token per CC).
  - NUMBER channel: unchanged from v2 (spatial-cluster tokenizer with
    dilation merge).

Why grouping helps:
  - Bigger image context for the per-token classifier (full line vs single
    fragment) → fewer mis-classifications.
  - Cleaner centroid (averaged over the full line, less biased by which
    portion is visible).
  - Fewer tokens per frame → less correlated outlier risk in DLT.

The grouping logic is a port of `src/homography/grid_solver_v2.py`'s
`group_yardline_pixels_cc` and `group_sideline_pixels`. Identical
algorithm, adapted to produce CC-tokenizer-style features.

Token feature layout unchanged:
    [type 1-hot (4),   centroid (2),   bbox (4),
     log_area (1),     orientation (2), ngs_x (1),
     has_ngs (1),      confidence (1)]   = 16 dims
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import cv2

# Re-export shared constants.
SRC_W = 1280
SRC_H = 720
MIN_CC_PX = 1
MIN_CC_PX_NUM = 1
LOG_AREA_DIVISOR = float(np.log(SRC_W * SRC_H))
TOKEN_FEATURE_DIM = 16

TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3

DEFAULT_DILATE_PX = 28

# Yardline grouping (port of group_yardline_pixels_cc).
YARD_MIN_PIXELS_PER_COMPONENT = 40
YARD_MIN_ASPECT_RATIO = 3.0
YARD_RHO_TOL_PX = 25.0
YARD_THETA_TOL_RAD = 0.08
YARD_MIN_FRAGMENTS_PER_LINE = 2
YARD_MIN_PIXELS_FOR_SINGLETON = 500
YARD_DEDUP_PEAK_X_PX = 50.0
YARD_MIN_PIXELS_PER_LINE = 100

# Sideline grouping — same (ρ, θ) collinearity as yardlines, but tuned for
# horizontal-ish sidelines that may have lower aspect ratio under steep
# perspective (redzone close-ups, etc.).
SIDE_MIN_PIXELS_PER_COMPONENT = 60
SIDE_MIN_ASPECT_RATIO = 2.0
SIDE_RHO_TOL_PX = 30.0
SIDE_THETA_TOL_RAD = 0.10
SIDE_MIN_FRAGMENTS_PER_LINE = 1     # singletons OK if big enough
SIDE_MIN_PIXELS_FOR_SINGLETON = 200
SIDE_MIN_PIXELS_PER_LINE = 100


# ─────────────────────────────────────────────────────────────────────────────
# Shared feature builders (identical to v2).
# ─────────────────────────────────────────────────────────────────────────────

def _orientation_from_pixels(ys: np.ndarray, xs: np.ndarray
                              ) -> tuple[float, float]:
    if len(ys) < 2:
        return 1.0, 0.0
    ys_f = ys.astype(np.float32); xs_f = xs.astype(np.float32)
    yc = ys_f.mean(); xc = xs_f.mean()
    dy = ys_f - yc; dx = xs_f - xc
    n = max(1, len(ys) - 1)
    s_yy = float((dy * dy).sum() / n)
    s_xx = float((dx * dx).sum() / n)
    s_yx = float((dy * dx).sum() / n)
    tr = s_yy + s_xx
    det = s_yy * s_xx - s_yx * s_yx
    sq = float(np.sqrt(max(0.0, tr * tr / 4 - det)))
    eig_max = tr / 2 + sq
    if abs(s_yx) > 1e-9:
        vy, vx = s_yx, eig_max - s_yy
    else:
        if s_xx >= s_yy:
            vy, vx = 0.0, 1.0
        else:
            vy, vx = 1.0, 0.0
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return 1.0, 0.0
    angle = float(np.arctan2(vy, vx))
    return float(np.cos(angle)), float(np.sin(angle))


def _build_feature(type_idx, cx, cy, x_min, y_min, x_max, y_max, area,
                    cos_t, sin_t, ngs_x_yards, has_ngs, confidence):
    feat = np.zeros(TOKEN_FEATURE_DIM, dtype=np.float32)
    feat[type_idx] = 1.0
    feat[4] = cx / SRC_W
    feat[5] = cy / SRC_H
    feat[6] = x_min / SRC_W
    feat[7] = y_min / SRC_H
    feat[8] = x_max / SRC_W
    feat[9] = y_max / SRC_H
    feat[10] = float(np.log(max(1, area))) / LOG_AREA_DIVISOR
    feat[11] = cos_t
    feat[12] = sin_t
    feat[13] = ngs_x_yards / 120.0
    feat[14] = 1.0 if has_ngs else 0.0
    feat[15] = float(confidence)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# v3 yardline grouping — port of group_yardline_pixels_cc.
# ─────────────────────────────────────────────────────────────────────────────

def _process_yardline_channel_grouped(prob_map: np.ndarray,
                                          return_pixels: bool = False):
    """Group CC fragments by PCA (ρ, θ) collinearity, one token per
    physical yardline. If return_pixels=True, also returns a parallel
    list of (ys, xs) pixel-set tuples (union of all member CCs)."""
    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ([], []) if return_pixels else []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)

    components: list[dict] = []
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < YARD_MIN_PIXELS_PER_COMPONENT:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < YARD_MIN_PIXELS_PER_COMPONENT:
            continue
        pts = np.column_stack([xs.astype(np.float64),
                                  ys.astype(np.float64)])
        center = pts.mean(axis=0)
        centered = pts - center
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6:
            continue
        if S[0] / S[1] < YARD_MIN_ASPECT_RATIO:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        if theta < 0:
            theta += np.pi
            rho = -rho
        components.append({
            "ys": ys, "xs": xs, "rho": rho, "theta": theta,
            "n": len(xs), "prob_map": prob_map,
        })

    if not components:
        return ([], []) if return_pixels else []

    # Greedy collinearity clustering.
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
            if theta_diff > YARD_THETA_TOL_RAD:
                continue
            wrap_flip = (abs(c["theta"] - d["theta"]) > np.pi / 2)
            d_rho_eff = -d["rho"] if wrap_flip else d["rho"]
            if abs(c["rho"] - d_rho_eff) > YARD_RHO_TOL_PX:
                continue
            used[j] = True
            cluster.append(d)
        clusters.append(cluster)

    out = []
    candidates = []    # (peak_x, n_pixels, feature, (all_ys, all_xs))
    for cluster in clusters:
        n_pix_total = sum(c["n"] for c in cluster)
        # Singleton guard: drop unmerged singletons unless big enough.
        if (len(cluster) < YARD_MIN_FRAGMENTS_PER_LINE
                and n_pix_total < YARD_MIN_PIXELS_FOR_SINGLETON):
            continue
        if n_pix_total < YARD_MIN_PIXELS_PER_LINE:
            continue
        all_xs = np.concatenate([c["xs"] for c in cluster])
        all_ys = np.concatenate([c["ys"] for c in cluster])
        cx = float(all_xs.mean()); cy = float(all_ys.mean())
        x_min = int(all_xs.min()); y_min = int(all_ys.min())
        x_max = int(all_xs.max()) + 1; y_max = int(all_ys.max()) + 1
        cos_t, sin_t = _orientation_from_pixels(all_ys, all_xs)
        conf = float(prob_map[all_ys, all_xs].mean())
        feat = _build_feature(
            type_idx=TYPE_YARD, cx=cx, cy=cy,
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            area=int(n_pix_total), cos_t=cos_t, sin_t=sin_t,
            ngs_x_yards=0.0, has_ngs=False,
            confidence=conf)
        candidates.append((cx, n_pix_total, feat, (all_ys, all_xs)))

    if not candidates:
        return ([], []) if return_pixels else []
    # Dedup: if two clusters' centroid x's are within DEDUP_PEAK_X_PX,
    # keep the one with more pixels.
    candidates.sort(key=lambda c: c[0])    # by centroid x
    keep = [True] * len(candidates)
    for i in range(len(candidates)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            if abs(candidates[i][0] - candidates[j][0]) > YARD_DEDUP_PEAK_X_PX:
                break
            # Within dedup distance — keep larger.
            if candidates[i][1] >= candidates[j][1]:
                keep[j] = False
            else:
                keep[i] = False
                break

    out_pix = []
    for ki, (_, _, feat, pix) in enumerate(candidates):
        if keep[ki]:
            out.append(feat)
            out_pix.append(pix)
    if return_pixels:
        return out, out_pix
    return out


# ─────────────────────────────────────────────────────────────────────────────
# v3 sideline grouping — port of group_sideline_pixels.
# ─────────────────────────────────────────────────────────────────────────────

def _peak_pick_1d(coords: np.ndarray, bin_width: float = 1.0,
                    min_sep: float = 20.0,
                    min_prom_frac: float = 0.10) -> tuple:
    """Histogram + scipy.signal.find_peaks with prominence filter.

    Direct port of src/homography/grid_solver_v2.py's _peak_pick_1d.
    """
    from scipy import signal as sp_signal
    if len(coords) == 0:
        return np.array([]), np.array([])
    c_min, c_max = float(coords.min()), float(coords.max())
    if c_max - c_min < 1e-6:
        return np.array([c_min]), np.array([len(coords)], dtype=float)
    n_bins = max(8, int(np.ceil((c_max - c_min) / bin_width)))
    hist, edges = np.histogram(coords, bins=n_bins, range=(c_min, c_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
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
    peaks, _ = sp_signal.find_peaks(
        smoothed, distance=distance_bins, prominence=prominence)
    return centers[peaks], smoothed[peaks]


def _process_sideline_channel_grouped(prob_map: np.ndarray,
                                          return_pixels: bool = False):
    """CC + (ρ, θ) collinearity grouping — same approach as yardlines.
    If return_pixels=True, also returns per-token (ys, xs) pixel sets."""
    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ([], []) if return_pixels else []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)

    components = []
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < SIDE_MIN_PIXELS_PER_COMPONENT:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < SIDE_MIN_PIXELS_PER_COMPONENT:
            continue
        pts = np.column_stack([xs.astype(np.float64),
                                  ys.astype(np.float64)])
        center = pts.mean(axis=0)
        centered = pts - center
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6:
            continue
        if S[0] / S[1] < SIDE_MIN_ASPECT_RATIO:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        if theta < 0:
            theta += np.pi
            rho = -rho
        components.append({"ys": ys, "xs": xs, "rho": rho,
                              "theta": theta, "n": len(xs)})
    if not components:
        return ([], []) if return_pixels else []

    used = [False] * len(components)
    clusters = []
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
            if theta_diff > SIDE_THETA_TOL_RAD:
                continue
            wrap_flip = (abs(c["theta"] - d["theta"]) > np.pi / 2)
            d_rho_eff = -d["rho"] if wrap_flip else d["rho"]
            if abs(c["rho"] - d_rho_eff) > SIDE_RHO_TOL_PX:
                continue
            used[j] = True
            cluster.append(d)
        clusters.append(cluster)

    out = []
    out_pix = []
    for cluster in clusters:
        n_pix_total = sum(c["n"] for c in cluster)
        if (len(cluster) < SIDE_MIN_FRAGMENTS_PER_LINE
                and n_pix_total < SIDE_MIN_PIXELS_FOR_SINGLETON):
            continue
        if n_pix_total < SIDE_MIN_PIXELS_PER_LINE:
            continue
        all_xs = np.concatenate([c["xs"] for c in cluster])
        all_ys = np.concatenate([c["ys"] for c in cluster])
        cx = float(all_xs.mean()); cy = float(all_ys.mean())
        x_min = int(all_xs.min()); y_min = int(all_ys.min())
        x_max = int(all_xs.max()) + 1; y_max = int(all_ys.max()) + 1
        cos_t, sin_t = _orientation_from_pixels(all_ys, all_xs)
        conf = float(prob_map[all_ys, all_xs].mean())
        out.append(_build_feature(
            type_idx=TYPE_SIDE, cx=cx, cy=cy,
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            area=int(n_pix_total), cos_t=cos_t, sin_t=sin_t,
            ngs_x_yards=0.0, has_ngs=False, confidence=conf))
        out_pix.append((all_ys, all_xs))
    if return_pixels:
        return out, out_pix
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hash (unchanged from v2 _process_simple_channel) and Number (unchanged
# from v2 _process_number_channel_spatial).
# ─────────────────────────────────────────────────────────────────────────────

# Re-use v2's hash + number processors + classifier helpers verbatim.
from cc_tokenizer_v2 import (    # noqa: E402, F401
    _process_simple_channel as _v2_process_simple_channel,
    _process_number_channel_spatial,
    null_classifier,
    make_legacy_classifier,
    make_gt_classifier,
    _make_classifier_crop,
    CLASSIFIER_CROP_SIZE,
)


def cc_tokens_from_frame_v3(
        masks: np.ndarray,
        number_classifier_fn: Callable | None = None,
        dilate_px: int = DEFAULT_DILATE_PX,
        return_aux: bool = False,
        ):
    """Tokenize one frame's v8 masks.

    Yardlines and sidelines are CC-grouped (one token per physical line).
    Hashes use v2's per-CC tokenizer. Numbers use v2's spatial-cluster
    tokenizer with dilation merge.

    Args / returns: identical to cc_tokens_from_frame_v2 (including the
    optional return_aux flag, which surfaces per-number-token top/bottom
    edge pixel coords for the H solver).
    """
    if number_classifier_fn is None:
        number_classifier_fn = null_classifier

    from cc_tokenizer_v2 import _yardline_spacing_from_mask
    yard_spacing = _yardline_spacing_from_mask(masks[..., 0])

    out_tokens: list[np.ndarray] = []
    if return_aux:
        yard_t, yard_pix = _process_yardline_channel_grouped(
            masks[..., 0], return_pixels=True)
        side_t, side_pix = _process_sideline_channel_grouped(
            masks[..., 1], return_pixels=True)
        hash_t, hash_pix = _v2_process_simple_channel(
            masks[..., 2], TYPE_HASH, return_pixels=True)
        num_t, num_edges, num_crops, num_pix = (
            _process_number_channel_spatial(
                masks[..., 3], number_classifier_fn,
                return_edges=True, yard_spacing_px=yard_spacing))
    else:
        yard_t = _process_yardline_channel_grouped(masks[..., 0])
        side_t = _process_sideline_channel_grouped(masks[..., 1])
        hash_t = _v2_process_simple_channel(masks[..., 2], TYPE_HASH)
        num_t = _process_number_channel_spatial(
            masks[..., 3], number_classifier_fn,
            yard_spacing_px=yard_spacing)
        yard_pix = side_pix = hash_pix = num_pix = None
        num_edges = num_crops = None
    out_tokens.extend(yard_t)
    out_tokens.extend(side_t)
    out_tokens.extend(hash_t)
    out_tokens.extend(num_t)

    if not out_tokens:
        tokens = np.zeros((0, TOKEN_FEATURE_DIM), dtype=np.float32)
    else:
        tokens = np.stack(out_tokens, axis=0).astype(np.float32)

    if return_aux:
        edges_arr = (np.array(num_edges, dtype=np.float32)
                       if num_edges else
                       np.zeros((0, 2, 2), dtype=np.float32))
        return tokens, {
            "num_edges": edges_arr,
            "num_crops": num_crops or [],
            "pixel_sets": {
                "yard": yard_pix,
                "side": side_pix,
                "hash": hash_pix,
                "num":  num_pix,
            },
        }
    return tokens
