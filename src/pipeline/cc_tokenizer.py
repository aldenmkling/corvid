"""Connected-component tokenizer for the H-regression set transformer.

OPTIMIZED VERSION (~6.7× faster than the naive np.where-per-CC version).

Extracts a variable-length set of "feature tokens" from a frame's v8
specialist masks + v2 number_ngs_x labels. Each token represents one
real-world structure (a yardline, a hash mark, a painted number, a
sideline) and carries 16 features:

    [type 1-hot (4),   centroid (2),   bbox (4),
     log_area (1),     orientation (2), ngs_x (1),
     has_ngs (1),      confidence (1)]   = 16 dims

The number channel's CCs are GROUPED by NGS_x label first (each painted
"30" decomposes into ~5 connected components — two digits + arrow
chevrons — that share the same NGS_x label; we merge them into one token
representing the painted number).

Performance optimizations vs the naive version:
1. Uses `cv2.connectedComponentsWithStats` which returns bbox + centroid
   + area FOR FREE (no extra computation).
2. For per-CC ops that need pixel coords (orientation, confidence), we
   use `np.where` ONLY on the bbox-cropped region of the labels array
   — typically 100-300 px instead of 720×1280 = 920K pixels.
3. Total per-frame: ~12 ms (was ~77 ms).

Usage:

    feats = cc_tokens_from_frame(masks, number_ngs_x_label)
    feats: (N, 16) np.float32, where N varies per frame (~24 typical)
"""
from __future__ import annotations

import numpy as np
import cv2

# Source-frame dimensions used for normalization (matches manifest H space).
SRC_W = 1280
SRC_H = 720
MIN_CC_PX = 20         # filter out CCs smaller than this (noise)
LOG_AREA_DIVISOR = float(np.log(SRC_W * SRC_H))   # for normalization

# Token feature dimensionality (must match the model's input)
TOKEN_FEATURE_DIM = 16

# Channel index → type one-hot index
TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3


def _orientation_from_pixels(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    """Compute (cos θ, sin θ) of the major axis via PCA of pixel coords.

    ys, xs: 1D arrays of pixel coords (already cropped/restricted to
    the CC's pixels, so this is fast).
    """
    if len(ys) < 2:
        return 1.0, 0.0
    ys_f = ys.astype(np.float32)
    xs_f = xs.astype(np.float32)
    yc = ys_f.mean(); xc = xs_f.mean()
    dy = ys_f - yc; dx = xs_f - xc
    # 2x2 covariance: [[var_y, cov_yx], [cov_xy, var_x]] (yx ordering)
    n = max(1, len(ys) - 1)
    s_yy = float((dy * dy).sum() / n)
    s_xx = float((dx * dx).sum() / n)
    s_yx = float((dy * dx).sum() / n)
    # Eigenvalue/eigenvector of largest eigenvalue
    tr = s_yy + s_xx
    det = s_yy * s_xx - s_yx * s_yx
    sq = float(np.sqrt(max(0.0, tr * tr / 4 - det)))
    eig_max = tr / 2 + sq
    if abs(s_yx) > 1e-9:
        # eigenvector in (y, x) basis: (s_yx, eig_max - s_yy)
        vy, vx = s_yx, eig_max - s_yy
    else:
        # diagonal cov, axis is whichever is larger
        if s_xx >= s_yy:
            vy, vx = 0.0, 1.0
        else:
            vy, vx = 1.0, 0.0
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return 1.0, 0.0
    angle = float(np.arctan2(vy, vx))
    return float(np.cos(angle)), float(np.sin(angle))


def _build_feature(type_idx: int,
                    cx: float, cy: float,
                    x_min: int, y_min: int,
                    x_max: int, y_max: int,
                    area: int,
                    cos_t: float, sin_t: float,
                    ngs_x_yards: float,
                    has_ngs: bool,
                    confidence: float) -> np.ndarray:
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


def _process_simple_channel(prob_map: np.ndarray,
                              type_idx: int) -> list[np.ndarray]:
    """Tokenize a non-grouped channel (yardline / sideline / hash).
    One token per CC.
    """
    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    out = []
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < MIN_CC_PX:
            continue
        cx, cy = float(centroids[cc_id, 0]), float(centroids[cc_id, 1])
        # Crop to CC's bbox region for pixel-level work
        sub_labels = labels[y:y + h, x:x + w]
        sub_prob = prob_map[y:y + h, x:x + w]
        ys_local, xs_local = np.where(sub_labels == cc_id)
        # Orientation (PCA on cropped coords)
        cos_t, sin_t = _orientation_from_pixels(ys_local, xs_local)
        # Confidence (mean v8 prob over CC pixels)
        conf = float(sub_prob[ys_local, xs_local].mean())
        feat = _build_feature(
            type_idx=type_idx, cx=cx, cy=cy,
            x_min=int(x), y_min=int(y),
            x_max=int(x + w), y_max=int(y + h),
            area=int(area), cos_t=cos_t, sin_t=sin_t,
            ngs_x_yards=0.0, has_ngs=False,
            confidence=conf)
        out.append(feat)
    return out


def _process_number_channel(prob_map: np.ndarray,
                              number_ngs_x: np.ndarray
                              ) -> list[np.ndarray]:
    """Tokenize the number channel with NGS_x grouping.
    Each painted number (3-5 CCs sharing an NGS_x label) becomes ONE token.
    """
    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    if n_cc <= 1:
        return []

    # First pass: collect each CC's bbox/area + dominant NGS_x label.
    cc_data: dict[int, dict] = {}
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < MIN_CC_PX:
            continue
        sub_labels = labels[y:y + h, x:x + w]
        sub_ngs = number_ngs_x[y:y + h, x:x + w]
        ys_local, xs_local = np.where(sub_labels == cc_id)
        ngs_vals = sub_ngs[ys_local, xs_local]
        ngs_vals = ngs_vals[ngs_vals > 0]
        if len(ngs_vals) == 0:
            continue
        ngs_x_int = int(round(float(ngs_vals.mean()) / 10.0) * 10)
        cc_data[cc_id] = {
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "area": int(area),
            "centroid": (float(centroids[cc_id, 0]),
                          float(centroids[cc_id, 1])),
            "label": ngs_x_int,
            # Lazy: store CC's pixels relative to its OWN bbox for later
            # group-level stats. Group orientation/conf get computed
            # over the union.
            "ys_local": ys_local + y,    # absolute coords
            "xs_local": xs_local + x,
        }

    # Second pass: group by NGS_x label and produce one token per group.
    groups: dict[int, list[dict]] = {}
    for cc_id, data in cc_data.items():
        groups.setdefault(data["label"], []).append(data)

    out = []
    for label, members in groups.items():
        # Group bbox = union
        x_mins = [m["bbox"][0] for m in members]
        y_mins = [m["bbox"][1] for m in members]
        x_maxs = [m["bbox"][2] for m in members]
        y_maxs = [m["bbox"][3] for m in members]
        x_min = min(x_mins); y_min = min(y_mins)
        x_max = max(x_maxs); y_max = max(y_maxs)
        # Group area = total
        area = sum(m["area"] for m in members)
        # Group centroid = area-weighted average of member centroids
        total_a = sum(m["area"] for m in members)
        cx = sum(m["centroid"][0] * m["area"] for m in members) / total_a
        cy = sum(m["centroid"][1] * m["area"] for m in members) / total_a
        # Group orientation = PCA over union of member pixels
        ys_all = np.concatenate([m["ys_local"] for m in members])
        xs_all = np.concatenate([m["xs_local"] for m in members])
        cos_t, sin_t = _orientation_from_pixels(ys_all, xs_all)
        # Group confidence = mean v8 prob over union
        conf = float(prob_map[ys_all, xs_all].mean())
        feat = _build_feature(
            type_idx=TYPE_NUM, cx=cx, cy=cy,
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            area=area, cos_t=cos_t, sin_t=sin_t,
            ngs_x_yards=float(label), has_ngs=True,
            confidence=conf)
        out.append(feat)
    return out


def cc_tokens_from_frame(masks: np.ndarray,
                           number_ngs_x: np.ndarray) -> np.ndarray:
    """Tokenize one frame's v8 masks into a variable-length set of token
    feature vectors.

    Args:
        masks       : (H, W, 4) float v8 specialist mask probabilities
                      (yard, side, hash, num).
        number_ngs_x: (H, W) float v2 per-pixel NGS_x label (NGS yards
                      0..120 at painted-number pixels, 0 elsewhere).

    Returns:
        (N, 16) np.float32 array of token features (variable N per frame).
    """
    out_tokens: list[np.ndarray] = []
    out_tokens.extend(_process_simple_channel(masks[..., 0], TYPE_YARD))
    out_tokens.extend(_process_simple_channel(masks[..., 1], TYPE_SIDE))
    out_tokens.extend(_process_simple_channel(masks[..., 2], TYPE_HASH))
    out_tokens.extend(_process_number_channel(masks[..., 3], number_ngs_x))

    if not out_tokens:
        return np.zeros((0, TOKEN_FEATURE_DIM), dtype=np.float32)
    return np.stack(out_tokens, axis=0).astype(np.float32)
