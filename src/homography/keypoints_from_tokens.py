"""Token-pixel-sets → keypoint correspondences.

Inputs from the tokenizer's aux dict:
  - per-token (ys, xs) pixel sets for each of yard / side / hash / num
  - per-num-token OBB midpoints (top_xy, bot_xy)
And from the classifier chain:
  - per-token class predictions

Per-class pixel pools are formed by concatenating the pixel sets of all
tokens predicted as that class. Lines are fit from those pooled pixels
exactly the way the classical pipeline does it.

Output: list of correspondences compatible with `KeypointTrackBank`.
"""
from __future__ import annotations

import numpy as np
import cv2

from .field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH
from .painted_numbers import NGS_Y_NEAR_INSIDE, NGS_Y_FAR_INSIDE


NGS_X_CLASS_TO_YARDS = {i: 10.0 + 5.0 * i for i in range(21)}
_NUMBER_NEAR_Y = NGS_Y_NEAR_INSIDE
_NUMBER_FAR_Y = NGS_Y_FAR_INSIDE

# Number-edge K-from-extreme (matches src/homography/painted_numbers
# .PER_GROUP_K_EXTREMES = 5): the inside edge is taken as the K-th most
# extreme pixel along the tangent direction. K stray outlier pixels at
# the extreme are ignored, so the edge sits "K pixels in from the worst-
# case pixel" robustly.
NUM_EDGE_K_EXTREMES = 5
# Reject number keypoints that fall this many pixels outside the
# number-token's pixel bbox (yardline / tangent went degenerate).
NUM_KP_BBOX_MARGIN_PX = 30

# Yardline-fit sanity: the fitted line is required to pass through at
# least YARD_FIT_MIN_MASK_COVERAGE of yardline-mask pixels along its
# length, sampled every YARD_FIT_SAMPLE_STEP_PX pixels in y. A line
# fitted to two merged yardlines runs DOWN THE GAP between them, so
# mask coverage along it drops sharply.
YARD_FIT_SAMPLE_STEP_PX = 2
YARD_FIT_HIT_RADIUS_PX = 1
YARD_FIT_MIN_MASK_COVERAGE = 0.70

# Yardline class-vs-position consistency: after fitting each yardline,
# its (predicted-class, image-x-at-center-y) pair should land on a
# roughly linear curve. A yardline whose label puts it far from that
# curve has a wrong class (the classifier "stretched" or "pulled apart"
# the spacing). Iteratively drop the worst offender until all residuals
# are below the threshold.
YARD_SPACING_RESIDUAL_PX = 25.0
YARD_SPACING_MIN_FITS = 3


# ── line fits ───────────────────────────────────────────────────────────────

def _fit_yardline_xy(ys, xs, min_pts=20):
    if len(ys) < min_pts:
        return None
    if float(np.var(ys)) < 1.0:
        return None
    b, a = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 1)
    return {"a": float(a), "b": float(b),
              "ymin": float(ys.min()), "ymax": float(ys.max()),
              "n": int(len(ys))}


def _fit_sideline_xy(ys, xs, min_pts=20):
    if len(xs) < min_pts:
        return None
    if float(np.var(xs)) < 1.0:
        return None
    b, a = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
    return {"a": float(a), "b": float(b),
              "xmin": float(xs.min()), "xmax": float(xs.max()),
              "n": int(len(xs))}


def _fit_hash_row_xy(ys, xs, min_pts=20):
    if len(xs) < min_pts:
        return None
    if float(np.var(xs)) < 1.0:
        return None
    m, c = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
    return {"m": float(m), "c": float(c),
              "xmin": float(xs.min()), "xmax": float(xs.max()),
              "n": int(len(xs))}


def _obb_top_bot(ys, xs):
    if len(ys) < 8:
        return None
    pts = np.stack([xs, ys], axis=-1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    corners = cv2.boxPoints(rect)
    order = np.argsort(corners[:, 1])
    top_mid = corners[order[:2]].mean(axis=0)
    bot_mid = corners[order[2:]].mean(axis=0)
    return top_mid, bot_mid


def _intersect_yard_with_horizontal(yard, m, c):
    denom = 1.0 - yard["b"] * m
    if abs(denom) < 1e-9:
        return None
    x = (yard["a"] + yard["b"] * c) / denom
    y = m * x + c
    return (x, y)


def _intersect_yard_with_sideline(yard, side):
    denom = 1.0 - yard["b"] * side["b"]
    if abs(denom) < 1e-9:
        return None
    x = (yard["a"] + yard["b"] * side["a"]) / denom
    y = side["a"] + side["b"] * x
    return (x, y)


def _line_mask_coverage(fit, yard_mask,
                              step_px=YARD_FIT_SAMPLE_STEP_PX,
                              radius=YARD_FIT_HIT_RADIUS_PX,
                              mask_thr=0.5):
    """Fraction of samples along a yardline fit `x = a + b·y` (in
    [ymin, ymax]) whose `radius`-neighborhood in `yard_mask` exceeds
    `mask_thr`. Used by bandaid 2 in `extract_keypoints`."""
    H, W = yard_mask.shape[:2]
    ymin = int(max(0, fit["ymin"]))
    ymax = int(min(H - 1, fit["ymax"]))
    if ymax <= ymin:
        return 0.0
    ys_check = np.arange(ymin, ymax + 1, step_px)
    xs_check = fit["a"] + fit["b"] * ys_check
    hit = 0; total = 0
    for y, x in zip(ys_check, xs_check):
        y = int(y); x = int(x)
        if not (0 <= y < H and 0 <= x < W):
            continue
        total += 1
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        if float(yard_mask[y0:y1, x0:x1].max()) > mask_thr:
            hit += 1
    if total == 0:
        return 0.0
    return hit / total


def _pool_pixels(pix_sets, classes, target_cls):
    """Concatenate (ys, xs) over all tokens with `classes == target_cls`."""
    ys_list = []; xs_list = []
    for i, cls in enumerate(classes):
        if int(cls) != int(target_cls):
            continue
        ys, xs = pix_sets[i]
        if len(ys) == 0:
            continue
        ys_list.append(ys); xs_list.append(xs)
    if not ys_list:
        return None
    return np.concatenate(ys_list), np.concatenate(xs_list)


def extract_keypoints(
    pixel_sets,           # dict with keys "yard","side","hash","num", each
                          # a list of (ys, xs) arrays (one per token)
    yard_classes,         # list[int] length = N_yard, 21-class NGS_x
    side_rows,            # list[int] length = N_side, 0/1
    hash_classes,         # list[int] length = N_hash, 21-class NGS_x
    hash_rows,            # list[int] length = N_hash, 0/1
    num_classes,          # list[int] length = N_num, 21-class NGS_x (RFB)
    num_rows,             # list[int] length = N_num, 0/1
    yard_mask=None,       # optional (H, W) float yardline probability map
                          # — enables bandaid 2 (mask-coverage check).
):
    """Fit lines from per-token pixel sets pooled by predicted class.
    Emit keypoints at intersections.

    Yardline-fit sanity (when `yard_mask` is provided): require the
    fitted line to pass through ≥ YARD_FIT_MIN_MASK_COVERAGE of
    yardline-mask pixels along its length. Catches the case where the
    tokenizer merges two adjacent yardlines into one token — the fit
    runs DOWN THE GAP between them, so mask coverage along it drops
    sharply.
    """

    yard_pix  = pixel_sets.get("yard")  or []
    side_pix  = pixel_sets.get("side")  or []
    hash_pix  = pixel_sets.get("hash")  or []
    num_pix   = pixel_sets.get("num")   or []

    yard_classes = list(yard_classes)
    side_rows    = list(side_rows)
    hash_classes = list(hash_classes)
    hash_rows    = list(hash_rows)
    num_classes  = list(num_classes)
    num_rows     = list(num_rows)

    # 1. Yardline fits — one per NGS_x class with enough pixels.
    # Sanity 1: drop fits with low yardline-mask coverage (the fit went
    # down the gap between two merged yardlines).
    yard_fits = {}
    yard_fit_drops = {"low_coverage": 0, "bad_spacing": 0}
    for cls in set(int(c) for c in yard_classes):
        pool = _pool_pixels(yard_pix, yard_classes, cls)
        if pool is None:
            continue
        ys, xs = pool
        fit = _fit_yardline_xy(ys, xs)
        if fit is None:
            continue
        if yard_mask is not None:
            cov = _line_mask_coverage(fit, yard_mask)
            fit["mask_coverage"] = float(cov)
            if cov < YARD_FIT_MIN_MASK_COVERAGE:
                yard_fit_drops["low_coverage"] += 1
                continue
        yard_fits[cls] = fit

    # Sanity 2: yardline (class, image-x-at-mid-y) pairs should lie on
    # a roughly linear curve. Iteratively drop the worst outlier until
    # all residuals are below YARD_SPACING_RESIDUAL_PX or we have too
    # few fits left to fit. Skip if fewer than YARD_SPACING_MIN_FITS.
    H_img = int(yard_mask.shape[0]) if yard_mask is not None else 720
    mid_y = H_img / 2.0
    while len(yard_fits) >= YARD_SPACING_MIN_FITS:
        classes = np.array(sorted(yard_fits.keys()), dtype=np.float64)
        xs_mid = np.array([yard_fits[int(c)]["a"]
                                  + yard_fits[int(c)]["b"] * mid_y
                                for c in classes], dtype=np.float64)
        # Linear fit x = m * cls + c0.
        A = np.stack([classes, np.ones_like(classes)], axis=1)
        m_fit, c0_fit = np.linalg.lstsq(A, xs_mid, rcond=None)[0]
        pred = m_fit * classes + c0_fit
        residuals = np.abs(xs_mid - pred)
        worst_i = int(np.argmax(residuals))
        if residuals[worst_i] <= YARD_SPACING_RESIDUAL_PX:
            break
        bad_cls = int(classes[worst_i])
        del yard_fits[bad_cls]
        yard_fit_drops["bad_spacing"] += 1

    # 2. Sideline fits — one per row.
    side_fits = {}
    for row in (0, 1):
        pool = _pool_pixels(side_pix, side_rows, row)
        if pool is None:
            continue
        ys, xs = pool
        fit = _fit_sideline_xy(ys, xs)
        if fit is not None:
            side_fits[row] = fit

    # 3. Hash row line fits — only used to give the number tangent a
    # robust slope direction. Hash keypoints themselves come from per-
    # hash centroids (below), NOT from this line fit.
    hash_row_lines = {}
    for row in (0, 1):
        pool = _pool_pixels(hash_pix, hash_rows, row)
        if pool is None:
            continue
        ys, xs = pool
        fit = _fit_hash_row_xy(ys, xs)
        if fit is not None:
            hash_row_lines[row] = fit

    # Tangent slope = slope of the nearest USABLE field line for that
    # row (hash row line preferred, sideline fallback). Both are global
    # fits over many pixels so the direction is stable.
    def _pick_row_slope(row):
        hl = hash_row_lines.get(row)
        if hl is not None:
            return float(hl["m"])
        sl = side_fits.get(row)
        if sl is not None:
            return float(sl["b"])    # y = a + b*x → slope is b
        return None

    num_row_slopes = {0: _pick_row_slope(0), 1: _pick_row_slope(1)}

    # ── Emit correspondences ──
    correspondences = []

    # 4. Per-hash keypoints. One keypoint per detected hash token,
    # at the hash mark's centroid, snapped onto its yardline fit (if
    # available) for sub-pixel x consistency.
    for i, (cls, row) in enumerate(zip(hash_classes, hash_rows)):
        cls = int(cls); row = int(row)
        if cls not in NGS_X_CLASS_TO_YARDS:
            continue
        ys, xs = hash_pix[i]
        if len(ys) == 0:
            continue
        cx = float(xs.mean()); cy = float(ys.mean())
        yfit = yard_fits.get(cls)
        if yfit is not None:
            # Snap centroid onto the yardline x at this y.
            px = yfit["a"] + yfit["b"] * cy
        else:
            px = cx
        ngs_x = NGS_X_CLASS_TO_YARDS[cls]
        ngs_y = HASH_Y_NEAR if row == 0 else HASH_Y_FAR
        kind = "near_hash" if row == 0 else "far_hash"
        correspondences.append({
            "pixel_u": np.array([px, cy], dtype=np.float64),
            "field": np.array([ngs_x, ngs_y], dtype=np.float64),
            "kind": kind, "source": "per_hash_centroid",
        })

    # 5. Yardline × sideline.
    for cls, yfit in yard_fits.items():
        if cls not in NGS_X_CLASS_TO_YARDS:
            continue
        ngs_x = NGS_X_CLASS_TO_YARDS[cls]
        for row, kind, ngs_y in ((0, "sideline_near", 0.0),
                                       (1, "sideline_far",  FIELD_WIDTH)):
            sfit = side_fits.get(row)
            if sfit is None: continue
            inter = _intersect_yard_with_sideline(yfit, sfit)
            if inter is None: continue
            correspondences.append({
                "pixel_u": np.array(inter, dtype=np.float64),
                "field": np.array([ngs_x, ngs_y], dtype=np.float64),
                "kind": kind, "source": "yard×sideline",
            })

    # 6. Per-NUMBER-TOKEN tangent × yardline.
    # Each individual digit gets its own keypoint based on its own
    # predicted row + NGS_x class. Same painted-yardline can produce
    # TWO keypoints (near + far) if both rows of the number are
    # detected.
    #   row 0 (bottom row in image, near sideline) → top of digit
    #     (smallest c in y = slope*x + c).
    #   row 1 (top row in image, far sideline)    → bottom of digit
    #     (largest c).
    per_num_anchors = []   # for viz: (cls, row, slope, c, ys, xs)
    for i, (cls, row) in enumerate(zip(num_classes, num_rows)):
        cls = int(cls); row = int(row)
        if cls not in NGS_X_CLASS_TO_YARDS:
            continue
        slope = num_row_slopes.get(row)
        if slope is None:
            continue
        yfit = yard_fits.get(cls)
        if yfit is None:
            continue
        ys, xs = num_pix[i]
        if len(ys) < NUM_EDGE_K_EXTREMES + 2:
            continue
        c_vals = (ys.astype(np.float64) -
                       slope * xs.astype(np.float64))
        # K-from-extreme inside-edge — matches the classical pipeline's
        # `edge_keypoint_per_group` (painted_numbers.py).
        K = min(NUM_EDGE_K_EXTREMES, max(0, len(c_vals) - 1))
        c_sorted = np.sort(c_vals)
        if row == 0:    # bottom row (near) → TOP of digit = K-th from min
            c = float(c_sorted[K])
        else:           # top row (far)    → BOTTOM of digit = K-th from max
            c = float(c_sorted[-(K + 1)])
        per_num_anchors.append((cls, row, slope, c, ys, xs))
        inter = _intersect_yard_with_horizontal(yfit, slope, c)
        if inter is None:
            continue
        # Sanity guard: keypoint should land near the digit's pixel bbox.
        x_kp, y_kp = inter
        if not (xs.min() - NUM_KP_BBOX_MARGIN_PX <= x_kp <= xs.max() + NUM_KP_BBOX_MARGIN_PX
                  and ys.min() - NUM_KP_BBOX_MARGIN_PX <= y_kp <= ys.max() + NUM_KP_BBOX_MARGIN_PX):
            continue
        ngs_x = NGS_X_CLASS_TO_YARDS[cls]
        ngs_y = _NUMBER_NEAR_Y if row == 0 else _NUMBER_FAR_Y
        kind = "number_near" if row == 0 else "number_far"
        correspondences.append({
            "pixel_u": np.array(inter, dtype=np.float64),
            "field": np.array([ngs_x, ngs_y], dtype=np.float64),
            "kind": kind, "source": "yard×num_tangent_per_token",
        })

    fits = {
        "yard_fits": yard_fits,
        "yard_fit_drops": yard_fit_drops,
        "side_fits": side_fits,
        "hash_row_lines": hash_row_lines,
        "num_row_slopes": num_row_slopes,
        "per_num_anchors": per_num_anchors,
    }
    return correspondences, fits
