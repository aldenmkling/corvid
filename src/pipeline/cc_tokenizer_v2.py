"""Connected-component tokenizer — v2.

Same I/O contract as cc_tokenizer.py (TOKEN_FEATURE_DIM=16 token), but the
NUMBER channel is tokenized via SPATIAL CC clustering rather than NGS_x label
grouping. This fixes a bug in v1 where two different painted numbers that
happened to share the same v2-classifier per-pixel NGS_x label (e.g. a "30"
near sideline and a "30" far sideline) would be merged into a single token
with a giant bbox spanning both rows.

Pipeline for the number channel:
  1. Threshold v8 number-mask probability → binary CC.
  2. Dilate the binary mask by ≈ DILATE_PX so that the multiple CCs within
     ONE painted number (digit, second digit, two arrow chevrons) merge,
     while CCs from DIFFERENT painted numbers stay separate.
  3. Run connected components on the dilated mask → spatial clusters.
  4. Per cluster, build the canonical 64×64 binary-mask crop (the input
     format the number classifier was trained on) and call a user-provided
     classifier callable to produce NGS_x.

Classifier interface — caller passes a `number_classifier_fn`:

    fn(crops: list[np.ndarray uint8 (64, 64)]) -> tuple[
        ngs_x_yards: np.ndarray (N,) float,
        has_ngs:     np.ndarray (N,) bool,
        confidence:  np.ndarray (N,) float,
    ]

Two helpers are provided:

  null_classifier(crops)
    placeholder. Returns ngs_x=0, has_ngs=False, confidence=0 for every
    crop. Use during pipeline development before the new classifier is
    chosen.

  make_legacy_classifier(number_ngs_x_map)
    factory. Returns a fn that uses the v2 per-pixel NGS_x map (mode-vote
    across each cluster's mask pixels) — same source of truth v1 used,
    but applied per-spatial-cluster instead of as a grouping key. Lets you
    isolate the effect of the tokenizer fix from the classifier swap.

Token feature layout is unchanged:
    [type 1-hot (4),   centroid (2),   bbox (4),
     log_area (1),     orientation (2), ngs_x (1),
     has_ngs (1),      confidence (1)]   = 16 dims
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Re-export shared constants so v8/v9 trainers can `from cc_tokenizer_v2 ...`
SRC_W = 1280
SRC_H = 720
# Pass every cluster through the pipeline — classifier+SR handles noise
# via low-confidence predictions, and tiny clusters might still be
# legitimate (small hash marks, distant numbers).
MIN_CC_PX = 1
MIN_CC_PX_NUM = 1
LOG_AREA_DIVISOR = float(np.log(SRC_W * SRC_H))
TOKEN_FEATURE_DIM = 16

TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3

# Dilation radius used to merge intra-painted-number CCs into one cluster
# without bridging adjacent painted numbers. Tuned on a 494-frame subset of
# the manifest by comparing cluster-count vs GT-projected-count: dilate-28
# matches expected within 2 on 98.2% of frames (median diff 0, mean abs
# error 0.62). Smaller radii (10-14) under-merge; the digit and arrow
# chevrons of a single painted number fragment across 3-5 clusters.
# This MUST stay aligned with build_round2_dataset.py (otherwise the
# classifier's training distribution drifts from inference distribution).
DEFAULT_DILATE_PX = 28

# Drop number-token groups whose pixel-count area is significantly below
# the per-frame median group area. Painted numbers are all the same size
# in NGS, so within a single camera shot all groups should be in a
# narrow band of pixel counts (modulo near/far perspective). Groups much
# smaller than the median are almost always partial / clipped glyphs at
# the frame edge or noise CCs the clusterer left ungrouped.
NUM_AREA_MEDIAN_FRAC = 0.5

# Number-CC grouping (replaces the old morphological-dilation merge).
# Mirrors src/homography/painted_numbers.detect_groups: CCs whose centroids
# are within `NUMBER_GROUP_DIST_FRAC * median_yardline_spacing_px` of each
# other (single-link) belong to the same painted number. This adapts to
# perspective scale and survives a player standing between two digits —
# the centroids of the digits are still close even if the mask is split.
NUMBER_GROUP_DIST_FRAC = 1.0
NUMBER_GROUP_FALLBACK_PX = 100
MIN_RAW_NUM_CC_PX = 30      # drop tiny noise CCs before clustering


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (re-implemented locally to keep this module self-contained).
# ─────────────────────────────────────────────────────────────────────────────

def _orientation_from_pixels(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
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


def _process_simple_channel(prob_map: np.ndarray, type_idx: int,
                                  return_pixels: bool = False):
    """Identical to v1 — yard / side / hash channels are unchanged.

    If return_pixels=True, also returns a list of (ys, xs) per token in
    the same order as the token list.
    """
    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ([], []) if return_pixels else []
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    out = []
    pix_sets = []
    for cc_id in range(1, n_cc):
        x, y, w, h, area = stats[cc_id]
        if area < MIN_CC_PX:
            continue
        cx, cy = float(centroids[cc_id, 0]), float(centroids[cc_id, 1])
        sub_labels = labels[y:y + h, x:x + w]
        sub_prob = prob_map[y:y + h, x:x + w]
        ys_local, xs_local = np.where(sub_labels == cc_id)
        ys_abs = ys_local + y
        xs_abs = xs_local + x
        cos_t, sin_t = _orientation_from_pixels(ys_local, xs_local)
        conf = float(sub_prob[ys_local, xs_local].mean())
        feat = _build_feature(
            type_idx=type_idx, cx=cx, cy=cy,
            x_min=int(x), y_min=int(y),
            x_max=int(x + w), y_max=int(y + h),
            area=int(area), cos_t=cos_t, sin_t=sin_t,
            ngs_x_yards=0.0, has_ngs=False,
            confidence=conf)
        out.append(feat)
        pix_sets.append((ys_abs, xs_abs))
    if return_pixels:
        return out, pix_sets
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Crop construction — matches the input format the number classifier was
# trained on: 64×64 grayscale (binary 0/255), pad to square then resize
# NEAREST so the digit shape is preserved. See data/number_classifier/round1.
# ─────────────────────────────────────────────────────────────────────────────

# Painted-yardline number crops are wider than tall (median 134x33 px).
# 128x32 crop preserves the ~4:1 aspect ratio closely.
CLASSIFIER_CROP_W = 128
CLASSIFIER_CROP_H = 32
CLASSIFIER_CROP_SIZE = (CLASSIFIER_CROP_W, CLASSIFIER_CROP_H)    # (W, H)


def _make_classifier_crop(bin_mask: np.ndarray,
                            x_min: int, y_min: int,
                            x_max: int, y_max: int,
                            cluster_label_map: np.ndarray | None = None,
                            cluster_id: int | None = None) -> np.ndarray:
    """Crop the cluster's mask to a 64×64 uint8 image suitable for the
    classifier.

    MATCHES build_round3_dataset.make_crop EXACTLY: tight bbox, NO padding-
    to-square, direct INTER_NEAREST resize to 64×64 (aspect-ratio distortion
    accepted). Aligns inference distribution with training distribution.
    """
    sub = bin_mask[y_min:y_max, x_min:x_max].astype(np.uint8)
    if cluster_label_map is not None and cluster_id is not None:
        sub_lab = cluster_label_map[y_min:y_max, x_min:x_max]
        sub = sub * (sub_lab == cluster_id).astype(np.uint8)
    sub = (sub > 0).astype(np.uint8) * 255
    if sub.size == 0:
        return np.zeros((CLASSIFIER_CROP_H, CLASSIFIER_CROP_W),
                          dtype=np.uint8)
    return cv2.resize(sub, (CLASSIFIER_CROP_W, CLASSIFIER_CROP_H),
                        interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────────────────────
# Number channel — spatial-clustering tokenizer.
# ─────────────────────────────────────────────────────────────────────────────

def _process_number_channel_spatial(
        prob_map: np.ndarray,
        number_classifier_fn: Callable[[list[np.ndarray]], tuple],
        dilate_px: int = DEFAULT_DILATE_PX,    # kept for back-compat; ignored
        return_edges: bool = False,
        yard_spacing_px: float | None = None,
        ):
    """One token per painted-number group, built via CC + scale-aware
    single-link clustering on centroids (mirrors
    src/homography/painted_numbers.detect_groups).

    Unlike the older morphological-dilation merge:
      • CCs are detected on the RAW binary mask (no dilation).
      • CCs whose centroids are within ~1× median yardline spacing
        (`yard_spacing_px`) are grouped via single-link clustering. A
        player standing between two digits doesn't break this — the
        centroids are still close even if the mask is split.
      • Each group's bbox is the union of all member CC pixels, so the
        bbox actually wraps the full painted glyph instead of just the
        unbroken portion of a single CC.

    If return_edges=True, returns (tokens, edges) where edges is a list
    parallel to the returned tokens: each entry is
    ((top_x, top_y), (bot_x, bot_y)) in pixel coords. Otherwise just
    returns the token list.
    """
    del dilate_px  # retained in signature for backward compatibility only

    bin_mask = (prob_map > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ([], [], [], []) if return_edges else []

    # 1. CC on the raw mask. Drop tiny noise CCs.
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    cc_pixels: list[tuple[np.ndarray, np.ndarray]] = []
    cc_centroids: list[list[float]] = []
    for cid in range(1, n_cc):
        if int(stats[cid, cv2.CC_STAT_AREA]) < MIN_RAW_NUM_CC_PX:
            continue
        ys, xs = np.where(lbl == cid)
        cc_pixels.append((ys, xs))
        cc_centroids.append([float(xs.mean()), float(ys.mean())])

    if not cc_pixels:
        return ([], [], [], []) if return_edges else []

    cc_centroids_np = np.asarray(cc_centroids, dtype=np.float64)

    # 2. Scale-aware cluster threshold.
    if yard_spacing_px is None or yard_spacing_px <= 0:
        cluster_thr = float(NUMBER_GROUP_FALLBACK_PX)
    else:
        cluster_thr = float(NUMBER_GROUP_DIST_FRAC * yard_spacing_px)

    # 3. Single-link clustering on centroids.
    if len(cc_centroids_np) == 1:
        group_ids = np.array([1], dtype=int)
    else:
        Z = linkage(pdist(cc_centroids_np), method="single")
        group_ids = fcluster(Z, t=cluster_thr, criterion="distance")

    # 4a. First pass: build raw groups so we can compute the per-frame
    # median area and filter the small (partial / clipped) outliers.
    raw_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for gid in np.unique(group_ids):
        member = np.where(group_ids == gid)[0]
        ys_abs = np.concatenate([cc_pixels[i][0] for i in member])
        xs_abs = np.concatenate([cc_pixels[i][1] for i in member])
        if len(ys_abs) < MIN_CC_PX_NUM:
            continue
        raw_groups.append((member, ys_abs, xs_abs))

    if not raw_groups:
        return ([], [], [], []) if return_edges else []

    median_area = float(np.median([len(g[1]) for g in raw_groups]))
    area_thr = NUM_AREA_MEDIAN_FRAC * median_area

    # 4b. Per-group geometry + crop, with median-area filter.
    cluster_records: list[dict] = []
    crops: list[np.ndarray] = []
    for member, ys_abs, xs_abs in raw_groups:
        if len(ys_abs) < area_thr:
            continue
        x_min = int(xs_abs.min()); y_min = int(ys_abs.min())
        x_max = int(xs_abs.max()) + 1; y_max = int(ys_abs.max()) + 1
        area = int(len(ys_abs))
        cx = float(xs_abs.mean()); cy = float(ys_abs.mean())

        cos_t, sin_t = _orientation_from_pixels(ys_abs, xs_abs)
        conf_mask = float(prob_map[ys_abs, xs_abs].mean())

        # Robust top/bottom edges via min-area rotated rect.
        pts = np.stack([xs_abs, ys_abs], axis=-1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        corners = cv2.boxPoints(rect)
        order = np.argsort(corners[:, 1])
        top_mid = corners[order[:2]].mean(axis=0)
        bot_mid = corners[order[2:]].mean(axis=0)

        # Build the group's pixel mask from union of member CCs.
        group_mask = np.zeros_like(bin_mask, dtype=np.uint8)
        for cc_idx in member:
            gys, gxs = cc_pixels[cc_idx]
            group_mask[gys, gxs] = 1
        # Tight crop, mask-only (set non-group pixels to 0).
        sub = group_mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
        if sub.size == 0:
            crop = np.zeros(
                (CLASSIFIER_CROP_H, CLASSIFIER_CROP_W), dtype=np.uint8)
        else:
            crop = cv2.resize(sub,
                                  (CLASSIFIER_CROP_W, CLASSIFIER_CROP_H),
                                  interpolation=cv2.INTER_NEAREST)

        cluster_records.append(dict(
            cx=cx, cy=cy, x_min=x_min, y_min=y_min,
            x_max=x_max, y_max=y_max,
            area=area, cos_t=cos_t, sin_t=sin_t,
            mask_conf=conf_mask,
            top_edge_xy=(float(top_mid[0]), float(top_mid[1])),
            bot_edge_xy=(float(bot_mid[0]), float(bot_mid[1])),
            ys_abs=ys_abs, xs_abs=xs_abs,    # for legacy classifier
        ))
        crops.append(crop)

    if not cluster_records:
        return ([], [], [], []) if return_edges else []

    # Second pass: batched classification. Classifier may return either
    # (ngs_x, has_ngs, conf) — keep all clusters, or
    # (ngs_x, has_ngs, conf, keep) — drop clusters where keep=False (e.g.
    # 10-class bg-aware classifier dropping background-predicted clusters).
    result = number_classifier_fn(crops, cluster_records)
    if len(result) == 4:
        ngs_x_yards, has_ngs, conf, keep = result
    else:
        ngs_x_yards, has_ngs, conf = result
        keep = np.ones(len(cluster_records), dtype=bool)
    ngs_x_yards = np.asarray(ngs_x_yards, dtype=np.float32)
    has_ngs = np.asarray(has_ngs, dtype=bool)
    conf = np.asarray(conf, dtype=np.float32)
    keep = np.asarray(keep, dtype=bool)

    out = []
    edges = []
    out_crops = []
    out_pix = []
    for r, crop, ngs, hn, c_cls, kp in zip(cluster_records, crops,
                                                 ngs_x_yards, has_ngs,
                                                 conf, keep):
        if not kp:
            continue
        # Final token confidence = mean v8 mask prob × classifier confidence,
        # so the model can distinguish a confidently-detected, confidently-
        # classified painted number from a noisy or uncertain one.
        token_conf = float(r["mask_conf"]) * float(c_cls if hn else 1.0)
        feat = _build_feature(
            type_idx=TYPE_NUM, cx=r["cx"], cy=r["cy"],
            x_min=r["x_min"], y_min=r["y_min"],
            x_max=r["x_max"], y_max=r["y_max"],
            area=r["area"],
            cos_t=r["cos_t"], sin_t=r["sin_t"],
            ngs_x_yards=float(ngs), has_ngs=bool(hn),
            confidence=token_conf)
        out.append(feat)
        edges.append((r["top_edge_xy"], r["bot_edge_xy"]))
        out_crops.append(crop)
        out_pix.append((r["ys_abs"], r["xs_abs"]))
    if return_edges:
        return out, edges, out_crops, out_pix
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Classifier callables (placeholder + legacy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def null_classifier(crops, cluster_records):
    """Placeholder. Returns ngs_x=0, has_ngs=0, confidence=0 for every cluster.

    Use this while pipeline-developing v9 before the real number classifier is
    chosen. Number tokens carry geometry only — the encoder gets zero NGS_x
    signal and must rely on positional context (yard / hash / side neighbors).
    """
    n = len(crops)
    return (np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=np.float32))


def make_gt_classifier(H_pixel: np.ndarray, K: np.ndarray,
                         dist: np.ndarray, image_h: int = 720,
                         image_w: int = 1280):
    """Project each spatial cluster's centroid through GT H to get its NGS_x.

    Round2-style labels: same source of truth used to build the round2 number-
    classifier dataset (verified manifest H + coordinates), but applied per
    cluster at v9 training time. This is teacher-forcing — at training the
    encoder sees clean GT anchor labels; at deploy-time, swap in the trained
    round2 classifier (MBConv / DS-ResNet10w / etc.) which approximates this.

    Implementation: cluster centroid (cx, cy) is in distorted source-image
    space. Undistort to camera coords, then apply H to get NGS coords. Quantize
    NGS_x to nearest 5-yard bucket. Set has_ngs=True only if the resulting
    NGS_x lands inside the painted-number range [10, 110].
    """
    # Cache H_inv: we go FROM image TO NGS, so we apply H not H_inv. H is
    # already (undistorted_pixel → NGS_yards), so forward direction.
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_arr = np.asarray(dist, dtype=np.float64).reshape(-1)
    H = np.asarray(H_pixel, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def _fn(crops, cluster_records):
        n = len(cluster_records)
        if n == 0:
            return (np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=np.float32))
        # Cluster centroids in distorted-image space.
        pts_dist = np.asarray([r["cx"] for r in cluster_records],
                                 dtype=np.float64).reshape(-1, 1)
        pts_dist_y = np.asarray([r["cy"] for r in cluster_records],
                                   dtype=np.float64).reshape(-1, 1)
        pts_dist = np.concatenate([pts_dist, pts_dist_y], axis=1)    # (N, 2)
        # Undistort each centroid to camera/undistorted-image coords.
        und = cv2.undistortPoints(
            pts_dist.reshape(-1, 1, 2), K, dist_arr, P=K).reshape(-1, 2)
        # Apply H to get NGS coords.
        und_h = np.concatenate(
            [und, np.ones((und.shape[0], 1))], axis=1)
        ngs_h = (H @ und_h.T).T
        ngs = ngs_h[:, :2] / ngs_h[:, 2:3]
        # Quantize NGS_x to nearest 5y. Painted numbers live at NGS_x ∈
        # {20, 30, 40, 50, 60, 70, 80, 90, 100} → 9 valid buckets.
        ngs_x = ngs[:, 0]
        ngs_x_q = np.round(ngs_x / 5.0) * 5.0
        # Has_ngs only if quantized value lands in painted-number range
        # AND original projection was reasonably close (within ~3 yards of
        # nearest 5y, conservatively — clusters far from any painted number
        # mean the H-projection put the cluster nowhere meaningful).
        valid = (ngs_x_q >= 10.0) & (ngs_x_q <= 110.0) & \
                  (np.abs(ngs_x - ngs_x_q) < 3.0)
        ngs_x_out = np.where(valid, ngs_x_q, 0.0).astype(np.float32)
        has = valid.astype(bool)
        # Confidence = 1 - (deviation from quantized bucket / 2.5y), so a
        # cluster that projects to exactly NGS_x=40 has conf=1, one that's
        # off by 2y has conf=0.2. Encoder can use this to weight anchors.
        dev = np.abs(ngs_x - ngs_x_q)
        conf = np.clip(1.0 - dev / 2.5, 0.0, 1.0)
        conf = np.where(valid, conf, 0.0).astype(np.float32)
        return ngs_x_out, has, conf

    return _fn


def make_legacy_classifier(number_ngs_x_map: np.ndarray):
    """Wrap the v2-classifier per-pixel NGS_x label map (the same data v1's
    tokenizer used as a grouping KEY) into a per-cluster majority-vote
    classifier.

    Lets you isolate the effect of the tokenizer fix from the classifier
    swap: feeding v9 the legacy classifier reuses the SAME label source as
    v1 — just applied per-spatial-cluster instead of as the grouping key.
    """
    def _fn(crops, cluster_records):
        ngs_x = np.zeros(len(crops), dtype=np.float32)
        has = np.zeros(len(crops), dtype=bool)
        conf = np.zeros(len(crops), dtype=np.float32)
        for i, r in enumerate(cluster_records):
            ys_abs = r["ys_abs"]; xs_abs = r["xs_abs"]
            vals = number_ngs_x_map[ys_abs, xs_abs]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            # Mode-vote rounded to nearest 5y bucket.
            quant = np.round(vals / 5.0) * 5.0
            uniq, counts = np.unique(quant, return_counts=True)
            j = int(counts.argmax())
            ngs_x[i] = float(uniq[j])
            has[i] = True
            conf[i] = float(counts[j]) / float(len(vals))
        return ngs_x, has, conf
    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────────────

def _yardline_spacing_from_mask(yard_prob: np.ndarray) -> float | None:
    """Median x-distance between adjacent yardline-mask CC centroids.

    Used as the scale parameter for number-CC single-link clustering.
    Returns None if fewer than 2 yardline CCs are detected (caller falls
    back to NUMBER_GROUP_FALLBACK_PX).
    """
    bin_mask = (yard_prob > 0.5).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    cxs = []
    for cid in range(1, n):
        if int(stats[cid, cv2.CC_STAT_AREA]) < 50:
            continue
        ys, xs = np.where(lbl == cid)
        cxs.append(float(xs.mean()))
    if len(cxs) < 2:
        return None
    cxs.sort()
    gaps = np.diff(np.array(cxs, dtype=np.float64))
    if len(gaps) == 0:
        return None
    # In v2 yardlines are per-CC (multiple CCs per painted yardline). Drop
    # very small gaps (within-yardline neighbors) before taking the
    # median — keep gaps that are > 0.4 × overall median. In v3 yardlines
    # are already grouped (one CC per yardline), in which case this filter
    # is a no-op.
    m0 = float(np.median(gaps))
    keep = gaps > 0.4 * m0
    return float(np.median(gaps[keep])) if keep.any() else m0


def cc_tokens_from_frame_v2(
        masks: np.ndarray,
        number_classifier_fn: Callable | None = None,
        dilate_px: int = DEFAULT_DILATE_PX,
        return_aux: bool = False,
        ):
    """Tokenize one frame's v8 masks into a variable-length set of token
    feature vectors.

    Args:
        masks                : (H, W, 4) v8 specialist mask probabilities
                                (yard, side, hash, num).
        number_classifier_fn : callable returning per-cluster
                                (ngs_x_yards, has_ngs, confidence) given
                                (crops, cluster_records). If None, uses
                                ``null_classifier`` as the placeholder.
        dilate_px            : dilation radius used to merge intra-painted-
                                number CCs. Default 10 (safe for 1280×720).
        return_aux           : when True, also returns an aux dict with
                                ``num_edges``: (N_num, 2, 2) array of
                                (top_xy, bot_xy) pixel coords aligned to
                                the N_num number tokens (the trailing
                                rows of the returned token array, in the
                                order they were emitted).

    Returns:
        (N, 16) np.float32 array of token features. If return_aux, also
        returns aux dict.
    """
    if number_classifier_fn is None:
        number_classifier_fn = null_classifier

    yard_spacing = _yardline_spacing_from_mask(masks[..., 0])

    out_tokens: list[np.ndarray] = []
    if return_aux:
        yard_t, yard_pix = _process_simple_channel(
            masks[..., 0], TYPE_YARD, return_pixels=True)
        side_t, side_pix = _process_simple_channel(
            masks[..., 1], TYPE_SIDE, return_pixels=True)
        hash_t, hash_pix = _process_simple_channel(
            masks[..., 2], TYPE_HASH, return_pixels=True)
        num_t, num_edges, num_crops, num_pix = _process_number_channel_spatial(
            masks[..., 3], number_classifier_fn,
            return_edges=True, yard_spacing_px=yard_spacing)
    else:
        yard_t = _process_simple_channel(masks[..., 0], TYPE_YARD)
        side_t = _process_simple_channel(masks[..., 1], TYPE_SIDE)
        hash_t = _process_simple_channel(masks[..., 2], TYPE_HASH)
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
