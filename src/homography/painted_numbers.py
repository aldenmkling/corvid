"""Painted-yardline-number detection + keypoint extraction.

Pipeline (per frame, all in undistorted-image space):
  1. predict_mask              — number UNet → binary mask
  2. detect_groups             — CC + scale-aware single-link clustering on
                                  centroids (cluster threshold = 1× median
                                  yardline spacing)
  3. assign_yardline           — per group, find closest yardline by
                                  perpendicular distance to its parametric fit
  4. NumberSideTracker.assign  — stabilize near/far across frames:
       • 2 groups at one yardline → relative image-y assignment is gold
       • 1 group + tracker history → match to closest tracked slot by image-y
       • 1 group + no history → per-frame classify (hash row → yardline mid)
  5. edge_keypoint_per_group   — one keypoint per painted number via 1D
                                  projection along perpendicular-to-hash-row
                                  direction (sideline fallback). K-th-from-
                                  extreme pixel value is the edge (robust
                                  to K stray pixels per group). NO global
                                  line fit: each painted number's edge is
                                  independently observed, no shared-line
                                  slope jitter.

Each keypoint's NGS coords are exact by construction:
  near: (yardline_x_ngs, NUMBER_Y_NEAR + 1)  = top of near digit toward field center  = 14.0
  far:  (yardline_x_ngs, NUMBER_Y_FAR  - 1)  = bottom of far digit toward field center = 39.33
"""

import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from src.homography.field_model import NUMBER_Y_NEAR, NUMBER_Y_FAR

# Inside-edge NGS y values (digits are 2yd tall, centered at NUMBER_Y_*)
NGS_Y_NEAR_INSIDE = NUMBER_Y_NEAR + 1.0   # 14.0   (top of near digit, toward field center)
NGS_Y_FAR_INSIDE = NUMBER_Y_FAR - 1.0     # 39.33 (bottom of far digit, toward field center)

# Tunables
PRED_THR = 0.5
MIN_CC_AREA = 30
GROUP_DIST_FRAC = 1.0
GROUP_DIST_FALLBACK_PX = 100
YARDLINE_PROX_FRAC = 0.6
YARDLINE_PROX_FALLBACK_PX = 60
# Per-group inside-edge: K outlier pixels at the extreme are ignored, so the
# edge sits "K pixels in from the worst-case pixel" robustly.
PER_GROUP_K_EXTREMES = 5

INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Number classifier (g_index → absolute NGS x via 9-class label) ─────────
CLASSIFIER_CLASSES = ["10L", "10R", "20L", "20R",
                       "30L", "30R", "40L", "40R", "50"]
NGS_X_BY_LABEL = {
    "10L": 20.0, "20L": 30.0, "30L": 40.0, "40L": 50.0, "50": 60.0,
    "40R": 70.0, "30R": 80.0, "20R": 90.0, "10R": 100.0,
}
CLS_INPUT_SIZE = 64
CLS_PIXEL_MEAN = 0.456
CLS_PIXEL_STD = 0.224
YD_PER_GRID_DEFAULT = 5.0
# G0Estimator defaults — tuned for stride-10 sampling in pre-scan
G0_FREEZE_THRESH = 5.0
G0_FREEZE_MARGIN = 2.0
G0_SAMPLE_STRIDE = 10
# Alternate freeze: enough unanimous votes with literally no runner-up.
# Captures clips with only one painted number visible per frame (e.g. close-up
# action where the camera misses most of the field).
G0_UNANIMOUS_MIN_FRAMES = 4
G0_UNANIMOUS_MIN_SCORE = 2.5


# ── Number UNet inference ──────────────────────────────────────────────────
_MODEL_CACHE = {}


def _load_unet(weights: str, device: torch.device):
    key = (weights, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                  in_channels=3, classes=1, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.to(device).eval()
    _MODEL_CACHE[key] = m
    return m


@torch.no_grad()
def predict_mask(frame_bgr: np.ndarray, weights: str,
                  device_str: str = "mps") -> np.ndarray:
    """Returns binary mask (uint8 0/255) at frame resolution. Run on the
    (distorted) source frame — this UNet was trained on raw camera output."""
    device = torch.device(device_str)
    model = _load_unet(weights, device)
    h0, w0 = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (INPUT_W, INPUT_H))
    x = (rgb_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (cv2.resize(prob, (w0, h0)) > PRED_THR).astype(np.uint8) * 255


# ── Geometry helpers ───────────────────────────────────────────────────────
def median_yardline_spacing_px(yardline_fits, y_ref):
    if len(yardline_fits) < 2:
        return None
    xs = sorted(f["a"] + f["b"] * y_ref for f in yardline_fits)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    return float(np.median(gaps))


def perp_dist_to_yardline(pt, fit):
    x, y = pt
    return abs(x - (fit["a"] + fit["b"] * y)) / np.sqrt(1 + fit["b"] ** 2)


def cluster_centroids(centroids, dist_thr):
    if len(centroids) <= 1:
        return np.arange(len(centroids), dtype=int)
    return fcluster(linkage(pdist(centroids), method="single"),
                     t=dist_thr, criterion="distance")


# ── Per-frame detection + grouping ─────────────────────────────────────────
def detect_groups(mask: np.ndarray, yardline_fits, image_h: int):
    """CC analysis + scale-aware single-link clustering. Returns
    (groups, cc_pixels, yl_prox_thr).

    groups: list of {id, cc_indices, centroid, ys_all, xs_all}
    cc_pixels: list of (ys, xs) tuples, one per CC (indexed by cc_indices)
    yl_prox_thr: threshold for yardline-proximity gating (in px)
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cc_pixels = []
    cc_centroids = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < MIN_CC_AREA:
            continue
        ys, xs = np.where(lbl == i)
        cc_pixels.append((ys, xs))
        cc_centroids.append([xs.mean(), ys.mean()])
    cc_centroids = np.array(cc_centroids) if cc_centroids else np.empty((0, 2))

    spacing = median_yardline_spacing_px(yardline_fits, image_h / 2)
    cluster_thr = (GROUP_DIST_FRAC * spacing) if spacing else GROUP_DIST_FALLBACK_PX
    yl_prox_thr = (YARDLINE_PROX_FRAC * spacing) if spacing else YARDLINE_PROX_FALLBACK_PX

    group_ids = cluster_centroids(cc_centroids, cluster_thr)
    groups = []
    for gid in (np.unique(group_ids) if len(group_ids) else []):
        idx = np.where(group_ids == gid)[0]
        ys_all = np.concatenate([cc_pixels[i][0] for i in idx])
        xs_all = np.concatenate([cc_pixels[i][1] for i in idx])
        groups.append({
            "id": int(gid),
            "cc_indices": idx.tolist(),
            "centroid": (float(xs_all.mean()), float(ys_all.mean())),
            "ys_all": ys_all, "xs_all": xs_all,
        })
    return groups, cc_pixels, yl_prox_thr


def assign_yardline(groups, yardline_fits, yl_prox_thr):
    """In-place: each group gets 'yardline_idx' (-1 if none within threshold)
    and 'yardline_fit' (None if -1)."""
    for grp in groups:
        gc = grp["centroid"]
        best = None
        for j, f in enumerate(yardline_fits):
            d = perp_dist_to_yardline(gc, f)
            if best is None or d < best[1]:
                best = (j, d, f)
        if best is None or best[1] > yl_prox_thr:
            grp["yardline_idx"] = -1
            grp["yardline_fit"] = None
        else:
            grp["yardline_idx"] = best[0]
            grp["yardline_fit"] = best[2]


def classify_side_per_frame(group_centroid, yardline_fit, hash_rows, image_h):
    """Returns 'near' or 'far'. Hash row > yardline midline > image half."""
    x_g, y_g = group_centroid
    near_y = far_y = None
    if hash_rows.get("near") is not None:
        m, c = hash_rows["near"]
        near_y = m * x_g + c
    if hash_rows.get("far") is not None:
        m, c = hash_rows["far"]
        far_y = m * x_g + c
    if near_y is not None and y_g > near_y:
        return "near"
    if far_y is not None and y_g < far_y:
        return "far"
    if yardline_fit is not None:
        y_mid = (yardline_fit["ymin"] + yardline_fit["ymax"]) * 0.5
        return "near" if y_g > y_mid else "far"
    return "near" if y_g > image_h * 0.5 else "far"


# ── Cross-frame side tracker ───────────────────────────────────────────────
class NumberSideTracker:
    """Per yardline g_index, two slots (near, far). Side identity carries
    across frames; new detections match to nearest slot by image-y."""

    def __init__(self):
        # tracks[g_index] = {"near": last_image_y, "far": last_image_y}
        self.tracks = defaultdict(dict)

    def assign(self, groups, classify_per_frame_fn):
        """In-place: each group with 'yardline_g_index' >= 0 gets a 'side'
        field. Groups with yardline_g_index == -1 fall back to per-frame
        classify."""
        by_yl = defaultdict(list)
        unassigned = []
        for grp in groups:
            g_idx = grp.get("yardline_g_index", -1)
            if g_idx == -1:
                unassigned.append(grp)
            else:
                by_yl[g_idx].append(grp)

        for grp in unassigned:
            grp["side"] = classify_per_frame_fn(grp)

        for g_idx, grps in by_yl.items():
            track = self.tracks[g_idx]
            if len(grps) >= 2:
                # Relative assignment is the gold-standard signal whenever 2+
                # groups exist at the same yardline. Smaller image-y → far.
                grps_sorted = sorted(grps, key=lambda g: g["centroid"][1])
                grps_sorted[0]["side"] = "far"
                grps_sorted[-1]["side"] = "near"
                # Any extras (rare) get classified individually
                for grp in grps_sorted[1:-1]:
                    grp["side"] = classify_per_frame_fn(grp)
            else:  # exactly 1 group at this yardline
                grp = grps[0]
                candidates = [(s, y) for s, y in track.items() if y is not None]
                if candidates:
                    closest = min(candidates,
                                    key=lambda t: abs(t[1] - grp["centroid"][1]))
                    grp["side"] = closest[0]
                else:
                    grp["side"] = classify_per_frame_fn(grp)
            for grp in grps:
                track[grp["side"]] = grp["centroid"][1]


# ── Per-group inside-edge keypoint ─────────────────────────────────────────
def get_reference_slope(side: str, hash_rows: dict, sideline_fits,
                          image_w: int, image_h: int):
    """Image-space slope of 'constant NGS y' lines on this side. Hash row
    is the closest NGS-y line to the painted number row → best reference.
    Sideline is the fallback when hashes aren't detected. Returns None if
    neither is available.

    sideline_fits format: list of {"a", "b", ...} where the line equation is
    y = a + b·x. So sideline slope = b.
    """
    if side == "far":
        if hash_rows.get("far") is not None:
            return float(hash_rows["far"][0])     # m of y = m·x + c
        for sf in sideline_fits or []:
            y_at_center = sf["a"] + sf["b"] * (image_w / 2)
            if y_at_center < image_h / 2:           # top half = far sideline
                return float(sf["b"])
    else:                                            # near
        if hash_rows.get("near") is not None:
            return float(hash_rows["near"][0])
        for sf in sideline_fits or []:
            y_at_center = sf["a"] + sf["b"] * (image_w / 2)
            if y_at_center > image_h / 2:           # bottom half = near sideline
                return float(sf["b"])
    return None


def _keypoint_within_group_bbox(x_kp, y_kp, grp, margin_px=30):
    """Sanity guard: the keypoint should land within (margin of) the group's
    pixel bounding box. If it's much further away, the geometry has gone
    degenerate (yardline tracker confused, mask noise, near-parallel lines)
    and emitting the keypoint would just corrupt downstream H solving."""
    xs, ys = grp["xs_all"], grp["ys_all"]
    return (xs.min() - margin_px <= x_kp <= xs.max() + margin_px
            and ys.min() - margin_px <= y_kp <= ys.max() + margin_px)


def edge_keypoint_per_group(grp, cc_pixels, m_ref,
                              k_extremes=PER_GROUP_K_EXTREMES):
    """Per-group inside-edge keypoint via 1D projection along the
    perpendicular-to-hash-row direction.

    For each pixel in the group, compute its perp-line intercept:
        d_i = y_i - m_ref · x_i
    where m_ref is the hash row slope (or sideline slope as fallback).
    Lines of constant d_i are parallel to the hash row; d_i indexes them
    by intercept. Larger d means larger image-y (toward field center for
    far; away from field center for near).

    Take the K-th-from-extreme d as the edge (robust to K stray pixels
    per group). Intersect the constant-d line (y = m·x + d) with the
    yardline (x = a + b·y):
        denom = 1 - m·b
        x_kp = (a + b·d_edge) / denom
        y_kp = (m·a + d_edge) / denom

    Returns image-space (x_kp, y_kp), or None if the group has no yardline
    fit, no reference slope, or the lines are degenerate.
    """
    fit = grp.get("yardline_fit")
    side = grp.get("side")
    if fit is None or side not in ("far", "near") or m_ref is None:
        return None
    a, b = fit["a"], fit["b"]

    all_xs, all_ys = [], []
    for ci in grp["cc_indices"]:
        ys, xs = cc_pixels[ci]
        all_xs.append(xs); all_ys.append(ys)
    if not all_xs:
        return None
    all_xs = np.concatenate(all_xs).astype(np.float64)
    all_ys = np.concatenate(all_ys).astype(np.float64)

    d = all_ys - m_ref * all_xs
    K = min(k_extremes, max(0, len(d) - 1))
    sorted_d = np.sort(d)
    if side == "far":
        d_edge = float(sorted_d[-(K + 1)])     # K-th from max
    else:                                        # near
        d_edge = float(sorted_d[K])              # K-th from min

    denom = 1.0 - m_ref * b
    if abs(denom) < 1e-9:
        return None
    x_kp = (a + b * d_edge) / denom
    y_kp = (m_ref * a + d_edge) / denom
    if not _keypoint_within_group_bbox(x_kp, y_kp, grp):
        return None
    return float(x_kp), float(y_kp)




# ── Top-level convenience ──────────────────────────────────────────────────
def process_frame(num_mask: np.ndarray,
                    yardline_fits, hash_rows, sideline_fits,
                    g_index_per_yardline,
                    image_h: int, image_w: int,
                    tracker: "NumberSideTracker"):
    """Full per-frame pipeline. Inputs are in undistorted-image space.

    Reference slope for the inside-edge tangent comes from same-frame hash
    rows (preferred) or sidelines (fallback). Per-CC tangent points are the
    pixels with extreme projection along perp-to-ref-slope; RANSAC fits the
    final tangent line through those candidates.

    Returns (keypoints, debug):
      keypoints: list of {image_xy, ngs_y, yardline_idx, side} ready to add
                  as homography correspondences (NGS x = g0 + 5*g_index).
      debug: groups, cc_pixels, far_line, near_line, ref slopes, yl_prox_thr.
    """
    groups, cc_pixels, yl_prox_thr = detect_groups(num_mask, yardline_fits, image_h)
    assign_yardline(groups, yardline_fits, yl_prox_thr)
    for grp in groups:
        if grp["yardline_idx"] >= 0 and grp["yardline_idx"] < len(g_index_per_yardline):
            grp["yardline_g_index"] = int(g_index_per_yardline[grp["yardline_idx"]])
        else:
            grp["yardline_g_index"] = -1

    def per_frame_classify(grp):
        return classify_side_per_frame(grp["centroid"], grp["yardline_fit"],
                                         hash_rows, image_h)
    tracker.assign(groups, per_frame_classify)

    # Per-group inside-edge keypoints — one keypoint per painted number group.
    m_ref_far = get_reference_slope("far", hash_rows, sideline_fits,
                                       image_w, image_h)
    m_ref_near = get_reference_slope("near", hash_rows, sideline_fits,
                                        image_w, image_h)
    keypoints = []
    for grp in groups:
        if grp.get("yardline_idx", -1) < 0:
            continue
        side = grp.get("side")
        if side not in ("far", "near"):
            continue
        m_ref = m_ref_far if side == "far" else m_ref_near
        pt = edge_keypoint_per_group(grp, cc_pixels, m_ref)
        if pt is None:
            continue
        x, y = pt
        if not (0 <= x < image_w and 0 <= y < image_h):
            continue
        ngs_y = NGS_Y_FAR_INSIDE if side == "far" else NGS_Y_NEAR_INSIDE
        keypoints.append({
            "image_xy": pt,
            "ngs_y": ngs_y,
            "yardline_idx": grp["yardline_idx"],
            "side": side,
        })

    return keypoints, {
        "groups": groups, "cc_pixels": cc_pixels,
        "m_ref_far": m_ref_far, "m_ref_near": m_ref_near,
        "yl_prox_thr": yl_prox_thr,
    }


# ── Number classifier: maps g_index → absolute NGS x via painted label ─────
class _MitClassifier(torch.nn.Module):
    """smp mit_b0 encoder + GAP + linear head. Same architecture as
    train_number_classifier.MitClassifier (kept private to this module)."""

    def __init__(self, encoder_name="mit_b0", num_classes=9, in_channels=1):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=None)
        feat_dim = self.encoder.out_channels[-1]
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[-1])


_CLASSIFIER_CACHE = {}


def load_classifier(weights: str, device: torch.device):
    """Returns (model, classes_list). Cached per (weights, device)."""
    key = (weights, str(device))
    if key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[key]
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m = _MitClassifier()
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    classes = ckpt.get("classes", CLASSIFIER_CLASSES)
    _CLASSIFIER_CACHE[key] = (m, classes)
    return m, classes


def crop_group_to_64(grp, image_h: int, image_w: int,
                       margin_px: int = 5,
                       size: int = CLS_INPUT_SIZE) -> np.ndarray:
    """Render a group's CC pixels as a 64×64 binary mask, padded square then
    resized. Same transform as the classifier's training data. Returns None
    if the group has no pixels."""
    xs, ys = grp["xs_all"], grp["ys_all"]
    if len(xs) == 0:
        return None
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[ys, xs] = 255
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(image_w, int(xs.max()) + margin_px + 1)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(image_h, int(ys.max()) + margin_px + 1)
    crop = mask[y0:y1, x0:x1]
    h_c, w_c = crop.shape
    if h_c > w_c:
        pad = h_c - w_c
        crop = np.pad(crop, ((0, 0), (pad // 2, pad - pad // 2)), mode='constant')
    elif w_c > h_c:
        pad = w_c - h_c
        crop = np.pad(crop, ((pad // 2, pad - pad // 2), (0, 0)), mode='constant')
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def classify_crops(crops, classifier, device: torch.device,
                     classes=CLASSIFIER_CLASSES):
    """Single batched forward. Returns (labels, confidences)."""
    if not crops:
        return [], []
    arr = np.stack(crops, axis=0).astype(np.float32) / 255.0
    arr = (arr - CLS_PIXEL_MEAN) / CLS_PIXEL_STD
    x = torch.from_numpy(arr).unsqueeze(1).to(device)
    logits = classifier(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    confs = probs[np.arange(len(pred_idx)), pred_idx]
    return [classes[i] for i in pred_idx], [float(c) for c in confs]


class G0Estimator:
    """Accumulates classifier votes for the clip-level g0_NGS_x anchor.

    Each painted-number group casts a vote via:
        g0_vote = NGS_X_BY_LABEL[label] − YD_PER_GRID · g_index
    Per frame:
        support(v) = Σ conf for groups voting v
        purity(v)  = support(v) / Σ conf (1.0 = unanimous)
        score(v)   = support(v) · purity(v)
    A frame contributes its winning candidate's score to a running per-g0
    accumulator. We freeze when the top accumulator score exceeds
    `freeze_thresh` AND beats the runner-up by `freeze_margin×`. If the clip
    ends without freezing, callers can use `summary()` for a flagged
    best-guess.
    """

    def __init__(self,
                  freeze_thresh: float = G0_FREEZE_THRESH,
                  freeze_margin: float = G0_FREEZE_MARGIN,
                  unanimous_min_frames: int = G0_UNANIMOUS_MIN_FRAMES,
                  unanimous_min_score: float = G0_UNANIMOUS_MIN_SCORE,
                  ngs_x_by_label=NGS_X_BY_LABEL,
                  yd_per_grid: float = YD_PER_GRID_DEFAULT):
        self.scores = defaultdict(float)
        self.frozen = False
        self.frozen_g0 = None
        self.frozen_at_frame = None
        self.n_frames_seen = 0   # frames passed in (sampled or not)
        self.n_frames_voted = 0  # frames that produced any vote
        self.ngs_x_by_label = ngs_x_by_label
        self.yd_per_grid = yd_per_grid
        self.freeze_thresh = freeze_thresh
        self.freeze_margin = freeze_margin
        self.unanimous_min_frames = unanimous_min_frames
        self.unanimous_min_score = unanimous_min_score

    def update(self, g_indices, labels, confs, frame_idx=None) -> bool:
        """Process one sampled frame's classifier output. Returns True iff
        the estimator just froze (i.e. caller should stop sampling)."""
        if self.frozen:
            return False
        self.n_frames_seen += 1
        per_g0 = defaultdict(float)
        total_conf = 0.0
        for g_idx, lbl, conf in zip(g_indices, labels, confs):
            ngs_x = self.ngs_x_by_label.get(lbl)
            if ngs_x is None or g_idx is None or g_idx < -100:
                continue
            g0 = ngs_x - self.yd_per_grid * float(g_idx)
            per_g0[float(g0)] += float(conf)
            total_conf += float(conf)
        if not per_g0 or total_conf <= 0:
            return False
        winner_g0, support = max(per_g0.items(), key=lambda kv: kv[1])
        purity = support / total_conf
        frame_score = support * purity
        self.scores[winner_g0] += frame_score
        self.n_frames_voted += 1
        sorted_vals = sorted(self.scores.values(), reverse=True)
        top = sorted_vals[0]
        runner = sorted_vals[1] if len(sorted_vals) >= 2 else 0.0
        # Standard freeze: high score AND clear margin over the runner-up.
        high_freeze = (top >= self.freeze_thresh
                        and top >= self.freeze_margin * max(runner, 1e-9))
        # Unanimous freeze: literally no runner-up after enough sampled
        # frames. Catches clips with only one visible painted number.
        unanimous_freeze = (runner == 0.0
                              and self.n_frames_voted >= self.unanimous_min_frames
                              and top >= self.unanimous_min_score)
        if high_freeze or unanimous_freeze:
            self.frozen = True
            self.frozen_g0 = max(self.scores.items(), key=lambda kv: kv[1])[0]
            self.frozen_at_frame = frame_idx
            return True
        return False

    def summary(self) -> dict:
        """Returns {g0, frozen, frame, score, runner_score, margin,
        n_voted, n_seen, fallback}.
        - frozen=True → confident anchor.
        - frozen=False & g0 not None → fallback best-guess (caller should flag).
        - g0=None → no votes at all (no painted numbers detected anywhere)."""
        if self.frozen:
            return {
                "g0": self.frozen_g0,
                "frozen": True,
                "fallback": False,
                "frame": self.frozen_at_frame,
                "score": float(self.scores[self.frozen_g0]),
                "runner_score": 0.0,  # filled below if there was one
                "margin": float("inf"),
                "n_voted": self.n_frames_voted,
                "n_seen": self.n_frames_seen,
            }
        if not self.scores:
            return {"g0": None, "frozen": False, "fallback": True,
                    "frame": None, "score": 0.0, "runner_score": 0.0,
                    "margin": 0.0, "n_voted": 0,
                    "n_seen": self.n_frames_seen}
        sorted_items = sorted(self.scores.items(),
                                key=lambda kv: kv[1], reverse=True)
        top_g0, top_score = sorted_items[0]
        runner_g0, runner_score = (sorted_items[1] if len(sorted_items) >= 2
                                    else (None, 0.0))
        return {
            "g0": float(top_g0),
            "frozen": False,
            "fallback": True,
            "frame": None,
            "score": float(top_score),
            "runner_score": float(runner_score),
            "margin": float(top_score) / max(float(runner_score), 1e-9),
            "n_voted": self.n_frames_voted,
            "n_seen": self.n_frames_seen,
        }
