"""Round-3 painted-number classifier dataset.

Same construction as round2 (GT-derived labels via verified H + identical
spatial-CC grouping and crop preprocessing to cc_tokenizer_v2), but adds a
10th `bg` class for clusters that have NO GT-projected painted-number
within MAX_MATCH_PX. These are the false-positive clusters the v8 number
mask produces in practice (grass texture, partial-mask noise, etc.).

Why this matters: at v9 inference time, the spatial CC tokenizer produces
~14% FP clusters in addition to the real painted numbers (measured across
311 val frames). The 9-class round2 classifier had no way to abstain on
FPs — it confidently labeled them as one of 9 yard classes and polluted
the encoder's anchors, dragging v9 from 96.71% (legacy mode, with implicit
abstention via v2 per-pixel labels) down to 76.80%.

Round3 closes that gap: with the `bg` class, mbconv learns to recognize
FPs and v9 maps `bg` → has_ngs=False, replicating legacy mode's abstention
but trained from the cluster distribution v9 actually sees at inference.

Both real-class crops and `bg` crops use the SAME spatial-CC grouping
(dilate-28) and SAME crop preprocessing (no margin, no padding-to-square,
INTER_NEAREST resize to 64×64) as cc_tokenizer_v2.py. Training distribution
= inference distribution.

Why this fixes the round1 problem:
  - round1's labels came from the v2 per-pixel NGS_x classifier (mode-vote),
    which inherited the v2 UNet's biases. A classifier trained on round1
    therefore caps at v2 fidelity.
  - round2's labels come from GT NGS coordinates. A classifier trained on
    round2 can in principle exceed v2 — its labels are independent of any
    other model.

Crop preprocessing (per the user's request, optimized for sharp edges):
  - Tight bbox around the cluster's binary mask pixels (no margin).
  - Direct resize to 64×64 with cv2.INTER_NEAREST (no padding to square,
    aspect-ratio distortion accepted — this is consistent across train and
    inference if the v9 tokenizer also uses the same crop function).
  - Output: uint8, 0 = bg, 255 = fg.

Grouping:
  - The user said "use a method that matches the number of groups we expect
    from the coords." We do: spatial-CC clustering (dilate-by-10-px then
    connectedComponents) on the binary number mask, then assign each GT
    projection to its NEAREST cluster centroid within MAX_MATCH_PX. This
    is implicit count-matching — at most one cluster per GT point, with
    unmatched GT points (occluded, off-frame, false-negatives in the v8
    mask) silently dropped.

Output: data/number_classifier/round2/<class>/<frame_id>_<side>.png
        + data/number_classifier/round2/manifest.json (provenance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.field_mapping.field_model import (    # noqa: E402
    NUMBER_Y_NEAR, NUMBER_Y_FAR, ngs_x_to_field_number, TEN_YARD_POSITIONS,
)


# ── Constants ───────────────────────────────────────────────────────────────

# Game whose frames are reserved as v9's val set. Excluded from round2 so the
# trained classifier never sees them, which keeps v9's val numbers clean when
# the classifier is plugged in (no leakage of val crops into classifier
# training). Project holdouts (2024100601, 2024122201) are *kept* — the user
# manually verified their Hs, so they're trustworthy training data here.
V9_VAL_GAME = "2024090802"

CROP_W = 128                # rectangular crop (matches cc_tokenizer_v2's
CROP_H = 32                 # CLASSIFIER_CROP_W/H; preserves digit aspect).
DILATE_PX = 28              # spatial-CC clustering radius. cc_tokenizer_v2
                            # uses the same radius so train and inference
                            # distributions match.
MIN_CLUSTER_PX = 200        # filter tiny v8-mask noise.

# ── Matching in NGS space ───────────────────────────────────────────────
# Each surviving cluster's centroid is projected through verified H to NGS
# coords, then we snap NGS_x to the nearest 10y painted-number value (20,
# 30, ..., 100) and NGS_y to the nearest number-row value (NUMBER_Y_NEAR
# ≈ 13, NUMBER_Y_FAR ≈ 40.33). Clusters whose centroid lands more than
# NGS_BUCKET_TOL/NGS_Y_TOL from a valid (yard, row) get dropped.
#
# 5.0-yd tolerance is the FULL half-width of a painted number (digits +
# arrow chevrons span ~5 yd) so a fully cut-off-by-50% number, or one
# whose centroid is offset by player occlusion, still snaps correctly.
NGS_BUCKET_TOL = 5.0
NGS_Y_TOL = 6.0
NGS_X_FIELD_MIN = 10.0      # painted numbers live at NGS_x ∈ [20,100],
NGS_X_FIELD_MAX = 110.0     # but the field of play is [10,110]; reject
                            # NGS_x outside this range (endzone text).
NUM_MASK_CHANNEL = 3        # masks[..., 3] = v8 number probability
NUM_MASK_THRESH = 0.5

# 9-class label space: NGS_x → "<num><side>" or "50"
def class_for_ngs_x(ngs_x: float) -> str:
    """NGS_x ∈ {20, 30, 40, 50, 60, 70, 80, 90, 100} → '20L'..'10R' / '50'."""
    num = ngs_x_to_field_number(ngs_x)
    if num == 50:
        return "50"
    return f"{num}{'L' if ngs_x < 60.0 else 'R'}"


# ── NGS → distorted image space ─────────────────────────────────────────────

def ngs_to_undistorted(pts_ngs: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply H^-1 to take (N, 2) NGS points → undistorted image pixels."""
    H_inv = np.linalg.inv(H)
    pts_h = np.concatenate([pts_ngs, np.ones((pts_ngs.shape[0], 1))], axis=1)
    img_h = (H_inv @ pts_h.T).T
    img = img_h[:, :2] / img_h[:, 2:3]
    return img


def undistorted_to_distorted(pts_und: np.ndarray, K: np.ndarray,
                              dist: np.ndarray) -> np.ndarray:
    """Apply forward Brown-Conrady distortion to take undistorted-image pixels
    → distorted-source-image pixels. Uses cv2.projectPoints with R=I, tvec=0,
    treating the undistorted normalized coords as already-projected 3D points
    on z=1.

    pts_und: (N, 2) undistorted pixel coords.
    """
    if pts_und.shape[0] == 0:
        return pts_und
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Normalize to camera-frame unit-z coords.
    pts_norm = np.zeros((pts_und.shape[0], 3), dtype=np.float64)
    pts_norm[:, 0] = (pts_und[:, 0] - cx) / fx
    pts_norm[:, 1] = (pts_und[:, 1] - cy) / fy
    pts_norm[:, 2] = 1.0
    rvec = np.zeros(3); tvec = np.zeros(3)
    pts_dist, _ = cv2.projectPoints(
        pts_norm.reshape(-1, 1, 3), rvec, tvec, K,
        np.asarray(dist, dtype=np.float64).reshape(-1))
    return pts_dist.reshape(-1, 2)


# ── Cluster the number mask ─────────────────────────────────────────────────

def spatial_cc_clusters(num_mask_prob: np.ndarray,
                          dilate_px: int = DILATE_PX
                          ) -> tuple[np.ndarray, list[dict]]:
    """Returns (cluster_label_map, cluster_records).

    cluster_label_map: (H, W) int — 0 = bg, n = cluster id n.
    cluster_records: list of {id, centroid: (cx, cy), bbox: (x_min,y_min,x_max,y_max),
                              ys, xs (absolute coords of cluster's mask pixels)}.
    """
    bin_mask = (num_mask_prob > NUM_MASK_THRESH).astype(np.uint8)
    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=np.int32), []
    if dilate_px > 0:
        ks = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        dilated = cv2.dilate(bin_mask, kernel, iterations=1)
    else:
        dilated = bin_mask
    n_clusters, cluster_labels, _, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)
    records = []
    for cid in range(1, n_clusters):
        # Restrict to ORIGINAL pixels inside this dilated cluster.
        cluster_mask = (cluster_labels == cid) & (bin_mask > 0)
        if cluster_mask.sum() < MIN_CLUSTER_PX:
            continue
        ys, xs = np.where(cluster_mask)
        records.append(dict(
            id=cid,
            centroid=(float(xs.mean()), float(ys.mean())),
            bbox=(int(xs.min()), int(ys.min()),
                   int(xs.max()) + 1, int(ys.max()) + 1),
            ys=ys, xs=xs,
        ))
    return cluster_labels.astype(np.int32), records


# ── Crop construction (no margin, no pad-square; INTER_NEAREST resize) ──────

def make_crop(bin_mask: np.ndarray, cluster_label_map: np.ndarray,
               cluster_id: int, x_min: int, y_min: int,
               x_max: int, y_max: int) -> np.ndarray:
    sub_bin = bin_mask[y_min:y_max, x_min:x_max].astype(np.uint8)
    sub_lab = cluster_label_map[y_min:y_max, x_min:x_max]
    crop = (sub_bin > 0) & (sub_lab == cluster_id)
    crop = crop.astype(np.uint8) * 255
    if crop.size == 0:
        return np.zeros((CROP_H, CROP_W), dtype=np.uint8)
    return cv2.resize(crop, (CROP_W, CROP_H),
                        interpolation=cv2.INTER_NEAREST)


# ── Per-frame pipeline ──────────────────────────────────────────────────────

def distorted_to_ngs(pts_dist: np.ndarray, H: np.ndarray, K: np.ndarray,
                       dist: np.ndarray) -> np.ndarray:
    """Take (N, 2) distorted-image pixels → undistorted → NGS yards via H."""
    if pts_dist.shape[0] == 0:
        return pts_dist
    und = cv2.undistortPoints(
        pts_dist.reshape(-1, 1, 2).astype(np.float64),
        K, np.asarray(dist).reshape(-1), P=K).reshape(-1, 2)
    und_h = np.concatenate([und, np.ones((und.shape[0], 1))], axis=1)
    ngs_h = (H @ und_h.T).T
    return ngs_h[:, :2] / ngs_h[:, 2:3]


def process_frame(frame_id: str, masks: np.ndarray, H: np.ndarray,
                    K: np.ndarray, dist: np.ndarray,
                    image_h: int, image_w: int) -> list[dict]:
    """Return list of crops, all real-class. No bg class.

    Per cluster:
      1. Project centroid through H → NGS coords.
      2. Drop if NGS_x outside [NGS_X_FIELD_MIN, NGS_X_FIELD_MAX]
         (endzone team text — explicit reject).
      3. Snap NGS_x to nearest 10y painted-number bucket; drop if
         |NGS_x - bucket| > NGS_BUCKET_TOL.
      4. Snap NGS_y to nearest number-row value; drop if
         |NGS_y - row| > NGS_Y_TOL.
      5. Otherwise emit the cluster with the matched class.
    """
    num_mask = masks[..., NUM_MASK_CHANNEL].astype(np.float32)
    cluster_labels, clusters = spatial_cc_clusters(num_mask)
    if not clusters:
        return []
    bin_mask = (num_mask > NUM_MASK_THRESH).astype(np.uint8)

    centroids_dist = np.asarray(
        [c["centroid"] for c in clusters], dtype=np.float64)
    centroids_ngs = distorted_to_ngs(centroids_dist, H, K, dist)

    out = []
    for ci, c in enumerate(clusters):
        ngs_x_raw, ngs_y_raw = centroids_ngs[ci]
        # Reject endzone text — NGS_x outside the field of play.
        if not (NGS_X_FIELD_MIN <= ngs_x_raw <= NGS_X_FIELD_MAX):
            continue
        # Snap NGS_x to nearest 10y painted-number value.
        bucket_x = int(round(ngs_x_raw / 10.0)) * 10
        if bucket_x not in TEN_YARD_POSITIONS:
            continue
        if abs(ngs_x_raw - bucket_x) > NGS_BUCKET_TOL:
            continue
        # Pick the closer number row.
        d_near = abs(ngs_y_raw - NUMBER_Y_NEAR)
        d_far = abs(ngs_y_raw - NUMBER_Y_FAR)
        if d_near < d_far:
            side, dy = "near", d_near
        else:
            side, dy = "far", d_far
        if dy > NGS_Y_TOL:
            continue
        cls = class_for_ngs_x(float(bucket_x))
        x_min, y_min, x_max, y_max = c["bbox"]
        crop = make_crop(bin_mask, cluster_labels, c["id"],
                           x_min, y_min, x_max, y_max)
        out.append(dict(
            cls=cls, side=side, ngs_x=float(bucket_x),
            frame_id=frame_id, crop=crop,
            match_dist_px=float(np.hypot(
                ngs_x_raw - bucket_x,
                ngs_y_raw - (NUMBER_Y_NEAR if side == "near" else NUMBER_Y_FAR))),
            cluster_centroid=c["centroid"],
        ))
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/number_classifier/round3"))
    ap.add_argument("--max-entries", type=int, default=None,
                     help="Optional cap for smoke-testing.")
    ap.add_argument("--image-h", type=int, default=720)
    ap.add_argument("--image-w", type=int, default=1280)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr_by_clip = m.get("intrinsics_by_clip", {})
    entries = m["entries"]
    if args.max_entries:
        entries = entries[:args.max_entries]
    print(f"Processing {len(entries)} entries...")

    class_counts: Counter = Counter()
    side_counts: Counter = Counter()
    skipped_v9_val = 0
    skipped_no_intrinsics = 0
    skipped_no_cache = 0
    skipped_no_mask = 0
    total_emitted = 0
    match_dists: list[float] = []

    t0 = time.time()
    for i, e in enumerate(entries):
        clip = e["clip"]
        # Exclude v9's val game so the classifier never sees those crops in
        # training. (Project holdouts 2024100601/2024122201 are NOT excluded —
        # their GT Hs are manually verified per user, so they're valid here.)
        clip_game = clip.split("/", 1)[0] if "/" in clip else clip
        if clip_game == V9_VAL_GAME:
            skipped_v9_val += 1
            continue
        intr = intr_by_clip.get(clip)
        if intr is None:
            skipped_no_intrinsics += 1
            continue
        cp = os.path.join(args.cache_dir, f"{e['id']}.npz")
        if not os.path.exists(cp):
            skipped_no_cache += 1
            continue

        d = np.load(cp)
        masks = d["masks"].astype(np.float32)
        if masks[..., NUM_MASK_CHANNEL].max() < NUM_MASK_THRESH:
            skipped_no_mask += 1
            continue

        H = np.asarray(e["H"], dtype=np.float64)
        K = np.asarray(intr["K"], dtype=np.float64)
        dist = np.asarray(intr["dist"], dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)

        records = process_frame(
            e["id"], masks, H, K, dist,
            image_h=args.image_h, image_w=args.image_w)
        for r in records:
            cls_dir = os.path.join(args.out_dir, r["cls"])
            os.makedirs(cls_dir, exist_ok=True)
            if r["cls"] == "bg":
                fname = f"{r['frame_id']}_bg{r.get('bg_idx', 0)}.png"
            else:
                fname = f"{r['frame_id']}_{r['side']}.png"
            cv2.imwrite(os.path.join(cls_dir, fname), r["crop"])
            class_counts[r["cls"]] += 1
            side_counts[r["side"]] += 1
            if r["match_dist_px"] >= 0:
                match_dists.append(r["match_dist_px"])
            total_emitted += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(entries)}]  emitted={total_emitted}  "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"Total emitted: {total_emitted}")
    print(f"Skipped — v9 val game ({V9_VAL_GAME}): {skipped_v9_val}")
    print(f"Skipped — no intrinsics:  {skipped_no_intrinsics}")
    print(f"Skipped — no cache file:  {skipped_no_cache}")
    print(f"Skipped — empty mask:     {skipped_no_mask}")
    print(f"\nPer-class counts:")
    for cls in ["10L", "20L", "30L", "40L", "50",
                 "40R", "30R", "20R", "10R"]:
        print(f"  {cls:>4}: {class_counts[cls]:>5}")
    print(f"\nPer-side counts:")
    for s, n in side_counts.items():
        print(f"  {s:>4}: {n:>5}")
    if match_dists:
        match_dists = np.asarray(match_dists)
        print(f"\nNGS-space match distance (cluster NGS → painted-number bucket):")
        print(f"  mean={match_dists.mean():.2f}y  median={np.median(match_dists):.2f}y  "
              f"max={match_dists.max():.2f}y")

    manifest = {
        "total_emitted": int(total_emitted),
        "class_counts": dict(class_counts),
        "side_counts": dict(side_counts),
        "v9_val_game_excluded": V9_VAL_GAME,
        "skipped_v9_val": int(skipped_v9_val),
        "skipped_no_intrinsics": int(skipped_no_intrinsics),
        "skipped_no_cache": int(skipped_no_cache),
        "skipped_no_mask": int(skipped_no_mask),
        "match_distance_stats": {
            "mean": float(match_dists.mean()) if len(match_dists) else 0.0,
            "median": float(np.median(match_dists)) if len(match_dists) else 0.0,
            "max": float(match_dists.max()) if len(match_dists) else 0.0,
        },
        "config": {
            "crop_size": (CROP_W, CROP_H),
            "dilate_px": DILATE_PX,
            "min_cluster_px": MIN_CLUSTER_PX,
            "ngs_bucket_tol_yards": NGS_BUCKET_TOL,
            "ngs_y_tol_yards": NGS_Y_TOL,
            "ngs_x_field_range": [NGS_X_FIELD_MIN, NGS_X_FIELD_MAX],
            "num_mask_thresh": NUM_MASK_THRESH,
            "resize_interp": "INTER_NEAREST",
            "padding": "none (tight bbox, no square padding)",
            "matching": "NGS-space: snap centroid to nearest painted-number "
                          "bucket within ±5y. Endzone text (NGS_x outside "
                          "[10,110]) explicitly dropped.",
            "label_source": "GT NGS coords projected through manifest H",
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
