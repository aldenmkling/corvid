#!/usr/bin/env python3
"""Auto-generate hash masks from existing point annotations.

Algorithm per hash point at (x, y):
  1. Estimate local yardline pixel width w_yl from the UNet yardline mask
  2. Crop disc of radius 3.5 * w_yl around the point
     (real hash is 6 yard-line-widths long; 3.5× radius leaves margin)
  3. Threshold for white pixels in crop (high intensity)
  4. Estimate local yardline direction (PCA on nearby yardline mask pixels)
  5. Subtract yardline mask from white pixels EXCEPT pixels within w_yl/2
     (= 2 inches) of the hash keypoint along the yardline direction.
     This preserves the part of the hash dash that overlaps the yardline,
     so the resulting mask is the full perpendicular dash, not a donut.
  6. Restrict result to disc radius (round mask).

Output: one binary mask PNG per frame, full image size.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Tunable constants. All in PIXEL units; will be scaled by w_yl per-hash.
CROP_RADIUS_K = 3.0              # disc crop radius (× w_yl). Real hash is
                                  # 6 × w_yl long; 3× radius gives ~1 × w_yl
                                  # margin around half-length.
BAND_HALF_K = 0.3                # band half-thickness in band_along direction
                                  # (× w_yl). Slab extends to disc edge
                                  # perpendicular to band_along.
WHITE_PERCENTILE = 70            # adaptive threshold: top (100-this)% by intensity
WHITE_FLOOR = 145                # absolute floor on threshold (0–255).
DIRECTION_WINDOW_PX = 25         # window for PCA-estimating yardline direction
# Rectangle parameters (synthetic perp-to-yardline rectangle anchored on
# keypoint; PCA on the noisy mask only refines length within bounds).
HASH_LENGTH_K = 6.0              # ideal hash length / w_yl (24" / 4")
HASH_WIDTH_K = 1.0               # ideal hash width  / w_yl (4" / 4")
HASH_LEN_MIN_K = 4.0             # clamp measured length to [4..8] × w_yl
HASH_LEN_MAX_K = 8.0
ROT_TOLERANCE_DEG = 10.0         # max deviation of long axis from perp-to-yl;
                                  # outside this, snap to perp direction


def _nearest_yardline_pixel(yardline_mask: np.ndarray,
                              x: int, y: int, search_r: int = 8):
    """If (x, y) isn't on the mask, find the nearest mask pixel within
    `search_r` px. Returns (x, y) on the mask or None."""
    h, w = yardline_mask.shape
    if 0 <= y < h and 0 <= x < w and yardline_mask[y, x]:
        return x, y
    for r in range(1, search_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and yardline_mask[ny, nx]:
                    return nx, ny
    return None


def estimate_yardline_width_perpendicular(yardline_mask: np.ndarray,
                                            x: int, y: int,
                                            along: np.ndarray,
                                            max_search: int = 25) -> int | None:
    """Yardline pixel width measured ALONG the perpendicular to the local
    yardline direction. Robust to tilt (horizontal scan over-estimates for
    tilted yardlines)."""
    h, w = yardline_mask.shape
    nearest = _nearest_yardline_pixel(yardline_mask, x, y)
    if nearest is None:
        return None
    x, y = nearest
    perp = np.array([-along[1], along[0]])

    def walk(sign):
        for d in range(1, max_search + 1):
            px = int(round(x + d * sign * perp[0]))
            py = int(round(y + d * sign * perp[1]))
            if not (0 <= px < w and 0 <= py < h):
                return d - 1
            if not yardline_mask[py, px]:
                return d - 1
        return max_search

    return walk(+1) + walk(-1) + 1


def rectangularize_hash(hash_crop: np.ndarray,
                          along: np.ndarray,
                          cx: int, cy: int,
                          w_yl: int) -> tuple[np.ndarray, dict]:
    """Connected-component cleanup (no rectangle fitting):
      - Find CCs in the noisy hash mask
      - Restrict to CCs within ~2 × w_yl of the keypoint (in image pixels)
      - If the largest CC dominates (>= 2× the next biggest), keep only it
      - Otherwise keep all 'sizable' fragments (>= min size)
      - Output = union of kept CC pixel masks

    Keeps the natural shape of the actual hash paint instead of forcing a
    rectangle on top of it.
    """
    h, w = hash_crop.shape
    info = {"n_kept": 0, "mode": "none"}

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hash_crop.astype(np.uint8), connectivity=8,
    )
    if n_labels <= 1:
        return np.zeros_like(hash_crop, dtype=np.uint8), info

    max_dist = 2.0 * w_yl
    # "More than a few pixels": floor at 5, scaling up with yardline width.
    min_size = max(5, int(round(0.5 * w_yl)))

    # Collect candidate CCs near the keypoint.
    candidates = []   # (label, area, dist)
    for k in range(1, n_labels):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_size:
            continue
        cx_k = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] / 2
        cy_k = stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] / 2
        d = float(np.hypot(cx_k - cx, cy_k - cy))
        if d <= max_dist:
            candidates.append((k, area, d))

    if not candidates:
        return np.zeros_like(hash_crop, dtype=np.uint8), info

    # Sort by area descending.
    candidates.sort(key=lambda c: c[1], reverse=True)

    if len(candidates) >= 2 and candidates[0][1] >= 2 * candidates[1][1]:
        # Largest dominates → keep only it.
        kept_labels = [candidates[0][0]]
        info["mode"] = "dominant"
    else:
        # Keep all sizable candidates (already filtered by min_size).
        kept_labels = [c[0] for c in candidates]
        info["mode"] = "fragments"

    clean = np.isin(labels, kept_labels).astype(np.uint8)
    info["n_kept"] = len(kept_labels)
    info["candidate_areas"] = [c[1] for c in candidates]
    return clean, info


def pca_direction_of_mask(mask_crop: np.ndarray) -> np.ndarray:
    """PCA on a binary mask's pixel positions. Returns unit principal-axis
    vector. Falls back to (0, 1) (vertical) if too few pixels."""
    ys, xs = np.where(mask_crop)
    if len(xs) < 5:
        return np.array([0.0, 1.0])
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        d = vt[0]
        return d / max(np.linalg.norm(d), 1e-9)
    except np.linalg.LinAlgError:
        return np.array([0.0, 1.0])


def estimate_yardline_direction(yardline_mask: np.ndarray,
                                  x: int, y: int,
                                  window: int = DIRECTION_WINDOW_PX) -> np.ndarray:
    """Bootstrap direction (rough). Used only to seed the perpendicular-
    width measurement; final direction is computed by PCA inside the
    actual hash crop."""
    h, w = yardline_mask.shape
    y0, y1 = max(0, y - window), min(h, y + window + 1)
    x0, x1 = max(0, x - window), min(w, x + window + 1)
    return pca_direction_of_mask(yardline_mask[y0:y1, x0:x1])


def build_hash_mask_at(frame: np.ndarray, x: int, y: int,
                        yardline_mask: np.ndarray,
                        sideline_mask: np.ndarray | None = None,
                        debug: bool = False) -> tuple[np.ndarray, dict] | None:
    """Generate a binary hash mask centered on (x, y). Returns (full_size_mask, info)
    or None if generation failed."""
    h_img, w_img = frame.shape[:2]

    # Bootstrap: rough direction + width to set crop size.
    along_init = estimate_yardline_direction(yardline_mask, x, y)
    w_yl = estimate_yardline_width_perpendicular(yardline_mask, x, y, along_init)
    if w_yl is None or w_yl < 2 or w_yl > 30:
        return None

    crop_r = int(round(CROP_RADIUS_K * w_yl))
    band_half = BAND_HALF_K * w_yl     # half-thickness of perp-to-yl band

    x0 = max(0, x - crop_r); x1 = min(w_img, x + crop_r + 1)
    y0 = max(0, y - crop_r); y1 = min(h_img, y + crop_r + 1)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    cy, cx = y - y0, x - x0   # keypoint coords inside crop

    # Final direction: PCA on yardline mask pixels INSIDE the crop. More
    # robust than a fixed-window estimate around the keypoint, which can
    # pull in adjacent yardlines when they're close.
    yl_crop_for_dir = yardline_mask[y0:y1, x0:x1]
    along = pca_direction_of_mask(yl_crop_for_dir)

    # White-pixel threshold (adaptive percentile + soft floor).
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr = max(WHITE_FLOOR, int(np.percentile(gray, WHITE_PERCENTILE)))
    white = (gray >= thr).astype(np.uint8)

    # Disc mask
    yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]].astype(np.float32)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    disc = (dist2 <= crop_r ** 2).astype(np.uint8)
    white = white & disc

    # Yardline + sideline masks within crop
    yl_crop = yardline_mask[y0:y1, x0:x1].astype(np.uint8)
    sl_crop = (sideline_mask[y0:y1, x0:x1].astype(np.uint8)
               if sideline_mask is not None else np.zeros_like(yl_crop))

    # Band direction: PCA on the *non-yardline* white pixels in the crop.
    # After stripping the yardline mask, what's left near the keypoint is
    # mainly the hash dash itself. Its principal axis is the dash's long
    # direction; the band slab should extend ALONG that direction (= thin
    # axis perpendicular to it). This is more robust than using the
    # yardline PCA, which can drift if multiple yardlines fall in the crop.
    dash_candidate = (white & (1 - yl_crop) & disc).astype(np.uint8)
    dash_dir = pca_direction_of_mask(dash_candidate)
    # band_along = perpendicular to dash direction (slab thin axis)
    band_along = np.array([-dash_dir[1], dash_dir[0]])
    # Force horizontal-ish orientation (flip so x-component is positive).
    if band_along[0] < 0:
        band_along = -band_along
    band_along = band_along / max(np.linalg.norm(band_along), 1e-9)

    dx_grid = xx - cx
    dy_grid = yy - cy
    proj_a = dx_grid * band_along[0] + dy_grid * band_along[1]
    preserve = (np.abs(proj_a) <= band_half).astype(np.uint8)

    # Subtract yardline pixels EXCEPT in preserve region.
    yl_to_remove = yl_crop & (1 - preserve)
    noisy_hash = white & (1 - yl_to_remove)
    # Also subtract sideline pixels (no preservation)
    noisy_hash = noisy_hash & (1 - sl_crop)

    # Rectangularize: fit an oriented rectangle to the noisy mask, render
    # a clean rectangular hash with constrained orientation/dimensions.
    clean_hash, rect_info = rectangularize_hash(noisy_hash, along, cx, cy, w_yl)

    full = np.zeros(frame.shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = clean_hash * 255

    info = {
        "w_yl": int(w_yl),
        "crop_r": crop_r,
        "band_half": float(band_half),
        "along": along.tolist(),
        "thr": int(thr),
        "n_pixels_kept": int(clean_hash.sum()),
        "rect": rect_info,
    }
    if debug:
        info["debug"] = {
            "crop": crop, "white": white * 255,
            "yl_crop": yl_crop * 255, "preserve": preserve * 255,
            "noisy_hash": noisy_hash * 255,
            "clean_hash": clean_hash * 255,
            "cx": cx, "cy": cy,
        }
    return full, info


def run_unet_yardline_mask(frame: np.ndarray, unet_weights: str,
                             device: str = "mps") -> np.ndarray:
    """One-frame helper: returns binary yardline mask at frame size."""
    from src.homography.grid_solver_v2 import run_unet
    yard_mask, _ = run_unet(frame, unet_weights, device=device)
    return (yard_mask > 0).astype(np.uint8)


def run_unet_full(frame: np.ndarray, unet_weights: str,
                    device: str = "mps") -> tuple[np.ndarray, np.ndarray]:
    from src.homography.grid_solver_v2 import run_unet
    y, s = run_unet(frame, unet_weights, device=device)
    return (y > 0).astype(np.uint8), (s > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint-dir", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train"))
    ap.add_argument("--unet-weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_round3_best.pth"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/train"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--viz-n", type=int, default=8,
                    help="Number of debug visualizations to save")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    with open(os.path.join(args.keypoint_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    n_frames = 0; n_hashes_total = 0; n_hashes_kept = 0
    viz_done = 0
    for ann in coco["annotations"]:
        info = images_by_id[ann["image_id"]]
        path = os.path.join(args.keypoint_dir, "images", info["file_name"])
        if not os.path.exists(path): continue
        frame = cv2.imread(path)
        if frame is None: continue

        yl_mask, sl_mask = run_unet_full(frame, args.unet_weights, args.device)
        # Resize masks back to frame size if UNet output a different resolution
        if yl_mask.shape != frame.shape[:2]:
            yl_mask = cv2.resize(yl_mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            sl_mask = cv2.resize(sl_mask, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        full_hash_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for p in ann["points"]:
            if p["channel"] != 1: continue   # hash channel
            n_hashes_total += 1
            x, y = int(round(p["x"])), int(round(p["y"]))
            res = build_hash_mask_at(
                frame, x, y, yl_mask, sl_mask,
                debug=(viz_done < args.viz_n),
            )
            if res is None: continue
            full, meta = res
            n_hashes_kept += 1
            full_hash_mask = np.maximum(full_hash_mask, full)

            # Save viz of first N
            if "debug" in meta and viz_done < args.viz_n:
                d = meta["debug"]
                k = max(1, 240 // (2 * meta["crop_r"] + 1))   # upscale for viewing
                def up(m): return cv2.resize(m, None, fx=k, fy=k,
                                                interpolation=cv2.INTER_NEAREST)
                src = up(d["crop"])
                w_ = up(cv2.cvtColor(d["white"], cv2.COLOR_GRAY2BGR))
                yl_ = up(cv2.cvtColor(d["yl_crop"], cv2.COLOR_GRAY2BGR))
                noisy_ = up(cv2.cvtColor(d["noisy_hash"], cv2.COLOR_GRAY2BGR))
                # Final panel: source crop + clean mask in see-through red
                clean_up = up(d["clean_hash"])     # uint8 mask, 0 or 255
                overlay = src.copy().astype(np.float32)
                m = (clean_up > 0)
                red = np.array([60, 60, 230], dtype=np.float32)
                overlay[m] = 0.45 * overlay[m] + 0.55 * red
                overlay = overlay.clip(0, 255).astype(np.uint8)
                viz = np.hstack([src, w_, yl_, noisy_, overlay])
                rect_info = meta.get("rect", {})
                fit_str = rect_info.get("mode", "?")
                cv2.putText(viz,
                            f"y={y} w_yl={meta['w_yl']} thr={meta['thr']} "
                            f"rect={fit_str}  src | white | yardline | noisy | overlay",
                            (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                vp = os.path.join(viz_dir, f"hash_{viz_done:02d}_y{y}.jpg")
                cv2.imwrite(vp, viz)
                viz_done += 1

        # Save per-frame hash mask
        out_path = os.path.join(args.out_dir, "masks",
                                  os.path.splitext(info["file_name"])[0] + ".png")
        cv2.imwrite(out_path, full_hash_mask)
        n_frames += 1
        if n_frames % 50 == 0:
            print(f"  frame {n_frames}  hashes processed {n_hashes_total}  "
                  f"kept {n_hashes_kept}")

    print(f"\n  {n_frames} frames; {n_hashes_total} hash points → "
          f"{n_hashes_kept} kept ({100 * n_hashes_kept / max(n_hashes_total, 1):.1f}%)")
    print(f"  masks: {args.out_dir}/masks/")
    print(f"  viz: {viz_dir}/  ({viz_done} samples)")


if __name__ == "__main__":
    main()
