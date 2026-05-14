"""Build a CLEANED unified-mask training dataset using verified-good
homographies from the dense field pool.

For each user-Y'd entry in the dense pool:
  1. Extract source frame from MP4 at that frame_idx
  2. Run all 4 specialists → raw masks (with their false positives)
  3. Project each pixel through the entry's known-good H → (NGS_x, NGS_y)
  4. AND the raw mask with a per-channel "allowed region" in NGS coords:
       - yard:   NGS_x within 1.0yd of nearest 5y multiple; NGS_y ∈ [-1, 54.33]
       - side:   NGS_y within 2.0yd of 0 OR 53.33; NGS_x ∈ [0, 120]
       - hash:   NGS_y within 0.5yd of 23.58 OR 29.75 AND NGS_x within 1yd of integer
       - number: NGS_y in [11.5, 14.5] OR [38.83, 41.83] AND NGS_x within 3.5yd
                 of multiple of 10 in [20, 100]
  5. Save (rgb, cleaned_masks_4ch) as .npz

Effectively distills H-aware QC into the masks: the unified model trained on
these will learn to output cleaner masks at inference (no field-detected hash
on a coach, no number-mask on a sideline ref, etc.) without needing H itself.

Usage:
    python scripts/data_prep/build_qc_unified_mask_dataset.py \\
        --pool-dir output/dense_field_pool \\
        --out-dir data/training/unified_masks/round2_qc \\
        --device mps
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography import painted_numbers
from src.homography.specialists import (
    LINE_WEIGHTS, HASH_WEIGHTS, NUMBER_WEIGHTS,
    run_specialists,
)
from src.field_mapping.field_model import (
    FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR, NUMBER_Y_NEAR, NUMBER_Y_FAR,
)


# ── Allowed-region rules per channel ──
# These tolerances are deliberately LOOSE — the goal is to throw out clear
# false positives (hash on a coach, number on a sideline ref), not to police
# pixel-perfect mask thickness. Tight tolerances over-cull legitimate
# yardline/sideline pixels because per-frame H is only accurate to ~1-2 yd
# and the specialist masks are 4-6 px wide.
def yard_mask_allowed(ngs_xy):
    """NGS_x within 1.0yd of nearest multiple of 5; NGS_y ∈ [-1, 54.33].
    Loosened from 0.5yd → 1.0yd after browsing post-QC thumbs showed the
    cleaned yard mask was much sparser than the actual yardlines visible
    in the source (H projection error + thin-mask AND under-cleaned)."""
    x, y = ngs_xy[..., 0], ngs_xy[..., 1]
    nearest_5 = np.round(x / 5.0) * 5.0
    return (np.abs(x - nearest_5) <= 1.0) & (y >= -1.0) & (y <= FIELD_WIDTH + 1.0)


def side_mask_allowed(ngs_xy):
    """NGS_y within 2.0yd of either sideline; NGS_x ∈ [0, 120].
    Iterated 0.5 → 1.0 → 2.0 yd. The polyfit-based side detection is wide
    and H projection error compounds, so the looser tolerance preserves
    nearly all legitimate sideline pixels while still rejecting obvious
    off-field FPs (refs in stands, etc.)."""
    x, y = ngs_xy[..., 0], ngs_xy[..., 1]
    near = np.abs(y - 0.0) <= 2.0
    far = np.abs(y - FIELD_WIDTH) <= 2.0
    return (near | far) & (x >= 0.0) & (x <= 120.0)


def hash_mask_allowed(ngs_xy):
    """NGS_y within 0.5yd of either hash row AND NGS_x within 1yd of integer."""
    x, y = ngs_xy[..., 0], ngs_xy[..., 1]
    near_y = np.abs(y - HASH_Y_NEAR) <= 0.5
    far_y = np.abs(y - HASH_Y_FAR) <= 0.5
    nearest_int = np.round(x)
    near_int = np.abs(x - nearest_int) <= 1.0
    in_field = (x >= 10.0) & (x <= 110.0)
    return (near_y | far_y) & near_int & in_field


def number_mask_allowed(ngs_xy):
    """NGS_y in number band ±0.5 AND NGS_x within 3.5yd of multiple of 10 in [20,100]."""
    x, y = ngs_xy[..., 0], ngs_xy[..., 1]
    near_y = (y >= NUMBER_Y_NEAR - 1.0 - 0.5) & (y <= NUMBER_Y_NEAR + 1.0 + 0.5)
    far_y = (y >= NUMBER_Y_FAR - 1.0 - 0.5) & (y <= NUMBER_Y_FAR + 1.0 + 0.5)
    nearest_10 = np.round(x / 10.0) * 10.0
    near_x = (np.abs(x - nearest_10) <= 3.5) & (nearest_10 >= 20.0) & (nearest_10 <= 100.0)
    return (near_y | far_y) & near_x


def project_pixels_to_ngs(H, h, w):
    """For each pixel (px, py) in an h×w grid, project through H to get
    (NGS_x, NGS_y). Returns (h, w, 2) float32."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pts = np.stack([xs.ravel().astype(np.float64),
                     ys.ravel().astype(np.float64),
                     np.ones(xs.size, dtype=np.float64)], axis=1)
    field = (H @ pts.T).T
    field = field[:, :2] / np.clip(field[:, 2:3], 1e-9, None)
    return field.reshape(h, w, 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=os.path.join(PROJECT_ROOT, "output/dense_field_pool"))
    ap.add_argument("--manifest-file", default=None,
                    help="Path to a manifest JSON (with 'entries' list) to use "
                         "directly. If set, --pool-dir's decisions.json filter "
                         "is bypassed (assumes manifest is already filtered).")
    ap.add_argument("--clips-root", default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data/training/unified_masks/round2_qc"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-entries", type=int, default=None,
                    help="For debug: stop after this many entries.")
    ap.add_argument("--save-thumbs", action="store_true",
                    help="Also save RGB+cleaned-mask overlay JPEGs for spot-check.")
    ap.add_argument("--reuse-rgb", action="store_true",
                    help="For each entry read 'rgb' from the existing "
                         "<out-dir>/raw/<id>.npz instead of decoding the MP4. "
                         "Skips video I/O — useful when iterating on H-clean "
                         "rules and you only need to re-run specialists.")
    ap.add_argument("--reclean-only", action="store_true",
                    help="Read rgb + raw_masks from existing "
                         "<out-dir>/raw/<id>.npz, apply NEW H-clean rules, "
                         "and overwrite the 'masks' field. Skips specialists "
                         "entirely — pure CPU/numpy, ~30s for 2500 entries. "
                         "Requires raw_masks to be present (from a prior build).")
    args = ap.parse_args()
    if args.reuse_rgb and args.reclean_only:
        sys.exit("--reuse-rgb and --reclean-only are mutually exclusive.")

    raw_dir = os.path.join(args.out_dir, "raw")
    thumb_dir = os.path.join(args.out_dir, "thumbs")
    os.makedirs(raw_dir, exist_ok=True)
    if args.save_thumbs:
        os.makedirs(thumb_dir, exist_ok=True)

    if args.manifest_file:
        # Use explicit manifest (assumed already filtered to Y entries)
        manifest = json.load(open(args.manifest_file))
        entries = manifest["entries"]
        print(f"Manifest: {args.manifest_file} → {len(entries)} entries")
    else:
        # Load full pool + decisions, filter to Y'd
        manifest = json.load(open(os.path.join(args.pool_dir, "manifest.json")))
        decisions_path = os.path.join(args.pool_dir, "decisions.json")
        decisions = json.load(open(decisions_path)) if os.path.exists(decisions_path) else {}
        entries = [e for e in manifest["entries"] if decisions.get(e["id"]) == "y"]
        print(f"Pool: {len(manifest['entries'])} entries, "
              f"decisions: {len(decisions)} → {len(entries)} Y'd entries")
    if args.max_entries:
        entries = entries[: args.max_entries]
        print(f"Limited to first {len(entries)} for debug")
    if not entries:
        sys.exit("No entries to process.")

    # Process
    n_done = n_skipped = n_failed = 0
    drop_stats = {"yard": [], "side": [], "hash": [], "number": []}
    t_start = time.time()

    for i, e in enumerate(entries, 1):
        out_path = os.path.join(raw_dir, f"{e['id']}.npz")

        # ── Source frame + raw masks ──
        if args.reclean_only:
            # CPU path: existing raw_masks + rgb, just re-AND with new rules
            if not os.path.exists(out_path):
                n_failed += 1; continue
            d = np.load(out_path)
            if "raw_masks" not in d.files:
                # Old-format file (pre-refactor) — can't reclean without raw
                n_failed += 1; continue
            frame = d["rgb"]
            raw_masks = d["raw_masks"]
            h, w = frame.shape[:2]
        else:
            if args.reuse_rgb:
                if not os.path.exists(out_path):
                    n_failed += 1; continue
                d = np.load(out_path)
                if "rgb" not in d.files:
                    n_failed += 1; continue
                frame = d["rgb"]
            else:
                # Default: decode from MP4 (legacy / fresh build)
                if os.path.exists(out_path):
                    n_skipped += 1
                    continue
                clip_path = os.path.join(args.clips_root, e["clip"])
                cap = cv2.VideoCapture(clip_path)
                if not cap.isOpened():
                    n_failed += 1; continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(e["frame_idx"]))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    n_failed += 1; continue
            h, w = frame.shape[:2]

            # Run specialists (only path that needs the GPU)
            yard, side, hash_ = run_specialists(
                frame, LINE_WEIGHTS, HASH_WEIGHTS, args.device)
            num = painted_numbers.predict_mask(frame, NUMBER_WEIGHTS, args.device)
            raw_masks = np.stack([
                (yard > 0).astype(np.uint8),
                (side > 0).astype(np.uint8),
                (hash_ > 0).astype(np.uint8),
                (num > 0).astype(np.uint8),
            ], axis=-1)

        # ── H-clean (always re-applied — that's the whole point) ──
        H = np.array(e["H"], dtype=np.float64)
        ngs = project_pixels_to_ngs(H, h, w)
        allowed = np.stack([
            yard_mask_allowed(ngs),
            side_mask_allowed(ngs),
            hash_mask_allowed(ngs),
            number_mask_allowed(ngs),
        ], axis=-1).astype(np.uint8)
        cleaned = (raw_masks & allowed).astype(np.uint8)

        # Track drop stats
        for ci, name in enumerate(["yard", "side", "hash", "number"]):
            raw_n = int(raw_masks[..., ci].sum())
            cln_n = int(cleaned[..., ci].sum())
            drop_frac = (raw_n - cln_n) / max(1, raw_n)
            drop_stats[name].append(drop_frac)

        # Always save BOTH raw_masks (for cheap future re-clean) and the
        # current cleaned masks (consumed by training).
        np.savez_compressed(out_path, rgb=frame,
                             raw_masks=raw_masks, masks=cleaned)

        if args.save_thumbs:
            thumb = frame.copy()
            ov = thumb.copy()
            ov[cleaned[..., 0] > 0] = (60, 60, 230)     # yard - red
            ov[cleaned[..., 1] > 0] = (60, 230, 60)     # side - green
            ov[cleaned[..., 2] > 0] = (230, 60, 60)     # hash - blue
            ov[cleaned[..., 3] > 0] = (60, 230, 230)    # number - yellow
            thumb = cv2.addWeighted(ov, 0.55, thumb, 0.45, 0)
            cv2.imwrite(os.path.join(thumb_dir, f"{e['id']}.jpg"), thumb,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        n_done += 1
        if i % 50 == 0 or i == len(entries):
            elapsed = time.time() - t_start
            eta = elapsed / max(1, i - n_skipped) * (len(entries) - i)
            print(f"  [{i}/{len(entries)}] new={n_done} cached={n_skipped} "
                  f"failed={n_failed}  ({elapsed:.0f}s elapsed, eta {eta:.0f}s)",
                  flush=True)

    # Drop-rate summary (how aggressive was the cleaning per channel)
    print(f"\nMean drop fractions per channel (raw → cleaned):")
    for name in ["yard", "side", "hash", "number"]:
        if drop_stats[name]:
            mean = np.mean(drop_stats[name])
            med = np.median(drop_stats[name])
            print(f"  {name}: mean={mean*100:.1f}%, median={med*100:.1f}%")

    # Build manifest — but NEVER overwrite if this was a partial run
    # (--max-entries) or we'd silently shrink the canonical manifest and
    # break downstream training (this is what blew up v4 the first time).
    if args.max_entries:
        print(f"\nSkipping manifest write (--max-entries={args.max_entries} "
              f"is a partial run — leaving existing manifest intact).")
    else:
        manifest_out = {
            "n_total": n_done + n_skipped,
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_pool": args.pool_dir,
            "source_decisions_y_count": len(entries),
            # Preserve H so the manifest can drive future rebuilds (--reuse-rgb,
            # --reclean-only) without needing the source pool around. ~0.5 KB
            # per entry; ~1 MB total at 2500 entries.
            "entries": [
                {"id": e["id"], "clip": e["clip"], "frame_idx": e["frame_idx"],
                 "game": e["clip"].split("/")[0],
                 "H": e["H"].tolist() if hasattr(e.get("H"), "tolist") else e.get("H")}
                for e in entries
            ],
        }
        out_manifest = os.path.join(args.out_dir, "dataset_manifest.json")
        with open(out_manifest, "w") as f:
            json.dump(manifest_out, f, indent=2)
        print(f"\nManifest -> {out_manifest}")
    print(f"  {n_done} new, {n_skipped} resumed, {n_failed} failed in "
          f"{time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
