#!/usr/bin/env python3
"""
Build a line-segmentation dataset from hand-review decisions.

Reads review_decisions.json (from review_line_labels.py) and:
  - keep   → save full frame + full mask
  - crop   → crop frame + mask to the saved bbox, save the crop
  - reject → skip
  - missing decision → skip

Overwrites data/line_detection/ — same layout as build_line_dataset.py.
Uses the same render_masks code (adaptive yard widths, snap, etc.).
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/data_prep"))

from build_line_dataset import render_masks, grab_frame

DEFAULT_POOL = os.path.join(PROJECT_ROOT, "output/self_sup_pool_10k")
DEFAULT_CLIPS = os.path.join(PROJECT_ROOT, "videos/clips")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data/line_detection")
VAL_FRACTION = 0.20
RNG_SEED = 42


def load_summary_rows(pool_dir):
    """Map frame_id → pool row (includes game/play/angle/frame_idx/h_path)."""
    path = os.path.join(pool_dir, "summary.csv")
    if not os.path.exists(path):
        print(f"missing {path}"); sys.exit(1)
    return {r["frame_id"]: r for r in csv.DictReader(open(path))
            if r.get("solved") == "True"}


def stratified_split_by_game(items, val_fraction, rng):
    by_game = defaultdict(list)
    for r in items:
        by_game[r["game"]].append(r)
    train, val = [], []
    for game in sorted(by_game):
        rows = by_game[game][:]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_fraction))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def write_split(items, pool_dir, clips_dir, split_dir):
    img_dir = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    n_written, n_failed = 0, 0
    for r in items:
        mp4 = os.path.join(clips_dir, r["game"], r["play"], f"{r['angle']}.mp4")
        frame = grab_frame(mp4, int(r["frame_idx"]))
        if frame is None:
            n_failed += 1; continue
        with open(os.path.join(pool_dir, r["h_path"]), "rb") as f:
            hd = pickle.load(f)
        yard, side = render_masks(frame, hd["H"], hd["k1"], hd["k2"])

        # Pack yard (R), side (G) into a 3-ch PNG. BGR ordering for cv2.imwrite.
        mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        mask[..., 2] = yard   # R
        mask[..., 1] = side   # G

        # Apply crop if present
        if r["decision"] == "crop":
            x, y, w, h = r["crop"]
            # Clamp to image bounds just in case
            x = max(0, x); y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)
            if w < 32 or h < 32:
                n_failed += 1; continue
            frame = frame[y:y+h, x:x+w]
            mask = mask[y:y+h, x:x+w]

        fid = r["frame_id"]
        cv2.imwrite(os.path.join(img_dir, f"{fid}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(mask_dir, f"{fid}.png"), mask)
        n_written += 1
        if n_written % 50 == 0:
            print(f"  wrote {n_written}...")
    return n_written, n_failed


def main(args):
    # Load decisions
    with open(args.decisions) as f:
        decisions = json.load(f)
    pool = load_summary_rows(args.pool_dir)

    items = []
    n_keep, n_crop, n_reject, n_skip, n_missing = 0, 0, 0, 0, 0
    for fid, dec in decisions.items():
        status = dec.get("status") if isinstance(dec, dict) else dec
        if status == "reject":
            n_reject += 1; continue
        if status not in ("keep", "crop"):
            n_skip += 1; continue
        if fid not in pool:
            n_missing += 1; continue
        row = dict(pool[fid])
        row["decision"] = status
        if status == "crop":
            row["crop"] = dec["crop"]
            n_crop += 1
        else:
            n_keep += 1
        items.append(row)

    print(f"decisions: {len(decisions)} total")
    print(f"  keep:    {n_keep}")
    print(f"  crop:    {n_crop}")
    print(f"  reject:  {n_reject}")
    print(f"  skipped (unknown status): {n_skip}")
    print(f"  missing from pool:        {n_missing}")
    print(f"→ {len(items)} usable frames")

    rng = random.Random(RNG_SEED)
    train, val = stratified_split_by_game(items, VAL_FRACTION, rng)
    print(f"split: {len(train)} train, {len(val)} val")

    os.makedirs(args.out_dir, exist_ok=True)

    # Remove any old contents
    for split in ("train", "valid"):
        for sub in ("images", "masks"):
            d = os.path.join(args.out_dir, split, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))

    print("\nwriting train split...")
    n_train, f_train = write_split(train, args.pool_dir, args.clips_dir,
                                     os.path.join(args.out_dir, "train"))
    print("\nwriting valid split...")
    n_val, f_val = write_split(val, args.pool_dir, args.clips_dir,
                                 os.path.join(args.out_dir, "valid"))

    per_game = defaultdict(lambda: {"train": 0, "val": 0})
    for r in train:
        per_game[r["game"]]["train"] += 1
    for r in val:
        per_game[r["game"]]["val"] += 1

    manifest = {
        "source": "review_decisions.json",
        "total_decisions": len(decisions),
        "keep": n_keep, "crop": n_crop, "reject": n_reject,
        "skipped_unknown": n_skip, "missing_from_pool": n_missing,
        "used": len(items),
        "train": n_train, "train_failed": f_train,
        "valid": n_val, "valid_failed": f_val,
        "per_game_counts": dict(per_game),
        "mask_format": "3ch PNG, R=yard, G=side, B=0 (BGR on disk)",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ndone. manifest → {args.out_dir}/manifest.json")
    print(f"train: {n_train} written, {f_train} failed")
    print(f"valid: {n_val} written, {f_val} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=DEFAULT_POOL)
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--decisions", default=None,
                    help="Path to review_decisions.json (default: <pool>/review_decisions.json)")
    args = ap.parse_args()
    if args.decisions is None:
        args.decisions = os.path.join(args.pool_dir, "review_decisions.json")
    main(args)
