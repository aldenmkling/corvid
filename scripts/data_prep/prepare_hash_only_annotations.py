#!/usr/bin/env python3
"""
Build a hash-only variant of the keypoint dataset.

Reads the existing 2-channel annotations (channel 0 = sideline, channel 1 =
hash) and writes a 1-channel variant where only hash points remain,
remapped to channel 0. Keeps frames WITHOUT hashes too — they act as hard
negatives (the loss sees an all-zero target and penalizes any firing).
Symlinks the image directories so we don't duplicate disk.

Usage:
  python scripts/data_prep/prepare_hash_only_annotations.py

Output:
  data/field_keypoints_hash_only/
    train/
      annotations.json  # hash-only, channel remapped to 0
      images            # symlink to data/field_keypoints/train/images
    valid/
      annotations.json
      images
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SRC = os.path.join(PROJECT_ROOT, "data/field_keypoints")
DEFAULT_DST = os.path.join(PROJECT_ROOT, "data/field_keypoints_hash_only")

HASH_CHANNEL = 1   # channel id in source annotations


def filter_split(src_split_dir, dst_split_dir):
    os.makedirs(dst_split_dir, exist_ok=True)
    src_ann_path = os.path.join(src_split_dir, "annotations.json")
    dst_ann_path = os.path.join(dst_split_dir, "annotations.json")

    with open(src_ann_path) as f:
        coco = json.load(f)

    kept_anns = []
    n_points_total = 0
    n_points_kept = 0
    n_hashless_frames = 0
    for ann in coco["annotations"]:
        pts = ann["points"]
        n_points_total += len(pts)
        hash_pts = [p for p in pts if p.get("channel") == HASH_CHANNEL]
        n_points_kept += len(hash_pts)
        if not hash_pts:
            n_hashless_frames += 1
        remapped = [{**p, "channel": 0} for p in hash_pts]
        # Keep the annotation even if empty — acts as a hard-negative frame.
        kept_anns.append({**ann, "points": remapped})

    out = {"images": coco["images"], "annotations": kept_anns}
    with open(dst_ann_path, "w") as f:
        json.dump(out, f)

    # Symlink images/ dir (avoid duplicating ~100MB of jpgs)
    src_img_dir = os.path.join(src_split_dir, "images")
    dst_img_link = os.path.join(dst_split_dir, "images")
    if os.path.islink(dst_img_link) or os.path.exists(dst_img_link):
        os.unlink(dst_img_link) if os.path.islink(dst_img_link) else None
    if not os.path.exists(dst_img_link):
        os.symlink(src_img_dir, dst_img_link)

    return dict(
        frames_in=len(coco["images"]),
        frames_out=len(coco["images"]),
        anns_in=len(coco["annotations"]),
        anns_out=len(kept_anns),
        hashless_frames=n_hashless_frames,
        points_in=n_points_total,
        points_out=n_points_kept,
    )


def main(args):
    for split in ("train", "valid"):
        src = os.path.join(args.source, split)
        dst = os.path.join(args.dest, split)
        print(f"[{split}] {src} → {dst}")
        stats = filter_split(src, dst)
        print(f"  frames:      {stats['frames_in']} → {stats['frames_out']}  "
              f"(kept all; {stats['hashless_frames']} are hard negatives with no hashes)")
        print(f"  annotations: {stats['anns_in']} → {stats['anns_out']}")
        print(f"  points:      {stats['points_in']} → {stats['points_out']}  "
              f"(kept hash-only; channel 1 → 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SRC)
    ap.add_argument("--dest", default=DEFAULT_DST)
    main(ap.parse_args())
