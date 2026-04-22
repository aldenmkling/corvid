#!/usr/bin/env python3
"""
Merge multiple Label Studio exports into a single train/valid split.

Pools all annotated frames across rounds of active learning + original set,
deduplicates by filename (newest export wins), and produces fresh train/valid
directories that the HRNet training script expects.
"""

import argparse
import json
import os
import random
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data_prep.export_keypoints_coco import (
    parse_ls_export, stratified_split, write_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exports",
        nargs="+",
        required=True,
        help="Paths to LS export JSONs, in order of precedence (later wins on conflicts).",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        required=True,
        help="Directories of source images, one per export (same order).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "field_keypoints"),
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.exports) != len(args.image_dirs):
        print("ERROR: --exports and --image-dirs must have the same number of entries")
        sys.exit(1)

    # Build a map filename → (annotation, image_dir) so each file lives in ONE spot.
    merged = {}
    for export_path, image_dir in zip(args.exports, args.image_dirs):
        annotations = parse_ls_export(export_path)
        print(f"Loaded {len(annotations)} annotations from {export_path}")
        for ann in annotations:
            # Later exports override earlier (no-op on first pass)
            merged[ann["file_name"]] = {**ann, "_image_dir": image_dir}

    annotations = list(merged.values())
    print(f"\nMerged total: {len(annotations)} unique frames")

    # For stratified split, parse_ls_export yields dicts with file_name etc;
    # we attached _image_dir but the existing split/write path expects only
    # the original keys. Remove _image_dir before writing.
    random.seed(args.seed)
    train, val = stratified_split(annotations, args.val_split, args.seed)
    print(f"Split: {len(train)} train, {len(val)} val")

    # Clean old train/valid if present
    for sub in ("train", "valid"):
        d = os.path.join(args.output, sub)
        if os.path.exists(d):
            shutil.rmtree(d)

    # Custom writer that copies each image from whatever dir it came from.
    def write_split_multi_src(split_anns, out_dir):
        img_out_dir = os.path.join(out_dir, "images")
        os.makedirs(img_out_dir, exist_ok=True)
        images = []
        coco_annotations = []
        for idx, ann in enumerate(split_anns):
            fname = ann["file_name"]
            src_dir = ann["_image_dir"]
            src_path = os.path.join(src_dir, fname)
            if not os.path.exists(src_path):
                print(f"  WARN: image not found: {src_path}")
                continue
            dst_path = os.path.join(img_out_dir, fname)
            shutil.copy2(src_path, dst_path)
            images.append({
                "id": idx, "file_name": fname,
                "width": ann["width"], "height": ann["height"],
            })
            coco_annotations.append({
                "id": idx, "image_id": idx, "points": ann["points"],
            })
        coco = {"images": images, "annotations": coco_annotations}
        with open(os.path.join(out_dir, "annotations.json"), "w") as f:
            json.dump(coco, f, indent=2)
        n_pts = sum(len(a["points"]) for a in coco_annotations)
        n_side = sum(1 for a in coco_annotations for p in a["points"] if p["channel"] == 0)
        n_hash = sum(1 for a in coco_annotations for p in a["points"] if p["channel"] == 1)
        print(f"  {len(images)} images, {n_pts} keypoints "
              f"({n_side} sideline, {n_hash} hash), "
              f"avg {n_pts / max(len(images), 1):.1f}/frame")

    print("\nTrain:")
    write_split_multi_src(train, os.path.join(args.output, "train"))
    print("\nVal:")
    write_split_multi_src(val, os.path.join(args.output, "valid"))

    print(f"\nDone. Dataset at {args.output}/train and {args.output}/valid")


if __name__ == "__main__":
    main()
