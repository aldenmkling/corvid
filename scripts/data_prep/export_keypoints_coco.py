#!/usr/bin/env python3
"""
Convert Label Studio keypoint export to training-ready format.

The 2-channel HRNet model takes:
  - Channel 0: sideline_intersection
  - Channel 1: hash_intersection

Each frame can have arbitrary numbers of peaks per channel. Output format
is a minimal COCO-like annotations.json that the training script reads:

  {
    "images": [{"id": N, "file_name": "foo.jpg", "width": W, "height": H}, ...],
    "annotations": [{"image_id": N, "points": [{"x": px, "y": py, "channel": 0|1, "visible": true}, ...]}, ...]
  }

Creates train/ and valid/ subdirectories with stratified split by game_id.

Usage:
    python scripts/data_prep/export_keypoints_coco.py \
        --ls-export data/field_keypoints/field_keypoints_hand_labeled.json \
        --images data/field_keypoints/annotation_images \
        --output data/field_keypoints \
        --val-split 0.2
"""

import os
import sys
import json
import shutil
import argparse
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_TO_CHANNEL = {
    "sideline_intersection": 0,
    "hash_intersection": 1,
}


def parse_ls_export(ls_path: str) -> list[dict]:
    """Parse Label Studio JSON export into per-image keypoint annotations.

    Returns list of {file_name, width, height, points: [{x, y, channel, visible}, ...]}
    """
    with open(ls_path) as f:
        data = json.load(f)

    results = []

    for task in data:
        # Extract filename from task data.image path
        image_path = task.get("data", {}).get("image", "")
        # Path looks like: /data/local-files/?d=annotation_images/filename.jpg
        file_name = os.path.basename(image_path)
        if not file_name:
            continue

        # Collect keypoints from annotations (not predictions)
        points = []
        width = None
        height = None

        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for r in ann.get("result", []):
                if r.get("type") != "keypointlabels":
                    continue

                value = r["value"]
                label = value["keypointlabels"][0]
                if label not in LABEL_TO_CHANNEL:
                    continue

                # LS uses percentage coordinates
                x_pct = value["x"]
                y_pct = value["y"]

                # Capture original image dimensions from the result metadata
                if width is None:
                    width = r.get("original_width", 1280)
                    height = r.get("original_height", 720)

                x_px = x_pct / 100.0 * width
                y_px = y_pct / 100.0 * height

                points.append({
                    "x": x_px,
                    "y": y_px,
                    "channel": LABEL_TO_CHANNEL[label],
                    "visible": True,
                })

        if not points:
            continue

        results.append({
            "file_name": file_name,
            "width": width or 1280,
            "height": height or 720,
            "points": points,
        })

    return results


def stratified_split(annotations: list[dict], val_split: float, seed: int = 42):
    """Split annotations by game_id (first 10 chars of filename)."""
    random.seed(seed)

    by_game = {}
    for ann in annotations:
        game_id = ann["file_name"].split("_")[0]
        by_game.setdefault(game_id, []).append(ann)

    train_anns = []
    val_anns = []

    for game_id, game_anns in by_game.items():
        random.shuffle(game_anns)
        n_val = max(1, int(round(len(game_anns) * val_split)))
        val_anns.extend(game_anns[:n_val])
        train_anns.extend(game_anns[n_val:])

    return train_anns, val_anns


def write_split(annotations: list[dict], img_src_dir: str, out_dir: str):
    """Write images and annotations.json for one split."""
    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)

    images = []
    coco_annotations = []

    for idx, ann in enumerate(annotations):
        fname = ann["file_name"]
        src_path = os.path.join(img_src_dir, fname)

        if not os.path.exists(src_path):
            print(f"  WARN: image not found: {src_path}")
            continue

        dst_path = os.path.join(img_out_dir, fname)
        shutil.copy2(src_path, dst_path)

        images.append({
            "id": idx,
            "file_name": fname,
            "width": ann["width"],
            "height": ann["height"],
        })

        coco_annotations.append({
            "id": idx,
            "image_id": idx,
            "points": ann["points"],
        })

    coco = {
        "images": images,
        "annotations": coco_annotations,
    }

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    n_pts = sum(len(a["points"]) for a in coco_annotations)
    n_sideline = sum(1 for a in coco_annotations for p in a["points"] if p["channel"] == 0)
    n_hash = sum(1 for a in coco_annotations for p in a["points"] if p["channel"] == 1)
    print(f"  {len(images)} images, {n_pts} keypoints ({n_sideline} sideline, {n_hash} hash)")
    print(f"  avg {n_pts / max(len(images), 1):.1f} keypoints/frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls-export", required=True,
                        help="Label Studio JSON export file")
    parser.add_argument("--images", required=True,
                        help="Directory containing source images")
    parser.add_argument("--output", default="data/field_keypoints",
                        help="Output directory (creates train/ and valid/)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ls_path = args.ls_export if os.path.isabs(args.ls_export) else os.path.join(PROJECT_ROOT, args.ls_export)
    img_dir = args.images if os.path.isabs(args.images) else os.path.join(PROJECT_ROOT, args.images)
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(PROJECT_ROOT, args.output)

    annotations = parse_ls_export(ls_path)
    print(f"Parsed {len(annotations)} annotated images")

    train, val = stratified_split(annotations, args.val_split, args.seed)
    print(f"Split: {len(train)} train, {len(val)} val")

    print("\nTrain:")
    write_split(train, img_dir, os.path.join(out_dir, "train"))
    print("\nVal:")
    write_split(val, img_dir, os.path.join(out_dir, "valid"))

    print(f"\nDone. Dataset at {out_dir}/train and {out_dir}/valid")


if __name__ == "__main__":
    main()
