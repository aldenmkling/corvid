#!/usr/bin/env python3
"""
Convert Label Studio keypoint export to COCO keypoints format.

Takes a Label Studio JSON export and produces the COCO keypoints format
expected by the training script:
  data/field_keypoints/{train,valid}/
    images/*.jpg
    annotations.json

Usage:
    python scripts/export_keypoints_coco.py \
        --ls-export data/field_keypoints/export.json \
        --images data/field_keypoints/annotation_images/ \
        --output dataset_keypoints \
        --val-split 0.2
"""

import os
import sys
import json
import shutil
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.homography.keypoint_schema import (
    KEYPOINTS, KEYPOINT_NAMES, NUM_KEYPOINTS, ID_BY_NAME,
)


def parse_ls_export(ls_path: str) -> list[dict]:
    """Parse Label Studio JSON export into per-image keypoint annotations.

    Returns list of {file_name, keypoints: {label: (x_pct, y_pct)}}
    """
    with open(ls_path) as f:
        data = json.load(f)

    results = []

    for task in data:
        file_name = task.get("file_upload", "")
        # Label Studio may use different keys for the image path
        if not file_name and "data" in task:
            file_name = task["data"].get("image", "")
        # Extract just the filename
        file_name = os.path.basename(file_name)

        keypoints = {}
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "keypointlabels":
                    continue
                value = result["value"]
                label = value["keypointlabels"][0]
                # x, y are percentages of image dimensions
                x_pct = value["x"]
                y_pct = value["y"]
                keypoints[label] = (x_pct, y_pct)

        if keypoints:
            results.append({
                "file_name": file_name,
                "keypoints": keypoints,
            })

    return results


def convert_to_coco(
    annotations: list[dict],
    img_dir: str,
    output_dir: str,
    img_w: int = 1280,
    img_h: int = 720,
):
    """Convert parsed annotations to COCO keypoints format."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    images = []
    coco_annotations = []

    for idx, ann in enumerate(annotations):
        fname = ann["file_name"]
        src_path = os.path.join(img_dir, fname)

        if not os.path.exists(src_path):
            print(f"Warning: image not found: {src_path}")
            continue

        # Copy image
        dst_path = os.path.join(output_dir, "images", fname)
        shutil.copy2(src_path, dst_path)

        images.append({
            "id": idx,
            "file_name": fname,
            "width": img_w,
            "height": img_h,
        })

        # Build keypoints array: [x0, y0, v0, x1, y1, v1, ...]
        kp_flat = []
        n_visible = 0

        for ki in range(NUM_KEYPOINTS):
            kp_name = KEYPOINT_NAMES[ki]
            if kp_name in ann["keypoints"]:
                x_pct, y_pct = ann["keypoints"][kp_name]
                x_px = x_pct / 100.0 * img_w
                y_px = y_pct / 100.0 * img_h
                kp_flat.extend([x_px, y_px, 2])  # 2 = visible and labeled
                n_visible += 1
            else:
                kp_flat.extend([0.0, 0.0, 0])  # 0 = not labeled

        coco_annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "keypoints": kp_flat,
            "num_keypoints": n_visible,
            "bbox": [0, 0, img_w, img_h],
            "area": img_w * img_h,
            "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": coco_annotations,
        "categories": [{
            "id": 1,
            "name": "field",
            "supercategory": "field",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [],
        }],
    }

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved {len(images)} images and annotations to {output_dir}")
    avg_kps = sum(a["num_keypoints"] for a in coco_annotations) / max(len(coco_annotations), 1)
    print(f"Average keypoints per frame: {avg_kps:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls-export", required=True,
                        help="Label Studio JSON export file")
    parser.add_argument("--images", required=True,
                        help="Directory containing source images")
    parser.add_argument("--output", default="data/field_keypoints",
                        help="Output directory (will create train/ and valid/ subdirs)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    img_dir = os.path.join(project_root, args.images) if not os.path.isabs(args.images) else args.images

    annotations = parse_ls_export(args.ls_export)
    print(f"Parsed {len(annotations)} annotated images from Label Studio export")

    # Stratified split by game_id (extracted from filename)
    by_game = {}
    for ann in annotations:
        game_id = ann["file_name"].split("_")[0]
        by_game.setdefault(game_id, []).append(ann)

    train_anns = []
    val_anns = []

    for game_id, game_anns in by_game.items():
        random.shuffle(game_anns)
        n_val = max(1, int(len(game_anns) * args.val_split))
        val_anns.extend(game_anns[:n_val])
        train_anns.extend(game_anns[n_val:])

    print(f"Split: {len(train_anns)} train, {len(val_anns)} val")

    convert_to_coco(train_anns, img_dir, os.path.join(output_dir, "train"))
    convert_to_coco(val_anns, img_dir, os.path.join(output_dir, "valid"))

    print(f"\nDone. Dataset ready at {output_dir}/")


if __name__ == "__main__":
    main()
