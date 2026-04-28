#!/usr/bin/env python3
"""Convert our `{points: [{x, y, channel, visible}], ...}` annotations
into MMPose RTMO-compatible COCO-pose format. Hash-only (ch=1).

Each hash becomes one "object" with a small bbox + 1 keypoint at its position.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints_rtmo")

BBOX_HALF = 16        # 32×32 px box around each hash; gives the detection
                      # head some context to localize against.


def convert_split(split_name: str):
    src_ann = os.path.join(SRC_DIR, split_name, "annotations.json")
    src_img_dir = os.path.join(SRC_DIR, split_name, "images")
    out_dir = os.path.join(OUT_DIR, split_name)
    out_ann = os.path.join(out_dir, "annotations.json")
    os.makedirs(out_dir, exist_ok=True)

    # Symlink images dir.
    out_img = os.path.join(out_dir, "images")
    if not os.path.exists(out_img):
        os.symlink(src_img_dir, out_img)

    with open(src_ann) as f:
        src = json.load(f)

    out_images = list(src["images"])
    out_categories = [{
        "id": 1, "name": "field_hash", "supercategory": "field",
        "keypoints": ["hash"], "skeleton": [],
    }]
    out_anns = []
    next_id = 1

    for ann in src["annotations"]:
        img_id = ann["image_id"]
        img = next((i for i in src["images"] if i["id"] == img_id), None)
        if img is None: continue
        for p in ann["points"]:
            if p["channel"] != 1: continue       # hash-only
            if not p.get("visible", True): continue
            x, y = float(p["x"]), float(p["y"])
            bx = max(0, x - BBOX_HALF)
            by = max(0, y - BBOX_HALF)
            bw = min(img["width"] - bx, 2 * BBOX_HALF)
            bh = min(img["height"] - by, 2 * BBOX_HALF)
            out_anns.append({
                "id": next_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [bx, by, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
                "num_keypoints": 1,
                "keypoints": [x, y, 2],   # 2 = visible
                "segmentation": [[]],
            })
            next_id += 1

    out = {
        "images": out_images,
        "annotations": out_anns,
        "categories": out_categories,
    }
    with open(out_ann, "w") as f:
        json.dump(out, f)
    print(f"  {split_name}: {len(out_images)} images, "
          f"{len(out_anns)} hash objects → {out_ann}")


def main():
    for split in ("train", "valid"):
        convert_split(split)


if __name__ == "__main__":
    main()
