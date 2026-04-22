#!/usr/bin/env python3
"""
Sanity-check the converted keypoint dataset.

Loads a few frames from the train/valid splits and overlays the annotated
keypoints (sideline=red, hash=green) to verify the conversion from Label
Studio to training format is correct.
"""

import os
import sys
import json
import cv2
import numpy as np
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "field_keypoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "dataset_check")

CHANNEL_COLORS = {
    0: (0, 0, 255),   # sideline = red (BGR)
    1: (0, 255, 0),   # hash = green
}
CHANNEL_NAMES = {0: "sideline", 1: "hash"}


def render_split(split_name: str, n_samples: int = 4):
    split_dir = os.path.join(DATA_DIR, split_name)
    ann_path = os.path.join(split_dir, "annotations.json")
    img_dir = os.path.join(split_dir, "images")

    with open(ann_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    n_total = len(coco["annotations"])
    print(f"\n=== {split_name} ({n_total} images) ===")

    random.seed(0)
    indices = random.sample(range(n_total), min(n_samples, n_total))

    for idx in indices:
        ann = coco["annotations"][idx]
        img_info = images_by_id[ann["image_id"]]
        fname = img_info["file_name"]

        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{idx}] FAILED to load {fname}")
            continue

        n_side = sum(1 for p in ann["points"] if p["channel"] == 0)
        n_hash = sum(1 for p in ann["points"] if p["channel"] == 1)
        print(f"  [{idx}] {fname}: {n_side} sideline, {n_hash} hash")

        # Draw keypoints
        for p in ann["points"]:
            color = CHANNEL_COLORS[p["channel"]]
            center = (int(round(p["x"])), int(round(p["y"])))
            cv2.circle(img, center, 5, color, 2)

        # Legend
        cv2.rectangle(img, (8, 8), (220, 60), (0, 0, 0), -1)
        cv2.circle(img, (20, 25), 5, CHANNEL_COLORS[0], 2)
        cv2.putText(img, "sideline", (32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (20, 50), 5, CHANNEL_COLORS[1], 2)
        cv2.putText(img, "hash", (32, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"{split_name}_{idx:03d}_{fname}")
        cv2.imwrite(out_path, img)

    # Distribution stats
    per_frame_side = []
    per_frame_hash = []
    for ann in coco["annotations"]:
        n_side = sum(1 for p in ann["points"] if p["channel"] == 0)
        n_hash = sum(1 for p in ann["points"] if p["channel"] == 1)
        per_frame_side.append(n_side)
        per_frame_hash.append(n_hash)

    print(f"  Sideline per frame: min={min(per_frame_side)} max={max(per_frame_side)} "
          f"mean={np.mean(per_frame_side):.1f} median={int(np.median(per_frame_side))}")
    print(f"  Hash per frame:     min={min(per_frame_hash)} max={max(per_frame_hash)} "
          f"mean={np.mean(per_frame_hash):.1f} median={int(np.median(per_frame_hash))}")

    # Coordinate range sanity check
    all_x = [p["x"] for ann in coco["annotations"] for p in ann["points"]]
    all_y = [p["y"] for ann in coco["annotations"] for p in ann["points"]]
    print(f"  X range: {min(all_x):.1f} to {max(all_x):.1f} (should be 0–1280)")
    print(f"  Y range: {min(all_y):.1f} to {max(all_y):.1f} (should be 0–720)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    render_split("train", n_samples=4)
    render_split("valid", n_samples=4)
    print(f"\nSaved visualizations to {OUT_DIR}/")


if __name__ == "__main__":
    main()
