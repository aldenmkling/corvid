#!/usr/bin/env python3
"""Calibration script: extract large crops around annotated hash points
and stitch them into a visual grid so we can eyeball:
  - hash mark perpendicular length (px)
  - hash mark parallel width (px)
  - local yardline width (px) at the hash position

Used to set scaling constants for the mask generator.

Picks a stratified sample (varying perspective: top of image = far / small,
bottom = near / large) so we see hash sizes at multiple zooms.
"""

import json
import os
import random

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KP_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints/train")
OUT = os.path.join(PROJECT_ROOT, "output/hash_calibration_panel.jpg")

CROP_HALF = 30           # 60×60 crops
N_PER_BUCKET = 4         # samples per perspective bucket
BUCKETS_Y = [(0, 240), (240, 480), (480, 720)]   # top / mid / bottom thirds
LABEL_HEIGHT = 18        # text label below each crop
random.seed(7)


def main():
    with open(os.path.join(KP_DIR, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    # Collect (frame_path, hash_point_xy) tuples bucketed by image-y
    buckets = {b: [] for b in BUCKETS_Y}
    for ann in coco["annotations"]:
        info = images_by_id[ann["image_id"]]
        path = os.path.join(KP_DIR, "images", info["file_name"])
        if not os.path.exists(path): continue
        for p in ann["points"]:
            if p["channel"] != 1: continue   # hash channel
            for b in BUCKETS_Y:
                if b[0] <= p["y"] < b[1]:
                    buckets[b].append((path, p["x"], p["y"]))
                    break

    samples = []
    for b in BUCKETS_Y:
        random.shuffle(buckets[b])
        samples.extend(buckets[b][:N_PER_BUCKET])
    print(f"  collected {len(samples)} sample crops")

    # Build grid: N_PER_BUCKET wide × len(BUCKETS_Y) tall
    cell_h = 2 * CROP_HALF + LABEL_HEIGHT
    cell_w = 2 * CROP_HALF
    grid_h = cell_h * len(BUCKETS_Y)
    grid_w = cell_w * N_PER_BUCKET
    panel = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

    for i, (path, x, y) in enumerate(samples):
        row = i // N_PER_BUCKET
        col = i % N_PER_BUCKET
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        x, y = int(round(x)), int(round(y))
        x0 = max(0, x - CROP_HALF); x1 = min(w, x + CROP_HALF)
        y0 = max(0, y - CROP_HALF); y1 = min(h, y + CROP_HALF)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0: continue
        # Pad to consistent cell size
        pad_top = (y - CROP_HALF < 0) * (CROP_HALF - y) if y - CROP_HALF < 0 else 0
        pad_left = (x - CROP_HALF < 0) * (CROP_HALF - x) if x - CROP_HALF < 0 else 0
        padded = np.zeros((2 * CROP_HALF, 2 * CROP_HALF, 3), dtype=np.uint8)
        padded[pad_top:pad_top + crop.shape[0],
               pad_left:pad_left + crop.shape[1]] = crop
        # Draw a small dot at the annotated point
        center = (CROP_HALF, CROP_HALF)
        cv2.drawMarker(padded, center, (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        # Place into grid
        cell_y = row * cell_h
        cell_x = col * cell_w
        panel[cell_y:cell_y + 2 * CROP_HALF,
              cell_x:cell_x + 2 * CROP_HALF] = padded
        # Label
        label = f"y={y}"
        cv2.putText(panel, label,
                    (cell_x + 4, cell_y + 2 * CROP_HALF + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, panel)
    print(f"  wrote {OUT}")
    print(f"  scale: each cell is {2*CROP_HALF}×{2*CROP_HALF} px (1 px image = 1 px panel)")


if __name__ == "__main__":
    main()
