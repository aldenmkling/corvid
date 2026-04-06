#!/usr/bin/env python3
"""
Pre-annotate training frames using YOLOv12x to bootstrap labeling.

Runs detection on all extracted frames and writes YOLO-format label files.
Human annotators then only need to correct mistakes (add missed players,
remove false positives) rather than labeling from scratch.

Output: YOLO format .txt files alongside images
  - One row per detection: class_id center_x center_y width height (normalized)
  - Class 0 = player

Usage:
  python scripts/pre_annotate.py [--conf-thresh 0.20]
"""

import argparse
import os
import glob

import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "images")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "labels")

PERSON_CLASS = 0  # COCO person class

# Sideline views have coaches/cameramen along the bottom edge.
# Filter out detections whose center y is in the bottom 15% of the frame.
SIDELINE_BOTTOM_CUTOFF = 0.85  # normalized y — ignore detections below this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-thresh", type=float, default=0.20,
                        help="Confidence threshold (lower = more detections to correct)")
    parser.add_argument("--model", default="models/best.pt")
    args = parser.parse_args()

    os.makedirs(LABELS_DIR, exist_ok=True)

    print(f"Loading {args.model}...")
    model = YOLO(args.model)

    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not images:
        print(f"No images found in {IMAGES_DIR}/")
        print("Run extract_training_frames.py first.")
        return

    print(f"Pre-annotating {len(images)} images (conf >= {args.conf_thresh})...")

    total_detections = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]

        results = model(frame, verbose=False)[0]
        boxes = results.boxes
        person_mask = boxes.conf >= args.conf_thresh
        person_mask &= boxes.cls == PERSON_CLASS
        person_boxes = boxes[person_mask]

        # Write YOLO format labels, filtering sideline personnel
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        is_sideline = "sideline" in os.path.basename(img_path)

        n = 0
        with open(label_path, "w") as f:
            for box in person_boxes.xywhn.cpu().numpy():
                cx, cy, bw, bh = box
                # Skip detections near the bottom of sideline views
                if is_sideline and cy > SIDELINE_BOTTOM_CUTOFF:
                    continue
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                n += 1

        total_detections += n

    avg = total_detections / len(images) if images else 0
    print(f"\nDone! {total_detections} total detections ({avg:.1f} per image)")
    print(f"Labels: {LABELS_DIR}/")
    print(f"\nTo review/correct annotations, open in Label Studio or CVAT")
    print(f"pointing at {IMAGES_DIR}/ with labels from {LABELS_DIR}/")


if __name__ == "__main__":
    main()
