#!/usr/bin/env python3
"""
Quick baseline test: run pretrained YOLO on All-22 frames to see
how well off-the-shelf person detection works.

Usage:
  python scripts/detect_test.py
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips", "2019092204")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "detect_test_v12x")

# Test on a few plays, grabbing frames at different points
TEST_PLAYS = [1, 10, 50]
# Grab frames at these percentages through each clip
SAMPLE_PCTS = [0.1, 0.3, 0.5]


def extract_frame(video_path: str, pct: float):
    """Extract a single frame at the given percentage through the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = int(total * pct)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pretrained YOLOv12x
    print("Loading YOLOv12x pretrained on COCO...")
    model = YOLO("yolo12x.pt")

    # COCO class 0 = person
    PERSON_CLASS = 0

    for play_num in TEST_PLAYS:
        play_dir = os.path.join(CLIPS_DIR, f"play_{play_num:03d}")
        if not os.path.exists(play_dir):
            print(f"  Play {play_num} not found, skipping")
            continue

        for view in ["sideline", "endzone"]:
            video_path = os.path.join(play_dir, f"{view}.mp4")
            if not os.path.exists(video_path):
                continue

            for pct in SAMPLE_PCTS:
                frame = extract_frame(video_path, pct)
                if frame is None:
                    continue

                # Run detection
                results = model(frame, verbose=False)[0]

                # Filter to person class
                boxes = results.boxes
                person_mask = boxes.cls == PERSON_CLASS
                person_boxes = boxes[person_mask]

                n_persons = len(person_boxes)
                confs = person_boxes.conf.cpu().numpy()

                # Draw detections
                annotated = frame.copy()
                for i, box in enumerate(person_boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = confs[i]
                    color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated, f"{conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )

                # Add summary text
                label = f"Play {play_num} {view} @{pct:.0%}: {n_persons} persons"
                cv2.putText(
                    annotated, label,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )

                out_name = f"play{play_num:03d}_{view}_{int(pct*100)}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, annotated)

                conf_str = ""
                if len(confs) > 0:
                    conf_str = f" (conf: {confs.min():.2f}-{confs.max():.2f})"
                print(f"  Play {play_num:3d} {view:8s} @{pct:.0%}: {n_persons:2d} persons{conf_str} → {out_name}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
