#!/usr/bin/env python3
"""
Run RF-DETR detection on a video clip and output a video with bounding boxes.

Designed to run on a RunPod GPU instance.

Usage:
  python infer_viz.py --weights rfdetr_best_ema.pth --video input.mp4 --output output.mp4
"""

import argparse
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="RF-DETR detection visualization")
    parser.add_argument("--weights", required=True, help="Path to RF-DETR weights (.pth)")
    parser.add_argument("--video", required=True, help="Input video clip")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--resolution", type=int, default=1280, help="Model resolution")
    args = parser.parse_args()

    from rfdetr import RFDETRLarge

    print(f"Loading model: {args.weights}")
    model = RFDETRLarge(pretrain_weights=args.weights, resolution=args.resolution)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h}, {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_num = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, threshold=args.threshold)

        # Draw boxes
        n_det = len(results.xyxy)
        for i in range(n_det):
            x1, y1, x2, y2 = [int(v) for v in results.xyxy[i]]
            conf = float(results.confidence[i])

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Confidence label
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Foot point (95% down)
            fx = int((x1 + x2) / 2)
            fy = int(y1 + 0.95 * (y2 - y1))
            cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)

        # HUD
        cv2.putText(frame, f"Frame {frame_num} | {n_det} players",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)
        frame_num += 1

        if frame_num % 30 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_num / elapsed
            print(f"  {frame_num}/{total} frames | {n_det} det | {fps_actual:.1f} fps")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone: {frame_num} frames in {elapsed:.1f}s ({frame_num/elapsed:.1f} fps)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
