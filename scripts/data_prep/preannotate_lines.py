#!/usr/bin/env python3
"""Run UNet on each selected AL frame and embed the predicted masks as
Label Studio Brush pre-annotations.

Direct mask preannotation — when the user opens a task in LS, the UNet's
yard mask + side mask are already painted as the active brush layers. They
correct: brush in missing pixels, erase wrong ones, switch label between
yardline/sideline as needed. What they paint IS the training mask, no
polyline conversion in the loop.

Usage:
    python scripts/data_prep/preannotate_lines.py \\
        --manifest data/line_detection/al_round3/ls_import.json \\
        --images-dir data/line_detection/al_round3/images \\
        --weights models/unet_line_round2_best.pth \\
        --device mps

Updates the manifest in-place (replaces or adds a `predictions` key per task).
"""

import argparse
import json
import os
import sys
import time
import uuid

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import run_unet
from label_studio_converter.brush import mask2rle


def brush_prediction_entry(mask: np.ndarray, label: str, w: int, h: int) -> dict:
    """Format a binary mask as a Label Studio brushlabels prediction entry.

    `mask` is uint8 (0/255 or 0/1); we re-binarize and threshold to be safe.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    rle = mask2rle(mask_u8)
    return {
        "id": uuid.uuid4().hex[:8],
        "from_name": "lines",
        "to_name": "image",
        "type": "brushlabels",
        "image_rotation": 0,
        "original_width": w,
        "original_height": h,
        "value": {
            "format": "rle",
            "rle": rle,
            "brushlabels": [label],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Label Studio import JSON to update in-place")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--weights", default=os.path.join(
        PROJECT_ROOT, "models/unet_line_round2_best.pth"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    with open(args.manifest) as f:
        tasks = json.load(f)
    print(f"  loaded {len(tasks)} tasks from {args.manifest}")

    n_with_pred = 0
    t0 = time.time()
    for i, task in enumerate(tasks):
        img_url = task["data"]["image"]
        # URL is /data/local-files/?d=al_round3/images/foo.jpg → extract filename
        rel = img_url.split("?d=")[-1] if "?d=" in img_url else img_url
        fname = os.path.basename(rel)
        path = os.path.join(args.images_dir, fname)
        if not os.path.exists(path):
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        yard_mask, side_mask = run_unet(frame, args.weights, device=args.device)

        result_entries = []
        if yard_mask.any():
            result_entries.append(brush_prediction_entry(yard_mask, "yardline", w, h))
        if side_mask.any():
            result_entries.append(brush_prediction_entry(side_mask, "sideline", w, h))

        if result_entries:
            # Put masks in `annotations` so they're EDITABLE on load (vs
            # `predictions` which require a click-to-convert per task).
            # We're treating the AL workflow as "edit + correct UNet output"
            # rather than "review predictions", so this matches the intent.
            task["annotations"] = [{
                "result": result_entries,
                "ground_truth": False,
                "was_cancelled": False,
                "lead_time": 0,
            }]
            # Drop any prior predictions key from earlier preannotation runs.
            task.pop("predictions", None)
            n_with_pred += 1
        else:
            task.pop("predictions", None)
            task.pop("annotations", None)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(tasks)}]  {(i+1)/elapsed:.1f} fps  "
                  f"{elapsed:.0f}s elapsed")

    with open(args.manifest, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  preannotated {n_with_pred}/{len(tasks)} tasks (brush masks)")
    print(f"  manifest updated → {args.manifest}")


if __name__ == "__main__":
    main()
