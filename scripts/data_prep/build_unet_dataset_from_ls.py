#!/usr/bin/env python3
"""Convert a Label Studio brush-export JSON into a UNet training dataset.

Input:
  - LS export JSON (one entry per task, with `annotations[0].result[]` of
    type `brushlabels` carrying RLE-encoded masks for `yardline` and
    `sideline` labels)
  - The candidate frame JPEGs (already in `images/`)

Output:
  <out>/train/images/<id>.jpg
  <out>/train/masks/<id>.png       (3ch PNG: R=yardline, G=sideline, B=0)
  <out>/valid/...                   (stratified 80/20 by game)

Mask format matches what `train_unet_lines.py` already expects, so the
existing trainer works on this dataset without modification.

Usage:
  python scripts/data_prep/build_unet_dataset_from_ls.py \\
      --ls-json data/line_detection/al_round3/line_detection_hand_labels.json \\
      --images-dir data/line_detection/al_round3/images \\
      --out-dir   data/line_detection/al_round3 \\
      --val-frac 0.20

Stratified split by game so every game appears in both splits — keeps the
val set a meaningful generalization check.
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np

from label_studio_converter.brush import decode_rle


YARDLINE_LABELS = {"yardline", "yard_line", "yard"}
SIDELINE_LABELS = {"sideline", "side_line", "side"}

GAME_RE = re.compile(r"(\d{10})_play_")  # 2024090802_play_xxx → "2024090802"


def get_game_from_filename(fname: str) -> str:
    m = GAME_RE.search(os.path.basename(fname))
    return m.group(1) if m else "unknown"


def get_image_filename(task: dict) -> str | None:
    """Resolve the image's basename from a task. LS stores it under data.image
    as a `/data/local-files/?d=al_round3/images/foo.jpg` URL or similar."""
    img = task.get("data", {}).get("image", "")
    if not img:
        return None
    # Strip prefix
    if "?d=" in img:
        img = img.split("?d=")[-1]
    return os.path.basename(img)


def decode_brush_mask(rle: list[int], h: int, w: int) -> np.ndarray:
    """Decode LS brush RLE → binary uint8 mask of shape (h, w).

    The encoder ran `np.repeat(arr_flat, 4)` to fake a 4-channel image (RGBA);
    the decoder gives back that same length, so we reshape to (h, w, 4) and
    take channel 0.
    """
    flat = decode_rle(rle)
    if len(flat) != h * w * 4:
        return np.zeros((h, w), dtype=np.uint8)
    return flat.reshape(h, w, 4)[..., 0]


def parse_task(task: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (yard_mask, side_mask), each binary (h, w) uint8 with values 0/255.

    Multiple regions of the same label are unioned together.
    """
    yard = np.zeros((h, w), dtype=np.uint8)
    side = np.zeros((h, w), dtype=np.uint8)
    annotations = task.get("annotations") or task.get("completions") or []
    for ann in annotations:
        for r in ann.get("result", []):
            if r.get("type") != "brushlabels":
                continue
            value = r.get("value", {})
            labels = value.get("brushlabels") or value.get("labels") or []
            if not labels:
                continue
            label = labels[0].lower()
            rle = value.get("rle") or []
            if not rle:
                continue
            mask = decode_brush_mask(rle, h, w)
            if label in YARDLINE_LABELS:
                yard = np.maximum(yard, (mask > 0).astype(np.uint8) * 255)
            elif label in SIDELINE_LABELS:
                side = np.maximum(side, (mask > 0).astype(np.uint8) * 255)
    return yard, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-json", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.ls_json) as f:
        tasks = json.load(f)
    print(f"  loaded {len(tasks)} tasks from {args.ls_json}")

    # Stratified split by game.
    by_game: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        fname = get_image_filename(t)
        if not fname:
            continue
        if not os.path.exists(os.path.join(args.images_dir, fname)):
            continue
        by_game[get_game_from_filename(fname)].append(t)

    rng = random.Random(args.seed)
    train_tasks, val_tasks = [], []
    for game, gtasks in sorted(by_game.items()):
        gtasks = gtasks.copy()
        rng.shuffle(gtasks)
        n_val = max(1, int(round(len(gtasks) * args.val_frac)))
        val_tasks.extend(gtasks[:n_val])
        train_tasks.extend(gtasks[n_val:])
    print(f"  split: {len(train_tasks)} train / {len(val_tasks)} val "
          f"(stratified by {len(by_game)} games)")

    for split, split_tasks in (("train", train_tasks), ("valid", val_tasks)):
        img_out = os.path.join(args.out_dir, split, "images")
        mask_out = os.path.join(args.out_dir, split, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        n_yard_total = n_side_total = 0
        for t in split_tasks:
            fname = get_image_filename(t)
            if fname is None:
                continue
            src_img = os.path.join(args.images_dir, fname)
            frame = cv2.imread(src_img)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            yard, side = parse_task(t, h, w)
            n_yard_total += int((yard > 0).sum())
            n_side_total += int((side > 0).sum())

            # Pack into 3-channel PNG matching build_line_dataset.py's format.
            combined = np.zeros((h, w, 3), dtype=np.uint8)
            combined[..., 2] = yard       # R = yardline
            combined[..., 1] = side       # G = sideline
            base = os.path.splitext(fname)[0]
            shutil.copy(src_img, os.path.join(img_out, f"{base}.jpg"))
            cv2.imwrite(os.path.join(mask_out, f"{base}.png"), combined)
        n = len(split_tasks)
        if n > 0:
            print(f"  {split}: {n} frames, "
                  f"avg yard pixels/frame {n_yard_total // n}, "
                  f"avg side pixels/frame {n_side_total // n}")


if __name__ == "__main__":
    main()
