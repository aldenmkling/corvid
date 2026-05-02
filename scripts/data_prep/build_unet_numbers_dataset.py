#!/usr/bin/env python3
"""Convert the Label Studio export of hand-corrected painted-number masks
into a UNet-trainable dataset (train/+valid/ split).

For each LS task:
  1. Resolve the source image path from data["image"] (LS local-files URL)
  2. Decode the brush RLE into a binary mask at the annotation's
     original_width × original_height
  3. Copy the source image to {train|valid}/images/{stem}.jpg
  4. Write the binary mask to {train|valid}/masks/{stem}.png

Train/valid split is seeded by stem so the same frame always lands in the
same partition. Default 90/10 split.

Usage:
    .venv-labelstudio/bin/python scripts/data_prep/build_unet_numbers_dataset.py
"""

import argparse
import json
import os
import shutil
import sys
import urllib.parse

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LS_VENV_SITE = os.path.join(
    PROJECT_ROOT, ".venv-labelstudio/lib/python3.11/site-packages")
if os.path.isdir(LS_VENV_SITE) and LS_VENV_SITE not in sys.path:
    sys.path.insert(0, LS_VENV_SITE)

from label_studio_converter.brush import decode_rle


def resolve_image_path(ls_image_url: str, doc_root: str) -> str:
    """Map LS local-files URL `/data/local-files/?d=<rel>` → absolute path."""
    parsed = urllib.parse.urlparse(ls_image_url)
    qs = urllib.parse.parse_qs(parsed.query)
    rel = qs["d"][0]
    return os.path.join(doc_root, rel)


def decode_brush_to_mask(value: dict, h: int, w: int) -> np.ndarray:
    """LS brush RLE → uint8 binary mask of shape (h, w)."""
    rle = value["rle"]
    out = decode_rle(rle)                # (h*w*4,) flat or (h, w, 4) — varies by version
    out = np.array(out, dtype=np.uint8)
    if out.ndim == 1:
        out = out.reshape(h, w, 4)
    if out.ndim == 3:
        # Alpha channel carries the brush coverage.
        out = out[:, :, 3]
    return (out > 0).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export",
                     default=os.path.join(PROJECT_ROOT,
                       "data/yardline_numbers/round1/number_hand_labels.json"))
    ap.add_argument("--ls-doc-root",
                     default=os.path.join(PROJECT_ROOT, "data"),
                     help="LS LOCAL_FILES_DOCUMENT_ROOT used at labeling time")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT,
                       "data/yardline_numbers/dataset_round1"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = json.load(open(args.export))
    print(f"  loaded {len(tasks)} tasks")

    # Build (stem, src_image_path, mask_array) tuples.
    pairs = []
    for t in tasks:
        anns = t.get("annotations") or t.get("completions") or []
        if not anns:
            continue
        ann = anns[0]
        if ann.get("was_cancelled"):
            continue
        results = ann.get("result", [])
        if not results:
            continue

        ls_url = t["data"]["image"]
        src_image = resolve_image_path(ls_url, args.ls_doc_root)
        if not os.path.exists(src_image):
            print(f"  [skip] missing source: {src_image}")
            continue

        h0 = results[0]["original_height"]
        w0 = results[0]["original_width"]
        # Union all brush regions on the task into one mask.
        mask = np.zeros((h0, w0), dtype=np.uint8)
        for r in results:
            if r.get("type") != "brushlabels":
                continue
            sub = decode_brush_to_mask(r["value"], h0, w0)
            mask = np.maximum(mask, sub)

        stem = os.path.splitext(os.path.basename(src_image))[0]
        pairs.append((stem, src_image, mask, h0, w0))

    print(f"  usable: {len(pairs)} (image, mask) pairs")

    # Stem-keyed deterministic split.
    rng = np.random.default_rng(args.seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(round(args.val_frac * len(pairs))))
    val_idx = set(indices[:n_val])

    splits = {"train": [], "valid": []}
    for i, p in enumerate(pairs):
        splits["valid" if i in val_idx else "train"].append(p)
    print(f"  split → train={len(splits['train'])}  valid={len(splits['valid'])}")

    # Write to disk. Clear out any existing dataset first.
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for split, items in splits.items():
        img_dir = os.path.join(args.out, split, "images")
        msk_dir = os.path.join(args.out, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)
        for stem, src_image, mask, h0, w0 in items:
            img = cv2.imread(src_image)
            if img is None:
                print(f"  [skip] cant read {src_image}"); continue
            ih, iw = img.shape[:2]
            if (ih, iw) != (h0, w0):
                # Mask was painted at LS-shown resolution; resize to image dims
                mask_full = cv2.resize(mask, (iw, ih),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                mask_full = mask
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img,
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(msk_dir, stem + ".png"), mask_full)
    print(f"  wrote dataset → {args.out}")
    print(f"  train: {len(splits['train'])}  valid: {len(splits['valid'])}")


if __name__ == "__main__":
    main()
