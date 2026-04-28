#!/usr/bin/env python3
"""Filter triage results into a tiny hash-mask training set.

Copies the source frames + auto-generated hash masks for everything
the user marked 'good' in `hash_triage_results.json` into
`data/hash_masks/round1/train/` (images + masks subdirs).
"""

import argparse
import json
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/triage/hash_triage_results.json"))
    ap.add_argument("--src-images", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train/images"))
    ap.add_argument("--src-masks", default=os.path.join(
        PROJECT_ROOT, "output/hash_mask_test/masks"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/round1"))
    args = ap.parse_args()

    with open(args.triage) as f:
        decisions = json.load(f)
    good = [k for k, v in decisions.items() if v == "good"]
    print(f"  {len(good)} 'good' frames")

    img_out = os.path.join(args.out_dir, "train/images")
    mask_out = os.path.join(args.out_dir, "train/masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    n = 0
    for frame in good:
        stem = os.path.splitext(frame)[0]
        src_img = os.path.join(args.src_images, frame)
        src_mask = os.path.join(args.src_masks, stem + ".png")
        if not (os.path.exists(src_img) and os.path.exists(src_mask)):
            continue
        shutil.copy2(src_img, os.path.join(img_out, frame))
        shutil.copy2(src_mask, os.path.join(mask_out, stem + ".png"))
        n += 1
    print(f"  copied {n}/{len(good)} → {args.out_dir}/train/")


if __name__ == "__main__":
    main()
