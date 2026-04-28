#!/usr/bin/env python3
"""Shift every PNG mask in a directory by N pixels in image space.

Used to correct a systematic vertical bias in the predicted hash masks
(model learned a small offset from the keypoint-annotated training labels).
Default shift is 2px up.
"""

import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shift-y", type=int, default=-2,
                    help="Pixel shift in image-y. Negative = up. Default -2.")
    ap.add_argument("--shift-x", type=int, default=0,
                    help="Pixel shift in image-x. Default 0.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = [f for f in os.listdir(args.in_dir)
             if f.endswith(".png") and not f.startswith("._")]
    for f in files:
        m = cv2.imread(os.path.join(args.in_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # np.roll wraps; zero out the wraparound region.
        shifted = np.roll(m, (args.shift_y, args.shift_x), axis=(0, 1))
        if args.shift_y < 0:
            shifted[args.shift_y:, :] = 0
        elif args.shift_y > 0:
            shifted[:args.shift_y, :] = 0
        if args.shift_x < 0:
            shifted[:, args.shift_x:] = 0
        elif args.shift_x > 0:
            shifted[:, :args.shift_x] = 0
        cv2.imwrite(os.path.join(args.out_dir, f), shifted)

    print(f"  shifted {len(files)} masks by (dy={args.shift_y}, dx={args.shift_x})")
    print(f"  out: {args.out_dir}/")


if __name__ == "__main__":
    main()
