"""Filter the auto-mined classifier dataset to remove incomplete crops.

A complete painted-number group should have:
  • At least MIN_MASK_PX active mask pixels (drops tiny single-fragment crops)
  • At least MIN_CCS connected components (a complete number is 2+ digits ± arrow,
    so 2 CCs minimum; lone-arrow or single-digit crops have just 1)

Crops failing either check get moved to ../round1_rejected/<class>/ (not deleted)
so we can spot-check or reinstate later.
"""
import argparse
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data/number_classifier/round1")

MIN_MASK_PX = 200      # of 64×64=4096 (5% fill); a complete number is ~10–35%
MIN_CCS = 2            # 2 digits minimum (3 if arrow is its own CC)


def passes(crop, min_mask_px=MIN_MASK_PX, min_ccs=MIN_CCS):
    n_px = int((crop > 127).sum())
    if n_px < min_mask_px:
        return False, f"low_px={n_px}"
    n_cc, _ = cv2.connectedComponents(
        (crop > 127).astype(np.uint8), connectivity=8)
    n_cc -= 1   # subtract background
    if n_cc < min_ccs:
        return False, f"low_cc={n_cc}"
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--reject-dir", default=None,
                     help="Where to move rejected crops (default: <root>_rejected)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reject_root = args.reject_dir or args.root.rstrip("/") + "_rejected"

    kept = defaultdict(int)
    dropped = defaultdict(int)
    drop_reasons = defaultdict(int)

    for cls in sorted(os.listdir(args.root)):
        cls_dir = os.path.join(args.root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.endswith(".png"):
                continue
            path = os.path.join(cls_dir, fn)
            crop = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            ok, reason = passes(crop)
            if ok:
                kept[cls] += 1
            else:
                dropped[cls] += 1
                drop_reasons[reason] += 1
                if not args.dry_run:
                    rej_dir = os.path.join(reject_root, cls)
                    os.makedirs(rej_dir, exist_ok=True)
                    shutil.move(path, os.path.join(rej_dir, fn))

    total_kept = sum(kept.values())
    total_dropped = sum(dropped.values())
    print(f"\n{'class':6s} {'kept':>6s} {'dropped':>8s}")
    for cls in sorted(set(kept) | set(dropped)):
        print(f"{cls:6s} {kept[cls]:6d} {dropped[cls]:8d}")
    print(f"{'total':6s} {total_kept:6d} {total_dropped:8d}")
    print(f"\nDrop reasons: {dict(drop_reasons)}")
    if args.dry_run:
        print(f"(dry-run — no files moved)")
    else:
        print(f"rejected crops → {reject_root}")


if __name__ == "__main__":
    main()
