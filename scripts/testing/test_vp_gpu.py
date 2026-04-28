#!/usr/bin/env python3
"""A/B test GPU-batched VP search vs the current CPU Fisher-score search.

For each sample frame, runs both implementations on the same UNet yard mask,
reports VP locations + |Δ| in pixels + per-call timings.

Usage:
    .venv/bin/python scripts/testing/test_vp_gpu.py --n 10
    .venv/bin/python scripts/testing/test_vp_gpu.py --device mps --warm-start
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import (
    _find_best_vanishing_point, _find_best_vp_gpu, _find_best_vp_cc,
    _mask_pixels, run_unet,
)

VAL_DIR = os.path.join(PROJECT_ROOT, "data/line_detection/valid")
UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_round2_best.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--subsample", type=int, default=5000)
    ap.add_argument("--warm-start", action="store_true",
                    help="Test warm-start path (uses CPU result as vp_init).")
    args = ap.parse_args()

    images = sorted(f for f in os.listdir(os.path.join(VAL_DIR, "images"))
                    if f.endswith(".jpg") and not f.startswith("._"))
    rng = random.Random(args.seed)
    rng.shuffle(images)
    picks = images[: args.n]

    # Warmup MPS shaders
    _ = _find_best_vp_gpu(np.random.rand(200, 2) * 720,
                          frame_shape=(720, 1280), device=args.device,
                          n_pixel_subsample=args.subsample)

    print(f"{'frame':<55}  {'cpu_vp':>18}  {'cc_vp':>18}  {'gpu_vp':>18}  "
          f"{'|cc-cpu|':>8}  {'|gpu-cpu|':>9}  "
          f"{'cpu_ms':>7}  {'cc_ms':>6}  {'gpu_ms':>7}")
    print("-" * 160)

    cpu_times, cc_times, gpu_times = [], [], []
    cc_deltas, gpu_deltas = [], []
    for fname in picks:
        fid = os.path.splitext(fname)[0]
        frame = cv2.imread(os.path.join(VAL_DIR, "images", fname))
        h, w = frame.shape[:2]
        yard, _side = run_unet(frame, UNET_WEIGHTS, device=args.device)
        pts = _mask_pixels(yard)
        if len(pts) < 200:
            print(f"{fid:<55}  too few pixels ({len(pts)})")
            continue

        # CPU baseline
        t0 = time.time()
        vp_cpu = _find_best_vanishing_point(pts, frame_shape=(h, w))
        t_cpu = (time.time() - t0) * 1000

        # CC + RANSAC
        t0 = time.time()
        vp_cc = _find_best_vp_cc(yard, frame_shape=(h, w))
        t_cc = (time.time() - t0) * 1000

        # GPU hybrid
        vp_init = vp_cpu if args.warm_start else None
        t0 = time.time()
        vp_gpu = _find_best_vp_gpu(pts, frame_shape=(h, w),
                                    vp_init=vp_init,
                                    n_pixel_subsample=args.subsample,
                                    device=args.device)
        t_gpu = (time.time() - t0) * 1000

        d_cc = (float("nan") if vp_cc is None
                else float(np.hypot(vp_cpu[0] - vp_cc[0], vp_cpu[1] - vp_cc[1])))
        d_gpu = float(np.hypot(vp_cpu[0] - vp_gpu[0], vp_cpu[1] - vp_gpu[1]))
        cpu_str = f"({vp_cpu[0]:.0f},{vp_cpu[1]:.0f})"
        cc_str = "FALLBACK" if vp_cc is None else f"({vp_cc[0]:.0f},{vp_cc[1]:.0f})"
        gpu_str = f"({vp_gpu[0]:.0f},{vp_gpu[1]:.0f})"
        print(f"{fid:<55}  {cpu_str:>18}  {cc_str:>18}  {gpu_str:>18}  "
              f"{d_cc:>8.1f}  {d_gpu:>9.1f}  "
              f"{t_cpu:>7.1f}  {t_cc:>6.1f}  {t_gpu:>7.1f}")
        cpu_times.append(t_cpu)
        cc_times.append(t_cc)
        gpu_times.append(t_gpu)
        if not np.isnan(d_cc): cc_deltas.append(d_cc)
        gpu_deltas.append(d_gpu)

    if cpu_times:
        print("-" * 160)
        print(f"{'mean':<55}  {'':>18}  {'':>18}  {'':>18}  "
              f"{np.mean(cc_deltas) if cc_deltas else float('nan'):>8.1f}  "
              f"{np.mean(gpu_deltas):>9.1f}  "
              f"{np.mean(cpu_times):>7.1f}  {np.mean(cc_times):>6.1f}  "
              f"{np.mean(gpu_times):>7.1f}")
        print(f"{'median':<55}  {'':>18}  {'':>18}  {'':>18}  "
              f"{np.median(cc_deltas) if cc_deltas else float('nan'):>8.1f}  "
              f"{np.median(gpu_deltas):>9.1f}  "
              f"{np.median(cpu_times):>7.1f}  {np.median(cc_times):>6.1f}  "
              f"{np.median(gpu_times):>7.1f}")


if __name__ == "__main__":
    main()
