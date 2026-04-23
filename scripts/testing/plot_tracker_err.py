#!/usr/bin/env python3
"""Run tracker on a clip, plot per-frame err + method over time."""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.tracker import HomographyTracker

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "tracker_rectify")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--anchor", type=float, required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    tag = f"{os.path.basename(os.path.dirname(args.clip))}_{os.path.splitext(os.path.basename(args.clip))[0]}"
    out_png = args.output or os.path.join(OUT_DIR, f"{tag}_err_timeline.png")

    tracker = HomographyTracker(WEIGHTS, use_track_bank=True)
    cap = cv2.VideoCapture(args.clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames, errs, methods, n_corr = [], [], [], []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        anchor_arg = args.anchor if i == 0 else None
        try:
            r = tracker.process_frame(frame, anchor_ngs_x=anchor_arg)
        except Exception as e:
            print(f"frame {i}: {e}")
            continue
        frames.append(i)
        errs.append(r.field_reproj_error_mean
                    if r.field_reproj_error_mean == r.field_reproj_error_mean
                    else np.nan)
        methods.append(r.method)
        n_corr.append(r.n_correspondences)
    cap.release()

    frames = np.asarray(frames)
    errs = np.asarray(errs)
    n_corr = np.asarray(n_corr)
    methods = np.asarray(methods)
    time_s = frames / fps

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    # Top: err per frame, colored by method
    color_by = {"full": "tab:blue", "delta": "tab:orange", "carry": "tab:red"}
    for method, color in color_by.items():
        mask = methods == method
        ax1.scatter(time_s[mask], errs[mask], s=10, c=color,
                    label=f"{method} (n={int(mask.sum())})", alpha=0.8)
    ax1.set_ylabel("Reprojection error (yd)")
    ax1.set_title(f"Tracker per-frame error — {tag}")
    ax1.axhline(0.5, color="red", ls="--", alpha=0.4, label="0.5 yd target")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    finite = np.isfinite(errs)
    if finite.any():
        ax1.set_ylim(0, max(1.0, np.nanmax(errs[finite]) * 1.1))

    # Middle: correspondences per frame
    ax2.plot(time_s, n_corr, color="tab:green", lw=1, alpha=0.7)
    ax2.set_ylabel("# correspondences")
    ax2.axhline(4, color="gray", ls="--", alpha=0.4, label="min for full H")
    ax2.axhline(2, color="gray", ls=":", alpha=0.4, label="min for delta")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    # Bottom: method strip chart
    method_to_y = {"full": 2, "delta": 1, "carry": 0}
    y = np.array([method_to_y[m] for m in methods])
    for method, color in color_by.items():
        mask = methods == method
        ax3.scatter(time_s[mask], y[mask], s=6, c=color, alpha=0.8)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["carry", "delta", "full"])
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_xlabel("Time (s)")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    # Also dump CSV for quick analysis
    csv_path = out_png.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("frame,time_s,method,n_corr,err_yd\n")
        for frame, t, m, n, e in zip(frames, time_s, methods, n_corr, errs):
            f.write(f"{frame},{t:.3f},{m},{n},{e:.4f}\n")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
