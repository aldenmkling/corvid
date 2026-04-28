#!/usr/bin/env python3
"""Test a CC-based sideline grouper on frame 240 (where the PCA+peak
grouper was dropping 75% of sideline pixels).

CC-based: same approach as yardlines —
  1. Connected components on side_mask.
  2. Filter by size + aspect ratio.
  3. Per-component PCA → (ρ, θ).
  4. Cluster collinear fragments (matching ρ, θ).
  5. Keep up to N strongest clusters by pixel count (default 2).
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver_v2 import run_unet, group_sideline_pixels

CLIP = os.path.join(PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4")
UNET = os.path.join(PROJECT_ROOT, "models/unet_line_round3_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output/rebuild/step_sideline_cc_frame240.jpg")

FRAMES_TO_TEST = [0, 100, 240, 270, 360, 390]


def group_sideline_pixels_cc(
    side_mask: np.ndarray,
    min_pixels_per_component: int = 40,
    min_aspect_ratio: float = 3.0,
    rho_tol_px: float = 25.0,
    theta_tol_rad: float = 0.08,
    max_lines: int = 2,
    min_pixels_per_line: int = 100,
):
    """CC + collinearity merge for sidelines. Returns up to `max_lines`
    strongest clusters by pixel count.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        side_mask.astype(np.uint8), connectivity=8,
    )
    comps = []
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_pixels_per_component:
            continue
        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = pts.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(pts - center, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-6 or S[0] / S[1] < min_aspect_ratio:
            continue
        direction = Vt[0]
        normal = np.array([-direction[1], direction[0]])
        rho = float(normal @ center)
        theta = float(np.arctan2(normal[1], normal[0]))
        if theta < 0:
            theta += np.pi; rho = -rho
        comps.append({"pixels": pts, "rho": rho, "theta": theta, "n": len(pts)})

    if not comps:
        return []

    # Cluster by (ρ, θ) similarity. Greedy: assign each comp to first
    # existing cluster within tol; else start a new cluster.
    clusters = []
    for c in comps:
        placed = False
        for cl in clusters:
            d_rho = abs(c["rho"] - cl["rho"])
            d_theta = abs(c["theta"] - cl["theta"])
            d_theta = min(d_theta, np.pi - d_theta)
            if d_rho <= rho_tol_px and d_theta <= theta_tol_rad:
                cl["pixels"].append(c["pixels"])
                cl["n"] += c["n"]
                # Update centroid by pixel-weighted average.
                w_old = cl["n"] - c["n"]; w_new = c["n"]
                cl["rho"] = (cl["rho"] * w_old + c["rho"] * w_new) / cl["n"]
                cl["theta"] = (cl["theta"] * w_old + c["theta"] * w_new) / cl["n"]
                placed = True; break
        if not placed:
            clusters.append({
                "pixels": [c["pixels"]],
                "rho": c["rho"], "theta": c["theta"], "n": c["n"],
            })

    # Filter by pixel count, sort strongest first, take top N.
    clusters = [cl for cl in clusters if cl["n"] >= min_pixels_per_line]
    clusters.sort(key=lambda cl: cl["n"], reverse=True)
    clusters = clusters[:max_lines]

    # Concatenate pixels per cluster.
    out = []
    for cl in clusters:
        all_pts = np.concatenate(cl["pixels"], axis=0)
        out.append({"pixels": all_pts, "n": cl["n"], "rho": cl["rho"],
                    "theta": cl["theta"]})
    return out


def main():
    cap = cv2.VideoCapture(CLIP)
    panels = []
    for fi in FRAMES_TO_TEST:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]
        _, side_mask = run_unet(frame, UNET, device="mps")
        raw = int((side_mask > 0).sum())

        old = group_sideline_pixels(side_mask)
        old_kept = sum(len(g.pixels) for g in old)

        new = group_sideline_pixels_cc(side_mask)
        new_kept = sum(g["n"] for g in new)

        # Build viz: 3-up vertical (raw, old, new).
        def render(label_text, painted_mask):
            img = frame.copy().astype(np.float32)
            m = painted_mask > 0
            img[m] = 0.35 * img[m] + 0.65 * np.array([0, 255, 255], dtype=np.float32)
            img = img.clip(0, 255).astype(np.uint8)
            cv2.putText(img, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return img

        # Row: [raw, old grouper, new CC grouper]
        raw_panel = render(f"f{fi}: raw mask  {raw}px", side_mask)

        old_mask = np.zeros_like(side_mask, dtype=bool)
        for g in old:
            xs = g.pixels[:, 0].astype(np.int32)
            ys = g.pixels[:, 1].astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            old_mask[ys[valid], xs[valid]] = True
        old_panel = render(
            f"old grouper: {len(old)} groups, {old_kept}/{raw}px kept",
            old_mask,
        )

        new_mask = np.zeros_like(side_mask, dtype=bool)
        for g in new:
            xs = g["pixels"][:, 0].astype(np.int32)
            ys = g["pixels"][:, 1].astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            new_mask[ys[valid], xs[valid]] = True
        new_panel = render(
            f"NEW CC grouper: {len(new)} groups, {new_kept}/{raw}px kept",
            new_mask,
        )

        panels.append(np.hstack([raw_panel, old_panel, new_panel]))
        print(f"  frame {fi}: raw={raw}  old={len(old)}grp/{old_kept}px  "
              f"new={len(new)}grp/{new_kept}px")

    cap.release()
    if not panels:
        print("no panels"); return
    full = np.vstack(panels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, full)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
