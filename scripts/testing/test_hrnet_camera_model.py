#!/usr/bin/env python3
"""
Run the fine-tuned HRNet detector on the real kickoff frame and use its
output to calibrate the camera model.

Uses per-channel confidence thresholds: sidelines get a tighter threshold
because HRNet's sideline recall is weaker but we need the few detections
we get to be high-quality.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.keypoint_detector import HRNetKeypointModel, _refine_peak
from src.homography.camera_model import (
    calibrate_camera, camera_state_to_homography, project_field_to_pixel,
)
from src.homography.field_model import (
    YARD_LINE_POSITIONS, FIELD_WIDTH, FIELD_LENGTH,
    HASH_Y_NEAR, HASH_Y_FAR,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
FRAME_PATH = os.path.join(PROJECT_ROOT, "output", "camera_model_test", "real_kickoff_raw.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

# Per-channel confidence thresholds (tune these)
SIDELINE_THRESH = 0.40
HASH_THRESH = 0.40

INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_hrnet_with_per_channel_threshold(frame, weights_path, sideline_thresh, hash_thresh,
                                         device="cpu"):
    """Run HRNet and extract peaks with different thresholds per channel."""
    device = torch.device(device)
    model = HRNetKeypointModel(num_channels=2)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    orig_h, orig_w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        heatmaps = torch.sigmoid(logits[0]).cpu().numpy()

    _, hm_h, hm_w = heatmaps.shape
    thresholds = {0: sideline_thresh, 1: hash_thresh}

    all_px, all_ch, all_conf = [], [], []
    for ch in [0, 1]:
        hm = heatmaps[ch]
        thresh = thresholds[ch]
        mask = hm >= thresh
        if not mask.any():
            continue
        labels, n = ndimage.label(mask)
        for comp_id in range(1, n + 1):
            comp_mask = labels == comp_id
            vals = hm * comp_mask
            peak_idx = vals.argmax()
            py, px_h = peak_idx // hm_w, peak_idx % hm_w
            peak_val = float(hm[py, px_h])
            ref_x, ref_y = _refine_peak(hm, py, px_h)
            px = ref_x / hm_w * orig_w
            py = ref_y / hm_h * orig_h
            all_px.append([px, py])
            all_ch.append(ch)
            all_conf.append(peak_val)

    return np.array(all_px), np.array(all_ch), np.array(all_conf)


def cluster_by_x(pxs, tolerance=60):
    if len(pxs) == 0:
        return []
    xs_sorted = sorted(pxs[:, 0].tolist())
    columns = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if x - columns[-1][-1] < tolerance:
            columns[-1].append(x)
        else:
            columns.append([x])
    return [np.mean(c) for c in columns]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame = cv2.imread(FRAME_PATH)
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    print(f"\nHRNet with thresholds: sideline={SIDELINE_THRESH}, hash={HASH_THRESH}")
    pxs, chs, confs = run_hrnet_with_per_channel_threshold(
        frame, WEIGHTS, SIDELINE_THRESH, HASH_THRESH,
    )

    sideline_mask = chs == 0
    hash_mask = chs == 1
    sideline_pxs = pxs[sideline_mask]
    hash_pxs = pxs[hash_mask]
    print(f"  Sideline: {len(sideline_pxs)} (confs: {sorted(confs[sideline_mask].round(3).tolist())})")
    print(f"  Hash:     {len(hash_pxs)}")

    # Draw all detections
    vis_all = frame.copy()
    for px, conf in zip(sideline_pxs, confs[sideline_mask]):
        cv2.circle(vis_all, tuple(px.astype(int)), 6, (0, 0, 255), 2)
        cv2.putText(vis_all, f"{conf:.2f}", (int(px[0])+5, int(px[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for px in hash_pxs:
        cv2.circle(vis_all, tuple(px.astype(int)), 5, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "hrnet_kickoff_detections.jpg"), vis_all)

    # Cluster hash detections into columns
    column_xs = cluster_by_x(hash_pxs, tolerance=60)
    print(f"\nFound {len(column_xs)} hash columns at x = {[f'{x:.0f}' for x in column_xs]}")
    if len(column_xs) < 3:
        print("  Too few columns, aborting")
        return

    # Assume contiguous yard lines, 5 yards apart. Base NGS x = 40 (from visual ID)
    base_ngs_x = 40
    # grid 0 = leftmost column = NGS 40
    # Compute grid positions for each column using the detected spacing
    median_spacing = np.median(np.diff(column_xs))
    print(f"  Median spacing: {median_spacing:.1f} px")

    column_to_ngs = {}
    for i, xc in enumerate(column_xs):
        column_to_ngs[xc] = base_ngs_x + i * 5
    print(f"  Identity: {[(int(c), ngs) for c, ngs in column_to_ngs.items()]}")

    def nearest_column_ngs(px, cols_to_ngs, tolerance=60):
        best_col = None
        best_d = float('inf')
        for cx in cols_to_ngs:
            d = abs(px - cx)
            if d < best_d:
                best_d = d
                best_col = cx
        return cols_to_ngs[best_col] if best_d < tolerance else None

    # Build correspondences
    pixel_pts = []
    field_pts = []

    # Hashes: far (y<340) or near (y>=340) — hard cut from observed t-values
    for px in hash_pxs:
        ngs_x = nearest_column_ngs(px[0], column_to_ngs)
        if ngs_x is None:
            continue
        ngs_y = HASH_Y_FAR if px[1] < 340 else HASH_Y_NEAR
        pixel_pts.append(list(px))
        field_pts.append([ngs_x, ngs_y])

    # Sidelines: they should cluster into 1-2 rows in the image (far sideline
    # near the top, near sideline near the bottom if visible). Detections that
    # are far from these dominant rows are likely noise.
    n_sideline_kept = 0
    n_sideline_rejected = 0

    if len(sideline_pxs) > 0:
        # Find dominant row(s) by histogramming y-values
        y_vals = sideline_pxs[:, 1]
        # Use a simple 1D clustering: sort and split on gaps > 50px
        sorted_y = np.sort(y_vals)
        clusters = [[sorted_y[0]]]
        for y in sorted_y[1:]:
            if y - clusters[-1][-1] < 50:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        # Keep only clusters with >= 3 members (likely a true sideline row)
        valid_rows = [np.mean(c) for c in clusters if len(c) >= 3]
        print(f"  Sideline y-clusters (>=3 members): {[f'{y:.0f}' for y in valid_rows]}")

        def assign_sideline_y(py):
            # near sideline (y=0) is toward the bottom of the frame (high py)
            # far sideline (y=FIELD_WIDTH) is toward the top (low py)
            # Decide based on whether py is in the upper or lower half of the image.
            return FIELD_WIDTH if py < h / 2 else 0.0

        for px in sideline_pxs:
            # Accept only if close to a valid row
            if not any(abs(px[1] - y) < 25 for y in valid_rows):
                n_sideline_rejected += 1
                continue
            ngs_x = nearest_column_ngs(px[0], column_to_ngs)
            if ngs_x is None:
                n_sideline_rejected += 1
                continue
            pixel_pts.append(list(px))
            field_pts.append([ngs_x, assign_sideline_y(px[1])])
            n_sideline_kept += 1

    pixel_pts = np.array(pixel_pts)
    field_pts = np.array(field_pts)
    print(f"\nCorrespondences: {len(pixel_pts)}")
    print(f"  Sideline kept: {n_sideline_kept} (rejected: {n_sideline_rejected})")
    print(f"  Hash:          {sum(1 for p in field_pts if p[1] in [HASH_Y_NEAR, HASH_Y_FAR])}")

    if len(pixel_pts) < 8:
        print("Not enough correspondences")
        return

    # Calibrate
    print("\n=== Calibration ===")
    cal = calibrate_camera(pixel_pts, field_pts, (h, w))
    pos = cal.extrinsics.position
    print(f"Position: Cx={pos[0]:.1f}, Cy={pos[1]:.1f}, Cz={pos[2]:.1f}")
    print(f"State: pan={cal.initial_state.pan:.4f}, tilt={cal.initial_state.tilt:.4f}, f={cal.initial_state.focal_length:.1f}")
    print(f"Roll: {np.degrees(cal.initial_state.roll):.2f}°")
    print(f"Reprojection RMSE: {cal.calibration_error:.2f} pixels")

    # Overlay
    H_result = camera_state_to_homography(
        cal.initial_state, cal,
        n_inliers=len(pixel_pts), n_correspondences=len(pixel_pts),
    )
    overlay = frame.copy()
    for x in YARD_LINE_POSITIONS:
        ys = np.linspace(0, FIELD_WIDTH, 20)
        pf = np.column_stack([np.full_like(ys, x), ys])
        pp = project_field_to_pixel(pf, cal.initial_state, cal, apply_dist=False)
        for i in range(len(pp) - 1):
            cv2.line(overlay, tuple(pp[i].astype(int)), tuple(pp[i+1].astype(int)),
                     (0, 255, 0), 1, cv2.LINE_AA)
    for y in [HASH_Y_NEAR, HASH_Y_FAR]:
        x_min = max(0, H_result.yard_range[0] - 5)
        x_max = min(FIELD_LENGTH, H_result.yard_range[1] + 5)
        xs = np.linspace(x_min, x_max, 50)
        pf = np.column_stack([xs, np.full_like(xs, y)])
        pp = project_field_to_pixel(pf, cal.initial_state, cal, apply_dist=False)
        for i in range(len(pp) - 1):
            cv2.line(overlay, tuple(pp[i].astype(int)), tuple(pp[i+1].astype(int)),
                     (0, 200, 200), 1, cv2.LINE_AA)
    for i in range(len(pixel_pts)):
        cv2.circle(overlay, tuple(pixel_pts[i].astype(int)), 5, (0, 0, 255), 2)
    projected = project_field_to_pixel(field_pts, cal.initial_state, cal, apply_dist=False)
    for i in range(len(projected)):
        cv2.circle(overlay, tuple(projected[i].astype(int)), 3, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "hrnet_kickoff_overlay.jpg"), overlay)
    print(f"\nSaved: hrnet_kickoff_detections.jpg, hrnet_kickoff_overlay.jpg")


if __name__ == "__main__":
    main()
