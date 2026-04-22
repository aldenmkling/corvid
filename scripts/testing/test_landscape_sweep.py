#!/usr/bin/env python3
"""
For each frame individually: calibrate the camera, then sweep over Cz
(camera height) and re-calibrate with Cz pinned at each value. At each
pinned Cz, the solver still has freedom in (Cx, Cy, pan, tilt, f, roll).
The resulting RMSE vs. Cz reveals how constrained the camera height is
by that frame's data.

The intersection of low-RMSE regions across frames is where joint
calibration would succeed. If the frames disagree, we see it here.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.testing.test_grid_solver_camera import (
    build_yard_line_groups, groups_to_correspondences,
    calibrate_distortion_from_lines,
)
from scripts.testing.test_yard_line_grouping import (
    run_hrnet, extract_peaks,
)
from src.homography.camera_model import (
    CameraIntrinsics, rotation_matrix, undistort_points,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "camera_model_test")

HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30


def process_frame(frame_path, base_ngs_x):
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    groups, angle_deg = build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs)
    pixel_pts, field_pts, labels = groups_to_correspondences(groups, base_ngs_x)

    side_row = [g['sideline'] for g in groups if g.get('sideline') is not None]
    far_row = [g['far_hash'] for g in groups if g.get('far_hash') is not None]
    near_row = [g['near_hash'] for g in groups if g.get('near_hash') is not None]
    line_sets = [np.array(x) for x in (side_row, far_row, near_row) if len(x) >= 3]

    return {
        "shape": (h, w),
        "pixel_pts": pixel_pts,
        "field_pts": field_pts,
        "labels": labels,
        "line_sets": line_sets,
        "tilt_deg": angle_deg,
    }


def calibrate_with_fixed_cz(pix_u, field_pts, frame_shape, cz_fixed):
    """Calibrate with camera height pinned. Returns (params, rmse)."""
    h, w = frame_shape
    cx, cy = w / 2.0, h / 2.0
    N = len(field_pts)

    # Params: Cx, Cy, pan, tilt, f, roll (Cz fixed)
    def residuals(params):
        Cx, Cy, pan, tilt, f, roll = params
        C = np.array([Cx, Cy, cz_fixed])
        R = rotation_matrix(pan, tilt, roll)
        t = -R @ C
        pts_3d = np.column_stack([field_pts, np.zeros(N)])
        p_cam = (R @ pts_3d.T).T + t
        behind = p_cam[:, 2] <= 0
        if behind.any():
            return np.full(2 * N, 1000.0)
        x_norm = p_cam[:, 0] / p_cam[:, 2]
        y_norm = p_cam[:, 1] / p_cam[:, 2]
        u_pred = f * x_norm + cx
        v_pred = f * y_norm + cy
        return np.concatenate([u_pred - pix_u[:, 0], v_pred - pix_u[:, 1]])

    x0 = np.array([60.0, -60.0, 0.0, -0.45, 1000.0, 0.0])
    lo = [20, -150, -np.pi/2, -np.pi/2, 200, -np.radians(15)]
    hi = [100, -15, np.pi/2, 0.0, 10000, np.radians(15)]
    result = least_squares(
        residuals, x0, bounds=(lo, hi),
        method="trf", loss="soft_l1", f_scale=5.0, max_nfev=2000,
    )
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return result.x, rmse


def main():
    frames = [
        ("real_kickoff_raw.jpg", 35.0, "F1 (kickoff)"),
        ("play002_first.jpg", 70.0, "F2 (red zone)"),
    ]

    processed = []
    for fname, ngs, label in frames:
        fr = process_frame(os.path.join(OUTPUT_DIR, fname), ngs)
        h, w = fr["shape"]
        focal_guess = float(max(h, w))
        # Distortion from this frame's line sets only
        if fr["line_sets"]:
            k1, k2 = calibrate_distortion_from_lines(fr["line_sets"], (h, w),
                                                       focal_guess)
        else:
            k1, k2 = 0.0, 0.0
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            k1, k2 = 0.0, 0.0
        intr = CameraIntrinsics(fx=focal_guess, fy=focal_guess,
                                 cx=w/2.0, cy=h/2.0, k1=k1, k2=k2)
        fr["pix_u"] = undistort_points(fr["pixel_pts"], intr)
        fr["label"] = label
        fr["k1"] = k1
        fr["k2"] = k2
        print(f"{label}: {len(fr['pixel_pts'])} correspondences, "
              f"tilt {fr['tilt_deg']:.1f}°, k1={k1:.4f}, k2={k2:.4f}")
        processed.append(fr)

    # Sweep Cz for each frame
    cz_values = np.arange(20, 75, 2.5)
    results = {}
    for fr in processed:
        rmses = []
        cx_list, cy_list, roll_list, f_list, pan_list, tilt_list = [], [], [], [], [], []
        for cz in cz_values:
            params, rmse = calibrate_with_fixed_cz(
                fr["pix_u"], fr["field_pts"], fr["shape"], cz,
            )
            rmses.append(rmse)
            cx_list.append(params[0])
            cy_list.append(params[1])
            pan_list.append(params[2])
            tilt_list.append(params[3])
            f_list.append(params[4])
            roll_list.append(np.degrees(params[5]))
        results[fr["label"]] = {
            "cz": cz_values, "rmse": rmses,
            "cx": cx_list, "cy": cy_list,
            "pan": pan_list, "tilt": tilt_list,
            "f": f_list, "roll": roll_list,
        }

    # Print table + find each frame's optimum
    print("\n=== Cz sweep summary ===")
    for label, res in results.items():
        best_idx = int(np.argmin(res["rmse"]))
        best_cz = res["cz"][best_idx]
        best_rmse = res["rmse"][best_idx]
        print(f"\n{label}:")
        print(f"  Best Cz={best_cz:.1f}, RMSE={best_rmse:.2f}px")
        print(f"  At optimum: Cx={res['cx'][best_idx]:.1f}, Cy={res['cy'][best_idx]:.1f}, "
              f"roll={res['roll'][best_idx]:.1f}°, f={res['f'][best_idx]:.0f}, "
              f"pan={res['pan'][best_idx]:.3f}, tilt={res['tilt'][best_idx]:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for label, res in results.items():
        ax.plot(res["cz"], res["rmse"], marker="o", label=label)
    ax.set_xlabel("Camera height Cz (yards)")
    ax.set_ylabel("Reprojection RMSE (pixels)")
    ax.set_title("RMSE vs. Cz per frame")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for label, res in results.items():
        ax.plot(res["cz"], res["cy"], marker="o", label=label)
    ax.set_xlabel("Cz (yards)")
    ax.set_ylabel("Cy (yards)")
    ax.set_title("Cy chosen vs. Cz")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for label, res in results.items():
        ax.plot(res["cz"], res["roll"], marker="o", label=label)
    ax.set_xlabel("Cz (yards)")
    ax.set_ylabel("Roll (degrees)")
    ax.set_title("Roll chosen vs. Cz")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for label, res in results.items():
        ax.plot(res["cz"], res["f"], marker="o", label=label)
    ax.set_xlabel("Cz (yards)")
    ax.set_ylabel("Focal length (pixels)")
    ax.set_title("Focal length chosen vs. Cz")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "landscape_sweep.png")
    fig.savefig(out_path, dpi=110)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
