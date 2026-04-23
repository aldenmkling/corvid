#!/usr/bin/env python3
"""Diagnostic: render the grid solver's output on a single frame.

All grid-solver logic lives in `src/homography/grid_solver.py`. This script
just loads an image, runs the pipeline, and draws the resulting yard-line
groups (pairs, singletons, attached sidelines) for visual inspection.
"""

import os
import sys
import argparse

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.grid_solver import (  # noqa: E402
    run_hrnet, extract_peaks, build_yard_lines,
    HASH_THRESH, SIDELINE_THRESH,
)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
DEFAULT_FRAME = os.path.join(PROJECT_ROOT, "output", "al_round2_preview", "kickoff_wide.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "grid_solver_diagnostics")


def draw_visualization(frame, yard_lines, sideline_pxs, sideline_confs,
                       used_sideline):
    """Render the groupings with per-line colors and labels."""
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Colors cycled over groups
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (255, 160, 80), (160, 80, 255), (80, 255, 160),
        (200, 200, 200), (255, 120, 160), (120, 160, 255),
    ]

    # Sort for consistent coloring by grid position
    sort_order = sorted(range(len(yard_lines)),
                        key=lambda i: yard_lines[i].get('grid_pos', 999))

    for i, idx in enumerate(sort_order):
        yl = yard_lines[idx]
        color = palette[i % len(palette)]
        is_singleton = yl.get('singleton', False)
        grid_ok = yl.get('grid_fit_ok', False)
        # Dim singletons that fail grid check so we can see which would be kept
        if is_singleton and not grid_ok:
            color = (100, 100, 100)

        fh = np.array(yl['far_hash']) if yl['far_hash'] is not None else None
        nh = np.array(yl['near_hash']) if yl['near_hash'] is not None else None
        sl = np.array(yl['sideline']) if yl['sideline'] is not None else None

        # Extend yard line across the frame (near → far → top of image)
        if fh is not None and nh is not None:
            dx = fh[0] - nh[0]
            dy = fh[1] - nh[1]  # negative
            if abs(dy) > 1e-6:
                cv2.line(vis, tuple(nh.astype(int)), tuple(fh.astype(int)),
                         color, 2)
                t = (0 - fh[1]) / dy
                top_pt = (int(fh[0] + t * dx), 0)
                cv2.line(vis, tuple(fh.astype(int)), top_pt, color, 1, cv2.LINE_AA)
                t2 = (h - nh[1]) / dy
                bot_pt = (int(nh[0] + t2 * dx), h)
                cv2.line(vis, tuple(nh.astype(int)), bot_pt, color, 1, cv2.LINE_AA)

        # Draw markers
        if fh is not None:
            cv2.circle(vis, tuple(fh.astype(int)), 7, color, 2)
        if nh is not None:
            cv2.circle(vis, tuple(nh.astype(int)), 7, color, 2)
        if sl is not None:
            cv2.circle(vis, tuple(sl.astype(int)), 10, color, 3)
            cv2.drawMarker(vis, tuple(sl.astype(int)), color,
                           cv2.MARKER_CROSS, 14, 2)

        # Label with grid position + singleton type
        label_pt = nh if nh is not None else fh
        if label_pt is None and sl is not None:
            label_pt = sl
        if label_pt is not None:
            gp = yl.get('grid_pos', '?')
            singleton_type = ''
            if is_singleton:
                if fh is None and nh is None and sl is not None:
                    singleton_type = 'S_side'
                elif fh is not None and nh is None:
                    singleton_type = 'S_far'
                elif nh is not None and fh is None:
                    singleton_type = 'S_near'
                else:
                    singleton_type = 'S'
            sl_mark = '+S' if (sl is not None and not is_singleton) else ''
            ok_mark = '' if grid_ok or not is_singleton else ' [rejected]'
            text = f"g{gp} {singleton_type}{sl_mark}{ok_mark}"
            pt_int = (int(label_pt[0]) + 8, int(label_pt[1]) + 20)
            cv2.putText(vis, text, pt_int, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

    # Draw unused sideline detections in gray (these weren't claimed by any yard line)
    for i in range(len(sideline_pxs)):
        if i not in used_sideline:
            pt = tuple(sideline_pxs[i].astype(int))
            cv2.drawMarker(vis, pt, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 10, 1)
            cv2.putText(vis, f"?{sideline_confs[i]:.2f}",
                        (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Legend
    legend = [
        "Solid line = near->far hash pair",
        "Dashed line + cross = projected target for sideline",
        "Big circle = matched sideline (attached)",
        "Gray X = unmatched sideline detection",
        "Label: g<grid_pos> [S=singleton] [+S=has sideline]",
    ]
    y0 = h - 15 * len(legend) - 10
    for i, line in enumerate(legend):
        cv2.putText(vis, line, (10, y0 + i * 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=DEFAULT_FRAME, help="Path to frame image")
    parser.add_argument("--out-name", default=None,
                        help="Basename for output (default: derived from input frame)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame_path = args.frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Failed to read {frame_path}")
        return
    h, w = frame.shape[:2]
    print(f"Frame: {frame_path}")
    print(f"Size: {w}x{h}")
    out_name = args.out_name or os.path.splitext(os.path.basename(frame_path))[0] + "_grouping.jpg"

    heatmaps = run_hrnet(frame, WEIGHTS)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))
    print(f"Detections: {len(sideline_pxs)} sideline (thresh {SIDELINE_THRESH}), "
          f"{len(hash_pxs)} hash (thresh {HASH_THRESH})")

    yard_lines, used_sideline = build_yard_lines(
        hash_pxs, hash_confs, sideline_pxs, sideline_confs,
    )

    print(f"\n{len(yard_lines)} yard-line groups (sorted by grid_pos):")
    for yl in sorted(yard_lines, key=lambda y: y.get('grid_pos', 999)):
        gp = yl.get('grid_pos', '?')
        fh = yl['far_hash']
        nh = yl['near_hash']
        sl = yl['sideline']
        marks = []
        if fh: marks.append(f"far@({fh[0]:.0f},{fh[1]:.0f})")
        if nh: marks.append(f"near@({nh[0]:.0f},{nh[1]:.0f})")
        if sl:
            perp = yl.get('sideline_perp_dist')
            perp_str = f",perp={perp:.1f}" if perp is not None else ""
            conf = yl.get('sideline_conf')
            conf_str = f",conf={conf:.2f}" if conf is not None else ""
            marks.append(f"side@({sl[0]:.0f},{sl[1]:.0f}{conf_str}{perp_str})")
        singleton = " [singleton]" if yl.get('singleton') else ""
        print(f"  g{gp}{singleton}: {', '.join(marks)}")

    vis = draw_visualization(frame, yard_lines, sideline_pxs, sideline_confs,
                             used_sideline)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, vis)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
