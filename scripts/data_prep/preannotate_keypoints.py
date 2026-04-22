#!/usr/bin/env python3
"""
Pre-annotate field keypoint frames using the HRNet + grid-solver pipeline.

Instead of dumping every raw HRNet peak, this runs the full grid-solver
pipeline (pair hashes → attach sidelines → check grid consistency of
singletons) and emits only the keypoints our real inference pipeline would
use. This gives the annotator a preview of the model's actual output and
lets them correct it where wrong.

Emitted per frame:
  - Both hashes from every paired yard-line group.
  - Sidelines attached to paired groups (on the yard line within tolerance).
  - Singleton hashes whose x fits the established grid spacing
    (`grid_fit_ok` in the yard-line group).

NOT emitted:
  - Singleton sidelines (no yard-line confirmation).
  - Singleton hashes that don't fit the grid.
  - Raw peaks that the grid solver dropped as noise.

Usage:
    python scripts/data_prep/preannotate_keypoints.py \\
        --image-dir data/field_keypoints/al_round2/images \\
        --output data/field_keypoints/al_round2/ls_import.json \\
        --ls-path-prefix al_round2/images
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# The grid solver lives with the tests for now — import from there.
_TEST_DIR = os.path.join(PROJECT_ROOT, "scripts", "testing")
sys.path.insert(0, _TEST_DIR)

from test_yard_line_grouping import (  # noqa: E402
    run_hrnet, extract_peaks, split_hash_rows, pair_hashes,
    find_sideline_on_yard_line, assign_grid_positions,
    compute_hash_pca, _row_coord, yardline_tilt_slope_from_pairs,
)


DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "field_keypoints",
                                   "annotation_images")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "field_keypoints",
                                "ls_import.json")

HASH_THRESH = 0.40
SIDELINE_THRESH = 0.30

# Label Studio labels (must match labeling_config.xml)
LABEL_SIDELINE = "sideline_intersection"
LABEL_HASH = "hash_intersection"


def run_grid_solver(frame, weights_path, device="cpu"):
    """Run HRNet + grid solver. Returns (groups, raw_hashes, raw_sidelines).

    Raw detections are returned too so the caller can fall back to them
    when no valid grid emerges.
    """
    h, w = frame.shape[:2]
    heatmaps = run_hrnet(frame, weights_path, device=device)
    sideline_pxs, sideline_confs = extract_peaks(heatmaps[0], SIDELINE_THRESH, (h, w))
    hash_pxs, hash_confs = extract_peaks(heatmaps[1], HASH_THRESH, (h, w))

    far_hashes, near_hashes = split_hash_rows(hash_pxs)
    pairs, unpaired_far, unpaired_near, _, _ = pair_hashes(far_hashes, near_hashes)

    groups = []
    used_sideline = set()
    for fh, nh in pairs:
        fh = np.asarray(fh)
        nh = np.asarray(nh)
        sl_idx, _ = find_sideline_on_yard_line(
            nh, fh, sideline_pxs, max_perp_distance=12,
        )
        sideline_pt = None
        if sl_idx is not None and sl_idx not in used_sideline:
            used_sideline.add(sl_idx)
            sideline_pt = sideline_pxs[sl_idx].tolist()
        groups.append({
            "far_hash": fh.tolist(),
            "near_hash": nh.tolist(),
            "sideline": sideline_pt,
            "singleton": False,
        })
    for fh in unpaired_far:
        groups.append({
            "far_hash": np.asarray(fh).tolist(),
            "near_hash": None,
            "sideline": None,
            "singleton": True,
        })
    for nh in unpaired_near:
        groups.append({
            "far_hash": None,
            "near_hash": np.asarray(nh).tolist(),
            "sideline": None,
            "singleton": True,
        })

    # Note: singleton sidelines intentionally not emitted (too clustered).
    assign_grid_positions(groups)
    return groups, hash_pxs, sideline_pxs


def _make_annotation(px, py, label, img_w, img_h, idx):
    return {
        "id": f"gs_{idx}",
        "type": "keypointlabels",
        "value": {
            "x": round(float(px) / img_w * 100.0, 2),
            "y": round(float(py) / img_h * 100.0, 2),
            "width": 0.5,
            "keypointlabels": [label],
        },
        "to_name": "image",
        "from_name": "keypoint",
        "original_width": img_w,
        "original_height": img_h,
    }


def groups_to_annotations(groups, img_w, img_h):
    """Return Label Studio keypoint annotations using the grid-solver output.

    Emits:
      - Paired groups: both hashes + any matched sideline
      - Singleton hashes (far or near) that fit the grid
      - Singleton sidelines that fit the grid (the ones confirmed to lie on
        the paired-sideline row by build_yard_line_groups)
    """
    annotations = []
    for g in groups:
        if g.get("singleton"):
            if not g.get("grid_fit_ok", False):
                continue
            if g.get("far_hash") is not None:
                annotations.append(_make_annotation(
                    g["far_hash"][0], g["far_hash"][1], LABEL_HASH,
                    img_w, img_h, len(annotations)))
            elif g.get("near_hash") is not None:
                annotations.append(_make_annotation(
                    g["near_hash"][0], g["near_hash"][1], LABEL_HASH,
                    img_w, img_h, len(annotations)))
            elif g.get("sideline") is not None:
                annotations.append(_make_annotation(
                    g["sideline"][0], g["sideline"][1], LABEL_SIDELINE,
                    img_w, img_h, len(annotations)))
            continue

        # Paired group
        if g.get("far_hash") is not None:
            annotations.append(_make_annotation(
                g["far_hash"][0], g["far_hash"][1], LABEL_HASH,
                img_w, img_h, len(annotations)))
        if g.get("near_hash") is not None:
            annotations.append(_make_annotation(
                g["near_hash"][0], g["near_hash"][1], LABEL_HASH,
                img_w, img_h, len(annotations)))
        if g.get("sideline") is not None:
            annotations.append(_make_annotation(
                g["sideline"][0], g["sideline"][1], LABEL_SIDELINE,
                img_w, img_h, len(annotations)))

    return annotations


def raw_peaks_to_annotations(hash_pxs, sideline_pxs, img_w, img_h):
    """Fallback: emit every raw detection. Used when the grid solver fails
    to establish a valid grid (< 2 paired hash groups)."""
    annotations = []
    for pt in hash_pxs:
        annotations.append(_make_annotation(
            pt[0], pt[1], LABEL_HASH, img_w, img_h, len(annotations)))
    for pt in sideline_pxs:
        annotations.append(_make_annotation(
            pt[0], pt[1], LABEL_SIDELINE, img_w, img_h, len(annotations)))
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                        help="Directory of .jpg frames to pre-annotate")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Label Studio import JSON output path")
    parser.add_argument("--ls-path-prefix", default="annotation_images",
                        help="Subdirectory path LS uses to resolve local files "
                             "relative to LOCAL_FILES_DOCUMENT_ROOT. "
                             "Example: 'al_round2/images'.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    print(f"Found {len(images)} images in {args.image_dir}")
    if not images:
        print("No images found.")
        return

    print(f"Using weights: {args.weights}")
    print(f"Grid-solver pipeline: HRNet → pair hashes → attach sidelines → "
          f"validate singletons by grid spacing")

    tasks = []
    total_detections = 0
    total_hash = 0
    total_sideline = 0
    n_fallback = 0

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Warning: could not read {fname}")
            continue
        h, w = frame.shape[:2]

        groups, raw_hashes, raw_sidelines = run_grid_solver(
            frame, args.weights, device=args.device,
        )
        # If the grid solver couldn't establish a valid grid (< 2 paired hash
        # groups), fall back to showing raw peak detections — the annotator
        # can at least see what HRNet found.
        n_paired = sum(1 for g in groups if not g.get("singleton"))
        if n_paired < 2:
            annotations = raw_peaks_to_annotations(raw_hashes, raw_sidelines, w, h)
            n_fallback += 1
        else:
            annotations = groups_to_annotations(groups, w, h)

        n_hash = sum(1 for a in annotations
                     if a["value"]["keypointlabels"][0] == LABEL_HASH)
        n_side = sum(1 for a in annotations
                     if a["value"]["keypointlabels"][0] == LABEL_SIDELINE)
        total_hash += n_hash
        total_sideline += n_side
        total_detections += len(annotations)

        ls_path = f"{args.ls_path_prefix}/{fname}" if args.ls_path_prefix else fname
        task = {
            "data": {"image": f"/data/local-files/?d={ls_path}"},
            "predictions": [{
                "result": annotations,
                "score": 0.0,
            }],
        }
        tasks.append(task)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} processed")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(tasks, f, indent=2)

    avg = total_detections / max(len(tasks), 1)
    print(f"\nDone. {len(tasks)} tasks, {total_detections} keypoints "
          f"({total_hash} hash, {total_sideline} sideline, {avg:.1f} avg/frame)")
    print(f"Fallback to raw peaks: {n_fallback}/{len(tasks)} frames "
          f"(grid solver couldn't establish a valid grid)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
