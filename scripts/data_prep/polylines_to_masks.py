#!/usr/bin/env python3
"""Convert Label Studio polyline annotations → binary line masks.

Input: Label Studio JSON export with polyline annotations. Each task has
zero or more polylines, each labeled `yardline` or `sideline`. We rasterize
them to a 2-channel binary mask (channel 0 = yardline, channel 1 = sideline)
matching the source image's resolution.

The mask format matches what UNet was trained on so the new annotations slot
into the existing training pipeline.

Usage:
    python scripts/data_prep/polylines_to_masks.py \\
        --json /path/to/label_studio_export.json \\
        --images-dir data/line_detection/al_round3/images \\
        --out-dir   data/line_detection/al_round3/masks \\
        --thickness 5

For each task, writes:
    <out-dir>/<image_basename>.png
where the PNG is a 3-channel image with:
    R = yardline mask × 255
    G = sideline mask × 255
    B = 0
(matches the 2-class color encoding used by build_line_dataset.py.)
"""

import argparse
import json
import os

import cv2
import numpy as np


YARDLINE_LABELS = {"yardline", "yard_line", "yard"}
SIDELINE_LABELS = {"sideline", "side_line", "side"}


def parse_polylines(task: dict, target_w: int, target_h: int):
    """Extract polylines from a single Label Studio task.

    Label Studio 1.23 has no PolyLine tag, so we use PolygonLabels for input
    and treat each polygon as an OPEN polyline on the rasterization side
    (the implicit closing edge gets dropped). Label Studio stores point coords
    as percentages (0-100) of image dims; we convert to pixel coords here.

    Returns dict {label_name: list of (N, 2) point arrays}.
    """
    out = {"yardline": [], "sideline": []}
    annotations = task.get("annotations") or task.get("completions") or []
    for ann in annotations:
        for r in ann.get("result", []):
            # Accept either polylinelabels (newer LS versions) or polygonlabels
            # (the workaround we use on 1.23). Both have the same point format.
            if r.get("type") not in ("polylinelabels", "polygonlabels"):
                continue
            value = r.get("value", {})
            labels = (value.get("polylinelabels") or value.get("polygonlabels")
                      or value.get("labels") or [])
            if not labels:
                continue
            label = labels[0].lower()
            if label in YARDLINE_LABELS:
                cls = "yardline"
            elif label in SIDELINE_LABELS:
                cls = "sideline"
            else:
                continue
            points_pct = value.get("points", [])
            if len(points_pct) < 2:
                continue
            pts_px = []
            for px_pct, py_pct in points_pct:
                pts_px.append([float(px_pct) * target_w / 100.0,
                                float(py_pct) * target_h / 100.0])
            out[cls].append(np.array(pts_px, dtype=np.float32))
    return out


def rasterize_polylines(polylines: list, shape: tuple[int, int], thickness: int):
    """Rasterize a list of (N, 2) polylines into a binary mask of `shape` (H, W)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for pts in polylines:
        if len(pts) < 2:
            continue
        pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts_int], isClosed=False, color=1, thickness=thickness)
    return mask


def get_image_path(task: dict, images_dir: str) -> str | None:
    """Resolve a task's image filename. LS stores it under various keys."""
    data = task.get("data", {})
    for key in ("image", "img", "frame"):
        v = data.get(key)
        if not v:
            continue
        # Strip Label Studio path prefixes (e.g. /data/local-files/?d=…)
        v = str(v)
        if "?d=" in v:
            v = v.split("?d=")[-1]
        # Try filename-only lookup in images_dir
        basename = os.path.basename(v)
        path = os.path.join(images_dir, basename)
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True,
                    help="Label Studio polyline-annotation export JSON")
    ap.add_argument("--images-dir", required=True,
                    help="Directory containing the source frames")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write mask PNGs")
    ap.add_argument("--thickness", type=int, default=5,
                    help="Stroke width in pixels for line rasterization")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.json) as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = [tasks]

    n_ok = 0
    n_skip = 0
    for task in tasks:
        img_path = get_image_path(task, args.images_dir)
        if img_path is None:
            n_skip += 1
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            n_skip += 1
            continue
        h, w = frame.shape[:2]
        polys = parse_polylines(task, target_w=w, target_h=h)
        yard_mask = rasterize_polylines(polys["yardline"], (h, w), args.thickness)
        side_mask = rasterize_polylines(polys["sideline"], (h, w), args.thickness)
        # Encode as 3-channel PNG: R=yardline, G=sideline, B=0
        # (matches build_line_dataset.py's expected mask format).
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[..., 2] = yard_mask * 255    # OpenCV is BGR; R channel == index 2
        out[..., 1] = side_mask * 255    # G channel
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.out_dir, f"{base}.png"), out)
        n_ok += 1

    print(f"  wrote {n_ok} masks, skipped {n_skip} (missing image / no polylines)")


if __name__ == "__main__":
    main()
