#!/usr/bin/env python3
"""
Set up Label Studio project with pre-annotated All-22 frames.

Converts YOLO-format labels to Label Studio JSON format and creates
a project configuration for reviewing/correcting bounding boxes.

Usage:
  1. Start Label Studio:  label-studio start
  2. Run this script:     python scripts/setup_label_studio.py
  3. Open http://localhost:8080 and import the generated JSON

Or use without Label Studio — this also generates a visual review
HTML page you can open directly in a browser.
"""

import json
import os
import glob
import base64

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "images")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")


def yolo_to_ls_bbox(cx, cy, w, h):
    """Convert YOLO normalized (cx, cy, w, h) to Label Studio (x, y, w, h) percent."""
    return {
        "x": (cx - w / 2) * 100,
        "y": (cy - h / 2) * 100,
        "width": w * 100,
        "height": h * 100,
    }


def generate_label_studio_json():
    """Generate Label Studio import JSON with pre-annotations."""
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    tasks = []

    for img_path in images:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))

        task = {
            "data": {
                "image": f"/data/local-files/?d=images/{img_name}",
            },
            "predictions": [{
                "model_version": "yolo12x_pretrained",
                "result": [],
            }],
        }

        if os.path.exists(label_path):
            with open(label_path) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, w, h = [float(x) for x in parts]
                    bbox = yolo_to_ls_bbox(cx, cy, w, h)
                    task["predictions"][0]["result"].append({
                        "id": f"box_{i}",
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "value": {
                            "rectanglelabels": ["player"],
                            **bbox,
                            "rotation": 0,
                        },
                    })

        tasks.append(task)

    output_path = os.path.join(OUTPUT_DIR, "label_studio_import.json")
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Label Studio import JSON: {output_path}")
    print(f"  {len(tasks)} tasks with pre-annotations")
    return output_path


def generate_review_html():
    """Generate a simple HTML page for visual review of annotations."""
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<title>All-22 Annotation Review</title>
<style>
body { font-family: monospace; background: #1a1a1a; color: #eee; margin: 20px; }
.frame { display: inline-block; margin: 10px; position: relative; }
.frame img { max-width: 640px; display: block; }
.frame .info { font-size: 12px; padding: 4px; background: #333; }
.controls { position: sticky; top: 0; background: #1a1a1a; padding: 10px; z-index: 10; }
h2 { color: #4CAF50; }
</style>
</head>
<body>
<div class="controls">
<h2>All-22 Pre-Annotation Review</h2>
<p>Review the bounding boxes below. Green = detected players.</p>
<p>Target: 22 players per frame (on-field only, no sideline personnel).</p>
<p>Use Label Studio for corrections: <code>label-studio start</code></p>
</div>
"""]

    # Only show a sample for the HTML review
    sample = images[::10]  # every 10th image
    for img_path in sample:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))

        # Read image and draw boxes
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]

        n_boxes = 0
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = [float(x) for x in parts]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    n_boxes += 1

        # Encode as base64 for embedding
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode()

        status = "OK" if 20 <= n_boxes <= 24 else "CHECK"
        color = "#4CAF50" if status == "OK" else "#FF9800"

        html_parts.append(f"""
<div class="frame">
  <img src="data:image/jpeg;base64,{b64}">
  <div class="info" style="color:{color}">{img_name} — {n_boxes} detections [{status}]</div>
</div>""")

    html_parts.append("</body></html>")

    output_path = os.path.join(OUTPUT_DIR, "review.html")
    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Visual review page: {output_path}")
    print(f"  Showing {len(sample)} sample frames (every 10th)")


def main():
    print("Setting up annotation review...\n")
    generate_label_studio_json()
    print()
    generate_review_html()

    print(f"\n{'='*60}")
    print("To start annotating:")
    print(f"  1. Open {os.path.join(OUTPUT_DIR, 'review.html')} for a quick visual check")
    print(f"  2. For corrections, start Label Studio:")
    print(f"     cd \"{PROJECT_ROOT}\"")
    print(f"     .venv/bin/label-studio start")
    print(f"  3. Create a project, set local storage to data/annotations/images/")
    print(f"  4. Import {os.path.join(OUTPUT_DIR, 'label_studio_import.json')}")
    print(f"  5. Review and correct bounding boxes")
    print(f"  6. Export as YOLO format when done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
