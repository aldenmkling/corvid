#!/usr/bin/env python3
"""Build a triage HTML page for reviewing auto-generated hash masks.

For each training frame with hash points, renders an overlay (source +
red-tinted mask) and bundles them into a single self-contained HTML
viewer with keyboard shortcuts (g=good, f=fix, u=undo, s=save JSON).

Open the resulting `triage.html` directly in a browser. Hit "Save" at
the end (or any time) to download a JSON of decisions per frame.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hash Mask Triage</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1c1c1c; color: #ddd;
       margin: 0; padding: 10px; }
#info { font-size: 14px; margin-bottom: 6px; }
#img  { display: block; max-width: 100%; max-height: calc(100vh - 110px);
       border: 2px solid #444; image-rendering: -webkit-optimize-contrast; }
#controls { margin-top: 8px; }
button { font-size: 14px; padding: 6px 12px; margin-right: 6px;
         background: #333; color: #ddd; border: 1px solid #555;
         border-radius: 4px; cursor: pointer; }
button:hover { background: #444; }
.help { color: #888; font-size: 12px; margin-top: 6px; }
.good { color: #6f6; } .fix  { color: #f88; }
</style></head><body>
<div id="info">loading...</div>
<img id="img" alt="overlay"/>
<div id="controls">
  <button onclick="decide('good')">Good [g]</button>
  <button onclick="decide('fix')">Needs Fix [f]</button>
  <button onclick="undo()">Undo [u]</button>
  <button onclick="downloadResults()">Save JSON [s]</button>
</div>
<div class="help">Keyboard: g = good, f = needs fix, u = undo, s = save.
Save anytime to download partial results.</div>
<script>
const MANIFEST = __MANIFEST__;
let idx = 0;
const history = [];
const decisions = {};
function show() {
  const img = document.getElementById('img');
  const info = document.getElementById('info');
  if (idx >= MANIFEST.length) {
    img.style.display = 'none';
    info.innerHTML = `Done! ${Object.keys(decisions).length} decided. Hit Save.`;
    return;
  }
  const m = MANIFEST[idx];
  const counts = Object.values(decisions).reduce((a,b)=>{a[b]=(a[b]||0)+1;return a},{});
  img.src = 'overlays/' + m.filename;
  info.innerHTML = `${idx+1} / ${MANIFEST.length} &mdash; ${m.frame} ` +
                   `(${m.n_hashes} hashes) &middot; ` +
                   `<span class="good">good=${counts.good||0}</span> ` +
                   `<span class="fix">fix=${counts.fix||0}</span>`;
}
function decide(label) {
  if (idx >= MANIFEST.length) return;
  history.push(MANIFEST[idx].frame);
  decisions[MANIFEST[idx].frame] = label;
  idx++;
  show();
}
function undo() {
  if (history.length === 0) return;
  const last = history.pop();
  delete decisions[last];
  idx = Math.max(0, idx - 1);
  show();
}
function downloadResults() {
  const blob = new Blob([JSON.stringify(decisions, null, 2)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'hash_triage_results.json';
  a.click();
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'g') decide('good');
  else if (e.key === 'f') decide('fix');
  else if (e.key === 'u') undo();
  else if (e.key === 's') downloadResults();
});
show();
</script></body></html>
"""


def build_overlay(img: np.ndarray, mask: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w > target_w:
        s = target_w / w
        img = cv2.resize(img, (target_w, int(round(h * s))))
        mask = cv2.resize(mask, (target_w, int(round(h * s))),
                           interpolation=cv2.INTER_NEAREST)
    out = img.copy().astype(np.float32)
    m = mask > 0
    red = np.array([60, 60, 230], dtype=np.float32)
    out[m] = 0.4 * out[m] + 0.6 * red
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint-dir", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train"))
    ap.add_argument("--mask-dir", default=os.path.join(
        PROJECT_ROOT, "output/hash_mask_test/masks"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/triage"))
    ap.add_argument("--target-width", type=int, default=1280)
    args = ap.parse_args()

    overlay_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    with open(os.path.join(args.keypoint_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    manifest = []
    n_skipped = 0
    for ann in coco["annotations"]:
        info = images_by_id[ann["image_id"]]
        n_hashes = sum(1 for p in ann["points"] if p["channel"] == 1)
        if n_hashes == 0:
            continue
        frame_path = os.path.join(args.keypoint_dir, "images", info["file_name"])
        mask_path = os.path.join(args.mask_dir,
                                  os.path.splitext(info["file_name"])[0] + ".png")
        if not os.path.exists(frame_path) or not os.path.exists(mask_path):
            n_skipped += 1
            continue
        img = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            n_skipped += 1
            continue
        ov = build_overlay(img, mask, args.target_width)
        out_name = os.path.splitext(info["file_name"])[0] + ".jpg"
        cv2.imwrite(os.path.join(overlay_dir, out_name), ov,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        manifest.append({
            "frame": info["file_name"],
            "filename": out_name,
            "n_hashes": n_hashes,
        })

    # Sort manifest by frame name for reproducible order.
    manifest.sort(key=lambda m: m["frame"])

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest))
    html_path = os.path.join(args.out_dir, "triage.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"  {len(manifest)} overlays  ({n_skipped} skipped)")
    print(f"  overlays: {overlay_dir}/")
    print(f"  open: {html_path}")


if __name__ == "__main__":
    main()
