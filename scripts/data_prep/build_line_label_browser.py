#!/usr/bin/env python3
"""Build an HTML browser for line-detection training frames.

For each frame in `data/line_detection/train`, render an overlay
(source image + yard (red) + side (green) mask) and bundle into a
clickable HTML page where you can:
  - flip through frames with arrow keys
  - click on the near sideline → record (frame, click_x, click_y)
  - hit `n` to mark "no near sideline visible"
  - hit `u` to undo
  - hit `s` to download a JSON of all decisions
"""

import argparse
import json
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Line Label Browser</title>
<style>
body { font-family: -apple-system, sans-serif; background: #1c1c1c; color: #ddd;
       margin: 0; padding: 8px; }
#info { font-size: 13px; margin-bottom: 6px; }
.canvas-wrap { position: relative; display: inline-block; border: 2px solid #444; }
#img { display: block; max-width: 100%; cursor: crosshair;
       image-rendering: -webkit-optimize-contrast; }
#crosshair { position: absolute; pointer-events: none; }
#yline { stroke: cyan; stroke-width: 1; }
#xline { stroke: cyan; stroke-width: 1; }
#marker { position: absolute; border: 2px solid yellow; border-radius: 50%;
          width: 14px; height: 14px; pointer-events: none;
          transform: translate(-50%, -50%); }
button { font-size: 13px; padding: 6px 10px; margin-right: 6px;
         background: #333; color: #ddd; border: 1px solid #555;
         border-radius: 4px; cursor: pointer; }
button:hover { background: #444; }
.help { color: #888; font-size: 11px; margin-top: 6px; }
.has-y { color: #6f6; }
.no-near { color: #f88; }
</style></head><body>
<div id="info">loading...</div>
<div class="canvas-wrap">
  <img id="img" alt="frame"/>
  <svg id="crosshair" width="100%" height="100%"
       style="position: absolute; top: 0; left: 0; pointer-events: none;">
    <line id="yline" x1="0" y1="0" x2="0" y2="0"/>
    <line id="xline" x1="0" y1="0" x2="0" y2="0"/>
  </svg>
  <div id="marker" style="display: none"></div>
</div>
<div style="margin-top: 8px;">
  <button onclick="prev()">Prev [←]</button>
  <button onclick="next()">Next [→]</button>
  <button onclick="markNoNear()">No near visible [n]</button>
  <button onclick="undo()">Undo [u]</button>
  <button onclick="downloadResults()">Save JSON [s]</button>
</div>
<div class="help">
  Click on the near sideline in the image to record its (x, y).
  ← / → navigate. n = "no near sideline visible". u = undo current frame.
  s = download JSON of all decisions.
</div>
<script>
const MANIFEST = __MANIFEST__;
let idx = 0;
const decisions = {};

function show() {
  const img = document.getElementById('img');
  const info = document.getElementById('info');
  const m = MANIFEST[idx];
  img.src = 'overlays/' + m.filename;
  const d = decisions[m.filename];
  let dStr = '<span style="color:#888">unmarked</span>';
  if (d === 'no_near') dStr = '<span class="no-near">no near visible</span>';
  else if (d) dStr = `<span class="has-y">y=${d.y} x=${d.x}</span>`;
  const counts = Object.values(decisions).reduce((a, v) => {
    if (v === 'no_near') a.no_near++; else if (v) a.with_y++;
    return a;
  }, {with_y: 0, no_near: 0});
  info.innerHTML = `${idx+1} / ${MANIFEST.length} &mdash; ${m.filename}  ` +
                    `(${m.w}×${m.h}) &middot; ${dStr} &middot; ` +
                    `<span class="has-y">marked=${counts.with_y}</span>  ` +
                    `<span class="no-near">no-near=${counts.no_near}</span>`;
  // Marker
  const marker = document.getElementById('marker');
  if (d && d !== 'no_near') {
    const wrap = document.querySelector('.canvas-wrap');
    const rect = img.getBoundingClientRect();
    const wrect = wrap.getBoundingClientRect();
    const sx = img.clientWidth / m.w;
    const sy = img.clientHeight / m.h;
    marker.style.display = 'block';
    marker.style.left = (rect.left - wrect.left + d.x * sx) + 'px';
    marker.style.top = (rect.top - wrect.top + d.y * sy) + 'px';
  } else {
    marker.style.display = 'none';
  }
}

function next() { if (idx < MANIFEST.length - 1) { idx++; show(); } }
function prev() { if (idx > 0) { idx--; show(); } }
function undo() {
  delete decisions[MANIFEST[idx].filename];
  show();
}
function markNoNear() {
  decisions[MANIFEST[idx].filename] = 'no_near';
  show();
  if (idx < MANIFEST.length - 1) { idx++; setTimeout(show, 30); }
}

document.getElementById('img').addEventListener('click', (e) => {
  const img = e.target;
  const m = MANIFEST[idx];
  const rect = img.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const x_orig = Math.round(cx * m.w / img.clientWidth);
  const y_orig = Math.round(cy * m.h / img.clientHeight);
  decisions[m.filename] = {x: x_orig, y: y_orig};
  show();
});

document.getElementById('img').addEventListener('mousemove', (e) => {
  const img = e.target;
  const wrap = document.querySelector('.canvas-wrap');
  const rect = img.getBoundingClientRect();
  const wrect = wrap.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const ox = rect.left - wrect.left;
  const oy = rect.top - wrect.top;
  const yline = document.getElementById('yline');
  const xline = document.getElementById('xline');
  yline.setAttribute('x1', ox);   yline.setAttribute('x2', ox + img.clientWidth);
  yline.setAttribute('y1', oy + cy); yline.setAttribute('y2', oy + cy);
  xline.setAttribute('y1', oy);   xline.setAttribute('y2', oy + img.clientHeight);
  xline.setAttribute('x1', ox + cx); xline.setAttribute('x2', ox + cx);
});

function downloadResults() {
  const blob = new Blob([JSON.stringify(decisions, null, 2)],
                         {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'near_sideline_y.json';
  a.click();
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') next();
  else if (e.key === 'ArrowLeft') prev();
  else if (e.key === 'n') markNoNear();
  else if (e.key === 'u') undo();
  else if (e.key === 's') downloadResults();
});
show();
</script></body></html>
"""


def build_overlay(img: np.ndarray, mask_bgr: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    yard = (mask_bgr[..., 2] > 127).astype(np.uint8)
    side = (mask_bgr[..., 1] > 127).astype(np.uint8)
    yard = cv2.resize(yard, (w, h), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy().astype(np.float32)
    for m, color in [(yard, [60, 60, 230]), (side, [60, 230, 60])]:
        idx = m > 0
        out[idx] = 0.55 * out[idx] + 0.45 * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/train"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/line_detection/label_browser"))
    args = ap.parse_args()

    img_dir = os.path.join(args.root, "images")
    mask_dir = os.path.join(args.root, "masks")
    overlay_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(img_dir)
                     if f.endswith(".jpg") and not f.startswith("._")])
    manifest = []
    for f in files:
        stem = os.path.splitext(f)[0]
        img_p = os.path.join(img_dir, f)
        mask_p = os.path.join(mask_dir, stem + ".png")
        if not os.path.exists(mask_p):
            continue
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p)
        if img is None or mask is None:
            continue
        ov = build_overlay(img, mask)
        out_name = f
        cv2.imwrite(os.path.join(overlay_dir, out_name), ov,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        manifest.append({"filename": out_name, "w": img.shape[1], "h": img.shape[0]})

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest))
    html_path = os.path.join(args.out_dir, "browser.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"  {len(manifest)} overlays")
    print(f"  open: {html_path}")


if __name__ == "__main__":
    main()
