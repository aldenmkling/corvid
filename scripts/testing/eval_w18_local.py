#!/usr/bin/env python3
"""Local benchmark of HRNet-W18 (hash-only, 1-channel) on the 179-frame val
set in data/field_keypoints/valid/. Mirrors `eval_threshold_pod.py` but
loads the W18 backbone and evaluates only the hash channel (ch=1).
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

INPUT_H, INPUT_W = 512, 896
HEATMAP_H, HEATMAP_W = 256, 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MATCH_RADIUS = 4.0           # heatmap-resolution px

WEIGHTS = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints/valid")


class HRNetW18Hash(nn.Module):
    """1-channel hash detector matching the architecture in
    src/homography/grid_solver_v2.py:_load_hash_w18."""
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "hrnet_w18", pretrained=False, features_only=True, out_indices=(0,),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.head(self.backbone(x)[0])


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def extract_peaks(heatmap, threshold):
    mask = heatmap >= threshold
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    out = []
    for cid in range(1, n + 1):
        m = labels == cid
        idx = (heatmap * m).argmax()
        y, x = idx // heatmap.shape[1], idx % heatmap.shape[1]
        out.append((float(x), float(y)))
    return out


def gt_hash_points(ann, img_w, img_h):
    """Return only channel-1 (hash) ground truth in heatmap coords."""
    pts = []
    for p in ann["points"]:
        if p["channel"] == 1:
            hx = p["x"] / img_w * HEATMAP_W
            hy = p["y"] / img_h * HEATMAP_H
            pts.append((hx, hy))
    return pts


def match(pred_peaks, gt_peaks, radius):
    if not pred_peaks and not gt_peaks: return 0, 0, 0
    if not pred_peaks: return 0, 0, len(gt_peaks)
    if not gt_peaks: return 0, len(pred_peaks), 0
    used = [False] * len(gt_peaks); tp = 0
    for px, py in pred_peaks:
        best_d = float("inf"); best_j = -1
        for j, (gx, gy) in enumerate(gt_peaks):
            if used[j]: continue
            d = np.hypot(px - gx, py - gy)
            if d < best_d: best_d = d; best_j = j
        if best_j >= 0 and best_d <= radius:
            used[best_j] = True; tp += 1
    return tp, len(pred_peaks) - tp, sum(1 for u in used if not u)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  loading {WEIGHTS}")
    print(f"  device: {device}")
    model = HRNetW18Hash()
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    with open(os.path.join(VAL_DIR, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    print(f"  val: {len(coco['annotations'])} images, "
          f"{sum(len([p for p in a['points'] if p['channel'] == 1]) for a in coco['annotations'])} hash GT points")

    cache = []
    with torch.no_grad():
        for ann in coco["annotations"]:
            info = images_by_id[ann["image_id"]]
            frame = cv2.imread(os.path.join(VAL_DIR, "images", info["file_name"]))
            if frame is None: continue
            h, w = frame.shape[:2]
            tensor = preprocess(frame).to(device)
            logits = model(tensor)
            heatmap = torch.sigmoid(logits[0, 0]).cpu().numpy()
            cache.append((heatmap, gt_hash_points(ann, w, h)))

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.70, 0.80]
    print()
    print(f"  {'thresh':>7}  {'tp':>5} {'fp':>5} {'fn':>5}  "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    results = []
    for t in thresholds:
        ttp = tfp = tfn = 0
        for heatmap, gt in cache:
            peaks = extract_peaks(heatmap, t)
            tp, fp, fn = match(peaks, gt, MATCH_RADIUS)
            ttp += tp; tfp += fp; tfn += fn
        P = ttp / max(ttp + tfp, 1)
        R = ttp / max(ttp + tfn, 1)
        F = 2 * P * R / max(P + R, 1e-10)
        results.append((t, P, R, F))
        print(f"  {t:>7.2f}  {ttp:>5} {tfp:>5} {tfn:>5}  "
              f"{P:>6.3f} {R:>6.3f} {F:>6.3f}")
    best = max(results, key=lambda r: r[3])
    print(f"\n  Best F1 = {best[3]:.3f} at thresh={best[0]:.2f}  "
          f"(P={best[1]:.3f}, R={best[2]:.3f})")


if __name__ == "__main__":
    main()
