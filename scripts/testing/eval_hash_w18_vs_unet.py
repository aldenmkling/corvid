#!/usr/bin/env python3
"""Compare HRNet-W18 hash detector vs mit_b0 hash UNet on the keypoint val set.

For each detector, sweep sigmoid thresholds and match predictions to GT
hash keypoints (channel == 1) using nearest-unused matching with a
distance cap.

Both compared at a match radius equivalent to ~11 image-px (the rough
size of a hash mark): 4 px in W18 heatmap coords (256×448) = 8 px in
UNet input coords (512×896).
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

import segmentation_models_pytorch as smp
import timm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

INPUT_H, INPUT_W = 512, 896
HEATMAP_H, HEATMAP_W = 256, 448      # W18 output (1/2 res of input)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CC_AREA = 4                       # drop noise CCs in UNet output

W18_WEIGHTS = os.path.join(PROJECT_ROOT, "models/hrnet_w18_hash_round1_best.pth")
UNET_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_hash_last.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "data/field_keypoints/valid")


class HRNetW18Hash(nn.Module):
    """Same architecture as src/homography/grid_solver_v2._load_hash_w18."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "hrnet_w18", pretrained=False, features_only=True, out_indices=(0,))
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


def extract_w18_peaks(heatmap, threshold):
    """W18 keypoints: argmax inside each above-threshold connected component."""
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


def extract_unet_centroids(mask_prob, threshold):
    """UNet hash mask → keypoints via CC centroids."""
    bin_mask = (mask_prob >= threshold).astype(np.uint8)
    if not bin_mask.any():
        return []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_CC_AREA:
            continue
        cx, cy = cents[k]
        out.append((float(cx), float(cy)))
    return out


def gt_hash_points(ann, img_w, img_h, target_w, target_h):
    pts = []
    for p in ann["points"]:
        if p["channel"] == 1:
            pts.append((p["x"] / img_w * target_w,
                        p["y"] / img_h * target_h))
    return pts


def match(pred, gt, radius):
    if not pred and not gt: return 0, 0, 0
    if not pred: return 0, 0, len(gt)
    if not gt: return 0, len(pred), 0
    used = [False] * len(gt); tp = 0
    for px, py in pred:
        best_d = float("inf"); best_j = -1
        for j, (gx, gy) in enumerate(gt):
            if used[j]: continue
            d = np.hypot(px - gx, py - gy)
            if d < best_d: best_d = d; best_j = j
        if best_j >= 0 and best_d <= radius:
            used[best_j] = True; tp += 1
    return tp, len(pred) - tp, sum(1 for u in used if not u)


def sweep(name, cache, extract_fn, radius, thresholds, ext_args=()):
    print()
    print(f"  ── {name} (radius={radius:.1f} px in this coord frame) ──")
    print(f"  {'thresh':>7}  {'tp':>5} {'fp':>5} {'fn':>5}  "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    results = []
    for t in thresholds:
        ttp = tfp = tfn = 0
        for prob, gt in cache:
            preds = extract_fn(prob, t, *ext_args)
            tp, fp, fn = match(preds, gt, radius)
            ttp += tp; tfp += fp; tfn += fn
        P = ttp / max(ttp + tfp, 1)
        R = ttp / max(ttp + tfn, 1)
        F = 2 * P * R / max(P + R, 1e-10)
        results.append((t, P, R, F))
        print(f"  {t:>7.2f}  {ttp:>5} {tfp:>5} {tfn:>5}  "
              f"{P:>6.3f} {R:>6.3f} {F:>6.3f}")
    best = max(results, key=lambda r: r[3])
    return best


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")

    print(f"  loading W18:    {os.path.basename(W18_WEIGHTS)}")
    w18 = HRNetW18Hash()
    ckpt = torch.load(W18_WEIGHTS, map_location=device, weights_only=False)
    w18.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    w18.to(device).eval()

    print(f"  loading UNet:   {os.path.basename(UNET_WEIGHTS)}")
    unet = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    ckpt = torch.load(UNET_WEIGHTS, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt.get("model_state_dict", ckpt))
    unet.to(device).eval()

    with open(os.path.join(VAL_DIR, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    n_gt = sum(len([p for p in a["points"] if p["channel"] == 1])
                for a in coco["annotations"])
    print(f"  val: {len(coco['annotations'])} frames, {n_gt} GT hash points")

    w18_cache, unet_cache = [], []
    with torch.no_grad():
        for ann in coco["annotations"]:
            info = images_by_id[ann["image_id"]]
            frame = cv2.imread(os.path.join(VAL_DIR, "images", info["file_name"]))
            if frame is None: continue
            h, w = frame.shape[:2]
            tensor = preprocess(frame).to(device)

            heatmap = torch.sigmoid(w18(tensor)[0, 0]).cpu().numpy()
            w18_cache.append((heatmap, gt_hash_points(ann, w, h,
                                                       HEATMAP_W, HEATMAP_H)))

            mask = torch.sigmoid(unet(tensor)[0, 0]).cpu().numpy()
            unet_cache.append((mask, gt_hash_points(ann, w, h,
                                                     INPUT_W, INPUT_H)))

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.70, 0.80]

    # W18 in heatmap coords (radius 4 = ~11 image px)
    best_w18 = sweep("HRNet-W18", w18_cache, extract_w18_peaks, 4.0, thresholds)
    # UNet in input coords (radius 8 = ~11 image px, same physical distance)
    best_unet = sweep("mit_b0 hash UNet", unet_cache,
                       extract_unet_centroids, 8.0, thresholds)

    print()
    print(f"  W18    best F1 = {best_w18[3]:.3f}  thresh={best_w18[0]:.2f}  "
          f"P={best_w18[1]:.3f}  R={best_w18[2]:.3f}")
    print(f"  mit_b0 best F1 = {best_unet[3]:.3f}  thresh={best_unet[0]:.2f}  "
          f"P={best_unet[1]:.3f}  R={best_unet[2]:.3f}")


if __name__ == "__main__":
    main()
