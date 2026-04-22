import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

NUM_CHANNELS = 2
INPUT_H, INPUT_W = 512, 896
HEATMAP_H, HEATMAP_W = 256, 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MATCH_RADIUS = 4.0


class HRNetKeypointModel(nn.Module):
    def __init__(self, num_channels=2):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w48", pretrained=False, features_only=True, out_indices=(0,))
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, 1, bias=True),
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
    peaks = []
    for comp_id in range(1, n + 1):
        comp_mask = labels == comp_id
        vals = heatmap * comp_mask
        idx = vals.argmax()
        y, x = idx // heatmap.shape[1], idx % heatmap.shape[1]
        peaks.append((float(x), float(y)))
    return peaks


def gt_points(ann, img_w, img_h):
    per_ch = {0: [], 1: []}
    for p in ann["points"]:
        hx = p["x"] / img_w * HEATMAP_W
        hy = p["y"] / img_h * HEATMAP_H
        ch = p["channel"]
        if ch in per_ch:
            per_ch[ch].append((hx, hy))
    return per_ch


def match(pred_peaks, gt_peaks, radius):
    if not pred_peaks and not gt_peaks:
        return 0, 0, 0
    if not pred_peaks:
        return 0, 0, len(gt_peaks)
    if not gt_peaks:
        return 0, len(pred_peaks), 0
    used = [False] * len(gt_peaks)
    tp = 0
    for px, py in pred_peaks:
        best_d = float("inf")
        best_j = -1
        for j, (gx, gy) in enumerate(gt_peaks):
            if used[j]:
                continue
            d = np.hypot(px - gx, py - gy)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= radius:
            used[best_j] = True
            tp += 1
    return tp, len(pred_peaks) - tp, sum(1 for u in used if not u)


def main():
    weights = sys.argv[1]
    val_dir = "/workspace/field_keypoints/valid"
    print("Loading", weights)
    device = torch.device("cuda")
    model = HRNetKeypointModel()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    with open(os.path.join(val_dir, "annotations.json")) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    n_images = len(coco["annotations"])
    print("Running on {} val images...".format(n_images))

    cache = []
    with torch.no_grad():
        for ann in coco["annotations"]:
            img_info = images_by_id[ann["image_id"]]
            frame = cv2.imread(os.path.join(val_dir, "images", img_info["file_name"]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            tensor = preprocess(frame).to(device)
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits[0]).cpu().numpy()
            cache.append((heatmaps, gt_points(ann, w, h)))

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    print()
    header = "{:>7}  {:>5} {:>5} {:>5}  {:>6} {:>6} {:>6}  |  {:>6} {:>6}  {:>6} {:>6}".format(
        "thresh", "tp", "fp", "fn", "prec", "rec", "F1",
        "sideP", "sideR", "hashP", "hashR",
    )
    print(header)
    results = []
    for thresh in thresholds:
        ttp = tfp = tfn = 0
        per_ch = {0: [0, 0, 0], 1: [0, 0, 0]}
        for heatmaps, gt in cache:
            for ch in [0, 1]:
                peaks = extract_peaks(heatmaps[ch], thresh)
                tp, fp, fn = match(peaks, gt[ch], MATCH_RADIUS)
                ttp += tp
                tfp += fp
                tfn += fn
                per_ch[ch][0] += tp
                per_ch[ch][1] += fp
                per_ch[ch][2] += fn
        prec = ttp / max(ttp + tfp, 1)
        rec = ttp / max(ttp + tfn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        sP = per_ch[0][0] / max(per_ch[0][0] + per_ch[0][1], 1)
        sR = per_ch[0][0] / max(per_ch[0][0] + per_ch[0][2], 1)
        hP = per_ch[1][0] / max(per_ch[1][0] + per_ch[1][1], 1)
        hR = per_ch[1][0] / max(per_ch[1][0] + per_ch[1][2], 1)
        results.append((thresh, prec, rec, f1))
        row = "{:>7.2f}  {:>5} {:>5} {:>5}  {:>6.3f} {:>6.3f} {:>6.3f}  |  {:>6.3f} {:>6.3f}  {:>6.3f} {:>6.3f}".format(
            thresh, ttp, tfp, tfn, prec, rec, f1, sP, sR, hP, hR,
        )
        print(row)
    best = max(results, key=lambda r: r[3])
    print("\nBest F1 = {:.3f} at thresh={}".format(best[3], best[0]))


if __name__ == "__main__":
    main()
