"""Spot-check the trained number classifier on holdout 2024122201 (SoFi).

For a few clips, run rectify pass-1 to detect painted-number groups, then
classify each with the trained mit_b0 classifier. Visualize: source frame
with each group's bbox + predicted label + confidence overlaid. We don't
know ground-truth labels (no g0 for these holdout clips), but we can
sanity-check by eye whether the predictions look right.
"""
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/testing"))

from src.homography import painted_numbers
from src.homography.distortion import CameraIntrinsics
from src.homography.grid_solver_v2 import group_yardline_pixels_cc
from rebuild_full_clip_viz import (
    YardlineTracker, group_sideline_pixels_cc as group_sideline_pixels,
)
from rebuild_step4_hashes_v2 import total_mse
from rectify_step2_per_frame import (
    run_specialists, LINE_WEIGHTS, HASH_WEIGHTS, NUMBER_WEIGHTS,
    fit_yardline_undistorted, fit_sideline_undistorted, detect_hash_rows,
    HashRowTracker, NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL, YD_PER_GRID,
)

CLASSIFIER_WEIGHTS = os.path.join(PROJECT_ROOT, "models/number_classifier_best.pth")
INPUT_SIZE = 64
PIXEL_MEAN = 0.456
PIXEL_STD = 0.224

SAMPLES = [
    ("2024122201", "play_001", 0),
    ("2024122201", "play_020", 60),
    ("2024122201", "play_050", 50),
    ("2024122201", "play_100", 90),
    ("2024122201", "play_120", 100),
    ("2024122201", "play_155", 60),
]
OUT_DIR = os.path.join(PROJECT_ROOT, "output/number_classifier_holdout")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Same architecture as train_number_classifier.MitClassifier
class MitClassifier(nn.Module):
    def __init__(self, encoder_name="mit_b0", num_classes=9, in_channels=1):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=None)
        feat_dim = self.encoder.out_channels[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[-1])


print(f"loading classifier {CLASSIFIER_WEIGHTS} on {device}")
ckpt = torch.load(CLASSIFIER_WEIGHTS, map_location=device, weights_only=False)
classifier = MitClassifier()
classifier.load_state_dict(ckpt["model_state_dict"])
classifier.to(device).eval()
CLASSES = ckpt.get("classes",
                     ["10L", "10R", "20L", "20R", "30L", "30R", "40L", "40R", "50"])
print(f"classes: {CLASSES}")


def crop_group_to_64(group, image_h, image_w, margin_px=5, size=INPUT_SIZE):
    xs = group["xs_all"]; ys = group["ys_all"]
    if len(xs) == 0:
        return None, None
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[ys, xs] = 255
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(image_w, int(xs.max()) + margin_px + 1)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(image_h, int(ys.max()) + margin_px + 1)
    crop = mask[y0:y1, x0:x1]
    h_c, w_c = crop.shape
    if h_c > w_c:
        pad = h_c - w_c
        crop = np.pad(crop, ((0, 0), (pad // 2, pad - pad // 2)), mode='constant')
    elif w_c > h_c:
        pad = w_c - h_c
        crop = np.pad(crop, ((pad // 2, pad - pad // 2), (0, 0)), mode='constant')
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, x1, y1)


@torch.no_grad()
def classify_batch(crops):
    if not crops:
        return [], []
    arr = np.stack(crops, axis=0).astype(np.float32) / 255.0
    arr = (arr - PIXEL_MEAN) / PIXEL_STD
    x = torch.from_numpy(arr).unsqueeze(1).to(device)
    logits = classifier(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    confs = probs[np.arange(len(pred_idx)), pred_idx]
    return [CLASSES[i] for i in pred_idx], confs.tolist()


def bootstrap(clip_path, device_str="mps"):
    cap = cv2.VideoCapture(clip_path)
    ok, frame0 = cap.read()
    if not ok:
        cap.release(); return None
    h, w = frame0.shape[:2]
    focal = float(max(h, w)); cx, cy = w / 2.0, h / 2.0
    yard0, side0, _ = run_specialists(frame0, LINE_WEIGHTS, HASH_WEIGHTS, device_str)
    yl0 = group_yardline_pixels_cc(yard0)
    sl0 = group_sideline_pixels(side0)
    line_pts = [g.pixels for g in yl0] + [g.pixels for g in sl0]
    line_kinds = ["yardline"] * len(yl0) + ["sideline"] * len(sl0)
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    yl_tracker = YardlineTracker(g_min=-30, g_max=30, frame_h=h)
    fits_yl0 = [fit_yardline_undistorted(g.pixels, intr) for g in yl0]
    yl_tracker.init_from(fits_yl0, cy)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return {"cap": cap, "h": h, "w": w, "K": K, "dist": dist, "k1": k1,
            "intr": intr, "yl_tracker": yl_tracker,
            "hash_tracker": HashRowTracker(image_w=w),
            "number_tracker": painted_numbers.NumberSideTracker()}


def step(state, frame, device_str):
    intr = state["intr"]; w = state["w"]; h = state["h"]
    K = state["K"]; dist = state["dist"]; k1 = state["k1"]
    yard, side, hash_ = run_specialists(frame, LINE_WEIGHTS, HASH_WEIGHTS, device_str)
    yl = group_yardline_pixels_cc(yard)
    sl = group_sideline_pixels(side)
    fits_yl = [fit_yardline_undistorted(g.pixels, intr) for g in yl]
    fits_sl = [fit_sideline_undistorted(g.pixels, intr) for g in sl]
    if state["yl_tracker"].last_fit:
        fits_kept, g_index, _, _ = state["yl_tracker"].update(fits_yl, h / 2.0)
    else:
        init = state["yl_tracker"].init_from(fits_yl, h / 2.0)
        if init is None:
            fits_kept, g_index = [], np.array([], dtype=int)
        else:
            fits_kept, g_index, _ = init
    rows_raw = detect_hash_rows(hash_, intr)
    rows = state["hash_tracker"].observe(rows_raw)
    num_mask_d = painted_numbers.predict_mask(frame, NUMBER_WEIGHTS, device_str)
    if abs(k1) > 1e-6:
        num_mask_u = cv2.undistort(num_mask_d, K, dist)
        num_mask_u = (num_mask_u > 127).astype(np.uint8) * 255
    else:
        num_mask_u = num_mask_d
    _, num_dbg = painted_numbers.process_frame(
        num_mask_u, fits_kept, rows, fits_sl, g_index, h, w,
        state["number_tracker"])
    return g_index, num_dbg["groups"], num_dbg["cc_pixels"]


for game, play, fi in SAMPLES:
    clip_path = os.path.join(PROJECT_ROOT, f"videos/clips/{game}/{play}/sideline.mp4")
    if not os.path.exists(clip_path):
        print(f"  [skip] missing {clip_path}"); continue
    state = bootstrap(clip_path, str(device))
    if state is None:
        print(f"  [skip] bootstrap failed"); continue
    cap = state["cap"]
    if fi >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        print(f"  [skip] fi {fi} >= clip length"); cap.release(); continue
    g_index, groups, _ = None, None, None
    for f_cur in range(fi + 1):
        ok, frame = cap.read()
        if not ok: break
        g_index, groups, _ = step(state, frame, str(device))
    cap.release()
    if not groups:
        print(f"  [no groups] {game}/{play} f{fi}"); continue
    h, w = state["h"], state["w"]

    # Crop + classify each group
    crops, group_refs = [], []
    for grp in groups:
        if grp.get("yardline_idx", -1) < 0: continue
        side = grp.get("side")
        if side not in ("near", "far"): continue
        crop, bbox = crop_group_to_64(grp, h, w)
        if crop is None: continue
        crops.append(crop)
        group_refs.append((grp, bbox, side))
    labels, confs = classify_batch(crops)

    # Visualize on undistorted frame
    K, dist, k1 = state["K"], state["dist"], state["k1"]
    frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
    vis = (frame_u * 0.6).astype(np.uint8)
    for (grp, bbox, side), lbl, conf in zip(group_refs, labels, confs):
        x0, y0, x1, y1 = bbox
        col = (0, 220, 220) if conf > 0.8 else (60, 60, 220)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        text = f"{lbl} {conf:.2f}"
        cv2.putText(vis, text, (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.putText(vis, f"{game}/{play} f{fi}  groups={len(group_refs)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    out = os.path.join(OUT_DIR, f"{game}_{play}_f{fi:04d}.png")
    cv2.imwrite(out, vis)
    n_high = sum(1 for c in confs if c > 0.8)
    print(f"  {game}/{play} f{fi}: groups={len(group_refs)}  "
          f"high-conf={n_high}  preds={[(l, f'{c:.2f}') for l, c in zip(labels, confs)]}")
print(f"\noutputs: {OUT_DIR}")
