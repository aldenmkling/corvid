"""Single 4-panel viz for play_065 (1920×1080).

Layout (2x2 grid, 960×540 per panel):
   TOP-LEFT  : source + classifier corrs
   TOP-RIGHT : tracker (boxes colored by team)
   BOT-LEFT  : rectified canvas (smoothed H)
   BOT-RIGHT : rectified canvas + tracking dots (team-colored, SG-smoothed)

Outputs:  output/play_065_visuals/4panel.mp4
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy.signal import savgol_filter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "training"))

from cc_tokenizer_v2 import (
    TYPE_NUM, TYPE_YARD, TYPE_SIDE, TYPE_HASH,
    SRC_W, SRC_H, null_classifier,
)
from cc_tokenizer_v3 import cc_tokens_from_frame_v3
from model_token_v10 import TokenClassifyV10
from model_token_v10b import TokenClassifyV10b
from train_rf_a import make_painted_logits_fn, encoder_features
from train_rf_b import RFB
from train_token_v10c_stage2 import (
    rfb_forward_with_features_and_row, PAINTED_TO_21,
)
from train_token_v6 import N_NGS_X_CLASSES

from src.homography.keypoints_from_tokens import extract_keypoints
from src.homography.h_tracker import (
    HomographyTrackerLite, smooth_hs, loo_filter_and_replace,
    detect_bad_runs,
)
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)
from src.detector import RFDETRDetector
from src.tracker import PlayerTracker
from src.team_classifier import select_long_tracks, classify_teams_color_pca

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANVAS_YDS_PER_PX = 1 / 8
CANVAS_W = int(FIELD_LENGTH / CANVAS_YDS_PER_PX)   # 960
CANVAS_H = int(FIELD_WIDTH / CANVAS_YDS_PER_PX)    # 426

PANEL_W, PANEL_H = 960, 540
OUT_W, OUT_H = 2 * PANEL_W, 2 * PANEL_H            # 1920×1080

RMSE_THR_YD = 0.30
LOO_THR_YD = 0.20
CARRY_STREAK_LOST = 3
BAD_RUN_MIN_LEN = 5
DOT_SG_WINDOW = 9
DOT_SG_POLY = 2

KIND_COLORS_BGR = {
    "near_hash":     (255, 200, 0),
    "far_hash":      (255, 200, 0),
    "sideline_near": (0, 165, 255),
    "sideline_far":  (0, 165, 255),
    "number_near":   (0, 255, 255),
    "number_far":    (0, 255, 255),
}
TEAM_COLORS_BGR = {
    "team_A": (60, 60, 235),
    "team_B": (235, 200, 60),
    "unknown": (140, 140, 140),
}


def load_unet(path, device):
    m = smp.Unet("mit_b0", encoder_weights=None, in_channels=3, classes=4)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m.to(device).eval()


def pre(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UW, UH))
    x = (rgb.astype(np.float32) / 255.0 - IMM) / IMS
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def pred_masks(unet, frame, device):
    x = pre(frame).to(device)
    p = torch.sigmoid(unet(x))[0].cpu().numpy()
    h0, w0 = frame.shape[:2]
    out = np.zeros((h0, w0, 4), dtype=np.float32)
    for ci in range(4):
        out[..., ci] = cv2.resize(p[ci], (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


class NewPipeline:
    def __init__(self, device):
        self.device = device
        self.unet = load_unet(os.path.join(
            PROJECT_ROOT, "models/unet_unified_v8_yardside_recover/best.pth"),
            device)
        s1 = torch.load(os.path.join(
            PROJECT_ROOT, "models/token_only_v10_phase1_pseudo/best.pth"),
            map_location="cpu", weights_only=False)
        sa = s1["args"]; self.sa = sa
        self.encoder = TokenClassifyV10(
            n_layers=sa["n_layers"], n_heads=sa["n_heads"],
            d_model=sa["d_model"], ffn_dim=sa["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.encoder.load_state_dict(s1["model_state_dict"])
        v10c_ck = torch.load(os.path.join(
            PROJECT_ROOT, "models/v10c_phase3_pseudo/best.pth"),
            map_location="cpu", weights_only=False)
        self.v10c = TokenClassifyV10b(
            n_layers=v10c_ck["args"]["n_layers"], n_heads=v10c_ck["args"]["n_heads"],
            d_model=v10c_ck["args"]["d_model"], ffn_dim=v10c_ck["args"]["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.v10c.load_state_dict(v10c_ck["model_state_dict"])
        rfb_ck = torch.load(os.path.join(
            PROJECT_ROOT, "models/rf_b_phase2_pseudo/best.pth"),
            map_location="cpu", weights_only=False)
        ra = rfb_ck["args"]
        self.rfb = RFB(d_enc=sa["d_model"], d_model=ra["d_model"],
                       n_heads=ra["n_heads"], ffn_dim=ra["ffn_dim"],
                       dropout=0.0, with_row=True).to(device).eval()
        self.rfb.load_state_dict(rfb_ck["model_state_dict"])
        self.crop_fn = make_painted_logits_fn(
            os.path.join(PROJECT_ROOT, "models/dsresnet10ww_round3_128x32/best.pth"),
            "dsresnet10ww", device)

    def __call__(self, frame_bgr, K, dist):
        masks_d = pred_masks(self.unet, frame_bgr, self.device)
        masks = cv2.undistort(masks_d.astype(np.float32), K, dist)
        tokens_np, aux = cc_tokens_from_frame_v3(masks, null_classifier, return_aux=True)
        if tokens_np.shape[0] == 0: return None
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = (type_idx == TYPE_NUM); is_yard = (type_idx == TYPE_YARD)
        is_side = (type_idx == TYPE_SIDE); is_hash = (type_idx == TYPE_HASH)

        toks_t = torch.from_numpy(tokens_np).unsqueeze(0).to(self.device)
        pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            enc_feat = encoder_features(self.encoder, toks_t, pad)[0]

        nac = np.full(tokens_np.shape[0], -1, dtype=np.int64)
        nar = np.zeros(tokens_np.shape[0], dtype=np.float32)
        rfb_pre_full = torch.zeros(1, tokens_np.shape[0], self.sa["d_model"],
                                    device=self.device)
        if is_num.any() and aux["num_crops"]:
            ni = np.where(is_num)[0]
            cl = torch.from_numpy(self.crop_fn(aux["num_crops"])).float().to(self.device)
            pad_n = torch.zeros(1, len(ni), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                rl, rr, rpp = rfb_forward_with_features_and_row(
                    self.rfb, enc_feat[ni].unsqueeze(0), cl.unsqueeze(0), pad_n)
            pp = rl[0].argmax(-1).cpu().numpy()
            p21 = PAINTED_TO_21.numpy()[pp]
            prw = torch.sigmoid(rr[0]).cpu().numpy()
            for j, ti in enumerate(ni):
                nac[ti] = int(p21[j]); nar[ti] = float(prw[j])
                rfb_pre_full[0, ti] = rpp[0, j]

        nca_t = torch.from_numpy(nac).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.v10c(toks_t, pad, num_class_gt=nca_t,
                            num_rfb_features=rfb_pre_full)
            p2 = out["logits_pass2"][0]
        ngs_cls = p2[:, :N_NGS_X_CLASSES].argmax(-1).cpu().numpy()
        rows = (p2[:, N_NGS_X_CLASSES] > 0).cpu().numpy().astype(int)

        kw = dict(pixel_sets=aux["pixel_sets"],
                  yard_classes=ngs_cls[is_yard].tolist(),
                  side_rows=rows[is_side].tolist(),
                  hash_classes=ngs_cls[is_hash].tolist(),
                  hash_rows=rows[is_hash].tolist(),
                  num_classes=nac[is_num].tolist(),
                  num_rows=(nar[is_num] > 0.5).astype(int).tolist())
        corrs, _ = extract_keypoints(yard_mask=masks[..., 0], **kw)
        return {"corrs": corrs}


def field_to_canvas(field_xy):
    x_yd, y_yd = field_xy[..., 0], field_xy[..., 1]
    px = x_yd / CANVAS_YDS_PER_PX
    py = CANVAS_H - (y_yd / CANVAS_YDS_PER_PX)
    return np.stack([px, py], axis=-1)


def dot_pixel_from_xyxy(xyxy):
    x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return np.array([0.5 * (x0 + x1), y0 + 0.95 * (y1 - y0)], dtype=np.float64)


def project_via_H(pts_pixel, H):
    if len(pts_pixel) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homo = np.column_stack([pts_pixel, np.ones(len(pts_pixel))])
    proj = (H @ homo.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return proj[:, :2] / w


def render_rectified(frame_bgr, H):
    A = np.array([
        [1.0 / CANVAS_YDS_PER_PX, 0.0, 0.0],
        [0.0, -1.0 / CANVAS_YDS_PER_PX, CANVAS_H],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    warped = cv2.warpPerspective(frame_bgr, A @ H, (CANVAS_W, CANVAS_H))
    grid = warped.copy()
    for y in (0.0, FIELD_WIDTH):
        p0 = field_to_canvas(np.array([0.0, y]))
        p1 = field_to_canvas(np.array([FIELD_LENGTH, y]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for x in range(10, 115, 5):
        p0 = field_to_canvas(np.array([x, 0.0]))
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH]))
        color = (0, 255, 0) if x % 10 == 0 else (0, 180, 0)
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 color, 1, cv2.LINE_AA)
    for x in (0.0, FIELD_LENGTH):
        p0 = field_to_canvas(np.array([x, 0.0]))
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (0, 255, 255), 1, cv2.LINE_AA)
    for y in (HASH_Y_NEAR, HASH_Y_FAR):
        p0 = field_to_canvas(np.array([10.0, y]))
        p1 = field_to_canvas(np.array([110.0, y]))
        cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)),
                 (255, 255, 0), 1, cv2.LINE_AA)
    return cv2.addWeighted(warped, 0.7, grid, 0.3, 0)


def render_field_background():
    """Clean field for the dots panel (no warped source)."""
    img = np.full((CANVAS_H, CANVAS_W, 3), (32, 92, 32), dtype=np.uint8)
    for x_lo, x_hi in [(0.0, 10.0), (110.0, FIELD_LENGTH)]:
        p0 = field_to_canvas(np.array([x_lo, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x_hi, FIELD_WIDTH])).astype(int)
        cv2.rectangle(img, (p0[0], p1[1]), (p1[0], p0[1]), (24, 70, 24), -1)
    for y in (0.0, FIELD_WIDTH):
        p0 = field_to_canvas(np.array([0.0, y])).astype(int)
        p1 = field_to_canvas(np.array([FIELD_LENGTH, y])).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (255, 255, 255), 2, cv2.LINE_AA)
    for x in range(10, 115, 5):
        p0 = field_to_canvas(np.array([x, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH])).astype(int)
        thick = 2 if x % 10 == 0 else 1
        cv2.line(img, tuple(p0), tuple(p1), (240, 240, 240), thick, cv2.LINE_AA)
    for x in (10.0, 110.0):
        p0 = field_to_canvas(np.array([x, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH])).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (0, 220, 220), 2, cv2.LINE_AA)
    for y in (HASH_Y_NEAR, HASH_Y_FAR):
        for x in range(11, 110):
            p = field_to_canvas(np.array([x, y])).astype(int)
            cv2.line(img, (p[0], p[1] - 4), (p[0], p[1] + 4),
                     (255, 255, 255), 1, cv2.LINE_AA)
    for x in range(20, 105, 10):
        yard_num = min(x - 10, 110 - x)
        label = str(yard_num)
        for ngs_y in (4.5, FIELD_WIDTH - 4.5):
            p = field_to_canvas(np.array([x, ngs_y])).astype(int)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            org = (int(p[0] - tw // 2), int(p[1] + th // 2))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_DUPLEX,
                       0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_DUPLEX,
                       0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def fit_panel(img, label, bad=False):
    """Resize img to PANEL_W × PANEL_H, label across the top, red border if bad."""
    h, w = img.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    off_x = (PANEL_W - new_w) // 2
    off_y = (PANEL_H - new_h) // 2
    panel[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    # Label top-left.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.rectangle(panel, (0, 0), (tw + 24, th + 18), (0, 0, 0), -1)
    cv2.putText(panel, label, (12, th + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if bad:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1),
                      (0, 0, 255), 6)
    return panel


def draw_bad_run_strip(composite):
    """Single top-strip across the whole 1920-wide composite when bad."""
    h, w = composite.shape[:2]
    msg = "BAD RUN  —  H bridged by polynomial through clean neighbors"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    bar_h = th + 24
    overlay = composite.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, bar_h), (0, 0, 0), -1)
    composite[:bar_h] = cv2.addWeighted(composite[:bar_h], 0.4,
                                          overlay[:bar_h], 0.6, 0)
    cv2.putText(composite, msg, ((w - tw) // 2, th + 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-meas-frac", type=float, default=0.5)
    ap.add_argument("--clip", default="videos/clips/2019092204/play_065/sideline.mp4",
                    help="Path to sideline.mp4, relative to project root.")
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults to "
                         "output/4panel_viz/<game>_<play>.mp4.")
    args = ap.parse_args()

    clip = (args.clip if os.path.isabs(args.clip)
            else os.path.join(PROJECT_ROOT, args.clip))
    rel_clip = os.path.relpath(clip, os.path.join(PROJECT_ROOT, "videos/clips"))
    parts = rel_clip.split(os.sep)
    tag = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "clip"
    if args.out is None:
        out_dir = os.path.join(PROJECT_ROOT, "output", "4panel_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path_final = os.path.join(out_dir, f"{tag}.mp4")
    else:
        out_path_final = (args.out if os.path.isabs(args.out)
                          else os.path.join(PROJECT_ROOT, args.out))
        os.makedirs(os.path.dirname(out_path_final) or ".", exist_ok=True)
    device = torch.device(args.device)

    manifest = json.load(open(os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json")))
    rel = rel_clip
    intr = manifest["intrinsics_by_clip"].get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)

    print("Loading classifier pipeline ...")
    pipe = NewPipeline(device)
    print("Loading detector + tracker ...")
    detector = RFDETRDetector(
        weights=os.path.join(PROJECT_ROOT, "models/rfdetr_best_ema.pth"),
        device=args.device, conf_thresh=0.3)
    tracker = PlayerTracker(device=args.device, frame_rate=30)
    h_tracker = HomographyTrackerLite()

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {n_total} frames @ {fps:.1f} fps")

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    frames_u = []
    per_frame = []
    track_results = []
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != SRC_W: fr = cv2.resize(fr, (SRC_W, SRC_H))
        fr_u = cv2.undistort(fr, K, dist)
        frames_u.append(fr_u)

        pp = pipe(fr, K, dist)
        if pp is None:
            per_frame.append(None)
        else:
            corrs = pp["corrs"]
            res = h_tracker.update(corrs, frame_idx=fi)
            per_frame.append({"H": res["H"], "method": res["method"],
                              "corrs": corrs, "rmse": res["rmse_yd"]})

        dets = detector.detect(fr_u)
        H_for_tracker = per_frame[-1]["H"] if per_frame[-1] else None
        tr = tracker.update(dets, fr_u, H=H_for_tracker, K=K, dist=dist)
        track_results.append(tr)

        if (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{fi+1}/{n_total}] {dt:.0f}s ({(fi+1)/dt:.1f} fps)")
    cap.release()
    print(f"  pass 1 done in {time.time()-t0:.0f}s")

    # Cutoff.
    cutoff = len(per_frame)
    streak = 0; run_start = None
    for i, r in enumerate(per_frame):
        m = r["method"] if r else "none"
        if m == "carry":
            if streak == 0: run_start = i
            streak += 1
            if streak >= CARRY_STREAK_LOST:
                cutoff = run_start; break
        else:
            streak = 0; run_start = None
    print(f"  cutoff: {cutoff}/{len(per_frame)}")

    # LOO filter + replacement + SG smoothing.
    Hs = [r["H"] if (r and r["H"] is not None) else None
          for r in per_frame[:cutoff]]
    rmses = [r["rmse"] if (r and r.get("rmse") is not None) else None
             for r in per_frame[:cutoff]]
    Hs_clean, red_mask, _ = loo_filter_and_replace(
        Hs, rmses=rmses,
        thr_loo_yd=LOO_THR_YD, thr_rmse_yd=RMSE_THR_YD)
    n_red = sum(red_mask)
    print(f"  LOO+rmse red flags: {n_red}/{cutoff}")
    bad_runs = detect_bad_runs(red_mask, min_length=BAD_RUN_MIN_LEN)
    in_bad_run = [False] * cutoff
    for s, e in bad_runs:
        for k in range(s, e): in_bad_run[k] = True
    if bad_runs:
        print(f"  BAD RUNS (≥{BAD_RUN_MIN_LEN} consec red): {bad_runs}")

    for i in range(cutoff):
        if per_frame[i] is not None and Hs_clean[i] is not None:
            per_frame[i]["H"] = Hs_clean[i]
    valid_idx = [i for i in range(cutoff) if Hs_clean[i] is not None]
    if len(valid_idx) >= 7:
        sm = smooth_hs([Hs_clean[i] for i in valid_idx], window=7, poly=2)
        for vi, si in enumerate(valid_idx):
            per_frame[si]["H"] = sm[vi]
        print(f"  Sav-Gol smoothed (w=7, p=2) over {len(valid_idx)} frames")

    # Team classification.
    trajectories = tracker.trajectories
    long_track_ids = select_long_tracks(
        trajectories, min_meas_frac=args.min_meas_frac, n_valid_frames=cutoff)
    print(f"  long tracks: {len(long_track_ids)}/{len(trajectories)}")
    print("  classifying teams (color PCA + median split) ...")
    team_labels, _ = classify_teams_color_pca(
        trajectories, clip, n_samples_per_track=12,
        long_track_ids=long_track_ids)
    n_a = sum(1 for v in team_labels.values() if v == "team_A")
    n_b = sum(1 for v in team_labels.values() if v == "team_B")
    print(f"  team_A: {n_a}  team_B: {n_b}")

    # Per-track field-coord dot series + SG smoothing.
    n_render = cutoff
    dot_field = {tid: np.full((n_render, 2), np.nan, dtype=np.float64)
                 for tid in long_track_ids}
    for fi in range(n_render):
        rec = per_frame[fi]
        if rec is None or rec["H"] is None: continue
        H = rec["H"]
        for p in track_results[fi].players:
            if p.track_id not in long_track_ids: continue
            dp = dot_pixel_from_xyxy(p.xyxy)
            fxy = project_via_H(dp[None], H)[0]
            if np.isfinite(fxy).all():
                dot_field[p.track_id][fi] = fxy

    n_smoothed = 0
    for tid, arr in dot_field.items():
        valid = ~np.isnan(arr[:, 0])
        if not valid.any(): continue
        segs = []
        in_seg = False; start = 0
        for i in range(n_render):
            if valid[i] and not in_seg: start = i; in_seg = True
            elif not valid[i] and in_seg: segs.append((start, i)); in_seg = False
        if in_seg: segs.append((start, n_render))
        for (s, e) in segs:
            L = e - s
            if L < DOT_SG_WINDOW: continue
            w = DOT_SG_WINDOW
            if w > L: w = L if L % 2 == 1 else L - 1
            if w < DOT_SG_POLY + 2: continue
            arr[s:e, 0] = savgol_filter(arr[s:e, 0], w, DOT_SG_POLY)
            arr[s:e, 1] = savgol_filter(arr[s:e, 1], w, DOT_SG_POLY)
            n_smoothed += 1
    print(f"  per-track SG-smoothed {n_smoothed} segments")

    # ── Pass 2: render single 4-panel composite ──────────────────────────────
    out_path = out_path_final
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))
    field_bg = render_field_background()

    print(f"[pass 2/2] rendering {n_render} frames -> {out_path}")
    t0 = time.time()
    for fi in range(n_render):
        fr_u = frames_u[fi]
        rec = per_frame[fi]
        bad = in_bad_run[fi]

        # TL: source + corrs.
        tl = fr_u.copy()
        if rec is not None and rec["corrs"]:
            for c in rec["corrs"]:
                x, y = int(c["pixel_u"][0]), int(c["pixel_u"][1])
                col = KIND_COLORS_BGR.get(c["kind"], (0, 255, 0))
                cv2.circle(tl, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(tl, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)

        # TR: tracker boxes, team-colored.
        tr_img = fr_u.copy()
        for p in track_results[fi].players:
            tid = p.track_id
            if tid not in long_track_ids: continue
            lab = team_labels.get(tid, "unknown")
            col = TEAM_COLORS_BGR[lab]
            x0, y0, x1, y1 = (int(v) for v in p.xyxy)
            cv2.rectangle(tr_img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
            tag = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(tr_img, (x0, y0 - th - 8), (x0 + tw + 8, y0),
                          col, -1)
            cv2.putText(tr_img, tag, (x0 + 4, y0 - 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # BL: rectified (smoothed H).
        if rec is None or rec["H"] is None:
            bl = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        else:
            bl = render_rectified(fr_u, rec["H"])

        # BR: rectified field with dots only.
        br = field_bg.copy()
        for tid in long_track_ids:
            fxy = dot_field[tid][fi]
            if not np.isfinite(fxy).all(): continue
            cxy = field_to_canvas(fxy)
            cx, cy = int(round(cxy[0])), int(round(cxy[1]))
            if not (0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H): continue
            lab = team_labels.get(tid, "unknown")
            col = TEAM_COLORS_BGR[lab]
            cv2.circle(br, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(br, (cx, cy), 7, col, -1, cv2.LINE_AA)

        p_tl = fit_panel(tl,    "SOURCE + CORRS", bad=bad)
        p_tr = fit_panel(tr_img, "TRACKER (team-colored)", bad=bad)
        p_bl = fit_panel(bl,    "RECTIFIED", bad=bad)
        p_br = fit_panel(br,    "TRACKING DOTS", bad=bad)

        comp = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        comp[:PANEL_H, :PANEL_W] = p_tl
        comp[:PANEL_H, PANEL_W:] = p_tr
        comp[PANEL_H:, :PANEL_W] = p_bl
        comp[PANEL_H:, PANEL_W:] = p_br

        if bad: draw_bad_run_strip(comp)
        writer.write(comp)

    writer.release()
    print(f"  pass 2 done in {time.time()-t0:.0f}s")
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
