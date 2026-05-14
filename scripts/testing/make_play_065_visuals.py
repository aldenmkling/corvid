"""Render 3 separate presentation visuals for play_065 (2019092204):

  1. source_with_corrs.mp4  — undistorted source frame with corrs dotted on,
                              color-coded by keypoint kind
  2. rectified.mp4          — rectified field canvas only
  3. tracker.mp4            — undistorted source with per-player bounding
                              boxes + track IDs

Uses the new phase-1/2/3 classifier with bank + bandaids + red-frame
neighbor-average replacement + Savitzky-Golay smoothing + carry cutoff
(matches viz_new_classifier_clip.py logic).
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "training"))

from cc_tokenizer_v2 import (    # noqa: E402
    TYPE_NUM, TYPE_YARD, TYPE_SIDE, TYPE_HASH,
    SRC_W, SRC_H, null_classifier,
)
from cc_tokenizer_v3 import cc_tokens_from_frame_v3   # noqa: E402
from model_token_v10 import TokenClassifyV10           # noqa: E402
from model_token_v10b import TokenClassifyV10b         # noqa: E402
from train_rf_a import make_painted_logits_fn, encoder_features  # noqa: E402
from train_rf_b import RFB                              # noqa: E402
from train_token_v10c_stage2 import (
    rfb_forward_with_features_and_row, PAINTED_TO_21,
)
from train_token_v6 import N_NGS_X_CLASSES              # noqa: E402

import src.homography.keypoints_from_tokens as kft     # noqa: E402
from src.homography.keypoints_from_tokens import extract_keypoints  # noqa: E402
from src.homography.h_tracker import (   # noqa: E402
    solve_h, HomographyTrackerLite, smooth_hs, loo_filter_and_replace,
    detect_bad_runs,
)
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)
from src.detector import RFDETRDetector
from src.tracker import PlayerTracker
from src.team_classifier import (
    select_long_tracks, classify_teams_color_pca,
)

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANVAS_YDS_PER_PX = 1 / 8
CANVAS_W = int(FIELD_LENGTH / CANVAS_YDS_PER_PX)
CANVAS_H = int(FIELD_WIDTH / CANVAS_YDS_PER_PX)

RMSE_THR_YD = 0.30
LOO_THR_YD = 0.20
CARRY_STREAK_LOST = 3
BAD_RUN_MIN_LEN = 5   # consecutive red frames → "BAD RUN" overlay


def draw_bad_run_banner(img):
    """Red border + 'BAD RUN — H bridged' banner at top of the frame."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    msg = "BAD RUN  (H bridged across this stretch)"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    cv2.rectangle(img, (15, 15), (15 + tw + 20, 15 + th + 18),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (15, 15), (15 + tw + 20, 15 + th + 18),
                  (0, 0, 255), 2)
    cv2.putText(img, msg, (25, 15 + th + 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

TEAM_COLORS_BGR = {
    "team_A": (60, 60, 235),     # red
    "team_B": (235, 200, 60),    # cyan-blue
    "unknown": (140, 140, 140),
}

# Color-coding for correspondence kinds.
KIND_COLORS_BGR = {
    # hash crossings → cyan
    "near_hash":     (255, 200, 0),
    "far_hash":      (255, 200, 0),
    # yardline×sideline crossings → orange
    "sideline_near": (0, 165, 255),
    "sideline_far":  (0, 165, 255),
    # number tangent crossings → yellow
    "number_near":   (0, 255, 255),
    "number_far":    (0, 255, 255),
}

# Track ID color palette (distinct hues).
TRACK_COLORS = [
    (255,   0,   0), (  0, 255,   0), (  0, 128, 255), (255, 255,   0),
    (255,   0, 255), (  0, 255, 255), (128,   0, 255), (255, 128,   0),
    (  0, 255, 128), (128, 255,   0), (255,   0, 128), (  0, 128, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (192, 192, 192),
    (255, 165,   0), (  0, 100, 255), (200, 255,   0), ( 75,   0, 130),
    (255,  20, 147), (  0, 191, 255), (244, 164,  96),
]


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
        sa = s1["args"]
        self.sa = sa
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
        if tokens_np.shape[0] == 0:
            return None
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = (type_idx == TYPE_NUM)
        is_yard = (type_idx == TYPE_YARD)
        is_side = (type_idx == TYPE_SIDE)
        is_hash = (type_idx == TYPE_HASH)

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

        # Bandaids ON.
        corrs, fits = extract_keypoints(yard_mask=masks[..., 0], **kw)
        return {"corrs": corrs, "n_yard_fits": len(fits["yard_fits"])}


def field_to_canvas(field_xy):
    x_yd, y_yd = field_xy[..., 0], field_xy[..., 1]
    px = x_yd / CANVAS_YDS_PER_PX
    py = CANVAS_H - (y_yd / CANVAS_YDS_PER_PX)
    return np.stack([px, py], axis=-1)


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
    for x in range(20, 105, 10):
        yard_num = min(x - 10, 110 - x)
        for ngs_y in (4.5, FIELD_WIDTH - 4.5):
            p = field_to_canvas(np.array([x, ngs_y])).astype(int)
            label = str(yard_num)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            org = (int(p[0] - tw // 2), int(p[1] + th // 2))
            cv2.putText(grid, label, org, cv2.FONT_HERSHEY_DUPLEX,
                       0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(grid, label, org, cv2.FONT_HERSHEY_DUPLEX,
                       0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.addWeighted(warped, 0.7, grid, 0.3, 0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    clip = os.path.join(PROJECT_ROOT,
                         "videos/clips/2019092204/play_065/sideline.mp4")
    out_dir = os.path.join(PROJECT_ROOT, "output", "play_065_visuals")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    manifest = json.load(open(os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json")))
    rel = os.path.relpath(clip, os.path.join(PROJECT_ROOT, "videos/clips"))
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

    # ── Pass 1: classifier H + detect + track per frame ──────────────────────
    frames_u = []           # undistorted source per frame
    per_frame = []          # dicts of classifier outputs
    track_results = []      # list of TrackingResult per frame
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != SRC_W: fr = cv2.resize(fr, (SRC_W, SRC_H))
        fr_u = cv2.undistort(fr, K, dist)
        frames_u.append(fr_u)

        # Classifier pipeline → corrs.
        pp = pipe(fr, K, dist)
        if pp is None:
            per_frame.append(None)
        else:
            corrs = pp["corrs"]
            res = h_tracker.update(corrs, frame_idx=fi)
            per_frame.append({
                "H": res["H"], "method": res["method"], "corrs": corrs,
                "rmse": res["rmse_yd"],
            })

        # Detector + tracker.
        detections = detector.detect(fr_u)
        H_for_tracker = per_frame[-1]["H"] if per_frame[-1] else None
        tr = tracker.update(detections, fr_u, H=H_for_tracker, K=K, dist=dist)
        track_results.append(tr)

        if (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{fi+1}/{n_total}] {dt:.0f}s ({(fi+1)/dt:.1f} fps)")
    cap.release()
    print(f"  pass 1 done in {time.time()-t0:.0f}s")

    # ── Cutoff: ≥3 consecutive carries → clip lost.
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

    # ── LOO red flag + replacement + SG smoothing on H trajectory ───────────
    Hs = [r["H"] if (r and r["H"] is not None) else None
          for r in per_frame[:cutoff]]
    rmses = [r["rmse"] if (r and r.get("rmse") is not None) else None
             for r in per_frame[:cutoff]]
    Hs_clean, red_mask, loo_resids = loo_filter_and_replace(
        Hs, rmses=rmses,
        thr_loo_yd=LOO_THR_YD, thr_rmse_yd=RMSE_THR_YD)
    n_red = sum(red_mask)
    print(f"  LOO+rmse red flags: {n_red}/{cutoff}  "
          f"(loo>{LOO_THR_YD} or rmse>{RMSE_THR_YD})")
    bad_runs = detect_bad_runs(red_mask, min_length=BAD_RUN_MIN_LEN)
    in_bad_run = [False] * cutoff
    for s, e in bad_runs:
        for k in range(s, e): in_bad_run[k] = True
    if bad_runs:
        print(f"  BAD RUNS (≥{BAD_RUN_MIN_LEN} consec red): "
              f"{bad_runs}  → bridged but flagged in viz")
    for i in range(cutoff):
        if per_frame[i] is not None and Hs_clean[i] is not None:
            per_frame[i]["H"] = Hs_clean[i]

    valid_idx = [i for i in range(cutoff) if Hs_clean[i] is not None]
    if len(valid_idx) >= 7:
        sm = smooth_hs([Hs_clean[i] for i in valid_idx], window=7, poly=2)
        for vi, si in enumerate(valid_idx):
            per_frame[si]["H"] = sm[vi]
        print(f"  Sav-Gol smoothed (w=7, p=2) over {len(valid_idx)} frames")

    # ── Team classify long tracks for tracker viz color ──────────────────────
    trajectories = tracker.trajectories
    long_track_ids = select_long_tracks(
        trajectories, min_meas_frac=0.5, n_valid_frames=cutoff)
    print(f"  long tracks: {len(long_track_ids)}/{len(trajectories)}")
    print("  classifying teams (color PCA + median split) ...")
    team_labels, _ = classify_teams_color_pca(
        trajectories, clip, n_samples_per_track=12,
        long_track_ids=long_track_ids)
    n_a = sum(1 for v in team_labels.values() if v == "team_A")
    n_b = sum(1 for v in team_labels.values() if v == "team_B")
    print(f"  team_A: {n_a}  team_B: {n_b}")

    # ── Pass 2: render 3 separate videos.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_render = min(cutoff, len(frames_u))

    out1 = os.path.join(out_dir, "source_with_corrs.mp4")
    out2 = os.path.join(out_dir, "rectified.mp4")
    out3 = os.path.join(out_dir, "tracker.mp4")

    rect_h = CANVAS_H
    rect_w = CANVAS_W
    writer1 = cv2.VideoWriter(out1, fourcc, fps, (SRC_W, SRC_H))
    writer2 = cv2.VideoWriter(out2, fourcc, fps, (rect_w, rect_h))
    writer3 = cv2.VideoWriter(out3, fourcc, fps, (SRC_W, SRC_H))

    print(f"[pass 2/2] rendering {n_render} frames to 3 videos ...")
    t0 = time.time()
    for fi in range(n_render):
        fr_u = frames_u[fi]
        rec = per_frame[fi]
        bad = fi < len(in_bad_run) and in_bad_run[fi]

        # ── Visual 1: source + color-coded corrs.
        img1 = fr_u.copy()
        if rec is not None and rec["corrs"]:
            for c in rec["corrs"]:
                x, y = int(c["pixel_u"][0]), int(c["pixel_u"][1])
                col = KIND_COLORS_BGR.get(c["kind"], (0, 255, 0))
                cv2.circle(img1, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(img1, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
        if bad: draw_bad_run_banner(img1)
        writer1.write(img1)

        # ── Visual 2: rectified canvas only.
        if rec is None or rec["H"] is None:
            img2 = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
        else:
            img2 = render_rectified(fr_u, rec["H"])
        if bad: draw_bad_run_banner(img2)
        writer2.write(img2)

        # ── Visual 3: source + tracker boxes colored by team.
        img3 = fr_u.copy()
        for p in track_results[fi].players:
            tid = p.track_id
            if tid not in long_track_ids:
                continue
            label = team_labels.get(tid, "unknown")
            col = TEAM_COLORS_BGR[label]
            x0, y0, x1, y1 = (int(v) for v in p.xyxy)
            cv2.rectangle(img3, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
            tag = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(img3, (x0, y0 - th - 8), (x0 + tw + 8, y0),
                          col, -1)
            cv2.putText(img3, tag, (x0 + 4, y0 - 4),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        if bad: draw_bad_run_banner(img3)
        writer3.write(img3)

    writer1.release(); writer2.release(); writer3.release()
    print(f"  pass 2 done in {time.time()-t0:.0f}s")
    print(f"\nOutputs:")
    print(f"  → {out1}")
    print(f"  → {out2}")
    print(f"  → {out3}")


if __name__ == "__main__":
    main()
