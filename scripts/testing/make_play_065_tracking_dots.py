"""Rectified field canvas with team-colored tracking dots only (play_065).

Pipeline:
  1. Run the phase 1/2/3 classifier per frame → corrs → H (with bandaids,
     bank, neighbor-average red replacement, Sav-Gol smoothing, carry cutoff).
  2. Run RF-DETR + custom Kalman tracker (same as make_play_065_visuals.py).
  3. After the pipeline: select_long_tracks → drop short tracks.
  4. classify_teams_team_colors over long tracks → team_A / team_B labels.
  5. Render: per frame, draw one dot per long-track tracked player on a
     rectified field canvas. Dot pixel = (x_center, y0 + 0.95 * (y1 - y0))
     — middle of bbox, 95% from top to bottom. Project through smoothed H.

Saves to output/play_065_visuals/tracking_dots.mp4.
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

from src.homography.keypoints_from_tokens import extract_keypoints  # noqa: E402
from src.homography.h_tracker import (  # noqa: E402
    HomographyTrackerLite, smooth_hs, loo_filter_and_replace,
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
BAD_RUN_MIN_LEN = 5


def draw_bad_run_banner(img):
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

# Per-track dot smoothing.
DOT_SG_WINDOW = 9   # ~0.3 sec @ 30 fps
DOT_SG_POLY = 2

# Team colors (BGR). Vivid + presentation-friendly.
TEAM_COLORS_BGR = {
    "team_A": (60, 60, 235),     # red
    "team_B": (235, 200, 60),    # cyan-blue
    "unknown": (140, 140, 140),  # grey
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

        corrs, fits = extract_keypoints(yard_mask=masks[..., 0], **kw)
        return {"corrs": corrs, "n_yard_fits": len(fits["yard_fits"])}


def field_to_canvas(field_xy):
    x_yd, y_yd = field_xy[..., 0], field_xy[..., 1]
    px = x_yd / CANVAS_YDS_PER_PX
    py = CANVAS_H - (y_yd / CANVAS_YDS_PER_PX)
    return np.stack([px, py], axis=-1)


def render_field_background():
    """Plain green field with white yardlines, hash marks, and numbers.

    No source frame warp — just a clean canvas to overlay dots on.
    """
    img = np.full((CANVAS_H, CANVAS_W, 3), (32, 92, 32), dtype=np.uint8)

    # Endzone fill (slightly darker).
    for x_lo, x_hi in [(0.0, 10.0), (110.0, FIELD_LENGTH)]:
        p0 = field_to_canvas(np.array([x_lo, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x_hi, FIELD_WIDTH])).astype(int)
        cv2.rectangle(img, (p0[0], p1[1]), (p1[0], p0[1]),
                      (24, 70, 24), -1)

    # Sidelines.
    for y in (0.0, FIELD_WIDTH):
        p0 = field_to_canvas(np.array([0.0, y])).astype(int)
        p1 = field_to_canvas(np.array([FIELD_LENGTH, y])).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (255, 255, 255), 2, cv2.LINE_AA)

    # 5-yard grid (thin) + 10-yard grid (thick).
    for x in range(10, 115, 5):
        p0 = field_to_canvas(np.array([x, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH])).astype(int)
        thick = 2 if x % 10 == 0 else 1
        cv2.line(img, tuple(p0), tuple(p1),
                 (240, 240, 240), thick, cv2.LINE_AA)

    # Goal lines (yellow).
    for x in (10.0, 110.0):
        p0 = field_to_canvas(np.array([x, 0.0])).astype(int)
        p1 = field_to_canvas(np.array([x, FIELD_WIDTH])).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (0, 220, 220), 2, cv2.LINE_AA)

    # Hash rows (small ticks every 1 yard along each row).
    for y in (HASH_Y_NEAR, HASH_Y_FAR):
        for x in range(11, 110):
            p = field_to_canvas(np.array([x, y])).astype(int)
            cv2.line(img, (p[0], p[1] - 4), (p[0], p[1] + 4),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # Numbers (top + bottom).
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


def dot_pixel_from_xyxy(xyxy):
    """Middle of bbox in x, 95% from top to bottom in y."""
    x0, y0, x1, y1 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return np.array([0.5 * (x0 + x1), y0 + 0.95 * (y1 - y0)], dtype=np.float64)


def project_via_H(pts_pixel, H):
    """Pts in undistorted image space → field coords via H. No undistort
    step (frames are already undistorted upstream)."""
    if len(pts_pixel) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homo = np.column_stack([pts_pixel, np.ones(len(pts_pixel))])
    proj = (H @ homo.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return proj[:, :2] / w


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-meas-frac", type=float, default=0.5)
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

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    per_frame = []
    track_results = []
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != SRC_W: fr = cv2.resize(fr, (SRC_W, SRC_H))
        fr_u = cv2.undistort(fr, K, dist)

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

        detections = detector.detect(fr_u)
        H_for_tracker = per_frame[-1]["H"] if per_frame[-1] else None
        tr = tracker.update(detections, fr_u, H=H_for_tracker, K=K, dist=dist)
        track_results.append(tr)

        if (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{fi+1}/{n_total}] {dt:.0f}s ({(fi+1)/dt:.1f} fps)")
    cap.release()
    print(f"  pass 1 done in {time.time()-t0:.0f}s")

    # ── Cutoff (≥3 consecutive carries → clip lost). ─────────────────────────
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
    Hs_clean, red_mask, _ = loo_filter_and_replace(
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

    # ── Filter to long tracks + team classify ────────────────────────────────
    trajectories = tracker.trajectories
    # n_valid_frames=cutoff → threshold is relative to H-valid frames
    # (matches _viz_tracker.py production setup).
    long_track_ids = select_long_tracks(
        trajectories, min_meas_frac=args.min_meas_frac,
        n_valid_frames=cutoff)
    print(f"  long tracks ({args.min_meas_frac:.2f} thresh, "
          f"ref={cutoff}): {len(long_track_ids)} / {len(trajectories)}")

    # Color-PCA + median split on chromatic-pixel signatures. Production
    # method from _viz_tracker.py — validated 11/11 / 100% agreement vs
    # position on play_114. Enforces balanced split (the 11/11 prior).
    print("  classifying teams (color PCA + median split) ...")
    team_labels, team_conf = classify_teams_color_pca(
        trajectories, clip, n_samples_per_track=12,
        long_track_ids=long_track_ids)
    n_a = sum(1 for v in team_labels.values() if v == "team_A")
    n_b = sum(1 for v in team_labels.values() if v == "team_B")
    n_unk = sum(1 for v in team_labels.values() if v == "unknown")
    print(f"  team_A: {n_a}    team_B: {n_b}    unknown: {n_unk}")

    # ── Build per-track field-coord time series, then SG-smooth ──────────────
    n_render = cutoff
    # dot_field[tid][fi] = (x, y) in NGS yards, or NaN.
    dot_field = {tid: np.full((n_render, 2), np.nan, dtype=np.float64)
                 for tid in long_track_ids}
    for fi in range(n_render):
        rec = per_frame[fi]
        if rec is None or rec["H"] is None:
            continue
        H = rec["H"]
        for p in track_results[fi].players:
            if p.track_id not in long_track_ids:
                continue
            dp = dot_pixel_from_xyxy(p.xyxy)
            fxy = project_via_H(dp[None], H)[0]
            if np.isfinite(fxy).all():
                dot_field[p.track_id][fi] = fxy

    # SG-smooth each track's contiguous segments (per axis).
    n_smoothed = 0
    for tid, arr in dot_field.items():
        valid = ~np.isnan(arr[:, 0])
        if not valid.any(): continue
        # Find contiguous non-NaN segments.
        segs = []
        in_seg = False; start = 0
        for i in range(n_render):
            if valid[i] and not in_seg:
                start = i; in_seg = True
            elif not valid[i] and in_seg:
                segs.append((start, i)); in_seg = False
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
    print(f"  SG-smoothed (w={DOT_SG_WINDOW}, p={DOT_SG_POLY}) "
          f"{n_smoothed} track-segments")

    # ── Pass 2: render. ──────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "tracking_dots.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (CANVAS_W, CANVAS_H))
    field_bg = render_field_background()

    print(f"[pass 2/2] rendering {n_render} frames to {out_path} ...")
    t0 = time.time()
    for fi in range(n_render):
        canvas = field_bg.copy()
        bad = fi < len(in_bad_run) and in_bad_run[fi]
        for tid in long_track_ids:
            fxy = dot_field[tid][fi]
            if not np.isfinite(fxy).all(): continue
            cxy = field_to_canvas(fxy)
            cx, cy = int(round(cxy[0])), int(round(cxy[1]))
            if not (0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H): continue
            label = team_labels.get(tid, "unknown")
            col = TEAM_COLORS_BGR[label]
            cv2.circle(canvas, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 7, col, -1, cv2.LINE_AA)
        if bad: draw_bad_run_banner(canvas)
        writer.write(canvas)

    writer.release()
    print(f"  pass 2 done in {time.time()-t0:.0f}s")
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
