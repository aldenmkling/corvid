"""Compare our tracker output for play_065 against NGS ground truth.

Pipeline:
  1. Run the full pipeline on play_065 (classifier H + tracker + LOO filter
     + SG smoothing). Per-track per-frame field_xy in NGS yards.
  2. Load NGS TSV for play 1643 — 22 players × frames at 10 Hz.
  3. Time-align: NGS `ball_snap` event marks t=0. Our clip starts before
     the snap (~frame 120 ≈ 4 sec in). Sweep our-frame-of-snap candidate
     in [60..180] at 10-Hz resolution; pick the offset that minimizes
     total Hungarian assignment cost.
  4. Position-only Hungarian: 22 NGS × N our_tracks cost matrix, cost =
     mean L2 distance (NGS yards) over overlapping aligned frames.
  5. Score each matched pair: position RMSE, speed RMSE & correlation,
     acceleration RMSE & correlation. (Speed from finite diff on positions
     for us; NGS has `s` column directly. Accel = derivative of speed.)
  6. Output:
       output/ngs_compare/play_065_per_player.csv  — per-player stats
       output/ngs_compare/play_065_summary.txt     — aggregate
       output/ngs_compare/play_065_compare.mp4     — side-by-side viz
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

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
    HomographyTrackerLite, smooth_hs, loo_filter_and_replace, detect_bad_runs,
)
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)
from src.detector import RFDETRDetector
from src.tracker import PlayerTracker
from src.team_classifier import select_long_tracks

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANVAS_YDS_PER_PX = 1 / 8
CANVAS_W = int(FIELD_LENGTH / CANVAS_YDS_PER_PX)
CANVAS_H = int(FIELD_WIDTH / CANVAS_YDS_PER_PX)

RMSE_THR_YD = 0.30
LOO_THR_YD = 0.20
CARRY_STREAK_LOST = 3
DOT_SG_WINDOW = 9
DOT_SG_POLY = 2

# Time-alignment search.
SWEEP_FRAME_CENTER = 120     # ~4 sec in @ 30 fps
SWEEP_FRAME_RADIUS = 60      # ±2 sec → snap candidate ∈ [60, 180]
NGS_FPS = 10
OUR_FPS = 30
STRIDE = OUR_FPS // NGS_FPS  # 3 — every 3rd our-frame samples a NGS frame

# Color palette for matched (player, track) pairs.
np.random.seed(42)
PAIR_COLORS = [tuple(int(c) for c in np.random.randint(60, 255, 3))
               for _ in range(64)]


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


def field_to_canvas(field_xy):
    x_yd, y_yd = field_xy[..., 0], field_xy[..., 1]
    px = x_yd / CANVAS_YDS_PER_PX
    py = CANVAS_H - (y_yd / CANVAS_YDS_PER_PX)
    return np.stack([px, py], axis=-1)


def render_field_background():
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


def run_pipeline(clip_path, device, manifest):
    """Run the full pipeline; return per-track per-frame field_xy at 30 fps.

    Returns: (cutoff, dot_field, track_meta)
      cutoff: number of usable frames
      dot_field: dict[track_id -> (cutoff, 2) NGS-yard array, NaN where missing]
      track_meta: dict[track_id -> {n_obs}]
    """
    rel = os.path.relpath(clip_path, os.path.join(PROJECT_ROOT, "videos/clips"))
    intr = manifest["intrinsics_by_clip"].get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)

    print("Loading classifier pipeline ...")
    pipe = NewPipeline(device)
    print("Loading detector + tracker ...")
    detector = RFDETRDetector(
        weights=os.path.join(PROJECT_ROOT, "models/rfdetr_best_ema.pth"),
        device=str(device), conf_thresh=0.3)
    tracker = PlayerTracker(device=str(device), frame_rate=OUR_FPS)
    h_tracker = HomographyTrackerLite()

    cap = cv2.VideoCapture(clip_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {n_total} frames @ {OUR_FPS} fps")

    per_frame, track_results = [], []
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
            per_frame.append({"H": res["H"], "method": res["method"],
                              "rmse": res["rmse_yd"]})
        dets = detector.detect(fr_u)
        H_for_tracker = per_frame[-1]["H"] if per_frame[-1] else None
        tr = tracker.update(dets, fr_u, H=H_for_tracker, K=K, dist=dist)
        track_results.append(tr)
        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{n_total}] {time.time()-t0:.0f}s")
    cap.release()
    print(f"  pass 1 in {time.time()-t0:.0f}s")

    # Cutoff (carry streak).
    cutoff = len(per_frame); streak = 0; run_start = None
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

    # LOO filter + replacement + SG.
    Hs = [r["H"] if (r and r["H"] is not None) else None
          for r in per_frame[:cutoff]]
    rmses = [r["rmse"] if (r and r.get("rmse") is not None) else None
             for r in per_frame[:cutoff]]
    Hs_clean, red_mask, _ = loo_filter_and_replace(
        Hs, rmses=rmses, thr_loo_yd=LOO_THR_YD, thr_rmse_yd=RMSE_THR_YD)
    valid_idx = [i for i in range(cutoff) if Hs_clean[i] is not None]
    if len(valid_idx) >= 7:
        sm = smooth_hs([Hs_clean[i] for i in valid_idx], window=7, poly=2)
        for vi, si in enumerate(valid_idx):
            Hs_clean[si] = sm[vi]

    # Filter to long tracks (drop spurious).
    trajectories = tracker.trajectories
    long_ids = select_long_tracks(
        trajectories, min_meas_frac=0.5, n_valid_frames=cutoff)
    print(f"  long tracks: {len(long_ids)}/{len(trajectories)}")

    # Build per-track per-frame field_xy at 30 fps (NaN where missing).
    dot_field = {tid: np.full((cutoff, 2), np.nan, dtype=np.float64)
                 for tid in long_ids}
    for fi in range(cutoff):
        H = Hs_clean[fi]
        if H is None: continue
        for p in track_results[fi].players:
            if p.track_id not in long_ids: continue
            dp = dot_pixel_from_xyxy(p.xyxy)
            fxy = project_via_H(dp[None], H)[0]
            if np.isfinite(fxy).all():
                dot_field[p.track_id][fi] = fxy

    # SG-smooth each contiguous segment per track.
    for tid, arr in dot_field.items():
        valid = ~np.isnan(arr[:, 0])
        in_seg = False; start = 0; segs = []
        for i in range(cutoff):
            if valid[i] and not in_seg: start = i; in_seg = True
            elif not valid[i] and in_seg: segs.append((start, i)); in_seg = False
        if in_seg: segs.append((start, cutoff))
        for s, e in segs:
            L = e - s
            if L < DOT_SG_WINDOW: continue
            w = DOT_SG_WINDOW
            if w > L: w = L if L % 2 == 1 else L - 1
            if w < DOT_SG_POLY + 2: continue
            arr[s:e, 0] = savgol_filter(arr[s:e, 0], w, DOT_SG_POLY)
            arr[s:e, 1] = savgol_filter(arr[s:e, 1], w, DOT_SG_POLY)

    track_meta = {tid: {"n_obs": int(np.sum(~np.isnan(dot_field[tid][:, 0])))}
                  for tid in long_ids}
    return cutoff, dot_field, track_meta


def load_ngs(tsv_path):
    """Return (ngs_by_player, snap_frame, players_meta).

    ngs_by_player: dict[nflId -> dict with arrays x[F], y[F], s[F], frames[F]
                                  where F = total NGS frames]
    snap_frame: NGS frame index of `ball_snap` event
    players_meta: dict[nflId -> {displayName, jerseyNumber, position, teamAbbr}]
    """
    df = pd.read_csv(tsv_path, sep="\t")
    # ball_snap event sits in one row; same frame across all players for that frame.
    snap_rows = df[df["event"] == "ball_snap"]
    if len(snap_rows) == 0:
        raise ValueError("No ball_snap event in NGS TSV")
    snap_frame = int(snap_rows["frame"].iloc[0])

    # NGS x is along the field (0-120 — same convention as our pipeline outputs).
    ngs_by_player = {}
    players_meta = {}
    for nfl_id, sub in df.groupby("nflId"):
        if pd.isna(nfl_id):
            continue
        sub = sub.sort_values("frame")
        ngs_by_player[int(nfl_id)] = {
            "frames": sub["frame"].astype(int).to_numpy(),
            "x": sub["x"].astype(float).to_numpy(),
            "y": sub["y"].astype(float).to_numpy(),
            "s": sub["s"].astype(float).to_numpy(),  # speed yd/s
        }
        row = sub.iloc[0]
        players_meta[int(nfl_id)] = {
            "displayName": str(row["displayName"]),
            "jerseyNumber": int(row["jerseyNumber"]) if pd.notna(row["jerseyNumber"]) else None,
            "position": str(row["position"]),
            "teamAbbr": str(row["teamAbbr"]),
        }
    return ngs_by_player, snap_frame, players_meta


def compute_cost_matrix(dot_field, ngs_by_player, our_snap_frame, snap_frame_ngs,
                          cutoff):
    """Build a (n_tracks, n_players) cost matrix at a given snap alignment.

    Cost[ti, pi] = mean L2 distance over frames where BOTH have valid samples.
    Cost = +inf if no overlap.

    Returns (cost, n_overlap, track_ids, player_ids).
    """
    track_ids = sorted(dot_field.keys())
    player_ids = sorted(ngs_by_player.keys())
    nT, nP = len(track_ids), len(player_ids)
    cost = np.full((nT, nP), np.inf)
    n_overlap = np.zeros((nT, nP), dtype=int)

    # Build aligned (our_frame, ngs_frame) pairs.
    # NGS frame at our_frame F: ngs_F = snap_frame_ngs + (F - our_snap_frame) / STRIDE
    # Only valid where (F - our_snap_frame) % STRIDE == 0.
    aligned_pairs = []
    for F in range(cutoff):
        delta = F - our_snap_frame
        if delta % STRIDE != 0: continue
        ngs_F = snap_frame_ngs + delta // STRIDE
        aligned_pairs.append((F, ngs_F))

    if not aligned_pairs:
        return cost, n_overlap, track_ids, player_ids

    # For each player, build a frame_idx → pos lookup.
    player_pos = {}
    for pid in player_ids:
        d = ngs_by_player[pid]
        f2pos = {int(f): (float(x), float(y))
                 for f, x, y in zip(d["frames"], d["x"], d["y"])}
        player_pos[pid] = f2pos

    for ti, tid in enumerate(track_ids):
        arr = dot_field[tid]
        for pi, pid in enumerate(player_ids):
            f2pos = player_pos[pid]
            dists = []
            for our_F, ngs_F in aligned_pairs:
                if np.isnan(arr[our_F, 0]): continue
                if ngs_F not in f2pos: continue
                xy_us = arr[our_F]
                xy_ngs = f2pos[ngs_F]
                d_yd = np.hypot(xy_us[0] - xy_ngs[0], xy_us[1] - xy_ngs[1])
                dists.append(d_yd)
            if dists:
                cost[ti, pi] = float(np.mean(dists))
                n_overlap[ti, pi] = len(dists)
    return cost, n_overlap, track_ids, player_ids


def sweep_snap_offset(dot_field, ngs_by_player, snap_frame_ngs, cutoff,
                       center=SWEEP_FRAME_CENTER, radius=SWEEP_FRAME_RADIUS):
    """Sweep candidate our-snap-frame in [center-radius, center+radius]
    at 10-Hz step (=STRIDE). Return (best_offset, best_total_cost, sweep_log).
    """
    best = (None, np.inf, None)
    log = []
    for offset in range(center - radius, center + radius + 1, STRIDE):
        cost, n_ov, track_ids, player_ids = compute_cost_matrix(
            dot_field, ngs_by_player, offset, snap_frame_ngs, cutoff)
        # Hungarian on the finite-cost subset.
        cost_safe = np.where(np.isfinite(cost), cost, 1e6)
        if cost_safe.shape[0] == 0 or cost_safe.shape[1] == 0:
            continue
        r, c = linear_sum_assignment(cost_safe)
        sel = cost_safe[r, c]
        # Only sum costs that aren't the sentinel.
        valid_match = sel < 1e6
        if not valid_match.any():
            continue
        total = float(np.sum(sel[valid_match]))
        n_matched = int(valid_match.sum())
        log.append((offset, total, n_matched))
        if total < best[1]:
            best = (offset, total, (r[valid_match], c[valid_match],
                                      cost, n_ov, track_ids, player_ids))
    return best, log


def render_compare(out_path, dot_field, ngs_by_player, players_meta,
                     our_snap_frame, snap_frame_ngs, cutoff, matches):
    """Side-by-side rectified canvas. Left = NGS, right = ours.
    Matched pairs share a color from PAIR_COLORS."""
    field_bg = render_field_background()
    panel_h, panel_w = CANVAS_H, CANVAS_W
    out_h, out_w = panel_h, 2 * panel_w + 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, OUR_FPS, (out_w, out_h))

    # Build colors: assign each matched pair a unique color.
    track_color = {}      # tid -> bgr
    player_color = {}     # pid -> bgr
    track_to_player = matches["track_to_player"]
    for idx, (tid, pid) in enumerate(track_to_player.items()):
        c = PAIR_COLORS[idx % len(PAIR_COLORS)]
        track_color[tid] = c
        player_color[pid] = c
    GRAY = (140, 140, 140)

    # NGS frame at our frame F: (F - our_snap_frame) / STRIDE + snap_frame_ngs.
    # We render at 30 fps; at frames where F-our_snap is not divisible by STRIDE,
    # we interpolate linearly between adjacent NGS samples.
    def ngs_pos_at_our_F(F, pid):
        d = ngs_by_player.get(pid)
        if d is None: return None
        delta = F - our_snap_frame
        ngs_F_float = snap_frame_ngs + delta / STRIDE
        lo_F = int(np.floor(ngs_F_float)); hi_F = lo_F + 1
        # Find positions at lo and hi.
        idx_lo = np.searchsorted(d["frames"], lo_F)
        idx_hi = np.searchsorted(d["frames"], hi_F)
        if idx_lo >= len(d["frames"]) or d["frames"][idx_lo] != lo_F: return None
        if idx_hi >= len(d["frames"]) or d["frames"][idx_hi] != hi_F:
            return float(d["x"][idx_lo]), float(d["y"][idx_lo])
        alpha = ngs_F_float - lo_F
        return (float(d["x"][idx_lo] * (1 - alpha) + d["x"][idx_hi] * alpha),
                float(d["y"][idx_lo] * (1 - alpha) + d["y"][idx_hi] * alpha))

    for F in range(cutoff):
        left = field_bg.copy()
        right = field_bg.copy()

        # NGS panel.
        for pid in ngs_by_player:
            pos = ngs_pos_at_our_F(F, pid)
            if pos is None: continue
            cxy = field_to_canvas(np.array(pos))
            cx, cy = int(round(cxy[0])), int(round(cxy[1]))
            if not (0 <= cx < panel_w and 0 <= cy < panel_h): continue
            col = player_color.get(pid, GRAY)
            cv2.circle(left, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(left, (cx, cy), 7, col, -1, cv2.LINE_AA)
            # Jersey label.
            jn = players_meta.get(pid, {}).get("jerseyNumber")
            if jn is not None:
                cv2.putText(left, str(jn), (cx + 10, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Our panel.
        for tid in dot_field:
            xy = dot_field[tid][F]
            if not np.isfinite(xy).all(): continue
            cxy = field_to_canvas(xy)
            cx, cy = int(round(cxy[0])), int(round(cxy[1]))
            if not (0 <= cx < panel_w and 0 <= cy < panel_h): continue
            col = track_color.get(tid, GRAY)
            cv2.circle(right, (cx, cy), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(right, (cx, cy), 7, col, -1, cv2.LINE_AA)

        # Labels.
        cv2.putText(left, "NGS (10 Hz, interp)", (12, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "OURS (30 fps, SG-smoothed)", (12, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        time_label = f"t = {(F - our_snap_frame)/OUR_FPS:+.2f}s"
        cv2.putText(left, time_label, (12, panel_h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        canvas = np.full((out_h, out_w, 3), (24, 24, 24), dtype=np.uint8)
        canvas[:, :panel_w] = left
        canvas[:, panel_w + 20:] = right
        writer.write(canvas)

    writer.release()


def score_pair(arr_us, ngs_data, our_snap_frame, snap_frame_ngs, cutoff):
    """Score one matched pair. Returns dict of stats."""
    # Build aligned series at 10 Hz.
    pairs = []
    f2pos = {int(f): (float(x), float(y), float(s))
             for f, x, y, s in zip(ngs_data["frames"],
                                     ngs_data["x"], ngs_data["y"],
                                     ngs_data["s"])}
    aligned_t, aligned_us, aligned_ngs, aligned_ngs_s = [], [], [], []
    for F in range(cutoff):
        delta = F - our_snap_frame
        if delta % STRIDE != 0: continue
        ngs_F = snap_frame_ngs + delta // STRIDE
        if ngs_F not in f2pos: continue
        if np.isnan(arr_us[F, 0]): continue
        t = delta / OUR_FPS
        aligned_t.append(t)
        aligned_us.append(arr_us[F])
        ngs_x, ngs_y, ngs_s = f2pos[ngs_F]
        aligned_ngs.append([ngs_x, ngs_y])
        aligned_ngs_s.append(ngs_s)
    if len(aligned_t) < 5:
        return None
    us_xy = np.array(aligned_us)
    ngs_xy = np.array(aligned_ngs)
    ngs_s = np.array(aligned_ngs_s)
    t_arr = np.array(aligned_t)
    # Position.
    pos_err = np.linalg.norm(us_xy - ngs_xy, axis=1)
    # Speed: central finite diff on our position; NGS provides s.
    if len(t_arr) >= 3:
        dt = np.diff(t_arr)
        vx = np.gradient(us_xy[:, 0], t_arr)
        vy = np.gradient(us_xy[:, 1], t_arr)
        us_s = np.hypot(vx, vy)
        # Accel = derivative of speed.
        us_a = np.gradient(us_s, t_arr)
        ngs_a = np.gradient(ngs_s, t_arr)
        speed_err = us_s - ngs_s
        speed_corr = float(np.corrcoef(us_s, ngs_s)[0, 1]) if us_s.std() > 1e-6 and ngs_s.std() > 1e-6 else float("nan")
        accel_err = us_a - ngs_a
        accel_corr = float(np.corrcoef(us_a, ngs_a)[0, 1]) if us_a.std() > 1e-6 and ngs_a.std() > 1e-6 else float("nan")
    else:
        speed_err = np.array([np.nan]); speed_corr = float("nan")
        accel_err = np.array([np.nan]); accel_corr = float("nan")
    return {
        "n_obs": len(t_arr),
        "pos_rmse_yd": float(np.sqrt(np.mean(pos_err ** 2))),
        "pos_mean_yd": float(np.mean(pos_err)),
        "pos_p50_yd": float(np.median(pos_err)),
        "pos_p90_yd": float(np.percentile(pos_err, 90)),
        "speed_rmse_yds": float(np.sqrt(np.mean(speed_err ** 2))),
        "speed_corr": speed_corr,
        "speed_mean_us_yds": float(np.mean(us_s)) if len(t_arr) >= 3 else float("nan"),
        "speed_mean_ngs_yds": float(np.mean(ngs_s)),
        "accel_rmse_yds2": float(np.sqrt(np.mean(accel_err ** 2))),
        "accel_corr": accel_corr,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    device = torch.device(args.device)

    clip_path = os.path.join(PROJECT_ROOT,
                              "videos/clips/2019092204/play_065/sideline.mp4")
    ngs_path = "/Users/aldenkling/Desktop/Personal Research/" \
               "ngs_highlights-master/play_data/2019_KC_2019092204_1643.tsv"
    # On RunPod, override:
    if not os.path.exists(ngs_path):
        alt = os.path.join(PROJECT_ROOT, "data", "ngs",
                            "2019_KC_2019092204_1643.tsv")
        if os.path.exists(alt):
            ngs_path = alt
        else:
            raise FileNotFoundError(f"NGS TSV not found: {ngs_path} or {alt}")
    out_dir = os.path.join(PROJECT_ROOT, "output", "ngs_compare")
    os.makedirs(out_dir, exist_ok=True)

    manifest = json.load(open(os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json")))

    # ── Run our pipeline ─────────────────────────────────────────────────────
    cutoff, dot_field, track_meta = run_pipeline(clip_path, device, manifest)

    # ── Load NGS ─────────────────────────────────────────────────────────────
    print("Loading NGS TSV ...")
    ngs_by_player, snap_frame_ngs, players_meta = load_ngs(ngs_path)
    print(f"  NGS players: {len(ngs_by_player)}  snap_frame: {snap_frame_ngs}")
    teams = set(m["teamAbbr"] for m in players_meta.values())
    print(f"  teams: {teams}")

    # ── Sweep snap alignment ─────────────────────────────────────────────────
    print(f"Sweeping our-snap-frame in "
          f"[{SWEEP_FRAME_CENTER - SWEEP_FRAME_RADIUS}, "
          f"{SWEEP_FRAME_CENTER + SWEEP_FRAME_RADIUS}] @ stride {STRIDE} ...")
    best, log = sweep_snap_offset(dot_field, ngs_by_player, snap_frame_ngs,
                                    cutoff)
    best_offset, best_total, payload = best
    if payload is None:
        print("  ALIGNMENT FAILED — no usable overlap")
        return
    rows, cols, cost, n_ov, track_ids, player_ids = payload
    print(f"  best our-snap-frame: {best_offset}  total cost: {best_total:.2f}")
    print(f"  matched pairs: {len(rows)}")
    print(f"  sweep log (offset, total_cost, n_matched):")
    for o, c, nm in log:
        marker = "  <-- best" if o == best_offset else ""
        print(f"    {o:4d}  cost={c:8.2f}  matched={nm}{marker}")

    # ── Build the per-pair mapping ───────────────────────────────────────────
    track_to_player = {}
    pair_scores = []
    for r, c in zip(rows, cols):
        tid = track_ids[r]; pid = player_ids[c]
        track_to_player[tid] = pid
        s = score_pair(dot_field[tid], ngs_by_player[pid],
                        best_offset, snap_frame_ngs, cutoff)
        if s is None: continue
        meta = players_meta[pid]
        s.update({
            "track_id": tid, "nfl_id": pid,
            "player": meta["displayName"], "jersey": meta["jerseyNumber"],
            "pos": meta["position"], "team": meta["teamAbbr"],
            "n_obs": s["n_obs"], "match_cost_yd": float(cost[r, c]),
            "n_overlap_frames": int(n_ov[r, c]),
        })
        pair_scores.append(s)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(pair_scores)
    front = ["track_id", "nfl_id", "player", "jersey", "pos", "team",
             "n_obs", "n_overlap_frames", "match_cost_yd",
             "pos_rmse_yd", "pos_mean_yd", "pos_p50_yd", "pos_p90_yd",
             "speed_rmse_yds", "speed_corr",
             "speed_mean_us_yds", "speed_mean_ngs_yds",
             "accel_rmse_yds2", "accel_corr"]
    df = df[[c for c in front if c in df.columns]]
    csv_path = os.path.join(out_dir, "play_065_per_player.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "play_065_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"# play_065 vs NGS (TSV: 2019_KC_2019092204_1643.tsv)\n\n")
        f.write(f"Our cutoff frames: {cutoff} @ {OUR_FPS} fps\n")
        f.write(f"NGS players: {len(ngs_by_player)}, snap_frame_ngs: {snap_frame_ngs}\n")
        f.write(f"Best our-snap-frame: {best_offset} (= {best_offset/OUR_FPS:.2f}s into clip)\n")
        f.write(f"Matched pairs: {len(pair_scores)}\n\n")
        if pair_scores:
            f.write(f"## Aggregate (median across matched players)\n")
            for c in ("pos_rmse_yd", "pos_p50_yd", "pos_p90_yd",
                      "speed_rmse_yds", "speed_corr",
                      "accel_rmse_yds2", "accel_corr"):
                vals = df[c].dropna().to_numpy()
                f.write(f"  {c:22s}  median={np.median(vals):.3f}  "
                        f"mean={np.mean(vals):.3f}\n")
            f.write(f"\n## Per-player\n")
            f.write(df.to_string(index=False))
    print(f"  → {summary_path}")

    # Console quick view.
    if pair_scores:
        print("\nAggregate:")
        for c in ("pos_rmse_yd", "pos_p50_yd", "pos_p90_yd",
                  "speed_rmse_yds", "speed_corr",
                  "accel_rmse_yds2", "accel_corr"):
            vals = df[c].dropna().to_numpy()
            print(f"  {c:22s}  median={np.median(vals):.3f}  "
                  f"mean={np.mean(vals):.3f}")

    # ── Render comparison viz ────────────────────────────────────────────────
    print("\n[render] side-by-side comparison viz ...")
    matches = {"track_to_player": track_to_player}
    viz_path = os.path.join(out_dir, "play_065_compare.mp4")
    render_compare(viz_path, dot_field, ngs_by_player, players_meta,
                     best_offset, snap_frame_ngs, cutoff, matches)
    print(f"  → {viz_path}")


if __name__ == "__main__":
    main()
