"""Compare the new LOO red-flag filter to the old (rmse + temp_div) filter
on multiple clips.

For each clip:
  • Run pass 1: classifier per frame → corrs → H + method + rmse
  • Compute temp_div (max corner reproj distance vs neighbor average, yds)
  • Compute LOO residual (raw H vs degree-2 poly through 3-on-each-side)
  • Old red:  rmse > 0.30 yd  OR  temp_div > 1.0 yd
  • New red:  loo > --thr (default 0.20 yd)
  • Build a 2x2 confusion of {old_red, new_red}

Prints per-clip table + aggregate. Dumps JSON to
output/h_filter_diagnostics.json with the per-frame data for offline use.

We don't have ground-truth "this frame's H is bad", so:
  • |old ∩ new| / |old|  → fraction of old TPs the new filter catches
  • |new \\ old|           → extra catches (likely missed FNs)
  • |old \\ new|           → frames the new filter missed (concerning)
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
from src.homography.h_tracker import HomographyTrackerLite

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RMSE_THR_YD = 0.30
TEMP_THR_YD = 1.0
CARRY_STREAK_LOST = 3


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
        if tokens_np.shape[0] == 0:
            return None
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


def proj_corners_field(H):
    pts = np.array([[100, 100], [SRC_W - 100, 100],
                    [SRC_W - 100, SRC_H - 100], [100, SRC_H - 100]],
                   dtype=np.float64)
    p = (H @ np.column_stack([pts, np.ones(4)]).T).T
    z = p[:, 2:3]; z[np.abs(z) < 1e-9] = 1e-9
    return p[:, :2] / z


def corner_residual_yd(H_a, H_b):
    if H_a is None or H_b is None: return None
    ca = proj_corners_field(H_a); cb = proj_corners_field(H_b)
    return float(np.max(np.linalg.norm(ca - cb, axis=1)))


def loo_smooth_at(H_seq, i, half=3, poly=2):
    n = len(H_seq)
    lo = max(0, i - half); hi = min(n, i + half + 1)
    xs = [j for j in range(lo, hi) if j != i]
    if len(xs) < poly + 1: return None
    ys = np.stack([H_seq[j].flatten() for j in xs], axis=0)
    xs_a = np.array(xs, dtype=np.float64)
    H_loo = np.zeros(9, dtype=np.float64)
    for c in range(9):
        coef = np.polyfit(xs_a, ys[:, c], poly)
        H_loo[c] = np.polyval(coef, i)
    return H_loo.reshape(3, 3)


def analyze_clip(pipe, clip_path, manifest, thr_loo):
    """Run pass 1 + flag analysis for one clip. Returns per-frame data + counts."""
    rel = os.path.relpath(clip_path, os.path.join(PROJECT_ROOT, "videos/clips"))
    intr = manifest["intrinsics_by_clip"].get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)

    h_tracker = HomographyTrackerLite()
    cap = cv2.VideoCapture(clip_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame = []
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != SRC_W: fr = cv2.resize(fr, (SRC_W, SRC_H))
        pp = pipe(fr, K, dist)
        if pp is None:
            per_frame.append({"H": None, "method": "none", "rmse": None})
        else:
            corrs = pp["corrs"]
            res = h_tracker.update(corrs, frame_idx=fi)
            per_frame.append({"H": res["H"], "method": res["method"],
                              "rmse": res["rmse_yd"]})
    cap.release()
    dt = time.time() - t0
    print(f"  pass 1 in {dt:.0f}s ({n_total/dt:.1f} fps)")

    # Cutoff (≥3 consecutive carries → clip lost).
    cutoff = n_total
    streak = 0; run_start = None
    for i, r in enumerate(per_frame):
        m = r["method"]
        if m == "carry":
            if streak == 0: run_start = i
            streak += 1
            if streak >= CARRY_STREAK_LOST:
                cutoff = run_start; break
        else:
            streak = 0; run_start = None

    Hs_raw = [r["H"] for r in per_frame]
    temp_div = [None] * len(per_frame)
    for i in range(1, len(per_frame) - 1):
        if Hs_raw[i] is None or Hs_raw[i-1] is None or Hs_raw[i+1] is None: continue
        ci = proj_corners_field(Hs_raw[i])
        cavg = 0.5 * (proj_corners_field(Hs_raw[i-1]) + proj_corners_field(Hs_raw[i+1]))
        temp_div[i] = float(np.max(np.linalg.norm(ci - cavg, axis=1)))

    # LOO residuals over [0, cutoff) where raw H exists.
    seq_idx = [i for i in range(cutoff)
               if per_frame[i]["H"] is not None]
    seq_H = [per_frame[i]["H"] for i in seq_idx]
    loo_resid = [None] * len(per_frame)
    for local_i, gi in enumerate(seq_idx):
        H_loo = loo_smooth_at(seq_H, local_i, half=3, poly=2)
        if H_loo is None: continue
        loo_resid[gi] = corner_residual_yd(per_frame[gi]["H"], H_loo)

    # Flags. ONLY over [0, cutoff); outside that, neither flag is meaningful.
    old_red = [False] * len(per_frame)
    new_red = [False] * len(per_frame)
    for i in range(cutoff):
        r = per_frame[i]
        # Old: missing H counts as red.
        if r["H"] is None:
            old_red[i] = True
        else:
            if r["rmse"] is not None and r["rmse"] > RMSE_THR_YD: old_red[i] = True
            if temp_div[i] is not None and temp_div[i] > TEMP_THR_YD: old_red[i] = True
        # New: missing H counts as red. Missing LOO at clip start/end (need
        # at least poly+1 = 3 neighbors) → leave False, can't judge.
        if r["H"] is None:
            new_red[i] = True
        elif loo_resid[i] is not None and loo_resid[i] > thr_loo:
            new_red[i] = True

    # Confusion.
    both = sum(1 for i in range(cutoff) if old_red[i] and new_red[i])
    old_only = sum(1 for i in range(cutoff) if old_red[i] and not new_red[i])
    new_only = sum(1 for i in range(cutoff) if new_red[i] and not old_red[i])
    neither = sum(1 for i in range(cutoff) if not old_red[i] and not new_red[i])

    return {
        "rel": rel, "n_total": n_total, "cutoff": cutoff,
        "old_red_count": both + old_only,
        "new_red_count": both + new_only,
        "both": both, "old_only": old_only, "new_only": new_only,
        "neither": neither,
        "per_frame": [
            {"frame": i, "method": per_frame[i]["method"],
             "rmse": per_frame[i]["rmse"], "temp_div": temp_div[i],
             "loo_resid": loo_resid[i], "old_red": old_red[i],
             "new_red": new_red[i]}
            for i in range(cutoff)
        ],
        "loo_summary": {
            "median": float(np.nanmedian([v for v in loo_resid if v is not None])) if any(v is not None for v in loo_resid) else None,
            "p90": float(np.percentile([v for v in loo_resid if v is not None], 90)) if any(v is not None for v in loo_resid) else None,
            "p95": float(np.percentile([v for v in loo_resid if v is not None], 95)) if any(v is not None for v in loo_resid) else None,
            "p99": float(np.percentile([v for v in loo_resid if v is not None], 99)) if any(v is not None for v in loo_resid) else None,
            "max":  float(max(v for v in loo_resid if v is not None)) if any(v is not None for v in loo_resid) else None,
        },
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--thr", type=float, default=0.20)
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output", "h_filter_diagnostics.json"))
    args = ap.parse_args()
    device = torch.device(args.device)

    clip_rels = [
        "2019092204/play_023",
        "2019092204/play_065",
        "2019092204/play_001",
        "2019102712/play_011",
        "2019102712/play_046",
        "2019102712/play_118",
        "2024090801/play_032",
        "2024091501/play_001",
    ]

    manifest = json.load(open(os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json")))

    print(f"Loading classifier pipeline (device={args.device}) ...")
    pipe = NewPipeline(device)
    print(f"Filter thresholds: old=(rmse>{RMSE_THR_YD} OR temp_div>{TEMP_THR_YD}),"
          f"  new=(loo>{args.thr:.2f})\n")

    results = []
    for clip_rel in clip_rels:
        clip_path = os.path.join(PROJECT_ROOT, "videos/clips", clip_rel,
                                  "sideline.mp4")
        if not os.path.exists(clip_path):
            print(f"[skip] {clip_rel} — file not found"); continue
        print(f"[{clip_rel}]")
        r = analyze_clip(pipe, clip_path, manifest, args.thr)
        results.append(r)
        lo = r["loo_summary"]
        print(f"  cutoff={r['cutoff']}/{r['n_total']}  "
              f"LOO  med={lo['median']:.3f}  p95={lo['p95']:.3f}  "
              f"p99={lo['p99']:.3f}  max={lo['max']:.3f}")
        print(f"  OLD red: {r['old_red_count']:3d}  "
              f"NEW red: {r['new_red_count']:3d}  "
              f"both: {r['both']:3d}  "
              f"old_only: {r['old_only']:3d}  "
              f"new_only: {r['new_only']:3d}")
        print()

    # Aggregate.
    tot_old = sum(r["old_red_count"] for r in results)
    tot_new = sum(r["new_red_count"] for r in results)
    tot_both = sum(r["both"] for r in results)
    tot_oo = sum(r["old_only"] for r in results)
    tot_no = sum(r["new_only"] for r in results)
    tot_neither = sum(r["neither"] for r in results)
    tot_frames = tot_both + tot_oo + tot_no + tot_neither

    print("=" * 72)
    print(f"AGGREGATE OVER {len(results)} CLIPS  ({tot_frames} valid frames)")
    print(f"  OLD red count : {tot_old}")
    print(f"  NEW red count : {tot_new}")
    print(f"  Both flagged  : {tot_both}")
    print(f"  Old-only      : {tot_oo}  (frames new filter missed)")
    print(f"  New-only      : {tot_no}  (extra catches by new filter)")
    print(f"  Neither       : {tot_neither}")
    print()
    if tot_old > 0:
        print(f"  recall of OLD flags by NEW: {tot_both/tot_old*100:.1f}%  "
              f"({tot_both}/{tot_old})")
    print(f"  extra flagged by NEW: {tot_no}")

    def fmt(v):
        return "  None" if v is None else f"{v:.3f}"

    # Where they disagree — show specific frames.
    print()
    print(f"── Frames in OLD ∩ NOT NEW (old caught, new missed) ──")
    n_shown = 0
    for r in results:
        for pf in r["per_frame"]:
            if pf["old_red"] and not pf["new_red"]:
                print(f"  {r['rel']:30s}  frame {pf['frame']:4d}  "
                      f"method={pf['method']:5s}  "
                      f"rmse={fmt(pf['rmse'])}  "
                      f"temp_div={fmt(pf['temp_div'])}  "
                      f"loo={fmt(pf['loo_resid'])}")
                n_shown += 1
                if n_shown >= 30: break
        if n_shown >= 30: break
    if n_shown == 0:
        print("  (none — new filter is a strict superset)")

    print()
    print(f"── Top 20 frames in NEW ∩ NOT OLD (extra catches) ──")
    extras = []
    for r in results:
        for pf in r["per_frame"]:
            if pf["new_red"] and not pf["old_red"]:
                extras.append((r["rel"], pf))
    extras.sort(key=lambda x: -(x[1]["loo_resid"] or 0))
    for rel, pf in extras[:20]:
        print(f"  {rel:30s}  frame {pf['frame']:4d}  "
              f"method={pf['method']:5s}  "
              f"rmse={fmt(pf['rmse'])}  temp_div={fmt(pf['temp_div'])}  "
              f"loo={fmt(pf['loo_resid'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"thr_loo": args.thr,
                   "rmse_thr": RMSE_THR_YD, "temp_thr": TEMP_THR_YD,
                   "clips": results}, f, indent=2)
    print(f"\n  → wrote {args.out}")


if __name__ == "__main__":
    main()
