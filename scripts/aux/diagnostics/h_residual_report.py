"""Investigate whether SG-fit residuals are a cleaner red-flag signal
than rmse/temp_div thresholds.

For each frame:
  • raw H from the classifier pipeline
  • global-SG H (window=7 poly=2 over all good frames)
  • leave-one-out poly H (degree-2 fit through 3 frames on each side,
    excluding the frame itself, evaluated at the frame)

Residual = max corner reprojection error (NGS yards) between raw H and
the smoothed / LOO H.

Prints:
  - median / 90th / 95th / 99th percentile residual
  - residual at the frames currently flagged red (rmse>0.30 or temp_div>1.0)
  - top 10 frames by LOO residual
  - whether the LOO residual would have caught the same frames a threshold
    catches

Run on RunPod (same pattern as the visuals scripts).
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.field_mapping.tokenizer import (
    TYPE_NUM, TYPE_YARD, TYPE_SIDE, TYPE_HASH,
    SRC_W, SRC_H, null_classifier,
)
from src.field_mapping.tokenizer import tokenize_frame as cc_tokens_from_frame_v3
from src.field_mapping.encoder import TokenEncoder as TokenClassifyV10
from src.field_mapping.token_labeler import TokenLabeler as TokenClassifyV10b
from src.field_mapping.crop_classifier import make_painted_logits_fn
from src.field_mapping.encoder import encoder_features
from src.field_mapping.number_refiner import NumberRefiner as RFB
from src.field_mapping.token_labeler import refine_number_tokens as rfb_forward_with_features_and_row, PAINTED_TO_21
from src.field_mapping.classes import N_NGS_X_CLASSES

from src.field_mapping.keypoints import extract_keypoints
from src.field_mapping.homography import HomographyTrackerLite, smooth_hs

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RMSE_THR_YD = 0.30
TEMP_THR_YD = 1.0
CARRY_STREAK_LOST = 3
SG_WINDOW = 7
SG_POLY = 2


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
        return {"corrs": corrs}


def proj_corners_field(H):
    """Project 4 image corners through H. Returns (4, 2) in H's output space."""
    pts = np.array([[100, 100], [SRC_W - 100, 100],
                    [SRC_W - 100, SRC_H - 100], [100, SRC_H - 100]],
                   dtype=np.float64)
    p = (H @ np.column_stack([pts, np.ones(4)]).T).T
    z = p[:, 2:3]; z[np.abs(z) < 1e-9] = 1e-9
    return p[:, :2] / z


def corner_residual_yd(H_a, H_b):
    """Max corner-reprojection distance between two H matrices, in yards."""
    if H_a is None or H_b is None: return None
    ca = proj_corners_field(H_a)
    cb = proj_corners_field(H_b)
    return float(np.max(np.linalg.norm(ca - cb, axis=1)))


def loo_smooth_at(H_seq, i, half=3, poly=2):
    """Leave-one-out poly fit at index i.

    Fit a degree-`poly` polynomial through indices [i-half..i+half] EXCEPT i,
    per-coefficient of the 3x3 H matrix. Evaluate at i. Returns the LOO H.

    H_seq is a list of 3x3 matrices (no Nones). i is an index into that list
    where the fit is centered. half=3 + poly=2 ⇒ 6-point degree-2 fit.
    """
    n = len(H_seq)
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    xs = [j for j in range(lo, hi) if j != i]
    if len(xs) < poly + 1:
        return None
    ys = np.stack([H_seq[j].flatten() for j in xs], axis=0)  # (k, 9)
    xs_a = np.array(xs, dtype=np.float64)
    H_loo = np.zeros(9, dtype=np.float64)
    for c in range(9):
        coef = np.polyfit(xs_a, ys[:, c], poly)
        H_loo[c] = np.polyval(coef, i)
    return H_loo.reshape(3, 3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--clip", default=os.path.join(
        PROJECT_ROOT, "videos/clips/2019092204/play_065/sideline.mp4"))
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output", "h_residual_analysis.json"))
    args = ap.parse_args()
    device = torch.device(args.device)

    manifest = json.load(open(os.path.join(
        PROJECT_ROOT, "data/manifests/h_pool_and_intrinsics.json")))
    rel = os.path.relpath(args.clip, os.path.join(PROJECT_ROOT, "videos/clips"))
    intr = manifest["intrinsics_by_clip"].get(rel, {})
    K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    dist = np.asarray(intr.get("dist", [0]*5), dtype=np.float64)

    print("Loading classifier pipeline ...")
    pipe = NewPipeline(device)
    h_tracker = HomographyTrackerLite()

    cap = cv2.VideoCapture(args.clip)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {n_total} frames")

    per_frame = []
    t0 = time.time()
    for fi in range(n_total):
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[1] != SRC_W: fr = cv2.resize(fr, (SRC_W, SRC_H))
        pp = pipe(fr, K, dist)
        if pp is None:
            per_frame.append({"H": None, "rmse": None, "method": "none"})
        else:
            corrs = pp["corrs"]
            res = h_tracker.update(corrs, frame_idx=fi)
            per_frame.append({
                "H": res["H"], "method": res["method"],
                "rmse": res["rmse_yd"],
            })
        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{n_total}] {time.time()-t0:.0f}s")
    cap.release()
    print(f"  pass 1 in {time.time()-t0:.0f}s")

    # Carry cutoff.
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
    print(f"  cutoff: {cutoff}/{n_total}")

    # temp_div on raw Hs.
    Hs_raw = [r["H"] for r in per_frame]
    temp_div = [None] * len(per_frame)
    for i in range(1, len(per_frame) - 1):
        if Hs_raw[i] is None or Hs_raw[i-1] is None or Hs_raw[i+1] is None: continue
        ci = proj_corners_field(Hs_raw[i])
        cavg = 0.5 * (proj_corners_field(Hs_raw[i-1]) + proj_corners_field(Hs_raw[i+1]))
        temp_div[i] = float(np.max(np.linalg.norm(ci - cavg, axis=1)))

    # Current red flag.
    def is_red(i):
        r = per_frame[i]
        if r["H"] is None: return True
        if r["rmse"] is not None and r["rmse"] > RMSE_THR_YD: return True
        if temp_div[i] is not None and temp_div[i] > TEMP_THR_YD: return True
        return False
    red_flags = [is_red(i) for i in range(len(per_frame))]
    red_idx = [i for i, r in enumerate(red_flags) if r]
    print(f"  current red flags: {len(red_idx)} frames "
          f"(threshold rmse>{RMSE_THR_YD} or temp_div>{TEMP_THR_YD})")

    # ---- Global SG smoothing over [0, cutoff) using only non-red frames ----
    # (matches the production pipeline's red replacement step.)
    valid_idx = [i for i in range(cutoff)
                 if Hs_raw[i] is not None and not red_flags[i]]
    Hs_valid = [Hs_raw[i] for i in valid_idx]
    print(f"  global SG: {len(Hs_valid)} non-red frames, "
          f"window={SG_WINDOW} poly={SG_POLY}")
    Hs_sg = smooth_hs(Hs_valid, window=SG_WINDOW, poly=SG_POLY)
    # Map back to full index range. For non-valid frames, leave None.
    H_sg_by_frame = [None] * len(per_frame)
    for k, gi in enumerate(valid_idx):
        H_sg_by_frame[gi] = Hs_sg[k]

    # Global residual: raw[i] vs H_sg_by_frame[i] (for valid_idx only).
    global_resid = [None] * len(per_frame)
    for i in valid_idx:
        global_resid[i] = corner_residual_yd(Hs_raw[i], H_sg_by_frame[i])

    # ---- Leave-one-out residuals over ALL frames where raw H exists ----
    # Build a contiguous sequence of frames with raw H (within [0, cutoff)).
    seq_idx = [i for i in range(cutoff) if Hs_raw[i] is not None]
    seq_H = [Hs_raw[i] for i in seq_idx]
    loo_resid = [None] * len(per_frame)
    for local_i, gi in enumerate(seq_idx):
        H_loo = loo_smooth_at(seq_H, local_i, half=3, poly=2)
        if H_loo is None: continue
        loo_resid[gi] = corner_residual_yd(Hs_raw[gi], H_loo)

    # ---- Stats ----
    def pct(arr, q):
        a = np.array([v for v in arr if v is not None])
        if len(a) == 0: return float("nan")
        return float(np.percentile(a, q))

    def show(name, arr):
        vals = [v for v in arr if v is not None]
        if not vals:
            print(f"  {name}: no data"); return
        a = np.array(vals)
        print(f"  {name}: n={len(a)}  median={pct(arr, 50):.3f}  "
              f"p90={pct(arr, 90):.3f}  p95={pct(arr, 95):.3f}  "
              f"p99={pct(arr, 99):.3f}  max={a.max():.3f}")

    print("\n── Residual distribution (yards) ──")
    show("global SG residual ", global_resid)
    show("leave-one-out resid ", loo_resid)

    def fmt(v):
        return "  None" if v is None else f"{v:.3f}"

    print("\n── Currently red-flagged frames ──")
    if not red_idx:
        print("  (none in this clip)")
    for i in red_idx:
        r = per_frame[i]
        print(f"  frame {i:4d}  method={r['method']:5s}  "
              f"rmse={fmt(r['rmse'])}  "
              f"temp_div={fmt(temp_div[i])}  "
              f"global_resid={fmt(global_resid[i])}  "
              f"loo_resid={fmt(loo_resid[i])}")

    # Top-10 highest LOO residuals overall.
    pairs = [(i, v) for i, v in enumerate(loo_resid) if v is not None]
    pairs.sort(key=lambda x: -x[1])
    print("\n── Top 10 frames by LOO residual ──")
    for i, v in pairs[:10]:
        r = per_frame[i]
        print(f"  frame {i:4d}  loo={v:.3f}  method={r['method']:5s}  "
              f"rmse={fmt(r['rmse'])}  temp_div={fmt(temp_div[i])}  "
              f"red_flag={red_flags[i]}")

    # Dump full per-frame data to JSON for offline analysis.
    out_data = {
        "clip": rel,
        "n_total": n_total,
        "cutoff": cutoff,
        "sg_window": SG_WINDOW,
        "sg_poly": SG_POLY,
        "rmse_thr_yd": RMSE_THR_YD,
        "temp_thr_yd": TEMP_THR_YD,
        "per_frame": [
            {
                "frame": i,
                "method": per_frame[i]["method"],
                "rmse": per_frame[i]["rmse"],
                "temp_div": temp_div[i],
                "global_resid_yd": global_resid[i],
                "loo_resid_yd": loo_resid[i],
                "red_flag": red_flags[i],
            }
            for i in range(len(per_frame))
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  → wrote {args.out}")


if __name__ == "__main__":
    main()
