"""LOO red-flag viz for play_065.

Stacked output:
  TOP    — undistorted source frame with classifier corrs (color-coded by
           kind: hash=cyan, sideline=orange, number=yellow)
  BOTTOM — rectified canvas (using RAW per-frame H, no smoothing) so we
           can see the wobble that the LOO catches

A red border + "RED FLAG (LOO=X.XX yd)" overlay is drawn whenever the
LOO residual exceeds --thr (default 0.20 yd). This is the ONLY red-flag
signal — no rmse, no temp_div, no other thresholds.

LOO residual: at each frame, fit a degree-2 polynomial per H-coefficient
through the 3 frames on each side (excluding the frame itself), evaluate
at the frame, measure max corner reprojection distance vs the raw H in
NGS yards.

Saves to output/play_065_visuals/loo_redflag.mp4.
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
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)

UH, UW = 512, 896
IMM = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CANVAS_YDS_PER_PX = 1 / 8
CANVAS_W = int(FIELD_LENGTH / CANVAS_YDS_PER_PX)
CANVAS_H = int(FIELD_WIDTH / CANVAS_YDS_PER_PX)

CARRY_STREAK_LOST = 3

KIND_COLORS_BGR = {
    "near_hash":     (255, 200, 0),
    "far_hash":      (255, 200, 0),
    "sideline_near": (0, 165, 255),
    "sideline_far":  (0, 165, 255),
    "number_near":   (0, 255, 255),
    "number_far":    (0, 255, 255),
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
    pts = np.array([[100, 100], [SRC_W - 100, 100],
                    [SRC_W - 100, SRC_H - 100], [100, SRC_H - 100]],
                   dtype=np.float64)
    p = (H @ np.column_stack([pts, np.ones(4)]).T).T
    z = p[:, 2:3]; z[np.abs(z) < 1e-9] = 1e-9
    return p[:, :2] / z


def corner_residual_yd(H_a, H_b):
    if H_a is None or H_b is None: return None
    ca = proj_corners_field(H_a)
    cb = proj_corners_field(H_b)
    return float(np.max(np.linalg.norm(ca - cb, axis=1)))


def loo_smooth_at(H_seq, i, half=3, poly=2):
    """Degree-`poly` poly per H coefficient through {i-half..i+half}\\{i}."""
    n = len(H_seq)
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    xs = [j for j in range(lo, hi) if j != i]
    if len(xs) < poly + 1:
        return None
    ys = np.stack([H_seq[j].flatten() for j in xs], axis=0)
    xs_a = np.array(xs, dtype=np.float64)
    H_loo = np.zeros(9, dtype=np.float64)
    for c in range(9):
        coef = np.polyfit(xs_a, ys[:, c], poly)
        H_loo[c] = np.polyval(coef, i)
    return H_loo.reshape(3, 3)


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
    return cv2.addWeighted(warped, 0.7, grid, 0.3, 0)


def stack(top, bot):
    h_t, w_t = top.shape[:2]
    h_b, w_b = bot.shape[:2]
    if w_t != w_b:
        bot = cv2.resize(bot, (w_t, int(h_b * w_t / w_b)))
        h_b, w_b = bot.shape[:2]
    out = np.zeros((h_t + h_b, w_t, 3), dtype=np.uint8)
    out[:h_t] = top; out[h_t:] = bot
    return out


def draw_red_flag(img, loo_val, thr):
    """Border + banner — top-left of the image."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    msg = f"RED FLAG (LOO={loo_val:.2f} yd, thr={thr:.2f})"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.rectangle(img, (15, 15), (15 + tw + 24, 15 + th + 24),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (15, 15), (15 + tw + 24, 15 + th + 24),
                  (0, 0, 255), 2)
    cv2.putText(img, msg, (27, 15 + th + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--thr", type=float, default=0.20,
                    help="LOO residual threshold in NGS yards")
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
    h_tracker = HomographyTrackerLite()

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {n_total} frames @ {fps:.1f} fps")

    frames_u = []
    per_frame = []
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
                              "corrs": corrs})
        if (fi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{fi+1}/{n_total}] {dt:.0f}s ({(fi+1)/dt:.1f} fps)")
    cap.release()
    print(f"  pass 1 done in {time.time()-t0:.0f}s")

    # Cutoff (≥3 consecutive carries → clip lost).
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

    # LOO residuals over frames where raw H exists (within [0, cutoff)).
    seq_idx = [i for i in range(cutoff)
               if per_frame[i] is not None and per_frame[i]["H"] is not None]
    seq_H = [per_frame[i]["H"] for i in seq_idx]
    loo_resid = [None] * len(per_frame)
    for local_i, gi in enumerate(seq_idx):
        H_loo = loo_smooth_at(seq_H, local_i, half=3, poly=2)
        if H_loo is None: continue
        loo_resid[gi] = corner_residual_yd(per_frame[gi]["H"], H_loo)

    red = [(v is not None and v > args.thr) for v in loo_resid]
    n_red = sum(red)
    print(f"  red flags (LOO > {args.thr:.2f} yd): {n_red} / {cutoff}")

    # Pass 2: render.
    out_path = os.path.join(out_dir, "loo_redflag.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    n_render = min(cutoff, len(frames_u))
    print(f"[pass 2/2] rendering {n_render} frames -> {out_path}")
    t0 = time.time()
    for fi in range(n_render):
        fr_u = frames_u[fi]
        rec = per_frame[fi]

        # TOP — source + corrs.
        top = fr_u.copy()
        if rec is not None and rec["corrs"]:
            for c in rec["corrs"]:
                x, y = int(c["pixel_u"][0]), int(c["pixel_u"][1])
                col = KIND_COLORS_BGR.get(c["kind"], (0, 255, 0))
                cv2.circle(top, (x, y), 6, col, -1, cv2.LINE_AA)
                cv2.circle(top, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)

        # BOTTOM — rectified canvas (raw H, no smoothing).
        if rec is None or rec["H"] is None:
            bot = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        else:
            bot = render_rectified(fr_u, rec["H"])

        canvas = stack(top, bot)
        if red[fi]:
            draw_red_flag(canvas, loo_resid[fi], args.thr)

        # HUD: frame + LOO value
        hud_loo = "None" if loo_resid[fi] is None else f"{loo_resid[fi]:.3f}"
        cv2.putText(canvas, f"frame {fi}/{n_render}  LOO={hud_loo} yd",
                    (10, canvas.shape[0] - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {fi}/{n_render}  LOO={hud_loo} yd",
                    (10, canvas.shape[0] - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(
                out_path, fourcc, fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

    if writer is not None: writer.release()
    print(f"  pass 2 done in {time.time()-t0:.0f}s")
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
