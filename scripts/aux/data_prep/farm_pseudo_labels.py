"""Farm pseudo-labels for self-supervised expansion of the training set.

Per frame:
  1. Run the NEW pipeline (UNet → tokenize → encoder → RFB → v10c)
     → tokens, per-token model predictions, H_new, signal stats.
  2. Run the CLASSICAL pipeline (UNet → line fits → number classifier →
     classical solve)  →  H_classical, signal stats.
  3. Apply the SAME multi-signal red-flag criteria to both pipelines.
  4. Pick a trusted H:
       prefer NEW if its flags are clean,
       fall back to CLASSICAL if its flags are clean,
       else skip this frame.
  5. Project every NEW-pipeline token's centroid through the trusted H
     → derive the TRUE class for that token (independent of what the
     model originally predicted).
  6. Write per-frame records to <out-dir>/<clip_id>.npz with:
       - tokens (N, 16)         the new-pipeline tokens (geometry)
       - true_class (N,)        H-projected NGS_x class (yard / hash / num
                                  snapped to even index)
       - true_row (N,)          H-projected near/far for side / hash / num
       - model_yard_cls (N,)    raw v10c pass2 NGS_x argmax (per yard tok)
       - model_num_cls (N,)     raw RFB-mapped NGS_x (per num tok)
       - model_row (N,)         raw v10c row prediction
       - H_used (3, 3)          the H we used to derive labels
       - h_source ("new"|"classical")
       - n_corrections          count of tokens whose model pred ≠ true

Yellow-flag corrections (bandaid catches mis-classified yardlines) are
captured automatically because the H is solved from the surviving fits;
the dropped token gets a NEW true label from H-projection here.

Red-flag frames where BOTH pipelines fail are skipped — there's no
reliable H to derive labels from.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

# Each worker is one process; on a pod with N vCPUs and N workers, each
# worker must single-thread its CPU work. OMP/MKL/OPENBLAS env vars are
# read at module import (numpy/torch); cv2 needs its own setter.
cv2.setNumThreads(1)
try:
    torch.set_num_threads(1)
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "data_prep"))
sys.path.insert(0, PROJECT_ROOT)

from src.field_mapping.tokenizer import (
    TYPE_NUM, TYPE_YARD, TYPE_SIDE, TYPE_HASH,
    SRC_W, SRC_H, null_classifier,
)
from src.field_mapping.tokenizer import tokenize_frame as cc_tokens_from_frame_v3
from src.field_mapping.encoder import TokenEncoder as TokenClassifyV10
from src.field_mapping.token_labeler import TokenLabeler as TokenClassifyV10b
from src.field_mapping.classes import N_NGS_X_CLASSES
from train_token_v10b_stage2 import PAINTED_TO_21
from src.field_mapping.crop_classifier import make_painted_logits_fn, encoder_features
from src.field_mapping.number_refiner import NumberRefiner as RFB
from src.field_mapping.token_labeler import rfb_forward_with_features_and_row
from src.field_mapping.keypoints import extract_keypoints
from src.field_mapping.homography import solve_h
from src.field_mapping.field_model import (
    FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
)
from src.homography.painted_numbers import (
    NGS_Y_NEAR_INSIDE, NGS_Y_FAR_INSIDE,
)
from src.homography.rectify import compute_homographies


UNIFIED_INPUT_H, UNIFIED_INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Red-flag thresholds (match viz_clip_flagged.py).
RMSE_THR_YD = 0.30
INLIER_FRAC_THR = 0.95
N_CORRS_THR = 12
N_YARD_FITS_THR = 5
TEMP_THR_YD = 1.5

# Painted (even) NGS_x classes the numbers can occupy.
PAINTED_CLASSES = list(range(2, 19, 2))


# ── UNet + new pipeline ────────────────────────────────────────────────────

def load_unified(weights_path, device):
    model = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                          in_channels=3, classes=4)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UNIFIED_INPUT_W, UNIFIED_INPUT_H))
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def predict_masks(model, frame_bgr, device):
    x = preprocess(frame_bgr).to(device)
    probs = torch.sigmoid(model(x))[0].cpu().numpy()
    h0, w0 = frame_bgr.shape[:2]
    out = np.zeros((h0, w0, 4), dtype=np.float32)
    for ci in range(4):
        out[..., ci] = cv2.resize(
            probs[ci], (w0, h0), interpolation=cv2.INTER_LINEAR)
    return out


def project_corners(H):
    pts = np.array([[100, 100], [SRC_W - 100, 100],
                       [SRC_W - 100, SRC_H - 100], [100, SRC_H - 100]],
                      dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(4)])
    p = (H @ pts_h.T).T
    z = p[:, 2:3]; z[np.abs(z) < 1e-9] = 1e-9
    return p[:, :2] / z


def red_flags(rmse, inlier_frac, n_corrs, n_yard_fits, temp_div):
    """Return list of flag strings (empty = clean)."""
    f = []
    if rmse is not None and rmse > RMSE_THR_YD:
        f.append(f"rmse={rmse:.2f}")
    if inlier_frac is not None and inlier_frac < INLIER_FRAC_THR:
        f.append(f"inl={inlier_frac:.2f}")
    if n_corrs < N_CORRS_THR:
        f.append(f"corrs={n_corrs}")
    if n_yard_fits is not None and n_yard_fits < N_YARD_FITS_THR:
        f.append(f"yfits={n_yard_fits}")
    if temp_div is not None and temp_div > TEMP_THR_YD:
        f.append(f"temp={temp_div:.1f}")
    return f


# ── Snap projected NGS coords to valid classes per token type ─────────────

def snap_yardline_or_hash(ngs_x_yd):
    """Snap a projected NGS_x (yards) to the nearest 5y class index in
    [0, 20]. Returns -1 if out of bounds."""
    cls = int(round((ngs_x_yd - 10.0) / 5.0))
    if cls < 0 or cls > 20:
        return -1
    return cls


def snap_number_painted(ngs_x_yd):
    """Snap a projected NGS_x (yards) to the nearest PAINTED yardline
    class (even index in [2, 18]). Returns -1 if out of bounds."""
    cls = int(round((ngs_x_yd - 10.0) / 5.0))
    if cls < 2 or cls > 18:
        return -1
    if cls % 2 == 1:
        # Odd → pick the nearer even neighbor.
        cls += 1 if (ngs_x_yd > 10 + cls * 5) else -1
    return cls if (cls >= 2 and cls <= 18 and cls % 2 == 0) else -1


def snap_side_row(ngs_y_yd):
    return 0 if ngs_y_yd < FIELD_WIDTH * 0.5 else 1


def snap_hash_row(ngs_y_yd):
    return 0 if abs(ngs_y_yd - HASH_Y_NEAR) < abs(ngs_y_yd - HASH_Y_FAR) else 1


def snap_number_row(ngs_y_yd):
    return 0 if abs(ngs_y_yd - NGS_Y_NEAR_INSIDE) < abs(ngs_y_yd - NGS_Y_FAR_INSIDE) else 1


# ── Per-frame new-pipeline runner ──────────────────────────────────────────

class NewPipeline:
    def __init__(self, args, device):
        self.device = device
        self.unified = load_unified(args.unified_weights, device)
        s1_ck = torch.load(args.stage1_ckpt, map_location="cpu",
                                  weights_only=False)
        self.sa = s1_ck["args"]
        self.encoder = TokenClassifyV10(
            n_layers=self.sa["n_layers"], n_heads=self.sa["n_heads"],
            d_model=self.sa["d_model"], ffn_dim=self.sa["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.encoder.load_state_dict(s1_ck["model_state_dict"])
        v10c_ck = torch.load(args.v10c_ckpt, map_location="cpu",
                                  weights_only=False)
        self.v10c = TokenClassifyV10b(
            n_layers=self.sa["n_layers"], n_heads=self.sa["n_heads"],
            d_model=self.sa["d_model"], ffn_dim=self.sa["ffn_dim"],
            dropout=0.0, token_dropout=0.0).to(device).eval()
        self.v10c.load_state_dict(v10c_ck["model_state_dict"])
        rfb_ck = torch.load(args.rfb_ckpt, map_location="cpu",
                                  weights_only=False)
        ra = rfb_ck["args"]
        self.rfb = RFB(d_enc=self.sa["d_model"], d_model=ra["d_model"],
                              n_heads=ra["n_heads"], ffn_dim=ra["ffn_dim"],
                              dropout=0.0, with_row=True).to(device).eval()
        self.rfb.load_state_dict(rfb_ck["model_state_dict"])
        self.crop_fn = make_painted_logits_fn(
            args.crop_ckpt, args.crop_arch, device)

    def __call__(self, frame, K, dist):
        """Returns dict with tokens, model predictions, H, signals."""
        masks_d = predict_masks(self.unified, frame, self.device)
        masks = cv2.undistort(masks_d.astype(np.float32), K, dist)
        tokens_np, aux = cc_tokens_from_frame_v3(
            masks, null_classifier, return_aux=True)
        if tokens_np.shape[0] == 0:
            return None
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = type_idx == TYPE_NUM
        is_yard = type_idx == TYPE_YARD
        is_side = type_idx == TYPE_SIDE
        is_hash = type_idx == TYPE_HASH
        tokens_t = torch.from_numpy(tokens_np).unsqueeze(0).to(self.device)
        pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool,
                                device=self.device)
        with torch.no_grad():
            enc_feat = encoder_features(self.encoder, tokens_t, pad)[0]
        nac = np.full(tokens_np.shape[0], -1, dtype=np.int64)
        nar = np.zeros(tokens_np.shape[0], dtype=np.float32)
        rfb_pre_full = torch.zeros(1, tokens_np.shape[0], self.sa["d_model"],
                                          device=self.device)
        if is_num.any() and aux["num_crops"]:
            ni = np.where(is_num)[0]
            cl = torch.from_numpy(self.crop_fn(
                aux["num_crops"])).float().to(self.device)
            pad_n = torch.zeros(1, len(ni), dtype=torch.bool,
                                       device=self.device)
            with torch.no_grad():
                rl, rr, rpp = rfb_forward_with_features_and_row(
                    self.rfb, enc_feat[ni].unsqueeze(0),
                    cl.unsqueeze(0), pad_n)
            pp = rl[0].argmax(-1).cpu().numpy()
            p21 = PAINTED_TO_21.numpy()[pp]
            pr = torch.sigmoid(rr[0]).cpu().numpy()
            for j, ti in enumerate(ni):
                nac[ti] = int(p21[j])
                nar[ti] = float(pr[j])
                rfb_pre_full[0, ti] = rpp[0, j]
        nca_t = torch.from_numpy(nac).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.v10c(tokens_t, pad, num_class_gt=nca_t,
                                num_rfb_features=rfb_pre_full)
            p2 = out["logits_pass2"][0]
        ngs_cls = p2[:, :N_NGS_X_CLASSES].argmax(-1).cpu().numpy()
        rows = (p2[:, N_NGS_X_CLASSES] > 0).cpu().numpy().astype(int)

        corrs, fits = extract_keypoints(
            pixel_sets=aux["pixel_sets"],
            yard_classes=ngs_cls[is_yard].tolist(),
            side_rows=rows[is_side].tolist(),
            hash_classes=ngs_cls[is_hash].tolist(),
            hash_rows=rows[is_hash].tolist(),
            num_classes=nac[is_num].tolist(),
            num_rows=(nar[is_num] > 0.5).astype(int).tolist(),
            yard_mask=masks[..., 0],
        )
        H, inl, rmse = solve_h(corrs)
        n_corrs = len(corrs)
        inlier_frac = (float(inl.sum() / max(1, n_corrs))
                            if inl is not None else None)
        return {
            "tokens": tokens_np,
            "type_idx": type_idx,
            "model_ngs_cls": ngs_cls,           # for yard / hash
            "model_num_cls": nac,                # for num (RFB)
            "model_row": rows,                   # for side / hash / num row
            "model_num_row_sig": nar,            # raw RFB sigmoid for num row
            "n_yard_fits": len(fits["yard_fits"]),
            "yard_drops": fits["yard_fit_drops"],
            "H": H,
            "rmse": rmse,
            "inlier_frac": inlier_frac,
            "n_corrs": n_corrs,
        }


# ── Snap a token's projected NGS coords to its valid label ────────────────

def derive_true_labels(tokens_np, type_idx, H_trusted):
    """For each token, project centroid through H_trusted, snap to a
    valid class for the token type. Returns:
        true_class (N,) int  — NGS_x class for yard/hash/num, row(0/1) for side
        true_row (N,) int    — row class for side/hash/num (irrelevant for yard)
    Returns -1 in true_class for any token whose projection is out of
    valid range.
    """
    N = tokens_np.shape[0]
    true_class = np.full(N, -1, dtype=np.int64)
    true_row = np.full(N, -1, dtype=np.int64)

    cx = tokens_np[..., 4] * SRC_W
    cy = tokens_np[..., 5] * SRC_H
    pts = np.stack([cx, cy, np.ones_like(cx)], axis=-1)
    proj = (H_trusted @ pts.T).T
    z = proj[:, 2:3]; z[np.abs(z) < 1e-9] = 1e-9
    ngs = proj[:, :2] / z

    for i in range(N):
        gx, gy = float(ngs[i, 0]), float(ngs[i, 1])
        tt = int(type_idx[i])
        if tt == TYPE_YARD:
            true_class[i] = snap_yardline_or_hash(gx)
        elif tt == TYPE_HASH:
            true_class[i] = snap_yardline_or_hash(gx)
            true_row[i] = snap_hash_row(gy)
        elif tt == TYPE_NUM:
            true_class[i] = snap_number_painted(gx)
            true_row[i] = snap_number_row(gy)
        elif tt == TYPE_SIDE:
            true_row[i] = snap_side_row(gy)
    return true_class, true_row


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True,
                       help="path(s) to clip mp4 (repeatable)")
    ap.add_argument("--clips-dir", default="videos/clips")
    ap.add_argument("--manifest", default="data/manifests/h_pool_and_intrinsics.json")
    ap.add_argument("--out-dir", default="data/training/pseudo_labels")
    ap.add_argument("--stage1-ckpt", default="models/token_only_v10_stage1_v3unew_gt/best.pth")
    ap.add_argument("--rfb-ckpt", default="models/rf_b_v3unew_gt/best.pth")
    ap.add_argument("--crop-ckpt", default="models/dsresnet10ww_round3_128x32/best.pth")
    ap.add_argument("--crop-arch", default="dsresnet10ww")
    ap.add_argument("--v10c-ckpt", default="models/v10c_stage2_v3unew_gt/best.pth")
    ap.add_argument("--unified-weights", default="models/unet_unified_v8_yardside_recover/best.pth")
    ap.add_argument("--classifier-weights",
                       default="models/number_classifier_best.pth",
                       help="classical pipeline number classifier")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading new pipeline...")
    new_pipe = NewPipeline(args, device)

    m_manifest = json.load(open(args.manifest))
    intr_by_clip = m_manifest["intrinsics_by_clip"]

    for clip_path in args.clip:
        rel = os.path.relpath(os.path.abspath(clip_path),
                                    os.path.abspath(args.clips_dir))
        clip_id = rel.replace("/", "_").replace(".mp4", "")
        out_path = os.path.join(args.out_dir, f"{clip_id}.npz")
        if os.path.exists(out_path):
            print(f"[skip] {clip_id} (exists)")
            continue

        intr = intr_by_clip.get(rel, {})
        K = np.asarray(intr.get("K", np.eye(3).tolist()), dtype=np.float64)
        dist = np.asarray(intr.get("dist", [0] * 5), dtype=np.float64)
        print(f"\n=== {clip_id} ===  intr={'yes' if intr else 'no'}")

        # NEW pipeline first (cheaper, ~30s/clip).
        print("  running new pipeline...")
        cap = cv2.VideoCapture(clip_path)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(n_total, args.max_frames) if args.max_frames else n_total
        new_records = []
        t0 = time.time()
        for fi in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != SRC_W or frame.shape[0] != SRC_H:
                frame = cv2.resize(frame, (SRC_W, SRC_H))
            rec = new_pipe(frame, K, dist)
            new_records.append(rec)
            if (fi + 1) % 50 == 0:
                print(f"    [{fi+1}/{n}]  ({time.time()-t0:.0f}s)")
        cap.release()
        print(f"  new pipeline done ({time.time()-t0:.0f}s)")

        # Temporal divergence pass for the new H trajectory.
        def temp_div(Hs):
            div = [None] * len(Hs)
            for i in range(1, len(Hs) - 1):
                if Hs[i] is None or Hs[i-1] is None or Hs[i+1] is None:
                    continue
                ci = project_corners(Hs[i])
                cavg = 0.5 * (project_corners(Hs[i-1])
                                    + project_corners(Hs[i+1]))
                div[i] = float(np.max(np.linalg.norm(ci - cavg, axis=1)))
            return div
        new_Hs = [r["H"] if r is not None else None for r in new_records]
        temp_new = temp_div(new_Hs)

        # Flag new frames up front so we know whether classical is needed.
        new_flags = []
        for fi in range(len(new_records)):
            rec = new_records[fi]
            if rec is None or rec["H"] is None:
                new_flags.append(["no_H"])
                continue
            new_flags.append(red_flags(
                rec["rmse"], rec["inlier_frac"], rec["n_corrs"],
                rec["n_yard_fits"], temp_new[fi]))
        n_new_red = sum(1 for f in new_flags if f)

        # Only run classical if there's any chance it would be picked.
        # Each frame independently — if new is clean for ALL frames we
        # skip classical entirely (~2-3x speedup per clip).
        cls_Hs = [None] * len(new_records)
        cls_result = None
        temp_cls = [None] * len(new_records)
        if n_new_red == 0:
            print(f"  classical skipped (new clean for all {len(new_records)} frames)")
        else:
            print(f"  running classical (new has {n_new_red} red frames)...")
            t0 = time.time()
            cls_result = compute_homographies(
                video_path=clip_path,
                classifier_weights=args.classifier_weights,
                device=args.device,
                max_frames=args.max_frames,
                manual_g0=None,
                verbose=False,
            )
            if cls_result is None:
                print(f"  classical failed to start — using new where clean only")
                cls_result = None
            else:
                cls_Hs = cls_result["Hs"]
                temp_cls = temp_div(cls_Hs)
                print(f"  classical: {len(cls_Hs)} frames, methods="
                        f"{cls_result['method_counts']}  ({time.time()-t0:.0f}s)")

        # Decide H per frame + derive labels.
        per_frame_records = []
        n_kept_new = n_kept_cls = n_skipped = 0
        for fi in range(len(new_records)):
            rec = new_records[fi]
            if rec is None:
                n_skipped += 1
                continue
            flags_new = new_flags[fi]

            # Default to "no classical info"; only fill if we actually ran it.
            H_cls = cls_Hs[fi] if fi < len(cls_Hs) else None
            flags_cls = ["no_cls"]
            if cls_result is not None and fi < len(cls_Hs):
                cls_n_corrs = (cls_result["frame_meta"][fi].get("n_corrs", 0)
                                    if "frame_meta" in cls_result and fi < len(cls_result["frame_meta"])
                                    else 0)
                cls_n_inl = (cls_result["frame_meta"][fi].get("n_inliers", 0)
                                  if "frame_meta" in cls_result and fi < len(cls_result["frame_meta"])
                                  else 0)
                cls_rmse = (cls_result["frame_meta"][fi].get("rmse_yd")
                                if "frame_meta" in cls_result and fi < len(cls_result["frame_meta"])
                                else None)
                cls_inl_frac = (cls_n_inl / max(1, cls_n_corrs)
                                      if cls_n_corrs > 0 else None)
                flags_cls = red_flags(
                    cls_rmse, cls_inl_frac, cls_n_corrs,
                    n_yard_fits=None, temp_div=temp_cls[fi])

            if rec["H"] is not None and not flags_new:
                H_trusted = rec["H"]
                source = "new"
                n_kept_new += 1
            elif H_cls is not None and not flags_cls:
                H_trusted = H_cls
                source = "classical"
                n_kept_cls += 1
            else:
                n_skipped += 1
                continue

            true_cls, true_row = derive_true_labels(
                rec["tokens"], rec["type_idx"], H_trusted)
            # Compute model's "predicted" class per token (same convention
            # as true_class) to compute n_corrections.
            model_cls = np.full(rec["tokens"].shape[0], -1, dtype=np.int64)
            for i in range(rec["tokens"].shape[0]):
                tt = int(rec["type_idx"][i])
                if tt == TYPE_YARD or tt == TYPE_HASH:
                    model_cls[i] = int(rec["model_ngs_cls"][i])
                elif tt == TYPE_NUM:
                    model_cls[i] = int(rec["model_num_cls"][i])
            valid = (true_cls >= 0) & (model_cls >= 0)
            n_corrections = int((valid & (true_cls != model_cls)).sum())

            per_frame_records.append({
                "frame_idx": fi,
                "tokens": rec["tokens"].astype(np.float32),
                "type_idx": rec["type_idx"].astype(np.int8),
                "true_class": true_cls.astype(np.int16),
                "true_row": true_row.astype(np.int8),
                "model_ngs_cls": rec["model_ngs_cls"].astype(np.int16),
                "model_num_cls": rec["model_num_cls"].astype(np.int16),
                "model_row": rec["model_row"].astype(np.int8),
                "H_used": H_trusted.astype(np.float64),
                "h_source": source,
                "n_corrections": n_corrections,
            })

        print(f"  decided: kept-new={n_kept_new}  kept-cls={n_kept_cls}  "
                f"skipped={n_skipped}")
        # Save per-clip npz (list of per-frame dicts → np.savez).
        if not per_frame_records:
            print(f"  (no usable frames — not writing {out_path})")
            continue
        np.savez_compressed(
            out_path,
            frame_idx=np.array(
                [r["frame_idx"] for r in per_frame_records], dtype=np.int32),
            h_source=np.array(
                [r["h_source"] for r in per_frame_records], dtype=object),
            n_corrections=np.array(
                [r["n_corrections"] for r in per_frame_records],
                dtype=np.int32),
            tokens=np.array(
                [r["tokens"] for r in per_frame_records], dtype=object),
            type_idx=np.array(
                [r["type_idx"] for r in per_frame_records], dtype=object),
            true_class=np.array(
                [r["true_class"] for r in per_frame_records], dtype=object),
            true_row=np.array(
                [r["true_row"] for r in per_frame_records], dtype=object),
            model_ngs_cls=np.array(
                [r["model_ngs_cls"] for r in per_frame_records], dtype=object),
            model_num_cls=np.array(
                [r["model_num_cls"] for r in per_frame_records], dtype=object),
            model_row=np.array(
                [r["model_row"] for r in per_frame_records], dtype=object),
            H_used=np.array(
                [r["H_used"] for r in per_frame_records], dtype=np.float64),
        )
        print(f"  wrote {out_path}  "
                f"({len(per_frame_records)} usable frames, "
                f"{sum(r['n_corrections'] for r in per_frame_records)} "
                f"total token corrections)")


if __name__ == "__main__":
    main()
