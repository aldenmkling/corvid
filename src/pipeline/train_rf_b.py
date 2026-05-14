"""RF-B: per-frame transformer over number tokens, fusing v10 encoder
features + crop classifier logits.

Same goal as RF-A but adds 1 layer of self-attention across number tokens
within a frame. Lets each number's prediction be refined by considering
the other numbers' (encoder + crop) features in the same scene.

Why this might break the RF-A ceiling (~98%):
  - RF-A is pointwise: each number's classification ignores its neighbors'
    image content. If a number is ambiguous (partial occlusion), seeing
    that neighbors are confidently "20" and "40" pins it as "30".
  - The v10 encoder did cross-token attention WITHOUT seeing crop logits,
    so its number features encode geometric structure but not "we just
    learned this neighbor is a 30." RF-B re-attends post-fusion.

Architecture:
    Per number token: concat(enc_feat[d=96], crop_logits[d=9]) → d_model
    1-layer TransformerEncoder over number tokens (per-frame, padded)
    Linear → 9-class

Crop classifier: dsresnet10w (frozen, 14K params, 90.04% per-cluster).
Encoder: v10 Stage 1 (frozen, 0.41M params).
RF-B head: ~10-30K params depending on d_model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer_v2 import (    # noqa: E402
    cc_tokens_from_frame_v2, null_classifier, TYPE_NUM,
)
from cc_tokenizer_v3 import cc_tokens_from_frame_v3    # noqa: E402
from train_token_v6 import build_targets    # noqa: E402
from train_dense_regression import split_by_game    # noqa: E402
from train_h_set_regressor import h_pixel_to_norm    # noqa: E402
from model_token_v10 import TokenClassifyV10    # noqa: E402
from train_rf_a import (    # noqa: E402
    map_21class_to_painted, encoder_features, _crops_for_number_tokens,
    make_painted_logits_fn, N_PAINTED_CLASSES,
)


# ─────────────────────────────────────────────────────────────────────────────
# RF-B model
# ─────────────────────────────────────────────────────────────────────────────

class RFB(nn.Module):
    def __init__(self, d_enc: int, n_classes: int = N_PAINTED_CLASSES,
                 d_model: int = 64, n_heads: int = 4,
                 ffn_dim: int = 128, dropout: float = 0.1,
                 with_row: bool = False):
        super().__init__()
        self.embed = nn.Linear(d_enc + n_classes, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, n_classes)
        self.with_row = with_row
        if with_row:
            self.head_row = nn.Linear(d_model, 1)

    def forward(self, enc_feat: torch.Tensor,
                  crop_logits: torch.Tensor,
                  padding_mask: torch.Tensor):
        """
        enc_feat:    (B, N, d_enc)
        crop_logits: (B, N, n_classes)
        padding_mask: (B, N) bool — True = padded.
        Returns: (B, N, n_classes) class logits, or
                 (class_logits, row_logits) if with_row=True (row shape (B, N)).
        """
        x = torch.cat([enc_feat, crop_logits], dim=-1)
        x = self.embed(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        cls = self.head(x)
        if self.with_row:
            return cls, self.head_row(x).squeeze(-1)
        return cls


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame data: keep number tokens grouped by frame for transformer batching.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_per_frame(entries, cache_dir, intrinsics_by_clip,
                            encoder_model, crop_logits_fn, device, d_enc,
                            tokenizer="v2"):
    """Returns list of dicts, one per frame:
       enc: (M, d_enc) — number tokens' encoder features
       crop: (M, 9)
       gt:  (M,) int (0..8)
       gt_row: (M,) float (0=near, 1=far)
    Frames with 0 valid number tokens are skipped.
    """
    out = []
    t0 = time.time()
    for ei, e in enumerate(entries):
        cp = os.path.join(cache_dir, f"{e['id']}.npz")
        if not os.path.exists(cp):
            continue
        d = np.load(cp)
        masks = d["masks"].astype(np.float32)
        # Undistort masks to undistorted-pixel space (matches manifest H).
        intr = (intrinsics_by_clip or {}).get(e["clip"], {})
        K = np.asarray(intr.get("K", np.eye(3)), dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        dist_arr = np.asarray(intr.get("dist", [0, 0, 0, 0, 0]),
                                  dtype=np.float64)
        masks = cv2.undistort(masks, K, dist_arr)
        tok_fn = (cc_tokens_from_frame_v3 if tokenizer == "v3"
                   else cc_tokens_from_frame_v2)
        tokens_np, aux = tok_fn(masks, null_classifier, return_aux=True)
        if tokens_np.shape[0] == 0:
            continue
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = (type_idx == TYPE_NUM)
        if not is_num.any():
            continue

        H_pixel = np.array(e["H"], dtype=np.float64)
        H_norm_gt = h_pixel_to_norm(H_pixel)
        tokens_t = torch.from_numpy(tokens_np).unsqueeze(0)
        H_t = torch.from_numpy(H_norm_gt.astype(np.float32)).unsqueeze(0)
        gt_21, row_target = build_targets(tokens_t, H_t)
        gt_9 = map_21class_to_painted(gt_21[0])
        row_per_token = row_target[0]

        pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool)
        enc_feat = encoder_features(encoder_model, tokens_t.to(device),
                                       pad.to(device))[0]
        # Crops come from the tokenizer aux dict (one per number token,
        # same order as the trailing-N_num rows of `tokens_np`).
        crops = aux["num_crops"]
        if not crops:
            continue
        crop_logits_np = crop_logits_fn(crops)
        crop_logits = torch.from_numpy(crop_logits_np).float()

        num_indices = np.where(is_num)[0]
        # Filter to clusters with valid GT (in painted range).
        enc_keep, crop_keep, gt_keep, gt_row_keep = [], [], [], []
        for j, ti in enumerate(num_indices):
            cls9 = int(gt_9[ti].item())
            if cls9 < 0:
                continue
            enc_keep.append(enc_feat[ti].cpu())
            crop_keep.append(crop_logits[j])
            gt_keep.append(cls9)
            gt_row_keep.append(float(row_per_token[ti].item()))
        if not gt_keep:
            continue
        out.append(dict(
            enc=torch.stack(enc_keep),
            crop=torch.stack(crop_keep),
            gt=torch.tensor(gt_keep, dtype=torch.long),
            gt_row=torch.tensor(gt_row_keep, dtype=torch.float32),
        ))

        if (ei + 1) % 200 == 0:
            print(f"  [{ei+1}/{len(entries)}]  frames={len(out)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"  total frames with numbers: {len(out)}  ({time.time()-t0:.1f}s)")
    return out


def collate_frames(batch):
    """Pad variable-length number-token lists per frame to per-batch max."""
    max_n = max(item["enc"].shape[0] for item in batch)
    B = len(batch)
    d_enc = batch[0]["enc"].shape[1]
    n_classes = batch[0]["crop"].shape[1]
    enc = torch.zeros(B, max_n, d_enc)
    crop = torch.zeros(B, max_n, n_classes)
    gt = torch.full((B, max_n), -1, dtype=torch.long)
    gt_row = torch.zeros(B, max_n)
    pad = torch.ones(B, max_n, dtype=torch.bool)
    has_row = "gt_row" in batch[0]
    for i, item in enumerate(batch):
        n = item["enc"].shape[0]
        enc[i, :n] = item["enc"]
        crop[i, :n] = item["crop"]
        gt[i, :n] = item["gt"]
        if has_row:
            gt_row[i, :n] = item["gt_row"]
        pad[i, :n] = False
    return dict(enc=enc, crop=crop, gt=gt, gt_row=gt_row, padding_mask=pad)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v10_stage1_gt_val/best.pth"))
    ap.add_argument("--crop-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/dsresnet10w_round3/best.pth"))
    ap.add_argument("--crop-arch", default="dsresnet10w")
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/rf_b_dsresnet"))
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-game", default="2024090802")
    ap.add_argument("--tokenizer", default="v2", choices=["v2", "v3"])
    ap.add_argument("--with-row", action="store_true",
                    help="Add row prediction head + BCE loss.")
    ap.add_argument("--row-weight", type=float, default=1.0,
                    help="Weight on row BCE relative to class CE.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load v10 encoder
    print(f"Loading v10 encoder from {args.v10_ckpt}...")
    v10_ck = torch.load(args.v10_ckpt, map_location="cpu", weights_only=False)
    v10_args = v10_ck["args"]
    encoder_model = TokenClassifyV10(
        n_layers=v10_args["n_layers"], n_heads=v10_args["n_heads"],
        d_model=v10_args["d_model"], ffn_dim=v10_args["ffn_dim"],
        dropout=0.0, token_dropout=0.0)
    encoder_model.load_state_dict(v10_ck["model_state_dict"])
    encoder_model.eval().to(device)
    for p in encoder_model.parameters():
        p.requires_grad = False
    d_enc = v10_args["d_model"]

    # Load crop classifier
    print(f"Loading crop classifier {args.crop_arch}...")
    crop_logits_fn = make_painted_logits_fn(
        args.crop_ckpt, args.crop_arch, device)

    # Manifest
    print("Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr = m.get("intrinsics_by_clip", {})
    train_e, val_e, _ = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)

    print(f"Building train cache (tokenizer={args.tokenizer})...")
    train_data = precompute_per_frame(
        train_e, args.cache_dir, intr, encoder_model, crop_logits_fn,
        device, d_enc, tokenizer=args.tokenizer)
    print("Building val cache...")
    val_data = precompute_per_frame(
        val_e, args.cache_dir, intr, encoder_model, crop_logits_fn,
        device, d_enc, tokenizer=args.tokenizer)

    # Crop-only baseline
    val_crop_correct = val_crop_total = 0
    for f in val_data:
        pred = f["crop"].argmax(-1)
        val_crop_correct += int((pred == f["gt"]).sum())
        val_crop_total += int(f["gt"].numel())
    crop_acc = val_crop_correct / max(val_crop_total, 1)
    print(f"\nCrop-only baseline (val): {crop_acc*100:.2f}% "
          f"({val_crop_correct}/{val_crop_total})")

    # Model
    rfb = RFB(d_enc=d_enc, d_model=args.d_model, n_heads=args.n_heads,
                ffn_dim=args.ffn_dim, dropout=args.dropout,
                with_row=args.with_row).to(device)
    n_params = sum(p.numel() for p in rfb.parameters())
    print(f"RFB: {n_params:,} params  ({n_params/1e3:.2f}K)  "
          f"d_model={args.d_model} n_heads={args.n_heads}  "
          f"with_row={args.with_row}")

    optim = torch.optim.AdamW(rfb.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01)

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val = 0.0

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_frames)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_frames)

    def _forward(batch_):
        enc = batch_["enc"].to(device)
        crop = batch_["crop"].to(device)
        pad = batch_["padding_mask"].to(device)
        if args.with_row:
            cls_l, row_l = rfb(enc, crop, pad)
            return cls_l, row_l
        return rfb(enc, crop, pad), None

    for epoch in range(args.epochs):
        rfb.train()
        train_loss = 0.0; n_b = 0; train_corr = 0; train_row_corr = 0
        for batch in train_loader:
            gt = batch["gt"].to(device)
            gt_row = batch["gt_row"].to(device)
            cls_logits, row_logits = _forward(batch)
            flat_logits = cls_logits.reshape(-1, N_PAINTED_CLASSES)
            flat_gt = gt.reshape(-1)
            valid = flat_gt >= 0
            if not valid.any():
                continue
            loss_cls = F.cross_entropy(flat_logits[valid], flat_gt[valid])
            loss = loss_cls
            if args.with_row:
                flat_row_logits = row_logits.reshape(-1)
                flat_row_gt = gt_row.reshape(-1)
                loss_row = F.binary_cross_entropy_with_logits(
                    flat_row_logits[valid], flat_row_gt[valid])
                loss = loss_cls + args.row_weight * loss_row
                train_row_corr += int(
                    ((flat_row_logits[valid] > 0).float()
                     == flat_row_gt[valid]).sum())
            optim.zero_grad(); loss.backward(); optim.step()
            train_loss += loss.item() * valid.sum().item()
            n_b += int(valid.sum())
            train_corr += int(
                (flat_logits[valid].argmax(-1) == flat_gt[valid]).sum())
        sched.step()
        train_loss /= max(n_b, 1)
        train_acc = train_corr / max(n_b, 1)
        train_row_acc = train_row_corr / max(n_b, 1) if args.with_row else 0.0

        rfb.eval()
        val_corr = val_tot = val_row_corr = 0
        with torch.no_grad():
            for batch in val_loader:
                gt = batch["gt"].to(device)
                gt_row = batch["gt_row"].to(device)
                cls_logits, row_logits = _forward(batch)
                flat_logits = cls_logits.reshape(-1, N_PAINTED_CLASSES)
                flat_gt = gt.reshape(-1)
                valid = flat_gt >= 0
                if valid.any():
                    val_corr += int(
                        (flat_logits[valid].argmax(-1) == flat_gt[valid]).sum())
                    val_tot += int(valid.sum())
                    if args.with_row:
                        flat_row_logits = row_logits.reshape(-1)
                        flat_row_gt = gt_row.reshape(-1)
                        val_row_corr += int(
                            ((flat_row_logits[valid] > 0).float()
                             == flat_row_gt[valid]).sum())
        val_acc = val_corr / max(val_tot, 1)
        val_row_acc = val_row_corr / max(val_tot, 1) if args.with_row else 0.0

        row_str = ""
        if args.with_row:
            row_str = (f"  row: train={train_row_acc*100:.2f}% "
                       f"val={val_row_acc*100:.2f}%")
        print(f"Ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"train={train_acc*100:.2f}%  val={val_acc*100:.2f}%  "
              f"crop={crop_acc*100:.2f}%{row_str}", flush=True)
        with open(log_path, "a") as f:
            json.dump({"epoch": epoch+1, "train_loss": train_loss,
                          "train_acc": train_acc, "val_acc": val_acc,
                          "train_row_acc": train_row_acc,
                          "val_row_acc": val_row_acc,
                          "crop_baseline": crop_acc}, f); f.write("\n")
        ckpt = {"model_state_dict": rfb.state_dict(), "epoch": epoch+1,
                "args": vars(args), "val_acc": val_acc,
                "val_row_acc": val_row_acc}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best val_acc = {val_acc*100:.2f}%", flush=True)

    print(f"\nDone. RF-B best: {best_val*100:.2f}%  "
          f"(crop alone {crop_acc*100:.2f}%, Δ {(best_val-crop_acc)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
