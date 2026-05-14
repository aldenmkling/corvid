"""H regression via set transformer over connected-component tokens.

Architecture:
  - Tokenize v8 specialist masks via cc_tokenizer (one token per real-world
    feature: each painted yardline, sideline segment, hash mark, painted
    yardline number).
  - Each token has 16 features: type, centroid, bbox, log_area,
    orientation, NGS_x label, has_NGS flag, confidence.
  - Token embedding MLP: 16 → 128.
  - Sinusoidal positional encoding from centroid.
  - Transformer encoder: 4 layers × 4 heads, d_model=128.
  - Learnable [CLS] token prepended; final FC head reads CLS → 8 H_norm
    elements (h22 fixed=1).
  - Variable token count per frame (per-batch dynamic padding).

Loss: MSE on the 8 free elements of the normalized homography matrix.

Phase 5b — replaces the failed Phase 5a v3/v4 (mit_b0 + global pool head).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer import (    # noqa: E402
    cc_tokens_from_frame, TOKEN_FEATURE_DIM, SRC_W, SRC_H,
)
from train_dense_regression import NGS_X_MAX, NGS_Y_MAX, split_by_game  # noqa: E402

# Constants for normalization (must match train_h_regressor.py)
_D_NGS = np.diag([1.0 / NGS_X_MAX, 1.0 / NGS_Y_MAX, 1.0])
_D_PIXEL_INV = np.diag([SRC_W, SRC_H, 1.0])


def h_pixel_to_norm(H_pixel: np.ndarray) -> np.ndarray:
    """Convert pixel-space H to normalized H (h22 = 1)."""
    H_n = _D_NGS @ H_pixel @ _D_PIXEL_INV
    return H_n / H_n[2, 2]


# ── Sinusoidal positional encoding from a 2D centroid ──
def make_2d_pos_enc(cx: torch.Tensor, cy: torch.Tensor,
                     dim: int = 128) -> torch.Tensor:
    """Args:
        cx, cy: (B, N) tensors in [0, 1].
        dim:    output dimension; must be a multiple of 4.
    Returns:
        (B, N, dim) sinusoidal positional encoding.
    """
    assert dim % 4 == 0, "PE dim must be divisible by 4"
    quarter = dim // 4
    div = torch.exp(
        torch.arange(0, quarter, device=cx.device, dtype=cx.dtype)
        * -(math.log(10000.0) / quarter))
    # Scale cx, cy by 2π so they cycle within [0, 1] range
    cx_t = cx.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    cy_t = cy.unsqueeze(-1) * 2.0 * math.pi * 100.0 * div
    pe_x = torch.cat([torch.sin(cx_t), torch.cos(cx_t)], dim=-1)
    pe_y = torch.cat([torch.sin(cy_t), torch.cos(cy_t)], dim=-1)
    return torch.cat([pe_x, pe_y], dim=-1)


# ── Model ──
class HSetRegressor(nn.Module):
    """Set-transformer for homography regression.

    forward(tokens, padding_mask) → H_pred: (B, 3, 3)

    tokens:        (B, N_max, 16)   raw CC features
    padding_mask:  (B, N_max)       True where padded
    """
    def __init__(self, n_layers: int = 4, n_heads: int = 4,
                 d_model: int = 128, ffn_dim: int = 256,
                 dropout: float = 0.1, mean_h_norm: np.ndarray = None):
        super().__init__()
        self.d_model = d_model

        # Token feature embedding: 16 → d_model via 2-layer MLP
        self.token_embed = nn.Sequential(
            nn.Linear(TOKEN_FEATURE_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Learnable [CLS] token (prepended at position 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Final FC head: CLS feature → 8 H_norm elements
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

        # Mean-H init: bias the head's last linear so that at init the
        # model predicts the dataset's average H.
        nn.init.normal_(self.head[-1].weight, std=0.001)
        with torch.no_grad():
            if mean_h_norm is not None:
                bias = torch.from_numpy(
                    mean_h_norm.flatten()[:8].astype(np.float32))
            else:
                bias = torch.zeros(8)
                bias[0] = 1.0; bias[4] = 1.0
            self.head[-1].bias.copy_(bias)

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor) -> torch.Tensor:
        B, N_max, _ = tokens.shape

        # Embed token features
        feat = self.token_embed(tokens)             # (B, N, d_model)

        # Add positional encoding from centroid (features 4 and 5)
        cx = tokens[..., 4]
        cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        # Zero out PE at padded positions (so gradient/output is clean)
        pe = pe * (~padding_mask).unsqueeze(-1).to(pe.dtype)
        feat = feat + pe

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        x = torch.cat([cls, feat], dim=1)           # (B, 1+N, d_model)

        # Extend padding mask: CLS is always valid
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer encoder
        out = self.encoder(x, src_key_padding_mask=full_mask)

        # Read CLS output, predict 8 H elements
        cls_out = out[:, 0]                         # (B, d_model)
        h_params = self.head(cls_out)               # (B, 8)

        # Build (B, 3, 3) H matrix with h22 = 1
        ones = torch.ones(B, 1, device=tokens.device, dtype=h_params.dtype)
        flat = torch.cat([h_params, ones], dim=1)   # (B, 9)
        return flat.view(B, 3, 3)


# ── Dataset ──
class HSetDataset(Dataset):
    """Returns (tokens, padding_mask=None as placeholder, H_norm_gt) per item.
    The collator handles per-batch padding.

    To compute val L1 in yards we also need (gt, valid, und_grid) for the
    pixel-wise reprojection check. We compute these on demand from
    cached masks + manifest H + per-clip undistort grid.
    """
    def __init__(self, entries, cache_dir, v2_input_dir,
                 intrinsics_by_clip):
        self.entries = entries
        self.cache_dir = cache_dir
        self.v2_input_dir = v2_input_dir
        self.intrinsics_by_clip = intrinsics_by_clip or {}
        # Pre-compute per-clip undistort grids (for val L1 computation only)
        from train_dense_regression import make_undistort_grid
        self._und_grids = {}
        clips_used = {e["clip"] for e in entries}
        for clip in clips_used:
            intr = self.intrinsics_by_clip.get(clip)
            if not intr:
                continue
            key = self._und_key(intr["K"], intr["dist"])
            if key not in self._und_grids:
                self._und_grids[key] = make_undistort_grid(
                    intr["K"], intr["dist"])

    @staticmethod
    def _und_key(K, dist):
        K = np.asarray(K).flatten()
        dist = np.asarray(dist).flatten()
        return (round(float(K[0]), 4), round(float(K[2]), 4),
                round(float(K[5]), 4), round(float(dist[0]), 6))

    def _grid_for_clip(self, clip):
        intr = self.intrinsics_by_clip.get(clip)
        if not intr:
            return None
        return self._und_grids.get(self._und_key(intr["K"], intr["dist"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        cp = os.path.join(self.cache_dir, f"{e['id']}.npz")
        v2p = os.path.join(self.v2_input_dir, f"{e['id']}.npz")
        d = np.load(cp)
        d2 = np.load(v2p)
        masks = d["masks"].astype(np.float32)
        num_ngs_x = d2["number_ngs_x"].astype(np.float32)

        tokens = cc_tokens_from_frame(masks, num_ngs_x)        # (N, 16)
        if tokens.shape[0] == 0:
            # Defensive: shouldn't happen in our data, but handle gracefully
            tokens = np.zeros((1, TOKEN_FEATURE_DIM), dtype=np.float32)

        H_pixel = np.array(e["H"], dtype=np.float64)
        H_norm_gt = h_pixel_to_norm(H_pixel)

        und_grid = self._grid_for_clip(e["clip"])
        if und_grid is None:
            und_grid = np.zeros((176, 320, 2), dtype=np.float32)

        return {
            "tokens": torch.from_numpy(tokens),
            "H_norm_gt": torch.from_numpy(H_norm_gt.astype(np.float32)),
            "und_grid": torch.from_numpy(und_grid),
        }


def collate_set(batch):
    """Pad variable-length token sets to per-batch max."""
    n_max = max(item["tokens"].shape[0] for item in batch)
    B = len(batch)
    tokens = torch.zeros(B, n_max, TOKEN_FEATURE_DIM)
    pad_mask = torch.ones(B, n_max, dtype=torch.bool)   # True where padded
    H_gt = torch.zeros(B, 3, 3)
    und_grids = torch.zeros(B, 176, 320, 2)
    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        tokens[i, :n] = item["tokens"]
        pad_mask[i, :n] = False
        H_gt[i] = item["H_norm_gt"]
        und_grids[i] = item["und_grid"]
    return {
        "tokens": tokens,
        "padding_mask": pad_mask,
        "H_norm_gt": H_gt,
        "und_grid": und_grids,
    }


# ── Loss + metrics ──
def mse_h_8(H_pred: torch.Tensor, H_gt: torch.Tensor) -> torch.Tensor:
    """MSE on the 8 free elements of the homography (h22 fixed)."""
    pred_8 = torch.cat([
        H_pred[:, 0, :].reshape(-1, 3),
        H_pred[:, 1, :].reshape(-1, 3),
        H_pred[:, 2, :2].reshape(-1, 2),
    ], dim=1)
    gt_8 = torch.cat([
        H_gt[:, 0, :].reshape(-1, 3),
        H_gt[:, 1, :].reshape(-1, 3),
        H_gt[:, 2, :2].reshape(-1, 2),
    ], dim=1)
    return F.mse_loss(pred_8, gt_8)


def val_l1_yards(model, loader, device):
    """Per-pixel val L1 in NGS yards, computed via pred H @ undistort_grid.

    This is the project's metric of record (sub-yard target). Same semantic
    as our other models' val L1 — comparable across runs.
    """
    from train_dense_regression import OUTPUT_H, OUTPUT_W
    model.eval()
    sum_l1 = 0.0; sum_n = 0.0
    inlier_05 = 0.0; inlier_1 = 0.0
    h_mse_sum = 0.0; n_batches = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch["padding_mask"].to(device)
            H_gt = batch["H_norm_gt"].to(device)
            und_grid = batch["und_grid"].to(device)        # (B, H, W, 2) normalized
            # NOTE: und_grid stored in pixel space — normalize for use
            und_grid_norm = und_grid.clone()
            und_grid_norm[..., 0] /= SRC_W
            und_grid_norm[..., 1] /= SRC_H

            H_pred = model(tokens, mask)
            h_mse_sum += mse_h_8(H_pred, H_gt).item()
            n_batches += 1

            # Apply H_pred to und_grid_norm → pred NGS_norm
            ones = torch.ones_like(und_grid_norm[..., :1])
            pts = torch.cat([und_grid_norm, ones], dim=-1)   # (B, H, W, 3)
            field = torch.einsum("bij,bhwj->bhwi", H_pred, pts)
            pred_norm = field[..., :2] / field[..., 2:3].clamp(min=1e-6)
            # Denormalize to yards
            pred_yd_x = pred_norm[..., 0] * NGS_X_MAX
            pred_yd_y = pred_norm[..., 1] * NGS_Y_MAX
            # GT pixel-space NGS (compute via H_gt)
            gt_field = torch.einsum("bij,bhwj->bhwi", H_gt, pts)
            gt_norm = gt_field[..., :2] / gt_field[..., 2:3].clamp(min=1e-6)
            gt_yd_x = gt_norm[..., 0] * NGS_X_MAX
            gt_yd_y = gt_norm[..., 1] * NGS_Y_MAX
            # Validity: gt within field bounds
            valid = ((gt_yd_x >= 0) & (gt_yd_x <= NGS_X_MAX) &
                      (gt_yd_y >= 0) & (gt_yd_y <= NGS_Y_MAX)).float()
            err_x = torch.abs(pred_yd_x - gt_yd_x)
            err_y = torch.abs(pred_yd_y - gt_yd_y)
            err = err_x + err_y
            sum_l1 += (err * valid).sum().item()
            inlier_05 += (((err_x < 0.5) & (err_y < 0.5)).float() * valid
                           ).sum().item()
            inlier_1 += (((err_x < 1.0) & (err_y < 1.0)).float() * valid
                          ).sum().item()
            sum_n += valid.sum().item()
    if sum_n == 0:
        return {}
    return {
        "l1_yd": sum_l1 / sum_n,
        "inlier_lt_0.5yd": inlier_05 / sum_n,
        "inlier_lt_1yd": inlier_1 / sum_n,
        "h_mse": h_mse_sum / max(1, n_batches),
    }


def compute_mean_h_norm(entries):
    return np.stack(
        [h_pixel_to_norm(np.array(e["H"], dtype=np.float64))
         for e in entries]).mean(axis=0)


# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--v2-input-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression_v2_inputs"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/h_set_regressor_v1"))
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ffn-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16",
                    choices=["off", "bf16", "fp16"])
    ap.add_argument("--val-game", default="2024090802")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest_file}")
    m = json.load(open(args.manifest_file))
    intrinsics_by_clip = m.get("intrinsics_by_clip", {})
    if not intrinsics_by_clip:
        sys.exit("Manifest missing 'intrinsics_by_clip'.")

    train_entries, val_entries, val_game = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)
    print(f"Split: train={len(train_entries)}  val={len(val_entries)}  "
          f"(val game = {val_game})")

    train_ds = HSetDataset(train_entries, args.cache_dir, args.v2_input_dir,
                              intrinsics_by_clip)
    val_ds = HSetDataset(val_entries, args.cache_dir, args.v2_input_dir,
                            intrinsics_by_clip)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))

    # Compute mean H_norm for head bias init
    mean_h_norm = compute_mean_h_norm(train_entries)
    print(f"\nMean H_norm:")
    print(mean_h_norm)

    device = torch.device(args.device)
    model = HSetRegressor(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, ffn_dim=args.ffn_dim,
        dropout=args.dropout, mean_h_norm=mean_h_norm,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: HSetRegressor(L={args.n_layers}, H={args.n_heads}, "
          f"d={args.d_model}) — {n_params:.2f}M params")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    use_amp = (args.amp != "off") and (args.device == "cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    print(f"AMP: {args.amp}  LR: {args.lr} → {args.lr_min}")

    best_val_l1 = float("inf")
    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0; n_batches = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            H_gt = batch["H_norm_gt"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                H_pred = model(tokens, mask)
                loss = mse_h_8(H_pred, H_gt)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item(); n_batches += 1
        sched.step()
        train_loss = loss_sum / max(1, n_batches)
        elapsed = time.time() - t0

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            v = val_l1_yards(model, val_loader, device)
        print(f"Epoch {epoch+1:2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val L1={v.get('l1_yd', float('nan')):.3f}yd  "
              f"<.5yd={v.get('inlier_lt_0.5yd', 0)*100:.1f}%  "
              f"<1yd={v.get('inlier_lt_1yd', 0)*100:.1f}%  "
              f"h_mse={v.get('h_mse', float('nan')):.4f}  "
              f"({elapsed:.0f}s)", flush=True)

        with open(log_path, "a") as f:
            json.dump({"epoch": epoch+1, "train_loss": train_loss,
                       "lr": sched.get_last_lr()[0], **v}, f)
            f.write("\n")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1, "val_l1_yd": v.get("l1_yd"),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if v.get("l1_yd", float("inf")) < best_val_l1:
            best_val_l1 = v["l1_yd"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"  ↑ new best (val L1 = {best_val_l1:.3f}yd)")

    print(f"\nDone. Best val L1: {best_val_l1:.3f} yd")


if __name__ == "__main__":
    main()
