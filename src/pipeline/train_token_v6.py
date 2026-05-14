"""v6: iterative cross-attention + raw cx,cy in cross-attn query.

Two changes from v5:
  1. Two-pass cross-attention.
     • Pass 1 (same as v5): query = encoder_out, anchors = number tokens.
       Yields intermediate per-token NGS_x predictions.
     • Pass 2: anchors = numbers (with their labels) + yards/hashes (with
       their pass-1 SOFT predicted NGS_x). Now every yard/hash that v5
       got right in pass 1 becomes a new anchor for the rest.

     This addresses the failure mode where v5 gets numbers correct
     but can't fully interpolate to yardlines: in pass 2, the visible
     yardlines themselves serve as additional anchors, breaking ties.

  2. Raw (cx, cy) added to the cross-attention query.
     Cross-attention queries currently come from the encoder output,
     which has been mixed across 4 attention layers — losing some of
     the per-token raw image position. We add a fresh 2D positional
     encoding to the query so Q-K matching can use image-x proximity.

Deep supervision: we apply the per-token classification loss to BOTH
passes. Otherwise the model could learn to ignore pass 1.

All other v5 ingredients kept: post-norm self-attention encoder,
token dropout 20%, weight decay 1e-3, 21-class NGS_x.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer import (    # noqa: E402
    TOKEN_FEATURE_DIM, TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM,
)
from train_dense_regression import NGS_X_MAX, NGS_Y_MAX, split_by_game  # noqa: E402
from train_h_set_regressor import (    # noqa: E402
    HSetDataset, collate_set, make_2d_pos_enc,
)
from h_pnl_set_regressor import (    # noqa: E402
    HASH_Y_NEAR_NORM, HASH_Y_FAR_NORM, NUM_Y_NEAR_NORM, NUM_Y_FAR_NORM,
)


NGS_X_CLASS_STEP = 5.0
NGS_X_CLASS_MIN = 10.0
NGS_X_CLASS_MAX = 110.0
N_NGS_X_CLASSES = int(round(
    (NGS_X_CLASS_MAX - NGS_X_CLASS_MIN) / NGS_X_CLASS_STEP)) + 1   # 21


def ngs_x_to_class(ngs_x_yards: torch.Tensor) -> torch.Tensor:
    idx = torch.round((ngs_x_yards - NGS_X_CLASS_MIN) / NGS_X_CLASS_STEP)
    return idx.long().clamp(0, N_NGS_X_CLASSES - 1)


# Class index → NGS_x in normalized [0, 1] (NGS_x / 120) — used for soft
# expected-value computation when feeding pass-1 predictions as pass-2 anchors.
def make_class_to_ngs_x_norm(device):
    classes = torch.arange(N_NGS_X_CLASSES, device=device, dtype=torch.float32)
    yards = NGS_X_CLASS_MIN + classes * NGS_X_CLASS_STEP            # (21,)
    return yards / NGS_X_MAX                                         # in [0, 1]


# ────────────────────────────────────────────────────────────────────────────
# Model.
# ────────────────────────────────────────────────────────────────────────────

class TokenClassifyV6(nn.Module):
    def __init__(self, n_layers: int = 4, n_heads: int = 4,
                 d_model: int = 128, ffn_dim: int = 256,
                 dropout: float = 0.1,
                 token_dropout: float = 0.2,
                 min_keep_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.token_dropout = token_dropout
        self.min_keep_tokens = min_keep_tokens

        self.token_embed = nn.Sequential(
            nn.Linear(TOKEN_FEATURE_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Anchor projection: (ngs_x_norm, cx, cy) → d_model.
        # SHARED between pass 1 and pass 2 (same input format).
        self.anchor_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )

        # Learned global anchor — always-valid extra key for empty-anchor
        # frames. SHARED between passes.
        self.global_anchor = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.global_anchor, std=0.02)

        # Two cross-attention blocks (separate weights — they do different
        # jobs: pass-1 reads only numbers, pass-2 reads everything).
        self.cross_attn_1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm_1 = nn.LayerNorm(d_model)
        self.cross_attn_2 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm_2 = nn.LayerNorm(d_model)

        # Head: shared between pass 1 and pass 2 outputs (same task)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, N_NGS_X_CLASSES + 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.zeros_(self.head[-1].bias)

    def _apply_token_dropout(self, padding_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout <= 0:
            return padding_mask
        valid = ~padding_mask
        drop = torch.rand(valid.shape, device=padding_mask.device) < \
                self.token_dropout
        drop = drop & valid
        new_pad = padding_mask | drop
        n_kept = (~new_pad).sum(dim=-1)
        safe = n_kept >= self.min_keep_tokens
        return torch.where(safe.unsqueeze(-1), new_pad, padding_mask)

    def _cross_attend(self, query, anchor_input, anchor_valid_mask,
                        attn_module, norm_module, raw_pe):
        """One cross-attention pass.

        query:               (B, N, d) — token features (encoder out / pass-1 out)
        anchor_input:        (B, N, 3) — (ngs_x_norm, cx, cy) per token
        anchor_valid_mask:   (B, N) — True for tokens that should serve as keys
        attn_module:         the MHA module to use
        norm_module:         the LayerNorm to apply after residual
        raw_pe:              (B, N, d) — fresh positional encoding to add to query

        Returns refined token features (B, N, d).
        """
        B, N, d = query.shape
        # Build keys/values from anchor input
        anchor_keys = self.anchor_proj(anchor_input)            # (B, N, d)
        # Append always-valid global anchor
        global_k = self.global_anchor.expand(B, -1, -1)         # (B, 1, d)
        full_kv = torch.cat([anchor_keys, global_k], dim=1)      # (B, N+1, d)
        cross_pad = torch.cat([
            ~anchor_valid_mask,                                  # True = invalid
            torch.zeros(B, 1, dtype=torch.bool, device=query.device),
        ], dim=1)

        # Add raw positional encoding to query (so Q-K can match by image-x)
        query_pos = query + raw_pe

        out, _ = attn_module(
            query=query_pos, key=full_kv, value=full_kv,
            key_padding_mask=cross_pad)
        return norm_module(query + out)                          # residual

    def forward(self, tokens: torch.Tensor,
                  padding_mask: torch.Tensor):
        """Returns dict with two sets of logits (pass-1, pass-2)."""
        B, N, _ = tokens.shape
        eff_padding = self._apply_token_dropout(padding_mask)
        valid = ~eff_padding

        # ── Token features ──
        feat = self.token_embed(tokens)
        cx = tokens[..., 4]; cy = tokens[..., 5]
        pe = make_2d_pos_enc(cx, cy, dim=self.d_model)
        pe_masked = pe * valid.unsqueeze(-1).to(pe.dtype)        # zero-out padded
        feat = feat + pe_masked

        # ── Self-attention encoder ──
        encoded = self.encoder(feat, src_key_padding_mask=eff_padding)

        # ── Pass 1: anchors = numbers only (label_ngs_x as anchor signal) ──
        type_idx = tokens[..., :4].argmax(dim=-1)
        is_num = (type_idx == TYPE_NUM) & valid
        is_yard = (type_idx == TYPE_YARD) & valid
        is_hash = (type_idx == TYPE_HASH) & valid

        label_ngs_x = tokens[..., 13]                            # in [0, 1]
        anchor_input_1 = torch.stack([label_ngs_x, cx, cy], dim=-1)

        x_pass1 = self._cross_attend(
            encoded, anchor_input_1, anchor_valid_mask=is_num,
            attn_module=self.cross_attn_1, norm_module=self.cross_norm_1,
            raw_pe=pe_masked,
        )
        logits_pass1 = self.head(x_pass1)

        # ── Pass 2 setup: build new anchors from pass-1 predictions ──
        # For numbers: keep label_ngs_x (known, trustworthy).
        # For yards/hashes: use pass-1 SOFT predicted NGS_x.
        #
        # Soft NGS_x = expected value over softmax probabilities. Differentiable.
        with torch.amp.autocast(enabled=False, device_type=tokens.device.type
                                if tokens.device.type in ("cuda", "cpu") else "cpu"):
            ngs_x_logits_pass1 = logits_pass1[..., :N_NGS_X_CLASSES].float()
            probs_pass1 = F.softmax(ngs_x_logits_pass1, dim=-1)   # (B, N, 21)
            class_ngs_x_norm = make_class_to_ngs_x_norm(tokens.device)
            soft_ngs_x = (probs_pass1 * class_ngs_x_norm).sum(dim=-1)  # (B, N)

        # Choose ngs_x source per token
        ngs_x_for_anchor = torch.where(is_num, label_ngs_x, soft_ngs_x)
        anchor_input_2 = torch.stack([ngs_x_for_anchor, cx, cy], dim=-1)

        # In pass 2, valid anchors = number ∪ yard ∪ hash (NOT sidelines —
        # they don't have meaningful NGS_x).
        is_anchor_2 = is_num | is_yard | is_hash

        # ── Pass 2 cross-attention ──
        x_pass2 = self._cross_attend(
            x_pass1, anchor_input_2, anchor_valid_mask=is_anchor_2,
            attn_module=self.cross_attn_2, norm_module=self.cross_norm_2,
            raw_pe=pe_masked,
        )
        logits_pass2 = self.head(x_pass2)

        return {"logits_pass1": logits_pass1, "logits_pass2": logits_pass2}


# ────────────────────────────────────────────────────────────────────────────
# Targets + loss (mostly copied from v5 — losses computed for each pass).
# ────────────────────────────────────────────────────────────────────────────

def project_centroids(tokens, H_norm_gt):
    cx = tokens[..., 4]; cy = tokens[..., 5]
    pts = torch.stack([cx, cy, torch.ones_like(cx)], dim=-1)
    field = torch.einsum("bij,bnj->bni", H_norm_gt, pts)
    denom = field[..., 2:3].clamp(min=1e-6)
    g = field[..., :2] / denom
    return g[..., 0].clamp(0, 1), g[..., 1].clamp(0, 1)


def build_targets(tokens, H_norm_gt):
    gx_n, gy_n = project_centroids(tokens, H_norm_gt)
    gx_yards = gx_n * NGS_X_MAX
    gy_yards_norm = gy_n
    type_idx = tokens[..., :4].argmax(dim=-1)
    ngs_x_class = ngs_x_to_class(gx_yards)
    # Note: numbers can only sit on the 9 painted yardlines (NGS_x classes
    # [2, 4, ..., 18]). Tokens whose centroid projects to an odd class
    # are filtered upstream at tokenization (NUM_EDGE_MARGIN in
    # cc_tokenizer_v2) so we don't need a snap here.

    side_row = (gy_yards_norm > 0.5).float()
    hash_near_d = (gy_yards_norm - HASH_Y_NEAR_NORM).abs()
    hash_far_d = (gy_yards_norm - HASH_Y_FAR_NORM).abs()
    hash_row = (hash_far_d < hash_near_d).float()
    num_near_d = (gy_yards_norm - NUM_Y_NEAR_NORM).abs()
    num_far_d = (gy_yards_norm - NUM_Y_FAR_NORM).abs()
    num_row = (num_far_d < num_near_d).float()

    is_side = (type_idx == TYPE_SIDE)
    is_hash = (type_idx == TYPE_HASH)
    is_num = (type_idx == TYPE_NUM)
    row_target = torch.where(
        is_side, side_row,
        torch.where(is_hash, hash_row,
                     torch.where(is_num, num_row,
                                  torch.zeros_like(side_row))))
    return ngs_x_class, row_target


def compute_pass_losses(tokens, logits, padding_mask, ngs_x_class, row_target):
    """Same loss structure as v5, applied per pass."""
    type_idx = tokens[..., :4].argmax(dim=-1)
    valid = ~padding_mask
    is_yard = (type_idx == TYPE_YARD) & valid
    is_side = (type_idx == TYPE_SIDE) & valid
    is_hash = (type_idx == TYPE_HASH) & valid
    is_num = (type_idx == TYPE_NUM) & valid

    ngs_x_logits = logits[..., :N_NGS_X_CLASSES]
    row_logit = logits[..., N_NGS_X_CLASSES]
    eps = 1e-6
    losses = {}

    use_x = is_yard | is_hash | is_num
    if use_x.any():
        sel = ngs_x_logits[use_x]
        tgt = ngs_x_class[use_x]
        ce_all = F.cross_entropy(sel, tgt, reduction="none")
        flat_y = is_yard[use_x]
        flat_h = is_hash[use_x]
        flat_n = is_num[use_x]
        losses["yard_ce"] = (ce_all * flat_y).sum() / flat_y.sum().clamp(min=eps)
        losses["hash_ce"] = (ce_all * flat_h).sum() / flat_h.sum().clamp(min=eps)
        losses["num_ce"] = (ce_all * flat_n).sum() / flat_n.sum().clamp(min=eps)
    else:
        z = torch.tensor(0.0, device=logits.device)
        losses["yard_ce"] = z; losses["hash_ce"] = z; losses["num_ce"] = z

    bce = F.binary_cross_entropy_with_logits(row_logit, row_target,
                                                reduction="none")
    losses["side_row"] = (bce * is_side.float()).sum() / \
                           is_side.float().sum().clamp(min=eps)
    losses["hash_row"] = (bce * is_hash.float()).sum() / \
                           is_hash.float().sum().clamp(min=eps)
    losses["num_row"] = (bce * is_num.float()).sum() / \
                          is_num.float().sum().clamp(min=eps)
    losses["total"] = (losses["yard_ce"] + losses["hash_ce"] + losses["num_ce"]
                       + losses["side_row"] + losses["hash_row"]
                       + losses["num_row"])
    return losses


# ────────────────────────────────────────────────────────────────────────────
# Eval (uses pass-2 outputs only — that's our final answer).
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_type_metrics(model, loader, device):
    model.eval()
    counts = dict(
        yard_correct=0, yard_off1=0, yard_off2=0, yard_total=0,
        hash_correct=0, hash_off1=0, hash_off2=0, hash_total=0,
        num_x_correct=0, num_x_off1=0, num_x_off2=0, num_x_total=0,
        side_c=0, side_t=0, hash_r_c=0, hash_r_t=0, num_r_c=0, num_r_t=0,
    )
    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["padding_mask"].to(device)
        H_gt = batch["H_norm_gt"].to(device)
        out = model(tokens, mask)
        logits = out["logits_pass2"]
        ngs_x_logits = logits[..., :N_NGS_X_CLASSES]
        row_logit = logits[..., N_NGS_X_CLASSES]
        ngs_x_class, row_target = build_targets(tokens, H_gt)
        type_idx = tokens[..., :4].argmax(dim=-1)
        valid = ~mask
        ngs_x_pred = ngs_x_logits.argmax(dim=-1)

        m = (type_idx == TYPE_YARD) & valid
        if m.any():
            d = (ngs_x_pred[m] - ngs_x_class[m]).abs()
            counts["yard_correct"] += int((d == 0).sum())
            counts["yard_off1"] += int((d <= 1).sum())
            counts["yard_off2"] += int((d <= 2).sum())
            counts["yard_total"] += int(m.sum())
        m = (type_idx == TYPE_HASH) & valid
        if m.any():
            d = (ngs_x_pred[m] - ngs_x_class[m]).abs()
            counts["hash_correct"] += int((d == 0).sum())
            counts["hash_off1"] += int((d <= 1).sum())
            counts["hash_off2"] += int((d <= 2).sum())
            counts["hash_total"] += int(m.sum())
        m = (type_idx == TYPE_NUM) & valid
        if m.any():
            d = (ngs_x_pred[m] - ngs_x_class[m]).abs()
            counts["num_x_correct"] += int((d == 0).sum())
            counts["num_x_off1"] += int((d <= 1).sum())
            counts["num_x_off2"] += int((d <= 2).sum())
            counts["num_x_total"] += int(m.sum())
        row_pred = (row_logit > 0)
        m = (type_idx == TYPE_SIDE) & valid
        counts["side_c"] += int((row_pred[m] == row_target[m].bool()).sum())
        counts["side_t"] += int(m.sum())
        m = (type_idx == TYPE_HASH) & valid
        counts["hash_r_c"] += int((row_pred[m] == row_target[m].bool()).sum())
        counts["hash_r_t"] += int(m.sum())
        m = (type_idx == TYPE_NUM) & valid
        counts["num_r_c"] += int((row_pred[m] == row_target[m].bool()).sum())
        counts["num_r_t"] += int(m.sum())

    def acc(c, c1, c2, t):
        if t == 0: return None
        return {"n": t, "top1": c/t, "within_5y": c1/t, "within_10y": c2/t}

    return {
        "yard_x": acc(counts["yard_correct"], counts["yard_off1"],
                        counts["yard_off2"], counts["yard_total"]),
        "hash_x": acc(counts["hash_correct"], counts["hash_off1"],
                        counts["hash_off2"], counts["hash_total"]),
        "num_x":  acc(counts["num_x_correct"], counts["num_x_off1"],
                        counts["num_x_off2"], counts["num_x_total"]),
        "side_row_acc": (counts["side_c"]/counts["side_t"]) if counts["side_t"] else None,
        "hash_row_acc": (counts["hash_r_c"]/counts["hash_r_t"]) if counts["hash_r_t"] else None,
        "num_row_acc":  (counts["num_r_c"]/counts["num_r_t"]) if counts["num_r_t"] else None,
    }


def fmt_x(s):
    if s is None: return "n=0"
    return (f"n={s['n']}  top1={s['top1']*100:.1f}%  "
            f"≤5y={s['within_5y']*100:.1f}%  ≤10y={s['within_10y']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--v2-input-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression_v2_inputs"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v6"))
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ffn-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", default="off",
                    choices=["off", "bf16", "fp16"])
    ap.add_argument("--val-game", default="2024090802")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token-dropout", type=float, default=0.2)
    ap.add_argument("--min-keep-tokens", type=int, default=4)
    ap.add_argument("--pass1-loss-weight", type=float, default=1.0,
                    help="Weight for pass-1 deep supervision (relative to pass-2)")
    ap.add_argument("--max-train-entries", type=int, default=None)
    ap.add_argument("--max-val-entries", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest_file}")
    m = json.load(open(args.manifest_file))
    intr = m.get("intrinsics_by_clip", {})
    train_e, val_e, val_game = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)
    if args.max_train_entries: train_e = train_e[:args.max_train_entries]
    if args.max_val_entries: val_e = val_e[:args.max_val_entries]
    print(f"Split: train={len(train_e)}  val={len(val_e)}  "
          f"(val game = {val_game})")

    train_ds = HSetDataset(train_e, args.cache_dir, args.v2_input_dir, intr)
    val_ds = HSetDataset(val_e, args.cache_dir, args.v2_input_dir, intr)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))

    device = torch.device(args.device)
    model = TokenClassifyV6(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        min_keep_tokens=args.min_keep_tokens).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: TokenClassifyV6 — {n_params:.2f}M params")
    print(f"Pass-1 loss weight: {args.pass1_loss_weight}  Pass-2 weight: 1.0")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)
    use_amp = (args.amp != "off") and (args.device == "cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val_top1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = {k: 0.0 for k in [
            "total", "p1_yard", "p1_hash", "p1_num",
            "p2_yard", "p2_hash", "p2_num",
            "p2_side_row", "p2_hash_row", "p2_num_row"]}
        n = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            H_gt = batch["H_norm_gt"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(tokens, mask)
                ngs_x_class, row_target = build_targets(tokens, H_gt)
                l1 = compute_pass_losses(tokens, out["logits_pass1"], mask,
                                            ngs_x_class, row_target)
                l2 = compute_pass_losses(tokens, out["logits_pass2"], mask,
                                            ngs_x_class, row_target)
                total = l2["total"] + args.pass1_loss_weight * l1["total"]
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sums["total"] += total.item()
            sums["p1_yard"] += l1["yard_ce"].item()
            sums["p1_hash"] += l1["hash_ce"].item()
            sums["p1_num"] += l1["num_ce"].item()
            sums["p2_yard"] += l2["yard_ce"].item()
            sums["p2_hash"] += l2["hash_ce"].item()
            sums["p2_num"] += l2["num_ce"].item()
            sums["p2_side_row"] += l2["side_row"].item()
            sums["p2_hash_row"] += l2["hash_row"].item()
            sums["p2_num_row"] += l2["num_row"].item()
            n += 1
        sched.step()
        elapsed = time.time() - t0

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            v = per_type_metrics(model, val_loader, device)

        # Composite val score: average of yard/hash top-1 (the metrics we want
        # to push to 99%+). Save best by THIS, not by train loss.
        val_score = ((v["yard_x"]["top1"] if v["yard_x"] else 0) +
                       (v["hash_x"]["top1"] if v["hash_x"] else 0)) / 2

        print(f"Ep {epoch+1:2d}/{args.epochs}  "
              f"L={sums['total']/n:.3f} "
              f"(p1: y={sums['p1_yard']/n:.2f} h={sums['p1_hash']/n:.2f}) "
              f"(p2: y={sums['p2_yard']/n:.2f} h={sums['p2_hash']/n:.2f} "
              f"sr={sums['p2_side_row']/n:.3f} "
              f"hr={sums['p2_hash_row']/n:.3f})  ({elapsed:.0f}s)", flush=True)
        print(f"   yard {fmt_x(v['yard_x'])}")
        print(f"   hash {fmt_x(v['hash_x'])}")
        print(f"   num  {fmt_x(v['num_x'])}")
        srx = v['side_row_acc']; hrx = v['hash_row_acc']; nrx = v['num_row_acc']
        print(f"   rows: side={srx*100 if srx else 0:.1f}%  "
              f"hash={hrx*100 if hrx else 0:.1f}%  "
              f"num={nrx*100 if nrx else 0:.1f}%  "
              f"  val_score={val_score*100:.2f}%", flush=True)

        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch+1, "lr": sched.get_last_lr()[0],
                **{f"train_{k}": v_/n for k, v_ in sums.items()},
                "val_score": val_score,
                "yard_top1": (v["yard_x"] or {}).get("top1"),
                "hash_top1": (v["hash_x"] or {}).get("top1"),
                "num_top1":  (v["num_x"]  or {}).get("top1"),
                "side_acc": v["side_row_acc"],
                "hash_row_acc": v["hash_row_acc"],
                "num_row_acc":  v["num_row_acc"],
            }, f)
            f.write("\n")

        ckpt = {"model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "args": vars(args),
                "val_score": val_score}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        # Save best by val score (yard/hash top-1 average), not train loss
        if val_score > best_val_top1:
            best_val_top1 = val_score
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best by val score = {val_score*100:.2f}%", flush=True)

    print(f"\nDone. Best val_score: {best_val_top1*100:.2f}%")


if __name__ == "__main__":
    main()
