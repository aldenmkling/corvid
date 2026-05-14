"""v8: smaller v6 + aggressive augmentation.

Goal: produce an encoder that GENERALIZES instead of memorizing the training
set. v6 had a 50× train/val gap (99.6% perfect on train, ~3% errors on val).
The catastrophic val frames consistently fail because the encoder doesn't
know how to handle close-up scenarios it never saw during training.

Architectural changes from v6:
  - d_model:  128 → 96    (-43% capacity)
  - ffn_dim:  256 → 192   (proportional to d_model)
  - n_layers: 4 (unchanged — preserve cross-token reasoning depth)
  - n_heads:  4 (unchanged)

Regularization changes:
  - token_dropout:  0.2 → 0.4  (drop 40% of tokens per training pass)
  - weight_decay:   1e-4 → 1e-3
  - epochs:         80 → 100  (stronger reg slows convergence)
  - LR:             1e-3 (unchanged)

NEW data augmentation in the training dataset (via AugmentedHSetDataset):
  - centroid jitter: ±2 pixels (in normalized coords)
  - bbox jitter:     ±2 pixels per corner
  - orientation jitter: rotate cos/sin by ±5° per token

These force the model to handle perturbations of the same scenes — preventing
memorization of exact token configurations.

Eval is identical to v6 (no augmentation at test time).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer import (    # noqa: E402
    SRC_W, SRC_H, TOKEN_FEATURE_DIM,
    TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM,
)
from train_dense_regression import NGS_X_MAX, NGS_Y_MAX, split_by_game  # noqa: E402
from train_h_set_regressor import HSetDataset, collate_set   # noqa: E402

# Import v6 model + losses (we just retrain with smaller config + aug)
from train_token_v6 import (    # noqa: E402
    TokenClassifyV6, build_targets, compute_pass_losses,
    per_type_metrics, fmt_x, N_NGS_X_CLASSES,
)


# ────────────────────────────────────────────────────────────────────────────
# Augmented dataset: applies geometric jitter + per-feature noise at training.
# ────────────────────────────────────────────────────────────────────────────

class AugmentedHSetDataset(Dataset):
    """Wraps HSetDataset with token-level augmentation.

    Augmentations (training only, controlled by `enabled` flag):
      • centroid jitter — gaussian noise on (cx, cy) in normalized [0, 1]
      • bbox jitter — gaussian noise on (x_min, y_min, x_max, y_max)
      • orientation jitter — rotate (cos_t, sin_t) by ±max_angle_deg

    Augmentation is applied per-token (independent noise for each token).
    Padded tokens (all zeros, type one-hot all zeros) are NOT augmented since
    the collate adds padding tokens after this wrapper.

    Note: Number tokens' label_ngs_x (feature 13) and has_ngs (feature 14)
    are NOT modified — those are ground truth labels we need preserved.
    """
    def __init__(self, base_ds: HSetDataset,
                 enabled: bool = True,
                 centroid_sigma: float = 2.0 / 1280.0,   # 2 px normalized
                 bbox_sigma: float = 2.0 / 1280.0,
                 max_angle_deg: float = 5.0):
        self.base = base_ds
        self.enabled = enabled
        self.centroid_sigma = centroid_sigma
        self.bbox_sigma = bbox_sigma
        self.max_angle_rad = max_angle_deg * math.pi / 180.0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if not self.enabled:
            return item
        tokens = item["tokens"]              # (N, 16) torch float
        if tokens.shape[0] == 0:
            return item
        N = tokens.shape[0]

        # ── Centroid jitter (features 4, 5) ──
        if self.centroid_sigma > 0:
            jit = torch.randn(N, 2) * self.centroid_sigma
            tokens[:, 4:6] = (tokens[:, 4:6] + jit).clamp(0.0, 1.0)

        # ── Bbox jitter (features 6, 7, 8, 9) ──
        if self.bbox_sigma > 0:
            jit = torch.randn(N, 4) * self.bbox_sigma
            tokens[:, 6:10] = (tokens[:, 6:10] + jit).clamp(0.0, 1.0)

        # ── Orientation jitter (features 11, 12) ──
        # Rotate (cos_t, sin_t) by random angle
        if self.max_angle_rad > 0:
            angle_per_token = (torch.rand(N) * 2 - 1) * self.max_angle_rad
            cos_a = torch.cos(angle_per_token)
            sin_a = torch.sin(angle_per_token)
            cos_t = tokens[:, 11].clone()
            sin_t = tokens[:, 12].clone()
            tokens[:, 11] = cos_t * cos_a - sin_t * sin_a
            tokens[:, 12] = cos_t * sin_a + sin_t * cos_a

        # Note: we don't modify type one-hot, log_area, label_ngs_x, has_ngs,
        # or confidence — those should reflect ground-truth identity.

        item["tokens"] = tokens
        return item


# ────────────────────────────────────────────────────────────────────────────
# Main.
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--v2-input-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression_v2_inputs"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v8"))
    # Smaller architecture
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--ffn-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    # Aggressive regularization
    ap.add_argument("--token-dropout", type=float, default=0.4,
                    help="Per-token Bernoulli drop prob (was 0.2 in v6)")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--centroid-sigma", type=float, default=2.0 / 1280.0)
    ap.add_argument("--bbox-sigma", type=float, default=2.0 / 1280.0)
    ap.add_argument("--max-angle-deg", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", default="off",
                    choices=["off", "bf16", "fp16"])
    ap.add_argument("--val-game", default="2024090802")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-entries", type=int, default=None)
    ap.add_argument("--max-val-entries", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr = m.get("intrinsics_by_clip", {})
    train_e, val_e, val_game = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)
    if args.max_train_entries: train_e = train_e[:args.max_train_entries]
    if args.max_val_entries: val_e = val_e[:args.max_val_entries]
    print(f"Split: train={len(train_e)}  val={len(val_e)}  "
          f"(val game = {val_game})")

    base_train = HSetDataset(train_e, args.cache_dir, args.v2_input_dir, intr)
    base_val = HSetDataset(val_e, args.cache_dir, args.v2_input_dir, intr)
    aug_train = AugmentedHSetDataset(
        base_train, enabled=True,
        centroid_sigma=args.centroid_sigma,
        bbox_sigma=args.bbox_sigma,
        max_angle_deg=args.max_angle_deg)
    # Val is NOT augmented — reflects real test distribution
    aug_val = AugmentedHSetDataset(base_val, enabled=False)

    train_loader = DataLoader(
        aug_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        aug_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=collate_set,
        persistent_workers=(args.num_workers > 0))

    device = torch.device(args.device)
    model = TokenClassifyV6(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: TokenClassifyV6 (v8 config) — {n_params:.2f}M params")
    print(f"  d_model={args.d_model}  ffn_dim={args.ffn_dim}  layers={args.n_layers}")
    print(f"  token_dropout={args.token_dropout}  weight_decay={args.weight_decay}")
    print(f"  centroid_sigma={args.centroid_sigma:.5f}  "
          f"bbox_sigma={args.bbox_sigma:.5f}  "
          f"max_angle={args.max_angle_deg}°")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)
    use_amp = (args.amp != "off") and (args.device == "cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = {k: 0.0 for k in ["total", "p1_yard", "p1_hash", "p1_num",
                                  "p2_yard", "p2_hash", "p2_num",
                                  "p2_side_row", "p2_hash_row", "p2_num_row"]}
        # Track train accuracy as well (compare train/val gap)
        train_yard_corr = train_yard_tot = 0
        train_hash_corr = train_hash_tot = 0
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
                total = l2["total"] + l1["total"]
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # Train accuracy tracking
            with torch.no_grad():
                ngs_x_logits = out["logits_pass2"][..., :N_NGS_X_CLASSES]
                ngs_x_pred = ngs_x_logits.argmax(dim=-1)
                type_idx = tokens[..., :4].argmax(dim=-1)
                valid = ~mask
                m = (type_idx == TYPE_YARD) & valid
                if m.any():
                    train_yard_corr += int((ngs_x_pred[m] == ngs_x_class[m]).sum())
                    train_yard_tot += int(m.sum())
                m = (type_idx == TYPE_HASH) & valid
                if m.any():
                    train_hash_corr += int((ngs_x_pred[m] == ngs_x_class[m]).sum())
                    train_hash_tot += int(m.sum())

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

        train_yard_acc = train_yard_corr / max(1, train_yard_tot)
        train_hash_acc = train_hash_corr / max(1, train_hash_tot)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            v = per_type_metrics(model, val_loader, device)
        val_score = ((v["yard_x"]["top1"] if v["yard_x"] else 0) +
                       (v["hash_x"]["top1"] if v["hash_x"] else 0)) / 2

        # Train/val gap (yard top-1)
        gap = train_yard_acc - (v["yard_x"]["top1"] if v["yard_x"] else 0)

        print(f"Ep {epoch+1:3d}/{args.epochs}  "
              f"L={sums['total']/n:.3f} "
              f"(p1: y={sums['p1_yard']/n:.2f} h={sums['p1_hash']/n:.2f}) "
              f"(p2: y={sums['p2_yard']/n:.2f} h={sums['p2_hash']/n:.2f}) "
              f"({elapsed:.0f}s)", flush=True)
        print(f"   train: yard={train_yard_acc*100:.1f}% hash={train_hash_acc*100:.1f}%  "
              f"val: yard={(v['yard_x']['top1'] if v['yard_x'] else 0)*100:.1f}% "
              f"hash={(v['hash_x']['top1'] if v['hash_x'] else 0)*100:.1f}%  "
              f"GAP yard: {gap*100:+.1f}%   "
              f"val_score={val_score*100:.2f}%", flush=True)

        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch+1, "lr": sched.get_last_lr()[0],
                "train_yard_acc": train_yard_acc,
                "train_hash_acc": train_hash_acc,
                "val_score": val_score,
                "yard_top1": (v["yard_x"] or {}).get("top1"),
                "hash_top1": (v["hash_x"] or {}).get("top1"),
                "num_top1": (v["num_x"] or {}).get("top1"),
                "side_acc": v["side_row_acc"],
                "hash_row_acc": v["hash_row_acc"],
                "num_row_acc": v["num_row_acc"],
                "train_val_gap_yard": gap,
            }, f)
            f.write("\n")

        ckpt = {"model_state_dict": model.state_dict(),
                "epoch": epoch+1, "args": vars(args),
                "val_score": val_score}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best val_score = {val_score*100:.2f}%", flush=True)

    print(f"\nDone. Best val_score: {best_val_score*100:.2f}%")


if __name__ == "__main__":
    main()
