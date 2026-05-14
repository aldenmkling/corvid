"""Phase 1 — encoder training on pseudo-labeled data.

Trains TokenClassifyV10 end-to-end with GT number anchors injected in
place of the (not-yet-trained) RFB, on the ~320K pseudo-labeled frames
in data/pseudo_labels/*.npz.

The pseudo-label npz already contains pre-tokenized data + labels:
  tokens     (n_frames, n_tok_per_frame, 16)   geometry
  type_idx   (n_frames, n_tok_per_frame)        0=yard, 1=side, 2=num, 3=hash
                                                (note: pseudo-label dump uses
                                                 cc_tokenizer's 0/1/2/3, but
                                                 our v2/v3 TYPE_HASH=2 and
                                                 TYPE_NUM=3 — we re-derive
                                                 from tokens[..., :4].argmax)
  true_class (n_frames, n_tok_per_frame)        NGS_x class for yard/hash/num,
                                                 -1 for side
  true_row   (n_frames, n_tok_per_frame)        0/1 row for side/hash/num,
                                                 -1 for yard

No UNet/mask processing at train time — labels are pre-computed.

Train/val split: by clip's game prefix (e.g., hold out 2024090802 as val).

Usage:
  python scripts/training/train_phase1_pseudo.py
  python scripts/training/train_phase1_pseudo.py --val-games 2024090802 --epochs 50
  python scripts/training/train_phase1_pseudo.py --max-train-clips 100 --epochs 5
       (sanity-check on a subset before the real run)
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
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from model_token_v10 import TokenClassifyV10  # noqa: E402


# Constants (inlined to avoid heavy import chain from train_token_v6).
TOKEN_FEATURE_DIM = 16
TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3
N_NGS_X_CLASSES = 21    # 10..110 yd, step 5


def compute_pass_losses(tokens, logits, padding_mask, ngs_x_class, row_target):
    """Per-pass token-level CE (NGS_x) + BCE (row). Pure torch — no
    dependencies. Mirrors train_token_v6.compute_pass_losses."""
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
        tgt = ngs_x_class[use_x].clamp(min=0)  # -1 → 0 (won't be selected)
        ce_all = F.cross_entropy(sel, tgt, reduction="none")
        flat_y = is_yard[use_x].float()
        flat_h = is_hash[use_x].float()
        flat_n = is_num[use_x].float()
        # Only count tokens with a real label (>= 0).
        valid_lbl = (ngs_x_class[use_x] >= 0).float()
        losses["yard_ce"] = (ce_all * flat_y * valid_lbl).sum() \
                              / (flat_y * valid_lbl).sum().clamp(min=eps)
        losses["hash_ce"] = (ce_all * flat_h * valid_lbl).sum() \
                              / (flat_h * valid_lbl).sum().clamp(min=eps)
        losses["num_ce"] = (ce_all * flat_n * valid_lbl).sum() \
                              / (flat_n * valid_lbl).sum().clamp(min=eps)
    else:
        z = torch.tensor(0.0, device=logits.device)
        losses["yard_ce"] = z; losses["hash_ce"] = z; losses["num_ce"] = z

    bce = F.binary_cross_entropy_with_logits(row_logit, row_target,
                                                reduction="none")
    row_valid = (row_target >= 0).float()
    losses["side_row"] = (bce * is_side.float() * row_valid).sum() \
                           / (is_side.float() * row_valid).sum().clamp(min=eps)
    losses["hash_row"] = (bce * is_hash.float() * row_valid).sum() \
                           / (is_hash.float() * row_valid).sum().clamp(min=eps)
    losses["num_row"] = (bce * is_num.float() * row_valid).sum() \
                          / (is_num.float() * row_valid).sum().clamp(min=eps)
    losses["total"] = (losses["yard_ce"] + losses["hash_ce"] + losses["num_ce"]
                       + losses["side_row"] + losses["hash_row"]
                       + losses["num_row"])
    return losses


# ── Dataset ─────────────────────────────────────────────────────────────────

class PseudoLabelTokenDataset(Dataset):
    """Flat per-frame view across all pseudo-label npz files.

    Caches every npz's per-frame arrays in RAM at __init__ (~285 MB for
    1280 clips × ~250 frames). Per __getitem__ just slices into the cache.

    blacklist: optional set of "clip_id#frame_idx" strings to drop (e.g.,
    bad-GT frames identified during hand review).
    """

    def __init__(self, npz_dir, clip_filter=None, cache_in_ram=True,
                 blacklist=None):
        self.npz_dir = npz_dir
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        if clip_filter is not None:
            files = [f for f in files if clip_filter(f)]
        self.files = files
        blacklist = blacklist or set()
        # flat index: list of (file_idx, in_file_idx)
        self.flat = []
        # per-file cached arrays (object arrays of per-frame entries)
        self.cache = []
        t0 = time.time()
        n_tok_total = 0
        n_blacklisted = 0
        for fi, f in enumerate(files):
            d = np.load(os.path.join(npz_dir, f), allow_pickle=True)
            tokens = d["tokens"]          # object array (n_frames,)
            type_idx = d["type_idx"]      # object array
            true_class = d["true_class"]  # object array
            true_row = d["true_row"]      # object array
            frame_idx_arr = d["frame_idx"] # (n_frames,)
            clip_id = f.replace(".npz", "")
            n_frames = len(tokens)
            self.cache.append({
                "tokens": tokens,
                "type_idx": type_idx,
                "true_class": true_class,
                "true_row": true_row,
            })
            for k in range(n_frames):
                key = f"{clip_id}#{int(frame_idx_arr[k])}"
                if key in blacklist:
                    n_blacklisted += 1
                    continue
                self.flat.append((fi, k))
                n_tok_total += int(tokens[k].shape[0])
        dt = time.time() - t0
        print(f"  {len(self.flat)} frames from {len(self.files)} clips "
              f"({n_tok_total} tokens) loaded in {dt:.1f}s "
              f"(blacklisted {n_blacklisted})", flush=True)

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx):
        fi, k = self.flat[idx]
        c = self.cache[fi]
        tokens = np.asarray(c["tokens"][k], dtype=np.float32)
        type_idx = np.asarray(c["type_idx"][k], dtype=np.int64)
        true_class = np.asarray(c["true_class"][k], dtype=np.int64)
        true_row = np.asarray(c["true_row"][k], dtype=np.int64)
        # Ensure type_idx matches tokens[..., :4].argmax for consistency
        # with how downstream losses derive `is_yard / is_hash / ...`.
        # (Sanity-check at __getitem__ keeps the data tight.)
        return {
            "tokens": tokens,
            "type_idx": type_idx,
            "true_class": true_class,
            "true_row": true_row,
        }


def collate_pseudo(batch):
    """Pad tokens to max-N in batch; build padding_mask, ngs_x_class,
    row_target tensors. -1 in label arrays propagates (loss masks handle
    irrelevant types)."""
    B = len(batch)
    Nmax = max(b["tokens"].shape[0] for b in batch)
    tokens = np.zeros((B, Nmax, TOKEN_FEATURE_DIM), dtype=np.float32)
    pad = np.ones((B, Nmax), dtype=bool)
    ngs_x = np.full((B, Nmax), -1, dtype=np.int64)
    row = np.full((B, Nmax), -1, dtype=np.int64)
    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        tokens[i, :n] = b["tokens"]
        pad[i, :n] = False
        ngs_x[i, :n] = b["true_class"]
        row[i, :n] = b["true_row"]
    return {
        "tokens": torch.from_numpy(tokens),
        "padding_mask": torch.from_numpy(pad),
        "ngs_x_class": torch.from_numpy(ngs_x),
        "row_target": torch.from_numpy(row),
    }


# ── Eval ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = {"yard": [0, 0], "hash": [0, 0], "num": [0, 0],
           "side_row": [0, 0], "hash_row": [0, 0], "num_row": [0, 0]}
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        pad = batch["padding_mask"].to(device, non_blocking=True)
        ngs_x = batch["ngs_x_class"].to(device, non_blocking=True)
        row = batch["row_target"].to(device, non_blocking=True)
        type_idx = tokens[..., :4].argmax(dim=-1)
        is_num = (type_idx == TYPE_NUM)
        # Teacher force: feed pseudo-label num class as anchor (matches
        # training; phase-1 model relies on GT anchor since RFB is absent).
        num_class_gt = torch.where(is_num, ngs_x, torch.full_like(ngs_x, -1))
        out = model(tokens, pad, num_class_gt=num_class_gt)
        logits = out["logits_pass2"]
        x_pred = logits[..., :N_NGS_X_CLASSES].argmax(dim=-1)
        row_pred = (logits[..., N_NGS_X_CLASSES] > 0).long()
        valid = ~pad
        for name, mask in (("yard", (type_idx == TYPE_YARD) & valid),
                            ("hash", (type_idx == TYPE_HASH) & valid),
                            ("num",  (type_idx == TYPE_NUM)  & valid)):
            ok = (x_pred == ngs_x) & mask & (ngs_x >= 0)
            tot[name][0] += int(ok.sum())
            tot[name][1] += int(((ngs_x >= 0) & mask).sum())
        for name, mask in (("side_row", (type_idx == TYPE_SIDE) & valid),
                            ("hash_row", (type_idx == TYPE_HASH) & valid),
                            ("num_row",  (type_idx == TYPE_NUM)  & valid)):
            ok = (row_pred == row) & mask & (row >= 0)
            tot[name][0] += int(ok.sum())
            tot[name][1] += int(((row >= 0) & mask).sum())
    return {k: (v[0] / max(1, v[1])) for k, v in tot.items()}, tot


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/pseudo_labels")
    ap.add_argument("--out-dir", default="models/token_only_v10_phase1_pseudo")
    ap.add_argument("--val-games", nargs="+", default=["2024090802"],
                   help="Games whose clips go to the validation set.")
    # Architecture (matches existing prod weights).
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--ffn-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token-dropout", type=float, default=0.4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    # Optim.
    ap.add_argument("--epochs", type=int, default=20,
                   help="With ~320K samples, 5-20 epochs is plenty (vs "
                        "the 100 used on 2.5K GT samples).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-clips", type=int, default=None,
                   help="Limit train clips for quick sanity runs.")
    ap.add_argument("--max-val-clips", type=int, default=None)
    ap.add_argument("--blacklist", default=None,
                   help="Path to JSON with {'bad_gt_keys': [...]} list of "
                        "'clip_id#frame_idx' strings to drop.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    npz_dir = os.path.join(PROJECT_ROOT, args.npz_dir)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    val_set = set(args.val_games)
    def is_val(name): return any(name.startswith(g + "_") for g in val_set)
    def is_train(name): return not is_val(name)

    if args.max_train_clips or args.max_val_clips:
        # Pre-shuffle the file lists to ensure max-N subset is representative.
        all_files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        train_files = [f for f in all_files if is_train(f)]
        val_files = [f for f in all_files if is_val(f)]
        if args.max_train_clips:
            random.Random(args.seed).shuffle(train_files)
            train_files = set(train_files[:args.max_train_clips])
            train_filter = lambda f: f in train_files  # noqa: E731
        else:
            train_filter = is_train
        if args.max_val_clips:
            random.Random(args.seed + 1).shuffle(val_files)
            val_files = set(val_files[:args.max_val_clips])
            val_filter = lambda f: f in val_files  # noqa: E731
        else:
            val_filter = is_val
    else:
        train_filter = is_train
        val_filter = is_val

    blacklist = set()
    if args.blacklist:
        bl_path = os.path.join(PROJECT_ROOT, args.blacklist)
        with open(bl_path) as f:
            bl_data = json.load(f)
        blacklist = set(bl_data.get("bad_gt_keys", []))
        print(f"Blacklist loaded: {len(blacklist)} frames to drop "
              f"(from {args.blacklist})")

    print("Loading train set...")
    train_ds = PseudoLabelTokenDataset(npz_dir, train_filter, blacklist=blacklist)
    print("Loading val set...")
    val_ds = PseudoLabelTokenDataset(npz_dir, val_filter, blacklist=blacklist)

    device = torch.device(args.device)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_pseudo,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_pseudo,
        persistent_workers=(args.num_workers > 0))

    model = TokenClassifyV10(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, ffn_dim=args.ffn_dim,
        dropout=args.dropout, token_dropout=args.token_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: TokenClassifyV10 — {n_params:.2f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(log_path, "w") as f:
        f.write("# phase 1 pseudo-label training\n")

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = {"total": 0, "yard": 0, "hash": 0, "num": 0,
                "side_row": 0, "hash_row": 0, "num_row": 0, "aux": 0}
        train_yard_corr = train_yard_tot = 0
        train_hash_corr = train_hash_tot = 0
        n_batches = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            pad = batch["padding_mask"].to(device, non_blocking=True)
            ngs_x = batch["ngs_x_class"].to(device, non_blocking=True)
            row = batch["row_target"].to(device, non_blocking=True)
            # Zero out token slots 13, 14 (per-tokenizer "input num class")
            # for number tokens — model relies on classifier/anchor for those.
            type_idx = tokens[..., :4].argmax(dim=-1)
            is_num_in = (type_idx == TYPE_NUM)
            tokens[..., 13] = torch.where(
                is_num_in, torch.zeros_like(tokens[..., 13]), tokens[..., 13])
            tokens[..., 14] = torch.where(
                is_num_in, torch.zeros_like(tokens[..., 14]), tokens[..., 14])

            # Anchor: pseudo-label num class for num tokens.
            num_class_gt = torch.where(
                is_num_in, ngs_x, torch.full_like(ngs_x, -1))
            row_target_f = row.float().clamp(min=0)  # -1 → 0; loss masks it

            out = model(tokens, pad, num_class_gt=num_class_gt)
            l1 = compute_pass_losses(tokens, out["logits_pass1"], pad,
                                          ngs_x, row_target_f)
            l2 = compute_pass_losses(tokens, out["logits_pass2"], pad,
                                          ngs_x, row_target_f)
            num_logits_flat = out["num_logits"][..., :N_NGS_X_CLASSES] \
                .reshape(-1, N_NGS_X_CLASSES)
            num_gt_flat = num_class_gt.reshape(-1)
            valid_num = (num_gt_flat >= 0) & (num_gt_flat < N_NGS_X_CLASSES)
            if valid_num.any():
                aux_loss = F.cross_entropy(
                    num_logits_flat[valid_num], num_gt_flat[valid_num])
            else:
                aux_loss = torch.tensor(0.0, device=tokens.device)
            total = l2["total"] + l1["total"] + aux_loss

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            with torch.no_grad():
                p2_x = out["logits_pass2"][..., :N_NGS_X_CLASSES].argmax(dim=-1)
                valid = ~pad
                m = (type_idx == TYPE_YARD) & valid & (ngs_x >= 0)
                if m.any():
                    train_yard_corr += int((p2_x[m] == ngs_x[m]).sum())
                    train_yard_tot += int(m.sum())
                m = (type_idx == TYPE_HASH) & valid & (ngs_x >= 0)
                if m.any():
                    train_hash_corr += int((p2_x[m] == ngs_x[m]).sum())
                    train_hash_tot += int(m.sum())

            sums["total"] += total.item()
            sums["yard"] += l2["yard_ce"].item()
            sums["hash"] += l2["hash_ce"].item()
            sums["num"] += l2["num_ce"].item()
            sums["side_row"] += l2["side_row"].item()
            sums["hash_row"] += l2["hash_row"].item()
            sums["num_row"] += l2["num_row"].item()
            sums["aux"] += aux_loss.item()
            n_batches += 1

        sched.step()
        dt = time.time() - t0
        train_yard_acc = train_yard_corr / max(1, train_yard_tot)
        train_hash_acc = train_hash_corr / max(1, train_hash_tot)

        val_acc, val_tot = evaluate(model, val_loader, device)
        score = 0.5 * (val_acc["yard"] + val_acc["hash"])

        msg = (f"Ep {epoch+1:3d}/{args.epochs}  L={sums['total']/n_batches:.3f} "
               f"(yard={sums['yard']/n_batches:.3f} hash={sums['hash']/n_batches:.3f} "
               f"num={sums['num']/n_batches:.3f} sr={sums['side_row']/n_batches:.3f} "
               f"hr={sums['hash_row']/n_batches:.3f} nr={sums['num_row']/n_batches:.3f} "
               f"aux={sums['aux']/n_batches:.3f}) {dt:.0f}s\n"
               f"   train: y={train_yard_acc*100:5.2f}% h={train_hash_acc*100:5.2f}%  "
               f"val: y={val_acc['yard']*100:5.2f}% h={val_acc['hash']*100:5.2f}% "
               f"n={val_acc['num']*100:5.2f}% sr={val_acc['side_row']*100:5.2f}% "
               f"hr={val_acc['hash_row']*100:5.2f}% nr={val_acc['num_row']*100:5.2f}%  "
               f"score={score*100:.2f}%")
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if score > best_score:
            best_score = score
            ck = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "val_score": score,
                "val_acc": val_acc,
            }
            torch.save(ck, os.path.join(out_dir, "best.pth"))
            print(f"   ✓ saved best (score={score*100:.2f}%)", flush=True)

    print(f"Done. Best val score: {best_score*100:.2f}%")


if __name__ == "__main__":
    main()
