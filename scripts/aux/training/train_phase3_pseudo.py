"""Phase 3 — v10c stage 2 training on pseudo-label data.

Frozen: phase-1 encoder + phase-2 RFB.
Trains: v10c (TokenClassifyV10b) cross-attention + heads.

Per frame:
  1. Encoder features for all tokens (frozen).
  2. RFB on num tokens (frozen) using encoder features + pre-extracted
     crop_logits → painted-class + row + pre-head features (used as
     num_rfb_features residual into v10c).
  3. v10c forward with num_class_gt = RFB argmax (mapped to 21-class
     NGS_x via PAINTED_TO_21) and num_rfb_features.
  4. Loss: pass1 + pass2 token CE/BCE on yard/hash/side (num CE+row
     dropped because RFB owns those; matches train_token_v10c_stage2).

Labels come from pseudo-label true_class / true_row.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from model_token_v10 import TokenClassifyV10   # noqa: E402
from model_token_v10b import TokenClassifyV10b   # noqa: E402
from train_rf_a import encoder_features, N_PAINTED_CLASSES   # noqa: E402
from train_rf_b import RFB   # noqa: E402
from train_token_v10c_stage2 import (   # noqa: E402
    rfb_forward_with_features_and_row, PAINTED_TO_21,
)

TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3
N_NGS_X_CLASSES = 21
TOKEN_FEATURE_DIM = 16


# Pure-torch loss reused from phase 1 (yard / hash CE + side/hash row BCE).
def compute_pass_losses(tokens, logits, padding_mask, ngs_x_class, row_target):
    type_idx = tokens[..., :4].argmax(dim=-1)
    valid = ~padding_mask
    is_yard = (type_idx == TYPE_YARD) & valid
    is_side = (type_idx == TYPE_SIDE) & valid
    is_hash = (type_idx == TYPE_HASH) & valid
    ngs_x_logits = logits[..., :N_NGS_X_CLASSES]
    row_logit = logits[..., N_NGS_X_CLASSES]
    eps = 1e-6
    losses = {}
    use_x = is_yard | is_hash
    if use_x.any():
        sel = ngs_x_logits[use_x]
        tgt = ngs_x_class[use_x].clamp(min=0)
        ce_all = F.cross_entropy(sel, tgt, reduction="none")
        flat_y = is_yard[use_x].float()
        flat_h = is_hash[use_x].float()
        valid_lbl = (ngs_x_class[use_x] >= 0).float()
        losses["yard_ce"] = (ce_all * flat_y * valid_lbl).sum() \
                            / (flat_y * valid_lbl).sum().clamp(min=eps)
        losses["hash_ce"] = (ce_all * flat_h * valid_lbl).sum() \
                            / (flat_h * valid_lbl).sum().clamp(min=eps)
    else:
        z = torch.tensor(0.0, device=logits.device)
        losses["yard_ce"] = z; losses["hash_ce"] = z
    bce = F.binary_cross_entropy_with_logits(row_logit, row_target,
                                              reduction="none")
    row_valid = (row_target >= 0).float()
    losses["side_row"] = (bce * is_side.float() * row_valid).sum() \
                          / (is_side.float() * row_valid).sum().clamp(min=eps)
    losses["hash_row"] = (bce * is_hash.float() * row_valid).sum() \
                          / (is_hash.float() * row_valid).sum().clamp(min=eps)
    losses["total"] = (losses["yard_ce"] + losses["hash_ce"]
                       + losses["side_row"] + losses["hash_row"])
    return losses


# ── Dataset (same structure as phase 2 — needs crops sidecar) ───────────────

class Phase3Dataset(Dataset):
    def __init__(self, npz_dir, crops_dir, clip_filter=None, blacklist=None):
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        if clip_filter is not None:
            files = [f for f in files if clip_filter(f)]
        blacklist = blacklist or set()
        self.cache = []
        self.flat = []
        t0 = time.time()
        n_drop = 0
        for f in files:
            d = np.load(os.path.join(npz_dir, f), allow_pickle=True)
            cp = os.path.join(crops_dir, f)
            if not os.path.exists(cp): continue
            c = np.load(cp, allow_pickle=True)
            self.cache.append({
                "tokens": d["tokens"],
                "type_idx": d["type_idx"],
                "true_class": d["true_class"],
                "true_row": d["true_row"],
                "frame_idx": d["frame_idx"],
                "num_token_idx": c["num_token_idx"],
                "crop_logits": c["crop_logits"],
            })
            clip_id = f.replace(".npz", "")
            for k in range(len(d["frame_idx"])):
                key = f"{clip_id}#{int(d['frame_idx'][k])}"
                if key in blacklist:
                    n_drop += 1; continue
                self.flat.append((len(self.cache) - 1, k))
        print(f"  {len(self.flat)} frames over {len(self.cache)} clips "
              f"(blacklisted {n_drop}) in {time.time()-t0:.1f}s", flush=True)

    def __len__(self): return len(self.flat)

    def __getitem__(self, idx):
        fi, k = self.flat[idx]
        c = self.cache[fi]
        return {
            "tokens": np.asarray(c["tokens"][k], dtype=np.float32),
            "true_class": np.asarray(c["true_class"][k], dtype=np.int64),
            "true_row": np.asarray(c["true_row"][k], dtype=np.int64),
            "num_token_idx": np.asarray(c["num_token_idx"][k], dtype=np.int64),
            "crop_logits": np.asarray(c["crop_logits"][k], dtype=np.float32),
        }


def collate(batch):
    B = len(batch)
    Nmax = max(b["tokens"].shape[0] for b in batch)
    Nmax_num = max(max(b["num_token_idx"].shape[0], 1) for b in batch)
    tokens = np.zeros((B, Nmax, TOKEN_FEATURE_DIM), dtype=np.float32)
    pad = np.ones((B, Nmax), dtype=bool)
    ngs_x = np.full((B, Nmax), -1, dtype=np.int64)
    row = np.full((B, Nmax), -1, dtype=np.int64)
    num_idx = np.zeros((B, Nmax_num), dtype=np.int64)
    num_pad = np.ones((B, Nmax_num), dtype=bool)
    crop_logits = np.zeros((B, Nmax_num, N_PAINTED_CLASSES), dtype=np.float32)
    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        tokens[i, :n] = b["tokens"]
        pad[i, :n] = False
        ngs_x[i, :n] = b["true_class"]
        row[i, :n] = b["true_row"]
        ni = b["num_token_idx"]; nn = len(ni)
        if nn == 0 or b["crop_logits"].shape[0] == 0:
            continue
        num_idx[i, :nn] = ni
        num_pad[i, :nn] = False
        crop_logits[i, :nn] = b["crop_logits"]
    return {
        "tokens": torch.from_numpy(tokens),
        "pad": torch.from_numpy(pad),
        "ngs_x": torch.from_numpy(ngs_x),
        "row": torch.from_numpy(row),
        "num_idx": torch.from_numpy(num_idx),
        "num_pad": torch.from_numpy(num_pad),
        "crop_logits": torch.from_numpy(crop_logits),
    }


@torch.no_grad()
def evaluate(encoder, rfb, v10c, loader, device, d_enc):
    v10c.eval()
    tot = {"yard": [0, 0], "hash": [0, 0],
           "side_row": [0, 0], "hash_row": [0, 0]}
    for batch in loader:
        tokens = batch["tokens"].to(device); pad = batch["pad"].to(device)
        ngs_x = batch["ngs_x"].to(device); row = batch["row"].to(device)
        num_idx = batch["num_idx"].to(device); num_pad = batch["num_pad"].to(device)
        crop_logits = batch["crop_logits"].to(device)
        ef = encoder_features(encoder, tokens, pad)
        idx_exp = num_idx.unsqueeze(-1).expand(-1, -1, d_enc)
        num_feat = torch.gather(ef, 1, idx_exp)
        cls_logits, _, pre_head = rfb_forward_with_features_and_row(
            rfb, num_feat, crop_logits, num_pad)
        num_pred_painted = cls_logits.argmax(dim=-1)
        num_pred_21 = PAINTED_TO_21.to(device)[num_pred_painted]
        # Scatter back to per-token tensors.
        B, N, _ = tokens.shape
        num_class_gt = torch.full((B, N), -1, dtype=torch.long, device=device)
        num_rfb_features = torch.zeros((B, N, d_enc), device=device)
        for i in range(B):
            for j in range(num_idx.shape[1]):
                if num_pad[i, j]: continue
                idx = int(num_idx[i, j])
                num_class_gt[i, idx] = int(num_pred_21[i, j])
                num_rfb_features[i, idx] = pre_head[i, j]
        out = v10c(tokens, pad, num_class_gt=num_class_gt,
                   num_rfb_features=num_rfb_features)
        logits = out["logits_pass2"]
        x_pred = logits[..., :N_NGS_X_CLASSES].argmax(dim=-1)
        row_pred = (logits[..., N_NGS_X_CLASSES] > 0).long()
        valid = ~pad
        type_idx = tokens[..., :4].argmax(dim=-1)
        for name, mask in (("yard", (type_idx == TYPE_YARD) & valid),
                            ("hash", (type_idx == TYPE_HASH) & valid)):
            ok = (x_pred == ngs_x) & mask & (ngs_x >= 0)
            tot[name][0] += int(ok.sum()); tot[name][1] += int(((ngs_x >= 0) & mask).sum())
        for name, mask in (("side_row", (type_idx == TYPE_SIDE) & valid),
                            ("hash_row", (type_idx == TYPE_HASH) & valid)):
            ok = (row_pred == row) & mask & (row >= 0)
            tot[name][0] += int(ok.sum()); tot[name][1] += int(((row >= 0) & mask).sum())
    return {k: v[0] / max(1, v[1]) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/pseudo_labels")
    ap.add_argument("--crops-dir", default="data/pseudo_labels_crops")
    ap.add_argument("--encoder-ckpt",
                   default="models/token_only_v10_phase1_pseudo/best.pth")
    ap.add_argument("--rfb-ckpt",
                   default="models/rf_b_phase2_pseudo/best.pth")
    ap.add_argument("--out-dir", default="models/v10c_phase3_pseudo")
    ap.add_argument("--val-games", nargs="+",
                   default=["2024090802", "2024100601", "2024122201"])
    ap.add_argument("--blacklist", default=None)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--ffn-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token-dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                   help="Path to a v10c best.pth to resume training from.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    val_set = set(args.val_games)
    is_val = lambda n: any(n.startswith(g + "_") for g in val_set)  # noqa: E731
    is_train = lambda n: not is_val(n)  # noqa: E731
    blacklist = set()
    if args.blacklist:
        bl = json.load(open(os.path.join(PROJECT_ROOT, args.blacklist)))
        blacklist = set(bl.get("bad_gt_keys", []))

    # Encoder.
    ck_e = torch.load(os.path.join(PROJECT_ROOT, args.encoder_ckpt),
                       map_location="cpu", weights_only=False)
    sa = ck_e["args"]
    encoder = TokenClassifyV10(
        n_layers=sa["n_layers"], n_heads=sa["n_heads"],
        d_model=sa["d_model"], ffn_dim=sa["ffn_dim"],
        dropout=0.0, token_dropout=0.0).to(device).eval()
    encoder.load_state_dict(ck_e["model_state_dict"])
    for p in encoder.parameters(): p.requires_grad_(False)
    d_enc = sa["d_model"]

    # RFB.
    ck_r = torch.load(os.path.join(PROJECT_ROOT, args.rfb_ckpt),
                       map_location="cpu", weights_only=False)
    ra = ck_r["args"]
    rfb = RFB(d_enc=d_enc, d_model=ra["d_model"], n_heads=ra["n_heads"],
              ffn_dim=ra["ffn_dim"], dropout=0.0, with_row=True).to(device).eval()
    rfb.load_state_dict(ck_r["model_state_dict"])
    for p in rfb.parameters(): p.requires_grad_(False)

    # v10c.
    v10c = TokenClassifyV10b(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, ffn_dim=args.ffn_dim,
        dropout=args.dropout, token_dropout=args.token_dropout).to(device)
    n_p = sum(p.numel() for p in v10c.parameters()) / 1e6
    print(f"v10c: {n_p:.2f}M params")
    if args.resume:
        resume_path = os.path.join(PROJECT_ROOT, args.resume) \
            if not os.path.isabs(args.resume) else args.resume
        rk = torch.load(resume_path, map_location="cpu", weights_only=False)
        v10c.load_state_dict(rk["model_state_dict"])
        prev_score = rk.get("val", {}).get("yard", 0)
        print(f"Resumed from {resume_path}  prev val yard={prev_score*100:.2f}%")

    print("Loading train..."); train_ds = Phase3Dataset(
        os.path.join(PROJECT_ROOT, args.npz_dir),
        os.path.join(PROJECT_ROOT, args.crops_dir),
        is_train, blacklist=blacklist)
    print("Loading val..."); val_ds = Phase3Dataset(
        os.path.join(PROJECT_ROOT, args.npz_dir),
        os.path.join(PROJECT_ROOT, args.crops_dir),
        is_val, blacklist=blacklist)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            persistent_workers=(args.num_workers > 0))

    optim = torch.optim.AdamW(v10c.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(out_dir, "train.log")
    open(log_path, "w").write("# phase 3 v10c training\n")

    best_score = -1.0
    for ep in range(args.epochs):
        v10c.train(); t0 = time.time()
        loss_sum = n_b = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device); pad = batch["pad"].to(device)
            ngs_x = batch["ngs_x"].to(device); row = batch["row"].to(device)
            num_idx = batch["num_idx"].to(device); num_pad = batch["num_pad"].to(device)
            crop_logits = batch["crop_logits"].to(device)
            # encoder + rfb pre-compute (frozen).
            with torch.no_grad():
                ef = encoder_features(encoder, tokens, pad)
                idx_exp = num_idx.unsqueeze(-1).expand(-1, -1, d_enc)
                num_feat = torch.gather(ef, 1, idx_exp)
                cls_logits, _, pre_head = rfb_forward_with_features_and_row(
                    rfb, num_feat, crop_logits, num_pad)
                num_pred_painted = cls_logits.argmax(dim=-1)
                num_pred_21 = PAINTED_TO_21.to(device)[num_pred_painted]
            # Scatter num predictions + features into per-token tensors.
            B, N, _ = tokens.shape
            num_class_gt = torch.full((B, N), -1, dtype=torch.long, device=device)
            num_rfb_features = torch.zeros((B, N, d_enc), device=device)
            for i in range(B):
                for j in range(num_idx.shape[1]):
                    if num_pad[i, j]: continue
                    idx = int(num_idx[i, j])
                    num_class_gt[i, idx] = int(num_pred_21[i, j])
                    num_rfb_features[i, idx] = pre_head[i, j]
            out = v10c(tokens, pad, num_class_gt=num_class_gt,
                       num_rfb_features=num_rfb_features)
            l1 = compute_pass_losses(tokens, out["logits_pass1"], pad,
                                       ngs_x, row.float().clamp(min=0))
            l2 = compute_pass_losses(tokens, out["logits_pass2"], pad,
                                       ngs_x, row.float().clamp(min=0))
            total = l1["total"] + l2["total"]
            optim.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(v10c.parameters(), 1.0)
            optim.step()
            loss_sum += total.item(); n_b += 1
        sched.step()
        v = evaluate(encoder, rfb, v10c, val_loader, device, d_enc)
        score = 0.5 * (v["yard"] + v["hash"])
        msg = (f"Ep {ep+1:3d}/{args.epochs} L={loss_sum/n_b:.3f} "
               f"{time.time()-t0:.0f}s  val: yard={v['yard']*100:.2f}% "
               f"hash={v['hash']*100:.2f}% sr={v['side_row']*100:.2f}% "
               f"hr={v['hash_row']*100:.2f}%  score={score*100:.2f}%")
        print(msg, flush=True); open(log_path, "a").write(msg + "\n")
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": v10c.state_dict(),
                        "args": vars(args), "epoch": ep + 1,
                        "val": v},
                       os.path.join(out_dir, "best.pth"))
            print(f"   ✓ saved (score={score*100:.2f}%)")
    print(f"Done. Best={best_score*100:.2f}%")


if __name__ == "__main__":
    main()
