"""Phase 2 — RF-B training on pseudo-label data.

Inputs (per number token):
  encoder_features  — from frozen phase-1 encoder
  crop_logits       — from frozen crop classifier (pre-extracted into
                       data/pseudo_labels_crops/<clip>.npz)
Targets:
  painted_class (9-class, mapped from NGS_x via PAINTED_TO_21)
  row (near/far, 0/1)

Encoder frozen. Only RF-B trains.

Token alignment: the pseudo-label npz's tokens and the crop sidecar's
crop_logits/num_token_idx are aligned by frame_idx + token order (the
extractor verified num-count match per frame). We pick out num tokens
using num_token_idx and pair them with crop_logits row-by-row.
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
from train_rf_a import encoder_features, N_PAINTED_CLASSES   # noqa: E402
from train_rf_b import RFB   # noqa: E402

TYPE_YARD, TYPE_SIDE, TYPE_HASH, TYPE_NUM = 0, 1, 2, 3
TOKEN_FEATURE_DIM = 16
# Mapping NGS_x class (0..20) → painted class (0..8) ; -1 if not painted.
NGS_X_TO_PAINTED = {2:0, 4:1, 6:2, 8:3, 10:4, 12:5, 14:6, 16:7, 18:8}


def ngs_x_to_painted_arr(arr):
    out = np.full_like(arr, -1, dtype=np.int64)
    for k, v in NGS_X_TO_PAINTED.items():
        out[arr == k] = v
    return out


# ── Dataset ─────────────────────────────────────────────────────────────────

class Phase2NumDataset(Dataset):
    """Per-frame view returning ONLY number tokens + their crop logits."""

    def __init__(self, npz_dir, crops_dir, clip_filter=None, blacklist=None):
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        if clip_filter is not None:
            files = [f for f in files if clip_filter(f)]
        blacklist = blacklist or set()
        self.files = files
        self.cache = []
        self.flat = []
        t0 = time.time()
        n_drop = 0
        for fi, f in enumerate(files):
            d = np.load(os.path.join(npz_dir, f), allow_pickle=True)
            cp = os.path.join(crops_dir, f)
            if not os.path.exists(cp):
                # Sidecar missing — skip this clip entirely.
                continue
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
            "type_idx": np.asarray(c["type_idx"][k], dtype=np.int64),
            "true_class": np.asarray(c["true_class"][k], dtype=np.int64),
            "true_row": np.asarray(c["true_row"][k], dtype=np.int64),
            "num_token_idx": np.asarray(c["num_token_idx"][k], dtype=np.int64),
            "crop_logits": np.asarray(c["crop_logits"][k], dtype=np.float32),
        }


def collate(batch):
    """Pad ALL tokens to max-N (for encoder forward). Separately stack num-only
    sub-batches into a (B, N_num_max, …) shape."""
    B = len(batch)
    Nmax = max(b["tokens"].shape[0] for b in batch)
    Nmax_num = max(max(b["num_token_idx"].shape[0], 1) for b in batch)
    tokens = np.zeros((B, Nmax, TOKEN_FEATURE_DIM), dtype=np.float32)
    pad = np.ones((B, Nmax), dtype=bool)
    num_pad = np.ones((B, Nmax_num), dtype=bool)
    num_idx_in_all = np.zeros((B, Nmax_num), dtype=np.int64)
    crop_logits = np.zeros((B, Nmax_num, N_PAINTED_CLASSES), dtype=np.float32)
    painted_gt = np.full((B, Nmax_num), -1, dtype=np.int64)
    row_gt = np.full((B, Nmax_num), -1, dtype=np.int64)
    for i, b in enumerate(batch):
        n = b["tokens"].shape[0]
        tokens[i, :n] = b["tokens"]
        pad[i, :n] = False
        ni = b["num_token_idx"]
        nn = len(ni)
        if nn == 0 or b["crop_logits"].shape[0] == 0:
            continue
        num_idx_in_all[i, :nn] = ni
        num_pad[i, :nn] = False
        crop_logits[i, :nn] = b["crop_logits"]
        # GT painted class + row, gathered at num positions.
        painted_gt[i, :nn] = ngs_x_to_painted_arr(b["true_class"][ni])
        row_gt[i, :nn] = b["true_row"][ni]
    return {
        "tokens": torch.from_numpy(tokens),
        "pad": torch.from_numpy(pad),
        "num_idx": torch.from_numpy(num_idx_in_all),
        "num_pad": torch.from_numpy(num_pad),
        "crop_logits": torch.from_numpy(crop_logits),
        "painted_gt": torch.from_numpy(painted_gt),
        "row_gt": torch.from_numpy(row_gt),
    }


@torch.no_grad()
def evaluate(encoder, rfb, loader, device):
    rfb.eval()
    cls_corr = cls_tot = row_corr = row_tot = 0
    for batch in loader:
        tokens = batch["tokens"].to(device); pad = batch["pad"].to(device)
        num_idx = batch["num_idx"].to(device); num_pad = batch["num_pad"].to(device)
        crop_logits = batch["crop_logits"].to(device)
        painted_gt = batch["painted_gt"].to(device); row_gt = batch["row_gt"].to(device)
        ef = encoder_features(encoder, tokens, pad)
        # Gather num features.
        B, Nn = num_idx.shape
        d_enc = ef.shape[-1]
        idx_exp = num_idx.unsqueeze(-1).expand(-1, -1, d_enc)
        num_feat = torch.gather(ef, 1, idx_exp)
        cls_logits, row_logits = rfb(num_feat, crop_logits, num_pad)
        valid_cls = (~num_pad) & (painted_gt >= 0)
        valid_row = (~num_pad) & (row_gt >= 0)
        cls_pred = cls_logits.argmax(dim=-1)
        row_pred = (row_logits > 0).long()
        cls_corr += int(((cls_pred == painted_gt) & valid_cls).sum())
        cls_tot += int(valid_cls.sum())
        row_corr += int(((row_pred == row_gt) & valid_row).sum())
        row_tot += int(valid_row.sum())
    return (cls_corr / max(1, cls_tot)), (row_corr / max(1, row_tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/pseudo_labels")
    ap.add_argument("--crops-dir", default="data/pseudo_labels_crops")
    ap.add_argument("--encoder-ckpt",
                   default="models/token_only_v10_phase1_pseudo/best.pth")
    ap.add_argument("--out-dir", default="models/rf_b_phase2_pseudo")
    ap.add_argument("--val-games", nargs="+",
                   default=["2024090802", "2024100601", "2024122201"])
    ap.add_argument("--blacklist", default=None)
    # RFB arch
    ap.add_argument("--d-model", type=int, default=96,
                   help="Match encoder d_model (96 for v10).")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    npz_dir = os.path.join(PROJECT_ROOT, args.npz_dir)
    crops_dir = os.path.join(PROJECT_ROOT, args.crops_dir)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    val_set = set(args.val_games)
    def is_val(name): return any(name.startswith(g + "_") for g in val_set)
    def is_train(name): return not is_val(name)

    blacklist = set()
    if args.blacklist:
        bl = json.load(open(os.path.join(PROJECT_ROOT, args.blacklist)))
        blacklist = set(bl.get("bad_gt_keys", []))
        print(f"Blacklist: {len(blacklist)} frames")

    # Encoder (frozen).
    ck = torch.load(os.path.join(PROJECT_ROOT, args.encoder_ckpt),
                    map_location="cpu", weights_only=False)
    sa = ck["args"]
    encoder = TokenClassifyV10(
        n_layers=sa["n_layers"], n_heads=sa["n_heads"],
        d_model=sa["d_model"], ffn_dim=sa["ffn_dim"],
        dropout=0.0, token_dropout=0.0).to(device).eval()
    encoder.load_state_dict(ck["model_state_dict"])
    for p in encoder.parameters(): p.requires_grad_(False)
    d_enc = sa["d_model"]

    print("Loading train set...")
    train_ds = Phase2NumDataset(npz_dir, crops_dir, is_train, blacklist=blacklist)
    print("Loading val set...")
    val_ds = Phase2NumDataset(npz_dir, crops_dir, is_val, blacklist=blacklist)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            persistent_workers=(args.num_workers > 0))

    rfb = RFB(d_enc=d_enc, d_model=args.d_model, n_heads=args.n_heads,
              ffn_dim=args.ffn_dim, dropout=args.dropout,
              with_row=True).to(device)
    n_p = sum(p.numel() for p in rfb.parameters()) / 1e3
    print(f"RFB: d_enc={d_enc} d_model={args.d_model} ({n_p:.1f}K params)")

    optim = torch.optim.AdamW(rfb.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(out_dir, "train.log")
    open(log_path, "w").write("# phase 2 RFB training\n")

    best_score = -1.0
    for ep in range(args.epochs):
        rfb.train()
        t0 = time.time()
        loss_sum = ce_sum = bce_sum = 0; n_b = 0
        cls_corr_t = cls_tot_t = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device); pad = batch["pad"].to(device)
            num_idx = batch["num_idx"].to(device); num_pad = batch["num_pad"].to(device)
            crop_logits = batch["crop_logits"].to(device)
            painted_gt = batch["painted_gt"].to(device); row_gt = batch["row_gt"].to(device)
            with torch.no_grad():
                ef = encoder_features(encoder, tokens, pad)
            B, Nn = num_idx.shape
            idx_exp = num_idx.unsqueeze(-1).expand(-1, -1, d_enc)
            num_feat = torch.gather(ef, 1, idx_exp)
            cls_logits, row_logits = rfb(num_feat, crop_logits, num_pad)
            valid_cls = (~num_pad) & (painted_gt >= 0)
            valid_row = (~num_pad) & (row_gt >= 0)
            ce = F.cross_entropy(cls_logits[valid_cls], painted_gt[valid_cls]) \
                if valid_cls.any() else torch.tensor(0.0, device=device)
            bce = F.binary_cross_entropy_with_logits(
                row_logits[valid_row], row_gt[valid_row].float()) \
                if valid_row.any() else torch.tensor(0.0, device=device)
            total = ce + bce
            optim.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(rfb.parameters(), 1.0)
            optim.step()
            loss_sum += total.item(); ce_sum += ce.item(); bce_sum += bce.item(); n_b += 1
            with torch.no_grad():
                cls_pred = cls_logits.argmax(dim=-1)
                cls_corr_t += int(((cls_pred == painted_gt) & valid_cls).sum())
                cls_tot_t += int(valid_cls.sum())
        sched.step()
        v_cls, v_row = evaluate(encoder, rfb, val_loader, device)
        train_acc = cls_corr_t / max(1, cls_tot_t)
        msg = (f"Ep {ep+1:3d}/{args.epochs}  "
               f"L={loss_sum/n_b:.3f} (ce={ce_sum/n_b:.3f} bce={bce_sum/n_b:.3f}) "
               f"{time.time()-t0:.0f}s  "
               f"train cls={train_acc*100:.2f}%  "
               f"val cls={v_cls*100:.2f}%  row={v_row*100:.2f}%")
        score = 0.5 * (v_cls + v_row)
        print(msg, flush=True); open(log_path, "a").write(msg + "\n")
        if score > best_score:
            best_score = score
            ck_out = {"model_state_dict": rfb.state_dict(),
                      "args": vars(args), "epoch": ep + 1,
                      "val_cls": v_cls, "val_row": v_row}
            torch.save(ck_out, os.path.join(out_dir, "best.pth"))
            print(f"   ✓ saved (score={score*100:.2f}%)")
    print(f"Done. Best={best_score*100:.2f}%")


if __name__ == "__main__":
    main()
