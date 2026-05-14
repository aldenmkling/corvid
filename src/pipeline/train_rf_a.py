"""RF-A: standalone pointwise fusion of v10 encoder features + crop logits.

Goal: measure how accurate a tiny pointwise classifier can be when given
v10's scene-aware encoded number features AND a standalone crop
classifier's per-cluster logits.

Per number token:
    input  = concat(v10_encoder_feature[d=96], crop_logits[d=9])
    output = refined 9-class softmax over painted-number values

Both v10's encoder and the crop classifier are loaded frozen — only the
fusion head trains. Per-cluster pointwise (no cross-token attention) —
the encoder already did all the cross-token work.

Compares against:
    crop classifier alone:  92.18% (dsresnet10w on val game)
    v10 inline classifier:  91.19% (encoder-only fusion)
    full SR pipeline:       97.93% (dsresnet10w + external SR)

If RF-A beats the SR pipeline, plug it into v10 Stage 2 as the anchor
source.
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
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer_v2 import (    # noqa: E402
    cc_tokens_from_frame_v2, null_classifier, TYPE_NUM, SRC_W, SRC_H,
    DEFAULT_DILATE_PX, _make_classifier_crop, MIN_CC_PX_NUM,
)
from train_token_v6 import (    # noqa: E402
    N_NGS_X_CLASSES, build_targets, ngs_x_to_class,
)
from train_dense_regression import NGS_X_MAX, split_by_game    # noqa: E402
from train_h_set_regressor import h_pixel_to_norm    # noqa: E402
from model_token_v10 import TokenClassifyV10    # noqa: E402
from train_scene_refiner import make_backbone_logits_fn    # noqa: E402


N_PAINTED_CLASSES = 9
PAINTED_CLASS_TO_NGS_X = [20, 30, 40, 50, 60, 70, 80, 90, 100]


def map_21class_to_painted(idx_21: torch.Tensor) -> torch.Tensor:
    """21-class NGS_x (5y from 10..110) → 9-class painted-number (10y from 20..100).

    Returns -1 for indices that don't sit on a painted-number value.
    21-class index of NGS_x=20 is round((20-10)/5)=2; NGS_x=30 → 4; etc.
    Painted-number 21-class indices: {2, 4, 6, 8, 10, 12, 14, 16, 18}.
    """
    out = torch.full_like(idx_21, -1)
    for cls9, idx21 in enumerate(range(2, 19, 2)):
        out = torch.where(idx_21 == idx21, torch.full_like(idx_21, cls9), out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RF-A model: pointwise MLP fusing encoder features + crop logits.
# ─────────────────────────────────────────────────────────────────────────────

class RFA(nn.Module):
    def __init__(self, d_enc: int, n_classes: int = N_PAINTED_CLASSES,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d_enc + n_classes, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, enc_feat: torch.Tensor,
                  crop_logits: torch.Tensor) -> torch.Tensor:
        x = torch.cat([enc_feat, crop_logits], dim=-1)
        return self.fuse(x)


# ─────────────────────────────────────────────────────────────────────────────
# Cache build: run frozen v10 encoder + crop classifier per number token.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_token_data(entries, cache_dir, intrinsics_by_clip,
                            encoder_model, crop_logits_fn, device,
                            d_enc: int):
    """Returns flat per-number-token tensors:
       enc_feats: (N_total, d_enc)
       crop_logits: (N_total, 9)  ← in painted-number class order (20→0..100→8)
       gt_class9: (N_total,)
       valid mask: (N_total,) — True if cluster has a valid GT (in painted range)
    """
    enc_list, crop_list, gt_list = [], [], []
    t0 = time.time()
    for ei, e in enumerate(entries):
        cp = os.path.join(cache_dir, f"{e['id']}.npz")
        if not os.path.exists(cp):
            continue
        d = np.load(cp)
        masks = d["masks"].astype(np.float32)
        # Tokenize via cc_tokenizer_v2 with null classifier (NGS_x=0 for nums).
        tokens_np = cc_tokens_from_frame_v2(masks, null_classifier)
        if tokens_np.shape[0] == 0:
            continue
        type_idx = tokens_np[..., :4].argmax(-1)
        is_num = (type_idx == TYPE_NUM)
        if not is_num.any():
            continue

        # GT classes (21-class) for all tokens, then map num tokens to 9-class.
        H_pixel = np.array(e["H"], dtype=np.float64)
        H_norm_gt = h_pixel_to_norm(H_pixel)
        tokens_t = torch.from_numpy(tokens_np).unsqueeze(0)
        H_t = torch.from_numpy(H_norm_gt.astype(np.float32)).unsqueeze(0)
        gt_21, _ = build_targets(tokens_t, H_t)
        gt_21 = gt_21[0]    # (N,)
        gt_9 = map_21class_to_painted(gt_21)

        # Run encoder (frozen) → per-token features (extract h_4 for numbers).
        pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool)
        enc_feat = encoder_features(encoder_model, tokens_t.to(device),
                                       pad.to(device))[0]    # (N, d)

        # Extract crops for number tokens, run crop classifier.
        bin_mask = (masks[..., 3] > 0.5).astype(np.uint8)
        num_indices = np.where(is_num)[0]
        if num_indices.size == 0:
            continue
        # Re-cluster to recover per-cluster crop label_map (cc_tokenizer
        # doesn't expose it). Mirror its logic exactly.
        crops = _crops_for_number_tokens(bin_mask, tokens_np[is_num])
        if not crops:
            continue
        crop_logits_21 = crop_logits_fn(crops)    # (M, 9) in painted order
        crop_logits = torch.from_numpy(crop_logits_21).float()

        # Collect.
        for j, ti in enumerate(num_indices):
            cls9 = int(gt_9[ti].item())
            if cls9 < 0:
                continue    # skip non-painted-number clusters
            enc_list.append(enc_feat[ti].cpu())
            crop_list.append(crop_logits[j])
            gt_list.append(cls9)

        if (ei + 1) % 200 == 0:
            print(f"  [{ei+1}/{len(entries)}]  collected={len(gt_list)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"  total tokens: {len(gt_list)}  ({time.time()-t0:.1f}s)")
    if not gt_list:
        return None, None, None
    return (torch.stack(enc_list), torch.stack(crop_list),
            torch.tensor(gt_list, dtype=torch.long))


def _crops_for_number_tokens(bin_mask: np.ndarray,
                                num_tokens: np.ndarray) -> list:
    """Re-extract per-number-token crops matching the current tokenizer.

    Replicates `cc_tokenizer_v2._process_number_channel_spatial`'s pipeline:
      1. CC on the raw binary mask (no dilation).
      2. Single-link clustering on centroids with cluster threshold =
         `NUMBER_GROUP_DIST_FRAC × yard_spacing` (estimated from the
         token-bbox heights as a scale proxy when no yardline-mask is
         available here).
      3. Median-area filter drops partial / clipped groups.
      4. Per kept group: tight-mask crop resized to (CLASSIFIER_CROP_W,
         CLASSIFIER_CROP_H).

    Order matches the number-token order produced by
    `cc_tokens_from_frame_v{2,3}`.
    """
    from cc_tokenizer_v2 import (
        MIN_RAW_NUM_CC_PX, NUM_AREA_MEDIAN_FRAC,
        NUMBER_GROUP_DIST_FRAC, NUMBER_GROUP_FALLBACK_PX,
        SRC_W, SRC_H, CLASSIFIER_CROP_W, CLASSIFIER_CROP_H,
    )
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if bin_mask.sum() == 0:
        return []

    # 1. CC on raw mask.
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8)
    cc_pixels = []
    cc_centroids = []
    for cid in range(1, n_cc):
        if int(stats[cid, cv2.CC_STAT_AREA]) < MIN_RAW_NUM_CC_PX:
            continue
        ys, xs = np.where(lbl == cid)
        cc_pixels.append((ys, xs))
        cc_centroids.append([float(xs.mean()), float(ys.mean())])

    if not cc_pixels:
        return []

    # Estimate yard spacing from num_tokens bbox-x spread.
    # The training scripts pass num_tokens already filtered to is_num,
    # so this is consistent with the cluster-threshold the tokenizer
    # used. Fall back to a constant if too few number tokens.
    if num_tokens.shape[0] >= 2:
        cxs = np.sort(num_tokens[:, 4] * SRC_W)
        gaps = np.diff(cxs)
        yard_spacing = float(np.median(gaps)) if len(gaps) else None
    else:
        yard_spacing = None

    if yard_spacing is None or yard_spacing <= 0:
        cluster_thr = float(NUMBER_GROUP_FALLBACK_PX)
    else:
        cluster_thr = float(NUMBER_GROUP_DIST_FRAC * yard_spacing)

    cc_arr = np.asarray(cc_centroids, dtype=np.float64)
    if len(cc_arr) == 1:
        group_ids = np.array([1], dtype=int)
    else:
        Z = linkage(pdist(cc_arr), method="single")
        group_ids = fcluster(Z, t=cluster_thr, criterion="distance")

    # 2. Build groups; median-area filter; emit crops in group-id order.
    raw_groups = []
    for gid in np.unique(group_ids):
        member = np.where(group_ids == gid)[0]
        ys_abs = np.concatenate([cc_pixels[i][0] for i in member])
        xs_abs = np.concatenate([cc_pixels[i][1] for i in member])
        raw_groups.append((member, ys_abs, xs_abs))
    if not raw_groups:
        return []
    median_area = float(np.median([len(g[1]) for g in raw_groups]))
    area_thr = NUM_AREA_MEDIAN_FRAC * median_area

    crops = []
    for member, ys_abs, xs_abs in raw_groups:
        if len(ys_abs) < area_thr:
            continue
        x_min = int(xs_abs.min()); y_min = int(ys_abs.min())
        x_max = int(xs_abs.max()) + 1; y_max = int(ys_abs.max()) + 1
        group_mask = np.zeros_like(bin_mask, dtype=np.uint8)
        for cc_idx in member:
            gys, gxs = cc_pixels[cc_idx]
            group_mask[gys, gxs] = 1
        sub = group_mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
        if sub.size == 0:
            crop = np.zeros((CLASSIFIER_CROP_H, CLASSIFIER_CROP_W),
                              dtype=np.uint8)
        else:
            crop = cv2.resize(sub,
                                  (CLASSIFIER_CROP_W, CLASSIFIER_CROP_H),
                                  interpolation=cv2.INTER_NEAREST)
        crops.append(crop)
    return crops


@torch.no_grad()
def encoder_features(model: TokenClassifyV10, tokens: torch.Tensor,
                        padding_mask: torch.Tensor) -> torch.Tensor:
    """Run the encoder portion of TokenClassifyV10 to get h_4 (per-token feat)."""
    # Replicate model.forward up through the encoder.
    from train_h_set_regressor import make_2d_pos_enc
    # Defensive zero-out of NGS_x for number tokens (matches v10's forward).
    type_idx = tokens[..., :4].argmax(dim=-1)
    is_num = (type_idx == TYPE_NUM)
    tokens = tokens.clone()
    tokens[..., 13] = torch.where(
        is_num, torch.zeros_like(tokens[..., 13]), tokens[..., 13])
    tokens[..., 14] = torch.where(
        is_num, torch.zeros_like(tokens[..., 14]), tokens[..., 14])

    feat = model.token_embed(tokens)
    cx = tokens[..., 4]; cy = tokens[..., 5]
    pe = make_2d_pos_enc(cx, cy, dim=model.d_model)
    feat = feat + pe * (~padding_mask).unsqueeze(-1).to(pe.dtype)
    return model.encoder(feat, src_key_padding_mask=padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Crop classifier wrapper that returns logits in PAINTED-NUMBER class order.
# ─────────────────────────────────────────────────────────────────────────────

def make_painted_logits_fn(crop_ckpt: str, arch: str, device: torch.device):
    """Wraps make_backbone_logits_fn (which returns 9-D logits in
    train_number_classifier.CLASSES order). Re-arrange to painted order
    20→0, 30→1, ..., 100→8."""
    from train_number_classifier import CLASSES as TRAIN_CLASSES
    base_fn = make_backbone_logits_fn(crop_ckpt, arch, device)
    # Map indices: train order → painted order.
    name_to_painted = {
        "10L": 0, "20L": 1, "30L": 2, "40L": 3, "50": 4,
        "40R": 5, "30R": 6, "20R": 7, "10R": 8,
    }
    perm = np.array([name_to_painted[c] for c in TRAIN_CLASSES], dtype=int)
    inv_perm = np.argsort(perm)    # painted-order index → original

    def _fn(crops):
        logits = base_fn(crops)    # (N, 9) in TRAIN_CLASSES order
        return logits[:, inv_perm]    # (N, 9) in painted order
    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Train.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v10_stage1_gt_val/best.pth"))
    ap.add_argument("--crop-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/dsresnet10w_round3/best.pth"))
    ap.add_argument("--crop-arch", default="dsresnet10w",
                     choices=["dsresnet10w", "dsresnet10", "mbconv",
                                "mbconv_mini", "tiny", "mininet"])
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/rf_a"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-game", default="2024090802")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load v10 encoder.
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
    print(f"  d_enc = {d_enc}")

    # Load crop classifier.
    print(f"Loading crop classifier {args.crop_arch} from {args.crop_ckpt}...")
    crop_logits_fn = make_painted_logits_fn(
        args.crop_ckpt, args.crop_arch, device)

    # Manifest split.
    print(f"Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr = m.get("intrinsics_by_clip", {})
    train_e, val_e, _ = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)
    print(f"Split: train={len(train_e)}  val={len(val_e)}")

    # Precompute features for both splits.
    print("Building train cache...")
    tr_enc, tr_crop, tr_gt = precompute_token_data(
        train_e, args.cache_dir, intr, encoder_model, crop_logits_fn,
        device, d_enc)
    print("Building val cache...")
    val_enc, val_crop, val_gt = precompute_token_data(
        val_e, args.cache_dir, intr, encoder_model, crop_logits_fn,
        device, d_enc)

    # Crop-only baseline accuracy on val.
    crop_pred = val_crop.argmax(-1)
    crop_acc = (crop_pred == val_gt).float().mean().item()
    print(f"\nCrop-only baseline (val): {crop_acc*100:.2f}% "
          f"({(crop_pred == val_gt).sum().item()}/{len(val_gt)})")

    # RF-A model.
    rfa = RFA(d_enc=d_enc, hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in rfa.parameters())
    print(f"RFA: {n_params:,} params  ({n_params/1e3:.2f}K)")

    optim = torch.optim.AdamW(rfa.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01)

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val = 0.0

    n_train = tr_gt.shape[0]
    for epoch in range(args.epochs):
        rfa.train()
        perm = torch.randperm(n_train)
        train_loss = 0.0; n_b = 0; train_corr = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s + args.batch_size]
            enc_b = tr_enc[idx].to(device)
            crop_b = tr_crop[idx].to(device)
            gt_b = tr_gt[idx].to(device)
            logits = rfa(enc_b, crop_b)
            loss = F.cross_entropy(logits, gt_b)
            optim.zero_grad(); loss.backward(); optim.step()
            train_loss += loss.item() * gt_b.numel()
            n_b += gt_b.numel()
            train_corr += int((logits.argmax(-1) == gt_b).sum())
        sched.step()
        train_loss /= max(n_b, 1)
        train_acc = train_corr / max(n_b, 1)

        # Val
        rfa.eval()
        with torch.no_grad():
            logits = rfa(val_enc.to(device), val_crop.to(device))
            val_pred = logits.argmax(-1).cpu()
        val_acc = (val_pred == val_gt).float().mean().item()

        print(f"Ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc*100:.2f}%  val_acc={val_acc*100:.2f}%  "
              f"crop_baseline={crop_acc*100:.2f}%")

        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch+1, "train_loss": train_loss,
                "train_acc": train_acc, "val_acc": val_acc,
                "crop_baseline": crop_acc,
            }, f); f.write("\n")

        ckpt = {"model_state_dict": rfa.state_dict(),
                "epoch": epoch+1, "args": vars(args),
                "val_acc": val_acc, "crop_baseline": crop_acc}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best val_acc = {val_acc*100:.2f}%")

    print(f"\nDone. RF-A best val_acc: {best_val*100:.2f}%  "
          f"(crop alone: {crop_acc*100:.2f}%, "
          f"Δ = {(best_val - crop_acc)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
