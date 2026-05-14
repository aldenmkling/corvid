"""v10c Stage 2 — clean architecture: numbers exit the cross-attention path.

Premise (vs v10b):
  v10b kept number tokens in the query stream but locked their features
  post-encoder + RF-B residual. The readout still produced num NGS_x and
  num row predictions, but those were never as good as RF-B's own outputs
  because the readout had to live inside the locked-features bottleneck.

v10c instead:
  - RF-B is extended to predict row alongside class (both are sourced
    upstream of cross-attention).
  - Number tokens are still passed through the encoder (so non-number
    tokens see them in self-attention) and serve as cross-attention
    anchors via num_class_gt = RF-B argmax.
  - Number rows + NGS_x are taken DIRECTLY from RF-B at inference.
  - Loss drops num_ce AND num_row (RF-B owns both).
  - Cross-attention readout for numbers still runs (locked features →
    head) but its outputs are ignored everywhere downstream. Architecturally
    the readout for numbers is a no-op stub.

Why this is the right shape before we add temporal state:
  - Numbers have a single source of truth (RF-B). State updates layer on
    top of RF-B, not on top of two competing predictors.
  - Yard/hash readout learns to reason from clean number anchors without
    being asked to also fix up number predictions.
"""
from __future__ import annotations

import argparse
import json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from cc_tokenizer_v2 import (    # noqa: E402
    cc_tokens_from_frame_v2, null_classifier, TYPE_NUM,
)
import cv2 as _cv2_v10c    # noqa: E402
from cc_tokenizer_v3 import cc_tokens_from_frame_v3    # noqa: E402
from train_token_v6 import (    # noqa: E402
    build_targets, compute_pass_losses, N_NGS_X_CLASSES,
)
from train_dense_regression import split_by_game    # noqa: E402
from train_h_set_regressor import h_pixel_to_norm    # noqa: E402
from model_token_v10 import TokenClassifyV10    # noqa: E402
from model_token_v10b import TokenClassifyV10b    # noqa: E402
from train_rf_a import (    # noqa: E402
    encoder_features, _crops_for_number_tokens,
    make_painted_logits_fn,
)
from train_rf_b import RFB    # noqa: E402
from train_token_v8 import AugmentedHSetDataset    # noqa: E402


PAINTED_TO_21 = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=torch.long)


@torch.no_grad()
def rfb_forward_with_features_and_row(rfb: RFB, enc_feat, crop_logits,
                                       padding_mask):
    """Run extended RF-B; return (class_logits, row_logits, pre_head_features).

    Requires rfb.with_row=True.
    """
    x = torch.cat([enc_feat, crop_logits], dim=-1)
    x = rfb.embed(x)
    x = rfb.transformer(x, src_key_padding_mask=padding_mask)
    pre_head = x
    cls_logits = rfb.head(x)
    row_logits = rfb.head_row(x).squeeze(-1)
    return cls_logits, row_logits, pre_head


class HSetDatasetV10cS2(Dataset):
    """Same precompute as v10b S2, plus row prediction from RF-B."""

    def __init__(self, entries, cache_dir, intrinsics_by_clip,
                 encoder_model, crop_logits_fn, rfb_model, device, d_enc,
                 conf_threshold: float = 0.0):
        self.entries = entries
        self.cache_dir = cache_dir
        self.intrinsics_by_clip = intrinsics_by_clip or {}
        self.encoder_model = encoder_model
        self.crop_logits_fn = crop_logits_fn
        self.rfb_model = rfb_model
        self.device = device
        self.d_enc = d_enc
        self.conf_threshold = conf_threshold
        self._cache: list = []
        self._build_cache()

    @torch.no_grad()
    def _build_cache(self):
        t0 = time.time()
        n_skipped = 0
        for ei, e in enumerate(self.entries):
            cp = os.path.join(self.cache_dir, f"{e['id']}.npz")
            if not os.path.exists(cp):
                n_skipped += 1; continue
            d = np.load(cp)
            masks = d["masks"].astype(np.float32)
            # Undistort masks → tokens in undistorted-pixel space (matches
            # manifest H + the new staged pipeline).
            intr = self.intrinsics_by_clip.get(e["clip"], {})
            K = np.asarray(intr.get("K", np.eye(3)), dtype=np.float64)
            if K.shape == (9,):
                K = K.reshape(3, 3)
            dist = np.asarray(intr.get("dist", [0, 0, 0, 0, 0]),
                                  dtype=np.float64)
            masks = _cv2_v10c.undistort(masks, K, dist)
            tokens_np, aux = cc_tokens_from_frame_v3(
                masks, null_classifier, return_aux=True)
            if tokens_np.shape[0] == 0:
                n_skipped += 1; continue
            type_idx = tokens_np[..., :4].argmax(-1)
            is_num = (type_idx == TYPE_NUM)

            num_anchor_class = np.full(tokens_np.shape[0], -1, dtype=np.int64)
            num_anchor_row = np.zeros(tokens_np.shape[0], dtype=np.float32)
            num_has_row = np.zeros(tokens_np.shape[0], dtype=bool)
            rfb_features = np.zeros(
                (tokens_np.shape[0], self.d_enc), dtype=np.float32)

            if is_num.any():
                tokens_t = torch.from_numpy(tokens_np).unsqueeze(0)
                pad = torch.zeros(1, tokens_np.shape[0], dtype=torch.bool)
                enc_feat = encoder_features(
                    self.encoder_model, tokens_t.to(self.device),
                    pad.to(self.device))[0]
                crops = aux["num_crops"]
                if crops:
                    crop_logits_np = self.crop_logits_fn(crops)
                    crop_logits = torch.from_numpy(crop_logits_np).float().to(
                        self.device)
                    num_indices = np.where(is_num)[0]
                    enc_num = enc_feat[num_indices].unsqueeze(0)
                    crop_num = crop_logits.unsqueeze(0)
                    pad_num = torch.zeros(
                        1, len(num_indices), dtype=torch.bool,
                        device=self.device)
                    rfb_logits, rfb_row_logits, rfb_pre_head = (
                        rfb_forward_with_features_and_row(
                            self.rfb_model, enc_num, crop_num, pad_num))
                    rfb_logits = rfb_logits[0]
                    rfb_row_logits = rfb_row_logits[0]
                    rfb_pre_head = rfb_pre_head[0]
                    rfb_probs = torch.softmax(rfb_logits, dim=-1)
                    classifier_conf = rfb_probs.max(-1).values.cpu().numpy()
                    pred_painted = rfb_logits.argmax(-1).cpu().numpy()
                    pred_21 = PAINTED_TO_21.numpy()[pred_painted]
                    pred_row = torch.sigmoid(rfb_row_logits).cpu().numpy()
                    rfb_pre_head_np = rfb_pre_head.cpu().numpy()
                    for j, ti in enumerate(num_indices):
                        # Confidence-threshold filter: skip low-confidence
                        # number anchors entirely (treat as "no anchor"
                        # rather than risk feeding a wrong NGS_x to
                        # cross-attention).
                        if classifier_conf[j] < self.conf_threshold:
                            continue
                        num_anchor_class[ti] = int(pred_21[j])
                        num_anchor_row[ti] = float(pred_row[j])
                        num_has_row[ti] = True
                        tokens_np[ti, 15] = float(
                            tokens_np[ti, 15] * classifier_conf[j])
                        rfb_features[ti] = rfb_pre_head_np[j]

            H_pixel = np.array(e["H"], dtype=np.float64)
            H_norm_gt = h_pixel_to_norm(H_pixel)
            self._cache.append({
                "tokens": torch.from_numpy(tokens_np),
                "H_norm_gt": torch.from_numpy(H_norm_gt.astype(np.float32)),
                "num_anchor_class": torch.from_numpy(num_anchor_class),
                "num_anchor_row": torch.from_numpy(num_anchor_row),
                "num_has_row": torch.from_numpy(num_has_row),
                "rfb_features": torch.from_numpy(rfb_features),
            })
            if (ei + 1) % 200 == 0:
                print(f"  [{ei+1}/{len(self.entries)}]  cached  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        elapsed = time.time() - t0
        print(f"  cached {len(self._cache)} frames "
              f"(skipped {n_skipped}) in {elapsed:.1f}s", flush=True)
        # Drop unpicklable refs.
        self.encoder_model = None
        self.crop_logits_fn = None
        self.rfb_model = None

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        item = self._cache[idx]
        return {
            "tokens": item["tokens"].clone(),
            "H_norm_gt": item["H_norm_gt"],
            "num_anchor_class": item["num_anchor_class"].clone(),
            "num_anchor_row": item["num_anchor_row"].clone(),
            "num_has_row": item["num_has_row"].clone(),
            "rfb_features": item["rfb_features"].clone(),
        }


def collate_v10cs2(batch):
    n_max = max(item["tokens"].shape[0] for item in batch)
    if n_max == 0:
        n_max = 1
    B = len(batch)
    F_dim = batch[0]["tokens"].shape[1]
    d_rfb = batch[0]["rfb_features"].shape[1]
    tokens = torch.zeros(B, n_max, F_dim, dtype=torch.float32)
    H_gt = torch.stack([item["H_norm_gt"] for item in batch])
    num_anchor = torch.full((B, n_max), -1, dtype=torch.long)
    num_anchor_row = torch.zeros(B, n_max, dtype=torch.float32)
    num_has_row = torch.zeros(B, n_max, dtype=torch.bool)
    rfb_feat = torch.zeros(B, n_max, d_rfb, dtype=torch.float32)
    pad = torch.ones(B, n_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        if n > 0:
            tokens[i, :n] = item["tokens"]
            num_anchor[i, :n] = item["num_anchor_class"]
            num_anchor_row[i, :n] = item["num_anchor_row"]
            num_has_row[i, :n] = item["num_has_row"]
            rfb_feat[i, :n] = item["rfb_features"]
            pad[i, :n] = False
    return {"tokens": tokens, "H_norm_gt": H_gt,
              "padding_mask": pad,
              "num_anchor_class": num_anchor,
              "num_anchor_row": num_anchor_row,
              "num_has_row": num_has_row,
              "rfb_features": rfb_feat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v10_stage1_gt_val/best.pth"))
    ap.add_argument("--rfb-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/rf_b_dsresnet_d96_ffn96_row/best.pth"))
    ap.add_argument("--crop-ckpt", default=os.path.join(
        PROJECT_ROOT, "models/dsresnet10w_round3/best.pth"))
    ap.add_argument("--crop-arch", default="dsresnet10w")
    ap.add_argument("--manifest-file", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--cache-dir", default=os.path.join(
        PROJECT_ROOT, "data/dense_regression/cache"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "models/token_only_v10c_stage2"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--centroid-sigma", type=float, default=2.0 / 1280.0)
    ap.add_argument("--bbox-sigma", type=float, default=2.0 / 1280.0)
    ap.add_argument("--max-angle-deg", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-game", default="2024090802")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                     help="Drop number anchors with RF-B confidence below "
                          "this threshold (treated as 'no anchor').")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading v10 Stage 1 from {args.stage1_ckpt}...")
    s1_ck = torch.load(args.stage1_ckpt, map_location="cpu",
                          weights_only=False)
    s1_args = s1_ck["args"]
    encoder_v10 = TokenClassifyV10(
        n_layers=s1_args["n_layers"], n_heads=s1_args["n_heads"],
        d_model=s1_args["d_model"], ffn_dim=s1_args["ffn_dim"],
        dropout=0.0, token_dropout=0.0).to(device)
    encoder_v10.load_state_dict(s1_ck["model_state_dict"])
    encoder_v10.eval()
    for p in encoder_v10.parameters():
        p.requires_grad = False
    d_enc = s1_args["d_model"]

    model = TokenClassifyV10b(
        n_layers=s1_args["n_layers"], n_heads=s1_args["n_heads"],
        d_model=s1_args["d_model"], ffn_dim=s1_args["ffn_dim"],
        dropout=s1_args["dropout"],
        token_dropout=s1_args["token_dropout"]).to(device)
    model.load_state_dict(s1_ck["model_state_dict"])
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in
               ("encoder.", "token_embed.", "anchor_proj.",
                "num_classifier.")):
            p.requires_grad = False
    model.global_anchor.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {n_train:,} / {n_total:,} params "
          f"({n_train/n_total*100:.1f}%)")

    print(f"Loading extended RF-B from {args.rfb_ckpt}...")
    rfb_ck = torch.load(args.rfb_ckpt, map_location="cpu",
                          weights_only=False)
    rfb_args = rfb_ck["args"]
    if not rfb_args.get("with_row", False):
        raise ValueError(
            f"v10c requires RF-B trained with --with-row. "
            f"{args.rfb_ckpt} was trained without it.")
    rfb = RFB(d_enc=d_enc, d_model=rfb_args["d_model"],
                n_heads=rfb_args["n_heads"], ffn_dim=rfb_args["ffn_dim"],
                dropout=0.0, with_row=True).to(device)
    rfb.load_state_dict(rfb_ck["model_state_dict"])
    rfb.eval()
    for p in rfb.parameters():
        p.requires_grad = False
    print(f"  RF-B d_model={rfb_args['d_model']}  "
          f"val_acc={rfb_ck.get('val_acc',0)*100:.2f}%  "
          f"val_row_acc={rfb_ck.get('val_row_acc',0)*100:.2f}%")
    if rfb_args["d_model"] != s1_args["d_model"]:
        raise ValueError(
            f"RF-B d_model ({rfb_args['d_model']}) must match v10 d_model "
            f"({s1_args['d_model']}) for residual addition.")

    print(f"Loading crop classifier {args.crop_arch}...")
    crop_logits_fn = make_painted_logits_fn(
        args.crop_ckpt, args.crop_arch, device)

    print("Loading manifest...")
    m = json.load(open(args.manifest_file))
    intr = m.get("intrinsics_by_clip", {})
    train_e, val_e, val_game = split_by_game(
        m["entries"], val_game=args.val_game, val_frac=0.1, seed=args.seed)
    print(f"Split: train={len(train_e)}  val={len(val_e)}")

    print(f"Building train cache (conf_threshold={args.conf_threshold})...")
    train_ds = HSetDatasetV10cS2(
        train_e, args.cache_dir, intr,
        encoder_v10, crop_logits_fn, rfb, device, d_enc,
        conf_threshold=args.conf_threshold)
    print("Building val cache...")
    val_ds = HSetDatasetV10cS2(
        val_e, args.cache_dir, intr,
        encoder_v10, crop_logits_fn, rfb, device, d_enc,
        conf_threshold=args.conf_threshold)

    aug_train = AugmentedHSetDataset(
        train_ds, enabled=True,
        centroid_sigma=args.centroid_sigma,
        bbox_sigma=args.bbox_sigma,
        max_angle_deg=args.max_angle_deg)
    aug_val = AugmentedHSetDataset(val_ds, enabled=False)

    train_loader = DataLoader(
        aug_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_v10cs2,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        aug_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_v10cs2,
        persistent_workers=(args.num_workers > 0))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)

    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    best_val_score = 0.0

    from cc_tokenizer_v2 import TYPE_YARD, TYPE_HASH

    # ── losses-only-on-non-numbers helper ──
    def total_no_num(L):
        # Drop num_ce + num_row entirely (RF-B owns numbers).
        return (L["yard_ce"] + L["hash_ce"]
                + L["side_row"] + L["hash_row"])

    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()
        model.token_embed.eval()
        model.anchor_proj.eval()
        model.num_classifier.eval()

        t0 = time.time()
        train_loss = 0.0; n_b = 0
        train_yard_corr = train_yard_tot = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["padding_mask"].to(device, non_blocking=True)
            H_gt = batch["H_norm_gt"].to(device, non_blocking=True)
            num_anchor = batch["num_anchor_class"].to(
                device, non_blocking=True)
            rfb_feat = batch["rfb_features"].to(device, non_blocking=True)
            ngs_x_class, row_target = build_targets(tokens, H_gt)
            out = model(tokens, mask, num_class_gt=num_anchor,
                          num_rfb_features=rfb_feat)
            l1 = compute_pass_losses(tokens, out["logits_pass1"], mask,
                                       ngs_x_class, row_target)
            l2 = compute_pass_losses(tokens, out["logits_pass2"], mask,
                                       ngs_x_class, row_target)
            total = total_no_num(l1) + total_no_num(l2)
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            train_loss += total.item()
            n_b += 1
            with torch.no_grad():
                ngs_x_pred = out["logits_pass2"][..., :N_NGS_X_CLASSES] \
                    .argmax(-1)
                type_idx = tokens[..., :4].argmax(-1)
                valid = ~mask
                m_ = (type_idx == TYPE_YARD) & valid
                if m_.any():
                    train_yard_corr += int(
                        (ngs_x_pred[m_] == ngs_x_class[m_]).sum())
                    train_yard_tot += int(m_.sum())
        sched.step()
        elapsed = time.time() - t0
        train_yard_acc = train_yard_corr / max(1, train_yard_tot)

        model.eval()
        # yard/hash come from readout; num NGS_x and num row come from RF-B
        # (already cached as num_anchor_class / num_anchor_row).
        val_correct = {"yard": 0, "hash": 0, "num": 0}
        val_total = {"yard": 0, "hash": 0, "num": 0}
        val_row_correct = {"side": 0, "hash": 0, "num": 0}
        val_row_total = {"side": 0, "hash": 0, "num": 0}
        from cc_tokenizer_v2 import TYPE_SIDE
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                mask = batch["padding_mask"].to(device)
                H_gt = batch["H_norm_gt"].to(device)
                num_anchor = batch["num_anchor_class"].to(device)
                num_anchor_row = batch["num_anchor_row"].to(device)
                rfb_feat = batch["rfb_features"].to(device)
                ngs_x_class, row_target = build_targets(tokens, H_gt)
                out = model(tokens, mask, num_class_gt=num_anchor,
                              num_rfb_features=rfb_feat)
                pred = out["logits_pass2"][..., :N_NGS_X_CLASSES].argmax(-1)
                row_logit = out["logits_pass2"][..., N_NGS_X_CLASSES]
                row_pred = (row_logit > 0).float()
                type_idx = tokens[..., :4].argmax(-1)
                valid = ~mask
                # Yards/hashes from readout.
                for tt, name in [(TYPE_YARD, "yard"), (TYPE_HASH, "hash")]:
                    m_ = (type_idx == tt) & valid
                    if m_.any():
                        val_correct[name] += int(
                            (pred[m_] == ngs_x_class[m_]).sum())
                        val_total[name] += int(m_.sum())
                # Numbers from RF-B (cached argmax).
                m_num = (type_idx == TYPE_NUM) & valid
                if m_num.any():
                    val_correct["num"] += int(
                        (num_anchor[m_num] == ngs_x_class[m_num]).sum())
                    val_total["num"] += int(m_num.sum())
                # Side row from readout.
                m_side = (type_idx == TYPE_SIDE) & valid
                if m_side.any():
                    val_row_correct["side"] += int(
                        (row_pred[m_side] == row_target[m_side]).sum())
                    val_row_total["side"] += int(m_side.sum())
                # Hash row from readout.
                m_hash = (type_idx == TYPE_HASH) & valid
                if m_hash.any():
                    val_row_correct["hash"] += int(
                        (row_pred[m_hash] == row_target[m_hash]).sum())
                    val_row_total["hash"] += int(m_hash.sum())
                # Num row from RF-B (cached sigmoid > 0.5).
                if m_num.any():
                    num_row_pred = (num_anchor_row[m_num] > 0.5).float()
                    val_row_correct["num"] += int(
                        (num_row_pred == row_target[m_num]).sum())
                    val_row_total["num"] += int(m_num.sum())
        yard_acc = val_correct["yard"] / max(1, val_total["yard"])
        hash_acc = val_correct["hash"] / max(1, val_total["hash"])
        num_acc = val_correct["num"] / max(1, val_total["num"])
        side_row_acc = val_row_correct["side"] / max(1, val_row_total["side"])
        hash_row_acc = val_row_correct["hash"] / max(1, val_row_total["hash"])
        num_row_acc = val_row_correct["num"] / max(1, val_row_total["num"])
        val_score = (yard_acc + hash_acc) / 2

        gap = train_yard_acc - yard_acc
        print(f"Ep {epoch+1:3d}/{args.epochs}  L={train_loss/n_b:.3f}  "
              f"({elapsed:.0f}s)  train_yard={train_yard_acc*100:.1f}%  "
              f"val: yard={yard_acc*100:.1f}% hash={hash_acc*100:.1f}% "
              f"num={num_acc*100:.1f}%  rows: s={side_row_acc*100:.1f}% "
              f"h={hash_row_acc*100:.1f}% n={num_row_acc*100:.1f}%  "
              f"GAP={gap*100:+.1f}%  score={val_score*100:.2f}%", flush=True)

        with open(log_path, "a") as f:
            json.dump({
                "epoch": epoch+1, "lr": sched.get_last_lr()[0],
                "train_loss": train_loss/n_b,
                "train_yard_acc": train_yard_acc,
                "yard_top1": yard_acc, "hash_top1": hash_acc,
                "num_top1": num_acc,
                "side_row_acc": side_row_acc,
                "hash_row_acc": hash_row_acc,
                "num_row_acc": num_row_acc,
                "val_score": val_score, "gap_yard": gap,
            }, f); f.write("\n")

        ckpt = {"model_state_dict": model.state_dict(),
                "epoch": epoch+1, "args": vars(args),
                "val_score": val_score}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"   ↑ new best val_score = {val_score*100:.2f}%",
                  flush=True)

    print(f"\nDone. v10c Stage 2 best val_score: {best_val_score*100:.2f}%  "
          f"(v10b Stage 2 baseline: 98.25%)")


if __name__ == "__main__":
    main()
