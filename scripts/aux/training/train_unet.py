"""Train the unified mask specialist.

Architecture:
  - smp.Segformer encoder (default mit_b0) + native MLP decoder
  - Input: 3 channels (RGB)
  - Output: 4 channels (binary masks for yard, side, hash, number)
  - Loss: per-channel BCE + Dice (sum)

Training data:
  - Pseudo-labels from data/unified_masks/round1/raw/<id>.npz
  - decisions.json filters to user-Y'd entries
  - Train/val split by game (no leak)

Validation:
  - Per-channel F1 (binarize at 0.5)
  - Per-channel Dice
  - Compared to source specialists' val F1 baselines

Usage:
    python scripts/training/train_unified_mask.py \\
        --pool-dir data/unified_masks/round1 \\
        --out-dir models/unet_unified_v1 \\
        --device cuda --epochs 50
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import segmentation_models_pytorch as smp


INPUT_H, INPUT_W = 512, 896             # match existing specialist input size
N_CHANNELS_OUT = 4                       # yard, side, hash, number
CHANNEL_NAMES = ["yard", "side", "hash", "number"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Augmentations ──
def aug_color_jitter(rgb, brightness=0.2, contrast=0.2, saturation=0.15):
    img = rgb.astype(np.float32)
    if brightness > 0:
        img = img * random.uniform(1 - brightness, 1 + brightness)
    if contrast > 0:
        m = img.mean()
        img = (img - m) * random.uniform(1 - contrast, 1 + contrast) + m
    if saturation > 0:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)[..., None]
        img = gray + (img - gray) * random.uniform(1 - saturation, 1 + saturation)
    return np.clip(img, 0, 255).astype(np.uint8)


def aug_random_hflip(rgb, masks, p=0.5):
    """Horizontal flip — fine here because masks are class-symmetric
    (a yardline mirror'd is still a yardline). Different from dense
    regression where flip would invert NGS_x."""
    if random.random() < p:
        rgb = cv2.flip(rgb, 1)
        masks = cv2.flip(masks, 1)
        if masks.ndim == 2:
            masks = masks[..., None]
    return rgb, masks


def aug_random_crop_resize(rgb, masks, scale_range=(0.7, 1.0)):
    """Mild crop. Lower scale_range than the dense regression model
    because mask supervision is per-pixel and we want to keep most of the
    image in view (especially the markings)."""
    h, w = rgb.shape[:2]
    s = random.uniform(*scale_range)
    crop_h = max(1, int(h * s))
    crop_w = max(1, int(w * s))
    y0 = random.randint(0, h - crop_h)
    x0 = random.randint(0, w - crop_w)
    rgb_c = rgb[y0:y0+crop_h, x0:x0+crop_w]
    masks_c = masks[y0:y0+crop_h, x0:x0+crop_w]
    rgb_r = cv2.resize(rgb_c, (w, h), interpolation=cv2.INTER_LINEAR)
    masks_r = cv2.resize(masks_c, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb_r, masks_r


# ── Dataset ──
class UnifiedMaskDataset(Dataset):
    def __init__(self, entries, raw_dir, augment=False, use_raw_masks=False,
                  shift_hash_up_px=0):
        self.entries = entries
        self.raw_dir = raw_dir
        self.augment = augment
        self.use_raw_masks = use_raw_masks   # if True, train on un-cleaned specialist outputs
        self.shift_hash_up_px = shift_hash_up_px   # shift hash mask UP N px (native coords)
                                                    # used to correct a systematic
                                                    # "predictions drift low" bias

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        d = np.load(os.path.join(self.raw_dir, f"{e['id']}.npz"))
        rgb = d["rgb"].copy()        # HxWx3 BGR uint8
        mask_key = "raw_masks" if self.use_raw_masks else "masks"
        if mask_key not in d.files:
            raise KeyError(
                f"npz {e['id']} missing key '{mask_key}'. "
                f"Available: {d.files}. If you want raw_masks, rebuild "
                f"the QC dataset (--reuse-rgb) so each npz has both fields.")
        masks = d[mask_key].copy()    # HxWx4 uint8 binary

        # Optional: shift the hash channel UP by N pixels (in native coords) to
        # correct a systematic "predictions drift low" offset. Done BEFORE
        # resize so N maps to native image pixels. The bottom N rows of the
        # hash channel become 0 (which is fine — there's never a hash mark
        # within N px of the bottom edge).
        if self.shift_hash_up_px > 0:
            n = self.shift_hash_up_px
            shifted = np.zeros_like(masks[..., 2])
            shifted[:-n, :] = masks[n:, :, 2]
            masks[..., 2] = shifted

        # Resize to network input size
        h, w = rgb.shape[:2]
        if (h, w) != (INPUT_H, INPUT_W):
            rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            masks = cv2.resize(masks, (INPUT_W, INPUT_H),
                                interpolation=cv2.INTER_NEAREST)

        if self.augment:
            rgb, masks = aug_random_hflip(rgb, masks)
            if random.random() < 0.5:
                rgb, masks = aug_random_crop_resize(rgb, masks)
            rgb = aug_color_jitter(rgb)

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_rgb = (rgb_rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb_rgb.transpose(2, 0, 1))
        masks_t = torch.from_numpy(masks.transpose(2, 0, 1).astype(np.float32))
        return rgb_t, masks_t


# ── Model ──
def build_model(encoder_name="mit_b0", decoder="unet"):
    """Build the unified mask model.

    decoder='unet' (default): mit_b0 encoder + U-Net decoder with skip
        connections. Better for thin-feature segmentation (hash marks,
        sidelines). ~5.5M total params.
    decoder='segformer': mit_b0 encoder + native MLP decoder. Lighter
        (~3.7M total) but loses precision on thin features. Used in v1.
    """
    common = dict(encoder_name=encoder_name,
                  encoder_weights="imagenet",
                  in_channels=3, classes=N_CHANNELS_OUT)
    if decoder == "unet":
        return smp.Unet(**common)
    if decoder == "segformer":
        return smp.Segformer(**common)
    raise ValueError(f"unknown decoder: {decoder}")


# ── Loss ──
# Per-channel BCE weights — upweight sparse/hard channels so they don't
# get drowned out by easy negatives. Dice is already per-channel-balanced.
# Order: (yard, side, hash, number)
# v1-v3 used (1, 1, 3, 1.5) when hash was the laggard channel. After v3 the
# hash channel reached specialist parity (~0.95) and the bottleneck shifted
# to yard/side; rebalanced to (1, 1.5, 2, 1): boost side (the worst gap to
# specialist), trim hash (already solved), drop number to base.
DEFAULT_CHANNEL_WEIGHTS = (1.0, 1.5, 2.0, 1.0)


def bce_dice_loss(logits, targets, channel_weights=DEFAULT_CHANNEL_WEIGHTS,
                    dice_weight=2.0, bce_weight=1.0):
    """Per-channel weighted BCE + per-channel soft Dice."""
    # Per-channel BCE (mean over batch + spatial), then weighted average across channels.
    bce_per_ch = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_per_ch = bce_per_ch.mean(dim=(0, 2, 3))    # → (C,)
    cw = torch.tensor(channel_weights, device=bce_per_ch.device, dtype=bce_per_ch.dtype)
    bce = (bce_per_ch * cw).sum() / cw.sum()
    # Per-channel Dice (already class-balanced — small classes weigh equally)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = (2 * inter + 1) / (union + 1)
    dice_loss = (1 - dice).mean()
    return bce_weight * bce + dice_weight * dice_loss


def bce_focal_tversky_loss(logits, targets,
                             channel_weights=DEFAULT_CHANNEL_WEIGHTS,
                             alpha=0.3, beta=0.7, gamma=4.0/3.0,
                             ft_weight=2.0, bce_weight=1.0):
    """Per-channel weighted BCE + per-channel Focal Tversky.

    Tversky = TP / (TP + α·FN + β·FP). Generalizes Dice (which has α=β=0.5).
    Focal modulation: (1 - Tversky)^γ — focuses on hard examples.

    Defaults are PRECISION-BIASED (α=0.3, β=0.7) — penalizes false positives
    more than false negatives. Right for our pipeline where a false-positive
    hash mark creates a wrong correspondence into RANSAC, vs a false negative
    just means one fewer correspondence (RANSAC handles sparse points fine).
    """
    bce_per_ch = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_per_ch = bce_per_ch.mean(dim=(0, 2, 3))
    cw = torch.tensor(channel_weights, device=bce_per_ch.device, dtype=bce_per_ch.dtype)
    bce = (bce_per_ch * cw).sum() / cw.sum()
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(0, 2, 3))
    fn = ((1 - probs) * targets).sum(dim=(0, 2, 3))
    fp = (probs * (1 - targets)).sum(dim=(0, 2, 3))
    tversky = (tp + 1) / (tp + alpha * fn + beta * fp + 1)
    focal_tversky = (1 - tversky) ** gamma
    ft_loss = focal_tversky.mean()
    return bce_weight * bce + ft_weight * ft_loss


# ── Validation metrics ──
@torch.no_grad()
def val_metrics(model, loader, device, thr=0.5):
    model.eval()
    n_pixels = 0
    tp = torch.zeros(N_CHANNELS_OUT, device=device)
    fp = torch.zeros(N_CHANNELS_OUT, device=device)
    fn = torch.zeros(N_CHANNELS_OUT, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        if out.shape[-2:] != y.shape[-2:]:
            out = F.interpolate(out, size=y.shape[-2:], mode="bilinear",
                                align_corners=False)
        pred = (torch.sigmoid(out) > thr).float()
        tp += (pred * y).sum(dim=(0, 2, 3))
        fp += (pred * (1 - y)).sum(dim=(0, 2, 3))
        fn += ((1 - pred) * y).sum(dim=(0, 2, 3))
        n_pixels += y.numel() // y.shape[1]
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    out = {"mean_f1": float(f1.mean().item())}
    for ci, name in enumerate(CHANNEL_NAMES):
        out[f"f1_{name}"] = float(f1[ci].item())
    return out


# ── Train/val split: hold out one game ──
def split_by_game(entries, val_game=None, val_frac=0.1, seed=42):
    by_game = defaultdict(list)
    for e in entries:
        by_game[e["game"]].append(e)
    games = sorted(by_game.keys())
    if val_game is None:
        sizes = sorted([(len(by_game[g]), g) for g in games])
        target = int(len(entries) * val_frac)
        chosen = next((g for sz, g in sizes if sz >= target), sizes[-1][1])
        val_game = chosen
    train, val = [], []
    for g in games:
        (val if g == val_game else train).extend(by_game[g])
    return train, val, val_game


# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=os.path.join(PROJECT_ROOT, "data/unified_masks/round1"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "models/unet_unified_v1"))
    ap.add_argument("--encoder", default="mit_b0",
                    choices=["mit_b0", "mit_b1", "mit_b2"])
    ap.add_argument("--decoder", default="unet",
                    choices=["unet", "segformer"],
                    help="U-Net (with skip connections) better for thin "
                         "features; SegFormer MLP lighter but coarser.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-game", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-from", default=None,
                    help="Path to a checkpoint to load weights from before "
                         "training starts (for fine-tuning, e.g. v3.last.pth).")
    ap.add_argument("--loss", default="bce_dice",
                    choices=["bce_dice", "bce_focal_tversky"],
                    help="bce_dice: original v1-v3 loss. "
                         "bce_focal_tversky: precision-biased fine-tune loss "
                         "(default α=0.3, β=0.7, γ=4/3).")
    ap.add_argument("--ft-alpha", type=float, default=0.3,
                    help="Focal Tversky α (FN weight). Lower = penalize FN less.")
    ap.add_argument("--ft-beta", type=float, default=0.7,
                    help="Focal Tversky β (FP weight). Higher = penalize FP more.")
    ap.add_argument("--ft-gamma", type=float, default=4.0/3.0,
                    help="Focal Tversky γ (focusing parameter). >1 focuses on hard.")
    ap.add_argument("--use-raw-masks", action="store_true",
                    help="Train on raw specialist outputs (no H-clean QC). "
                         "Each npz must contain a 'raw_masks' field — generated "
                         "by build_qc_unified_mask_dataset.py with --reuse-rgb.")
    ap.add_argument("--channel-weights", type=float, nargs=4,
                    metavar=("YARD", "SIDE", "HASH", "NUMBER"),
                    default=None,
                    help="Override per-channel BCE weights. Default uses "
                         f"DEFAULT_CHANNEL_WEIGHTS = {DEFAULT_CHANNEL_WEIGHTS}.")
    ap.add_argument("--shift-hash-up-px", type=int, default=0,
                    help="Shift hash channel mask UP N pixels (native frame "
                         "coords) before resize. Used to correct a systematic "
                         "'hash predictions drift low' offset via fine-tune.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load manifest. If decisions.json exists, filter to Y'd entries;
    # otherwise assume the manifest is already pre-filtered (e.g., the QC
    # dataset which is built from already-Y'd entries).
    manifest = json.load(open(os.path.join(args.pool_dir, "dataset_manifest.json")))
    decisions_path = os.path.join(args.pool_dir, "decisions.json")
    if os.path.exists(decisions_path):
        decisions = json.load(open(decisions_path))
        entries = [e for e in manifest["entries"] if decisions.get(e["id"]) == "y"]
        print(f"Manifest: {len(manifest['entries'])} entries, "
              f"decisions: {len(decisions)} → {len(entries)} Y'd entries")
    else:
        entries = list(manifest["entries"])
        print(f"Manifest: {len(entries)} entries (no decisions.json — using all)")

    train_entries, val_entries, val_game = split_by_game(
        entries, val_game=args.val_game, val_frac=0.1, seed=args.seed)
    print(f"Split: train={len(train_entries)}  val={len(val_entries)}  "
          f"(val game = {val_game})")

    raw_dir = os.path.join(args.pool_dir, "raw")
    train_ds = UnifiedMaskDataset(train_entries, raw_dir, augment=True,
                                    use_raw_masks=args.use_raw_masks,
                                    shift_hash_up_px=args.shift_hash_up_px)
    val_ds = UnifiedMaskDataset(val_entries, raw_dir, augment=False,
                                  use_raw_masks=args.use_raw_masks,
                                  shift_hash_up_px=args.shift_hash_up_px)
    if args.use_raw_masks:
        print("⚠ Training on RAW specialist masks (no H-clean QC).")
    if args.shift_hash_up_px > 0:
        print(f"⚠ Hash mask will be shifted UP {args.shift_hash_up_px}px "
              f"(native coords) — fine-tune mode.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    device = torch.device(args.device)
    model = build_model(encoder_name=args.encoder, decoder=args.decoder).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {args.decoder.upper()}({args.encoder}, in_ch=3, "
          f"out_ch={N_CHANNELS_OUT}) — {n_params:.1f}M params")

    if args.init_from:
        print(f"Loading init weights from {args.init_from} ...")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"  loaded (epoch={ckpt.get('epoch', '?')}, "
              f"val_f1={ckpt.get('val_f1', '?')})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    cw = tuple(args.channel_weights) if args.channel_weights else DEFAULT_CHANNEL_WEIGHTS
    if args.loss == "bce_dice":
        def loss_fn(logits, targets):
            return bce_dice_loss(logits, targets, channel_weights=cw)
        print(f"Loss: BCE + Dice (per-channel weights={cw})")
    elif args.loss == "bce_focal_tversky":
        ft_a, ft_b, ft_g = args.ft_alpha, args.ft_beta, args.ft_gamma
        def loss_fn(logits, targets):
            return bce_focal_tversky_loss(
                logits, targets, channel_weights=cw,
                alpha=ft_a, beta=ft_b, gamma=ft_g)
        print(f"Loss: BCE + Focal Tversky (α={ft_a}, β={ft_b}, γ={ft_g:.3f}, "
              f"per-channel weights={cw})")
    else:
        raise ValueError(args.loss)

    best_val_f1 = 0.0
    log_path = os.path.join(args.out_dir, "training_log.jsonl")
    open(log_path, "w").close()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear",
                                    align_corners=False)
            loss = loss_fn(out, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item()
            n_batches += 1
        sched.step()
        train_loss = loss_sum / max(1, n_batches)
        elapsed = time.time() - t0

        v = val_metrics(model, val_loader, device)
        print(f"Epoch {epoch+1:2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val mean_f1={v['mean_f1']:.4f}  "
              f"yard={v['f1_yard']:.3f}  side={v['f1_side']:.3f}  "
              f"hash={v['f1_hash']:.3f}  num={v['f1_number']:.3f}  "
              f"({elapsed:.0f}s)", flush=True)

        with open(log_path, "a") as f:
            json.dump({"epoch": epoch+1, "train_loss": train_loss, **v,
                       "lr": sched.get_last_lr()[0]}, f)
            f.write("\n")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "encoder": args.encoder,
            "in_channels": 3, "classes": N_CHANNELS_OUT,
            "channel_names": CHANNEL_NAMES,
            "epoch": epoch + 1,
            "val_f1": v["mean_f1"],
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if v["mean_f1"] > best_val_f1:
            best_val_f1 = v["mean_f1"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"  ↑ new best (mean F1 = {best_val_f1:.4f})")

    print(f"\nDone. Best val mean F1: {best_val_f1:.4f}")
    print(f"Checkpoints + log -> {args.out_dir}")


if __name__ == "__main__":
    main()
