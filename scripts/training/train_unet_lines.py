#!/usr/bin/env python3
"""
Train a UNet for 2-class line segmentation (yard lines + sidelines).

Input: distorted (raw) frame at 512x896.
Output: 2-channel logits (yard / sideline), per-pixel sigmoid.
Loss: 0.5 * BCE + 0.5 * Dice per channel, averaged across classes.

Dataset layout expected (built by scripts/data_prep/build_line_dataset.py):
  <dataset>/
    train/images/<id>.jpg
    train/masks/<id>.png   (3-ch PNG: B=yard, G=side, R=0)
    valid/images/<id>.jpg
    valid/masks/<id>.png

Usage on RunPod:
    python train_unet_lines.py --dataset /workspace/line_detection \
        --epochs 100 --batch-size 16 --lr 1e-3 --output /workspace/output
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False


INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-class loss weights: [yard, side]. Sidelines get 3x weight — there are
# fewer sideline pixels per frame and the channel learns slower otherwise.
CLASS_WEIGHTS = (1.0, 3.0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class LineSegDataset(Dataset):
    """Frames with 2-channel line masks (yard, side)."""

    def __init__(self, root, augment=False, grayscale=False):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.grayscale = grayscale
        # macOS tar can leave `._filename.jpg` AppleDouble metadata sidecars
        # in extracted dirs. Skip them — they aren't real images.
        candidate_ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(self.img_dir))
                         if f.endswith(".jpg") and not f.startswith("._")]
        # Drop samples that are missing / unreadable — prevents worker crashes
        # mid-epoch. Report any drops.
        good, dropped = [], []
        for fid in candidate_ids:
            img_ok = os.path.getsize(os.path.join(self.img_dir, f"{fid}.jpg")) > 0 \
                     if os.path.exists(os.path.join(self.img_dir, f"{fid}.jpg")) else False
            mask_ok = os.path.exists(os.path.join(self.mask_dir, f"{fid}.png"))
            if img_ok and mask_ok:
                good.append(fid)
            else:
                dropped.append(fid)
        self.ids = good
        if dropped:
            print(f"WARN: dropped {len(dropped)} samples from {root} "
                  f"(missing/empty files). first: {dropped[0]}")
        self.augment = augment and HAS_ALB
        if self.augment:
            # Aggressive augmentation: lots of crop variation, flips, and color
            # jitter to combat small-dataset overfitting. Mask interpolation is
            # NEAREST throughout to keep labels crisp.
            self.tf = A.Compose([
                # Crop a random scale/aspect region, resize to input. Keeps
                # 95%+ of images useful but forces the model to see each scene
                # at many zooms/positions.
                A.RandomResizedCrop(
                    size=(INPUT_H, INPUT_W),
                    scale=(0.55, 1.0),
                    ratio=(1.6, 2.0),        # stay close to 16:9
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=6, border_mode=cv2.BORDER_REFLECT_101,
                         interpolation=cv2.INTER_LINEAR, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                                       val_shift_limit=15, p=0.5),
                A.CLAHE(clip_limit=(1.0, 3.0), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.15),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.tf = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{fid}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{fid}.png")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cv2.imread returned None for {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if getattr(self, "grayscale", False):
            # Convert to grayscale and replicate to 3 channels so model
            # input shape is unchanged. Tests whether color matters.
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([gray, gray, gray], axis=-1)
        mask = cv2.imread(mask_path)
        if mask is None:
            raise FileNotFoundError(f"cv2.imread returned None for {mask_path}")
        # Mask is a 3-ch PNG produced by build_line_dataset.py. cv2.imread
        # returns it in BGR order: ch 0 = B (empty), ch 1 = G (side line),
        # ch 2 = R (yard line).
        yard = (mask[..., 2] > 127).astype(np.float32)
        side = (mask[..., 1] > 127).astype(np.float32)
        mask_2ch = np.stack([yard, side], axis=-1)  # (H, W, 2)

        if self.augment:
            out = self.tf(image=img, mask=mask_2ch)
            img_t = out["image"].float()
            mask_t = out["mask"].permute(2, 0, 1).float()  # (2, H, W)
        else:
            img = cv2.resize(img, (INPUT_W, INPUT_H))
            mask_2ch = cv2.resize(mask_2ch, (INPUT_W, INPUT_H),
                                  interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.from_numpy(img).float()
            mask_t = torch.from_numpy(np.transpose(mask_2ch, (2, 0, 1))).float()
        return img_t, mask_t


# ── Loss ─────────────────────────────────────────────────────────────────────

def dice_per_class(pred, target, eps=1e-6):
    """Return per-class (1 - soft Dice). pred is sigmoid'd. Shape: (C,)."""
    dims = (0, 2, 3)                 # sum over batch + spatial
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice                  # (C,)


def combined_loss(logits, target, class_weights=None):
    """0.5 BCE + 0.5 Dice, weighted per class.

    BCE is computed per-pixel per-class, averaged over batch+spatial, then
    combined across classes with class_weights. Dice is already per-class.
    """
    C = logits.shape[1]
    if class_weights is None:
        w = torch.ones(C, device=logits.device)
    else:
        w = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)

    # Per-class BCE
    bce_elem = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")          # (B, C, H, W)
    bce_per_class = bce_elem.mean(dim=(0, 2, 3))   # (C,)

    # Per-class soft Dice
    pred = torch.sigmoid(logits)
    dice_per = dice_per_class(pred, target)        # (C,)

    per_class = 0.5 * bce_per_class + 0.5 * dice_per   # (C,)
    # Weighted mean (so loss magnitude is comparable to unweighted)
    return (per_class * w).sum() / w.sum()


# ── Metrics ──────────────────────────────────────────────────────────────────

def per_class_metrics(logits, target, thresh=0.5, eps=1e-6):
    """Returns dict with per-class and mean F1/P/R (pixel-level)."""
    pred = (torch.sigmoid(logits) > thresh).float()
    tp = (pred * target).sum(dim=(0, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(0, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(0, 2, 3))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return {
        "yard_p": p[0].item(), "yard_r": r[0].item(), "yard_f1": f1[0].item(),
        "side_p": p[1].item(), "side_r": r[1].item(), "side_f1": f1[1].item(),
        "mean_f1": f1.mean().item(),
    }


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, class_weights, scaler=None):
    model.train()
    total, n = 0.0, 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = combined_loss(logits, masks, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = combined_loss(logits, masks, class_weights)
            loss.backward()
            optimizer.step()
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, class_weights):
    model.eval()
    total_loss, n = 0.0, 0
    agg = {k: 0.0 for k in ["yard_p", "yard_r", "yard_f1",
                            "side_p", "side_r", "side_f1", "mean_f1"]}
    batches = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss = combined_loss(logits, masks, class_weights)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        m = per_class_metrics(logits, masks)
        for k in agg:
            agg[k] += m[k]
        batches += 1
    return total_loss / max(n, 1), {k: v / max(batches, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Root dir containing train/ and valid/ subdirs")
    ap.add_argument("--output", default="/workspace/output")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--amp", action="store_true", help="Use mixed-precision on CUDA")
    ap.add_argument("--class-weights", default=None,
                    help="Comma-separated per-class loss weights [yard,side]. "
                         f"Default {CLASS_WEIGHTS}")
    ap.add_argument("--grayscale", action="store_true",
                    help="Convert input to grayscale and replicate to 3 channels.")
    ap.add_argument("--device", default=None,
                    help="Force device: 'mps', 'cuda', or 'cpu'. Auto-detects if unset.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  Encoder: {args.encoder}  "
          f"Grayscale: {args.grayscale}")
    if not args.no_augment and not HAS_ALB:
        print("WARN: albumentations not installed — falling back to no augment")

    train_ds = LineSegDataset(os.path.join(args.dataset, "train"),
                                augment=not args.no_augment,
                                grayscale=args.grayscale)
    val_ds = LineSegDataset(os.path.join(args.dataset, "valid"),
                              augment=False, grayscale=args.grayscale)
    print(f"train: {len(train_ds)}, valid: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"resumed from {args.resume}")

    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    if args.class_weights:
        class_weights = tuple(float(v) for v in args.class_weights.split(","))
    else:
        class_weights = CLASS_WEIGHTS
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"class weights (yard, side): {class_weights}")

    best_f1 = 0.0
    log_path = os.path.join(args.output, "training_log.json")
    log_entries = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch <= args.warmup_epochs:
            warm = epoch / args.warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups,
                                   [args.lr * args.encoder_lr_mult, args.lr]):
                pg["lr"] = base_lr * warm

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                      class_weights_t, scaler)
        val_loss, m = eval_epoch(model, val_loader, device, class_weights_t)
        if epoch > args.warmup_epochs:
            scheduler.step()

        lr_now = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0
        msg = (f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s): "
               f"train={train_loss:.4f}, val={val_loss:.4f}, "
               f"mean_f1={m['mean_f1']:.3f}  "
               f"yard P/R/F1={m['yard_p']:.2f}/{m['yard_r']:.2f}/{m['yard_f1']:.2f}  "
               f"side P/R/F1={m['side_p']:.2f}/{m['side_r']:.2f}/{m['side_f1']:.2f}  "
               f"lr={lr_now:.2e}")
        print(msg, flush=True)
        log_entries.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "lr": lr_now, **m})
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)

        if m["mean_f1"] > best_f1:
            best_f1 = m["mean_f1"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "metrics": m, "args": vars(args)},
                       os.path.join(args.output, "best.pth"))
            print(f"  -> new best mean_f1 = {best_f1:.3f}")

        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch, "metrics": m, "args": vars(args)},
                   os.path.join(args.output, "last.pth"))

    print(f"\nDone. Best mean_f1: {best_f1:.3f}")
    print(f"Weights: {args.output}/best.pth")


if __name__ == "__main__":
    main()
