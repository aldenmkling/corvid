#!/usr/bin/env python3
"""Train unified 3-channel UNet (yard + side + hash).

Mirrors `train_unet_lines.py` (BCE + Dice, augmentations, cosine LR with
warmup) but with 3 output channels. Mask format is BGR PNG: B=hash,
G=side, R=yard (built by `build_unified_dataset.py`).
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

# Per-class loss weights: [yard, side, hash]. Side/hash get higher weight
# because they have far fewer pixels per frame than yard lines.
CLASS_WEIGHTS = (1.0, 3.0, 3.0)


class UnifiedSegDataset(Dataset):
    """Frames with 3-channel masks: yard (R), side (G), hash (B)."""

    def __init__(self, root, augment=False):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(self.img_dir))
               if f.endswith(".jpg") and not f.startswith("._")]
        self.ids = [i for i in ids
                    if os.path.exists(os.path.join(self.mask_dir, f"{i}.png"))]
        self.augment = augment and HAS_ALB
        if self.augment:
            self.tf = A.Compose([
                A.RandomResizedCrop(
                    size=(INPUT_H, INPUT_W),
                    scale=(0.55, 1.0),
                    ratio=(1.6, 2.0),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=6, border_mode=cv2.BORDER_REFLECT_101,
                         interpolation=cv2.INTER_LINEAR, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                            contrast_limit=0.3, p=0.7),
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
        img = cv2.imread(os.path.join(self.img_dir, f"{fid}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_bgr = cv2.imread(os.path.join(self.mask_dir, f"{fid}.png"))
        # cv2 returns BGR: ch 0 = B = hash, ch 1 = G = side, ch 2 = R = yard
        yard = (mask_bgr[..., 2] > 127).astype(np.float32)
        side = (mask_bgr[..., 1] > 127).astype(np.float32)
        hash_ = (mask_bgr[..., 0] > 127).astype(np.float32)
        mask_3ch = np.stack([yard, side, hash_], axis=-1)   # (H, W, 3)

        if self.augment:
            out = self.tf(image=img, mask=mask_3ch)
            img_t = out["image"].float()
            mask_t = out["mask"].permute(2, 0, 1).float()
        else:
            img = cv2.resize(img, (INPUT_W, INPUT_H))
            mask_3ch = cv2.resize(mask_3ch, (INPUT_W, INPUT_H),
                                    interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.from_numpy(img).float()
            mask_t = torch.from_numpy(np.transpose(mask_3ch, (2, 0, 1))).float()
        return img_t, mask_t


def dice_per_class(pred, target, eps=1e-6):
    dims = (0, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return 1 - (2 * inter + eps) / (union + eps)


def combined_loss(logits, target, class_weights):
    bce_elem = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    bce_per_class = bce_elem.mean(dim=(0, 2, 3))
    pred = torch.sigmoid(logits)
    dice_per = dice_per_class(pred, target)
    per_class = 0.5 * bce_per_class + 0.5 * dice_per
    return (per_class * class_weights).sum() / class_weights.sum()


def per_class_metrics(logits, target, thresh=0.5, eps=1e-6):
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
        "hash_p": p[2].item(), "hash_r": r[2].item(), "hash_f1": f1[2].item(),
        "mean_f1": f1.mean().item(),
    }


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
    agg_keys = ["yard_p", "yard_r", "yard_f1",
                 "side_p", "side_r", "side_f1",
                 "hash_p", "hash_r", "hash_f1",
                 "mean_f1"]
    agg = {k: 0.0 for k in agg_keys}
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
                    help="Root with train/ and valid/ subdirs.")
    ap.add_argument("--output", default="/workspace/output")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--encoder", default="mit_b0")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--class-weights", default=None,
                    help="Comma-separated [yard,side,hash]. Default 1,3,3.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else
                               ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}  Encoder: {args.encoder}")

    train_ds = UnifiedSegDataset(os.path.join(args.dataset, "train"),
                                   augment=not args.no_augment)
    val_ds = UnifiedSegDataset(os.path.join(args.dataset, "valid"), augment=False)
    print(f"train: {len(train_ds)}, valid: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                      in_channels=3, classes=3, activation=None).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
        cw = tuple(float(v) for v in args.class_weights.split(","))
    else:
        cw = CLASS_WEIGHTS
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    print(f"class weights (yard, side, hash): {cw}")

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
                                       cw_t, scaler)
        val_loss, m = eval_epoch(model, val_loader, device, cw_t)
        if epoch > args.warmup_epochs:
            scheduler.step()

        lr_now = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0
        msg = (f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s): "
               f"train={train_loss:.4f}, val={val_loss:.4f}, "
               f"mean_f1={m['mean_f1']:.3f}  "
               f"yard {m['yard_p']:.2f}/{m['yard_r']:.2f}/{m['yard_f1']:.2f}  "
               f"side {m['side_p']:.2f}/{m['side_r']:.2f}/{m['side_f1']:.2f}  "
               f"hash {m['hash_p']:.2f}/{m['hash_r']:.2f}/{m['hash_f1']:.2f}  "
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


if __name__ == "__main__":
    main()
