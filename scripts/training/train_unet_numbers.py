#!/usr/bin/env python3
"""Single-channel mit_b0 UNet for painted-number segmentation on raw
(distorted) broadcast frames.

Mirrors train_unet_hash.py / train_unet_lines.py — same arch, loss, aug
pipeline. Only differences: val split is required (we have hand-cleaned
ground truth so per-pixel F1 is meaningful), best-model checkpoint is
tracked.

Dataset layout (built by build_unet_numbers_dataset.py):
    train/images/<id>.jpg     train/masks/<id>.png
    valid/images/<id>.jpg     valid/masks/<id>.png
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


class NumberSegDataset(Dataset):
    def __init__(self, root, augment=False):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(self.img_dir))
               if f.endswith(".jpg") and not f.startswith("._")]
        self.ids = [fid for fid in ids
                    if os.path.exists(os.path.join(self.mask_dir, f"{fid}.png"))]
        self.augment = augment and HAS_ALB
        if self.augment:
            self.tf = A.Compose([
                A.RandomResizedCrop(
                    size=(INPUT_H, INPUT_W),
                    scale=(0.6, 1.0),
                    ratio=(1.6, 2.0),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=4, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                             contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20,
                                       val_shift_limit=15, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
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
        mask = cv2.imread(os.path.join(self.mask_dir, f"{fid}.png"),
                            cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            out = self.tf(image=img, mask=mask)
            return out["image"].float(), out["mask"].unsqueeze(0).float()

        img = cv2.resize(img, (INPUT_W, INPUT_H))
        mask = cv2.resize(mask, (INPUT_W, INPUT_H),
                            interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        img = np.transpose(img, (2, 0, 1))
        return (torch.from_numpy(img).float(),
                torch.from_numpy(mask).unsqueeze(0).float())


def dice_loss(pred_sigmoid, target, eps=1e-6):
    inter = (pred_sigmoid * target).sum()
    return 1 - (2 * inter + eps) / (pred_sigmoid.sum() + target.sum() + eps)


def loss_fn(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    return 0.5 * bce + 0.5 * dice_loss(pred, target)


@torch.no_grad()
def eval_epoch(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    total_loss = 0.0
    n = 0
    for imgs, masks in loader:
        imgs = imgs.to(device); masks = masks.to(device)
        logits = model(imgs)
        total_loss += loss_fn(logits, masks).item() * imgs.size(0)
        n += imgs.size(0)
        pred = (torch.sigmoid(logits) > threshold).float()
        tp += (pred * masks).sum().item()
        fp += (pred * (1 - masks)).sum().item()
        fn += ((1 - pred) * masks).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return total_loss / max(n, 1), {"precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     help="Root with train/{images,masks}/ and valid/{images,masks}/")
    ap.add_argument("--output", default="output/unet_numbers_round1")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--encoder", default="mit_b0")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available()
                                else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  Encoder: {args.encoder}")

    if not HAS_ALB:
        print("WARN: albumentations not installed — no augmentation")
    train_ds = NumberSegDataset(os.path.join(args.dataset, "train"), augment=True)
    val_ds = NumberSegDataset(os.path.join(args.dataset, "valid"), augment=False)
    print(f"train: {len(train_ds)}, valid: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=False,
                                drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                       in_channels=3, classes=1, activation=None).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    log = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        scheduler.step()
        train_loss = total / max(n, 1)
        val_loss, m = eval_epoch(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        msg = (f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.1f}s)  "
               f"train={train_loss:.4f}  val={val_loss:.4f}  "
               f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
               f"lr={lr_now:.2e}")
        print(msg, flush=True)
        log.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "lr": lr_now, **m})
        with open(os.path.join(args.output, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        torch.save({"model_state_dict": model.state_dict(),
                     "epoch": epoch, "args": vars(args)},
                    os.path.join(args.output, "last.pth"))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model_state_dict": model.state_dict(),
                         "epoch": epoch, "args": vars(args), "val_f1": best_f1},
                        os.path.join(args.output, "best.pth"))
            print(f"  ↑ new best F1={best_f1:.3f}", flush=True)
    print(f"\nDone. best val F1 = {best_f1:.3f}. Weights: {args.output}/best.pth")


if __name__ == "__main__":
    main()
