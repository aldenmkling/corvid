#!/usr/bin/env python3
"""Tiny single-channel UNet for hash-mark segmentation.

Trains on auto-generated hash masks the user accepted as 'good' during
triage. Used to bootstrap better pre-annotations on the rest of the
frames so manual fix work is reduced.

Same arch / loss / aug pipeline as `train_unet_lines.py` but with a
single output channel and no val split (val signal here is downstream
re-triage, not pixel F1 against rule-based labels).
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


class HashDataset(Dataset):
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
            img_t = out["image"].float()
            mask_t = out["mask"].unsqueeze(0).float()
        else:
            img = cv2.resize(img, (INPUT_W, INPUT_H))
            mask = cv2.resize(mask, (INPUT_W, INPUT_H),
                              interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.from_numpy(img).float()
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


def dice_loss(pred_sigmoid, target, eps=1e-6):
    inter = (pred_sigmoid * target).sum()
    return 1 - (2 * inter + eps) / (pred_sigmoid.sum() + target.sum() + eps)


def loss_fn(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    return 0.5 * bce + 0.5 * dice_loss(pred, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Root containing train/{images,masks}/")
    ap.add_argument("--output", default="output/unet_hash_round1")
    ap.add_argument("--epochs", type=int, default=80)
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
    train_ds = HashDataset(os.path.join(args.dataset, "train"), augment=True)
    print(f"train: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False,
                              drop_last=True)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                      in_channels=3, classes=1, activation=None).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    log = []
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
        avg = total / max(n, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        msg = (f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.1f}s)  "
               f"loss={avg:.4f}  lr={lr_now:.2e}")
        print(msg, flush=True)
        log.append({"epoch": epoch, "loss": avg, "lr": lr_now})
        with open(os.path.join(args.output, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch, "args": vars(args)},
                   os.path.join(args.output, "last.pth"))
    print(f"\nDone. Weights: {args.output}/last.pth")


if __name__ == "__main__":
    main()
