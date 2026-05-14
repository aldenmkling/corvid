#!/usr/bin/env python3
"""mit_b0 classifier for painted-yardline-number masks.

Input: 64×64 1-channel binary mask of a grouped painted-number CC set
       (output of painted_numbers grouping → bbox crop → pad-square → resize).
Output: 9-class softmax over absolute NGS-position labels:
       {10L, 20L, 30L, 40L, 50, 40R, 30R, 20R, 10R}.

Each label encodes a specific NGS x position:
  10L=20  20L=30  30L=40  40L=50  50=60  40R=70  30R=80  20R=90  10R=100

Training data layout (ImageFolder-style):
  data/number_classifier/round1/<class>/<filename>.png

Loss: class-weighted cross-entropy (inverse frequency) + label smoothing 0.1
      (absorbs ~5–10% expected label noise from auto-mining).
Augmentation: small affine (rotate ±5°, scale 0.9–1.1×, translate ±2 px).
              No h-flip (would mirror L↔R), no v-flip (mirrors near↔far).
Architecture: timm mit_b0, in_chans=1, num_classes=9, ImageNet pretrained.
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
from torch.utils.data import Dataset, DataLoader, Subset

import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False


class MitClassifier(nn.Module):
    """smp mit_b0 encoder + global-avg-pool + linear classifier head.
    Same encoder family as our line/hash/number UNets — keeps our stack
    consistent on a single encoder backbone."""

    def __init__(self, encoder_name="mit_b0", num_classes=9, in_channels=1,
                  weights="imagenet"):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=weights)
        feat_dim = self.encoder.out_channels[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[-1])


CLASSES = ["10L", "10R", "20L", "20R", "30L", "30R", "40L", "40R", "50"]
NUM_CLASSES = len(CLASSES)
INPUT_SIZE = 64
# For 1-channel input, timm averages 3-channel ImageNet pretrain stats. Use
# the average of the RGB mean/std as a single-channel reference.
PIXEL_MEAN = 0.456
PIXEL_STD = 0.224


class NumberClassifierDataset(Dataset):
    def __init__(self, root, classes=CLASSES, augment=False,
                 input_size: int = INPUT_SIZE,
                 input_w: int | None = None,
                 input_h: int | None = None,
                 morph_aug: bool = False,
                 cutout_aug: bool = False):
        # Support rectangular crops via explicit (input_w, input_h);
        # default to square input_size when not provided.
        if input_w is None:
            input_w = input_size
        if input_h is None:
            input_h = input_size
        self.input_w = input_w
        self.input_h = input_h
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                if fn.endswith(".png") and not fn.startswith("."):
                    self.samples.append(
                        (os.path.join(cls_dir, fn), self.cls_to_idx[cls]))
        self.augment = augment and HAS_ALB
        self.input_size = input_size
        self.morph_aug = morph_aug
        self.cutout_aug = cutout_aug
        if self.augment:
            tf_list = [
                A.Affine(rotate=(-5, 5), scale=(0.9, 1.1),
                          translate_px=(-2, 2),
                          interpolation=cv2.INTER_NEAREST,
                          p=0.8),
            ]
            if self.cutout_aug:
                tf_list.append(A.CoarseDropout(
                    num_holes_range=(1, 2),
                    hole_height_range=(4, 10),
                    hole_width_range=(4, 10),
                    fill=0, p=0.5))
            tf_list += [
                A.Normalize(mean=(PIXEL_MEAN,), std=(PIXEL_STD,),
                              max_pixel_value=255.0),
                ToTensorV2(),
            ]
            self.tf = A.Compose(tf_list)
        else:
            self.tf = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        if img.shape != (self.input_h, self.input_w):
            img = cv2.resize(img, (self.input_w, self.input_h),
                              interpolation=cv2.INTER_NEAREST)
        if self.morph_aug and self.augment and np.random.rand() < 0.5:
            # Random ±1px morphological dilation OR erosion (binary mask
            # quality variation).
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if np.random.rand() < 0.5:
                img = cv2.dilate(img, kernel, iterations=1)
            else:
                img = cv2.erode(img, kernel, iterations=1)
        if self.augment:
            img_t = self.tf(image=img)["image"].float()
        else:
            x = (img.astype(np.float32) / 255.0 - PIXEL_MEAN) / PIXEL_STD
            img_t = torch.from_numpy(x).unsqueeze(0).float()
        return img_t, label


def compute_class_weights(samples, n_classes):
    """Inverse-frequency weights, normalized so mean weight = 1."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    w = 1.0 / counts
    w *= n_classes / w.sum()
    return torch.from_numpy(w).float()


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    per_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        for c in range(NUM_CLASSES):
            mask = labels == c
            per_total[c] += int(mask.sum().item())
            per_correct[c] += int(((pred == labels) & mask).sum().item())
    acc = correct / max(n, 1)
    avg_loss = total_loss / max(n, 1)
    per_acc = per_correct / np.maximum(per_total, 1)
    return avg_loss, acc, per_acc, per_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     help="Root with <class>/*.png subdirs")
    ap.add_argument("--output", default="output/number_classifier_round1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="mit_b0")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    else:
        # NOTE: MPS produces NaN with smp's mit_b0 (attention layer
        # incompatibility). Use CUDA on RunPod, or CPU locally for smoke tests.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Encoder: {args.encoder}")
    if not HAS_ALB:
        print("WARN: albumentations not installed — no augmentation")

    full_aug = NumberClassifierDataset(args.dataset, CLASSES, augment=True)
    full_clean = NumberClassifierDataset(args.dataset, CLASSES, augment=False)
    n = len(full_aug)
    print(f"Total samples: {n}")
    if n == 0:
        print("ERROR: no samples found"); return

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)
    n_val = int(round(n * args.val_frac))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())

    train_ds = Subset(full_aug, train_idx)
    val_ds = Subset(full_clean, val_idx)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=False,
                                drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    class_weights = compute_class_weights(
        [full_aug.samples[i] for i in train_idx], NUM_CLASSES).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(2)}")

    model = MitClassifier(encoder_name=args.encoder,
                            num_classes=NUM_CLASSES,
                            in_channels=1, weights="imagenet").to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing)

    log = []
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        val_loss, val_acc, val_per_class, val_total = eval_epoch(
            model, val_loader, device, criterion)
        lr_now = optimizer.param_groups[0]["lr"]

        msg = (f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.1f}s)  "
               f"train={train_loss:.4f}  val={val_loss:.4f}  "
               f"val_acc={val_acc:.3f}  lr={lr_now:.2e}")
        print(msg, flush=True)
        pc_str = "  ".join(
            f"{CLASSES[i]}={val_per_class[i]:.2f}(n={val_total[i]})"
            for i in range(NUM_CLASSES)
        )
        print(f"    {pc_str}", flush=True)

        log.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": float(val_acc), "lr": lr_now,
            "per_class_acc": [float(x) for x in val_per_class],
            "per_class_n": [int(x) for x in val_total],
        })
        with open(os.path.join(args.output, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        torch.save({"model_state_dict": model.state_dict(),
                     "epoch": epoch, "args": vars(args), "classes": CLASSES},
                    os.path.join(args.output, "last.pth"))
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save({"model_state_dict": model.state_dict(),
                         "epoch": epoch, "args": vars(args), "classes": CLASSES,
                         "val_acc": best_val_acc},
                        os.path.join(args.output, "best.pth"))
            print(f"    ↑ new best val_acc={best_val_acc:.3f}", flush=True)

    print(f"\nDone. best val_acc = {best_val_acc:.3f}. Weights: {args.output}/best.pth")


if __name__ == "__main__":
    main()
