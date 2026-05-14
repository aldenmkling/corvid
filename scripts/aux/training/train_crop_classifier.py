"""Train the number-crop classifier (DSResNet10ww) on
data/number_classifier/round3_128x32.

Self-contained: imports the model class from src/pipeline/crop_classifier.py
and the constants from src/pipeline/number_classifier_constants.py.

Runs locally on MPS by default (~5 min for 80 epochs on this data) or
CUDA on RunPod. Output:
  models/dsresnet10ww_round3_128x32/best.pth
  models/dsresnet10ww_round3_128x32/last.pth
  models/dsresnet10ww_round3_128x32/train.log
  models/dsresnet10ww_round3_128x32/training_log.jsonl

Hyperparams mirror the original 2026-05-10 training run (80 ep, lr 1e-3 →
1e-5 cosine, batch 64, weight_decay 5e-4, label_smoothing 0.1, dropout 0.2,
morph + cutout augmentation).

Usage:
  python scripts/aux/training/train_crop_classifier.py --device mps
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "pipeline"))

from crop_classifier import DSResNet10ww   # noqa: E402
from number_classifier_constants import (   # noqa: E402
    CLASSES, NUM_CLASSES, PIXEL_MEAN, PIXEL_STD,
)


INPUT_W = 128
INPUT_H = 32


def _augment(img: np.ndarray) -> np.ndarray:
    """Light augmentations: morph dilation/erosion, cutout, small jitter."""
    # Morph aug.
    if random.random() < 0.4:
        k = random.choice([1, 2])
        op = random.choice([cv2.MORPH_DILATE, cv2.MORPH_ERODE])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2 + 1, k * 2 + 1))
        img = cv2.morphologyEx(img, op, kernel)
    # Cutout aug: 1-2 black boxes.
    if random.random() < 0.5:
        h, w = img.shape
        for _ in range(random.randint(1, 2)):
            cw = random.randint(3, 12)
            ch = random.randint(3, 10)
            cx = random.randint(0, w - cw)
            cy = random.randint(0, h - ch)
            img = img.copy()
            img[cy:cy + ch, cx:cx + cw] = 0
    # Small shift.
    if random.random() < 0.3:
        sx = random.randint(-3, 3)
        sy = random.randint(-2, 2)
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img


class CropDataset(Dataset):
    def __init__(self, paths_labels, augment=False):
        self.items = paths_labels
        self.augment = augment

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, lbl = self.items[i]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((INPUT_H, INPUT_W), dtype=np.uint8)
        if img.shape != (INPUT_H, INPUT_W):
            img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        if self.augment:
            img = _augment(img)
        x = img.astype(np.float32) / 255.0
        x = (x - PIXEL_MEAN) / PIXEL_STD
        return torch.from_numpy(x).unsqueeze(0), int(lbl)


def gather(root):
    items = []
    for ci, c in enumerate(CLASSES):
        for p in glob.glob(os.path.join(root, c, "*.png")):
            items.append((p, ci))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dataset", default=os.path.join(
        PROJECT_ROOT, "data", "number_classifier", "round3_128x32"))
    ap.add_argument("--output", default=os.path.join(
        PROJECT_ROOT, "models", "dsresnet10ww_round3_128x32"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    items = gather(args.dataset)
    random.shuffle(items)
    n_val = int(round(len(items) * args.val_frac))
    val = items[:n_val]; train = items[n_val:]

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, l in train: counts[l] += 1
    class_weights = (counts.sum() / (NUM_CLASSES * counts)).astype(np.float32)
    class_weights = np.clip(class_weights, 0.1, 5.0)

    train_ds = CropDataset(train, augment=True)
    val_ds   = CropDataset(val,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=False)

    model = DSResNet10ww(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr_min)
    cw = torch.from_numpy(class_weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw,
                                    label_smoothing=args.label_smoothing)

    log_path = os.path.join(args.output, "train.log")
    jsonl_path = os.path.join(args.output, "training_log.jsonl")
    best_pth = os.path.join(args.output, "best.pth")
    last_pth = os.path.join(args.output, "last.pth")

    with open(log_path, "w") as f:
        f.write(f"Arch: dsresnet10ww   Device: {args.device}   Output: {args.output}\n")
        f.write(f"Classes ({NUM_CLASSES}): {CLASSES}\n")
        f.write(f"Total samples: {len(items)}\n")
        f.write(f"Train: {len(train)}  Val: {len(val)}\n")
        f.write(f"Model: dsresnet10ww  params={n_params:,} ({n_params/1000:.2f}K)\n")
        f.write(f"Class weights: {class_weights.tolist()}\n")
    print(open(log_path).read())

    open(jsonl_path, "w").close()
    best_acc = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, ok, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); optim.step()
            loss_sum += float(loss.detach()) * x.size(0)
            ok += int((logits.argmax(1) == y).sum())
            tot += x.size(0)
        train_loss = loss_sum / tot
        train_acc = ok / tot

        model.eval()
        v_tot, v_ok, v_loss = 0, 0, 0.0
        per_class_ok = np.zeros(NUM_CLASSES, dtype=np.int64)
        per_class_tot = np.zeros(NUM_CLASSES, dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                v_loss += float(loss_fn(logits, y)) * x.size(0)
                preds = logits.argmax(1)
                v_ok += int((preds == y).sum())
                v_tot += x.size(0)
                for i in range(NUM_CLASSES):
                    m = (y == i)
                    per_class_tot[i] += int(m.sum())
                    per_class_ok[i] += int(((preds == y) & m).sum())
        val_loss = v_loss / v_tot
        val_acc = v_ok / v_tot
        sched.step()

        msg = (f"Ep {ep:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
               f"train_acc={train_acc*100:.2f}%  val_loss={val_loss:.4f}  "
               f"val_acc={val_acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            msg += "\n   ↑ new best val_acc = " + f"{val_acc*100:.2f}%"
        print(msg)
        with open(log_path, "a") as f: f.write(msg + "\n")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({
                "epoch": ep, "lr": optim.param_groups[0]["lr"],
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "per_class_acc": (per_class_ok /
                                   np.maximum(per_class_tot, 1)).tolist(),
                "per_class_total": per_class_tot.tolist(),
            }) + "\n")

        ckpt = {"model_state_dict": model.state_dict(),
                "epoch": ep, "args": vars(args), "classes": CLASSES,
                "val_acc": val_acc, "arch": "dsresnet10ww",
                "n_params": n_params}
        torch.save(ckpt, last_pth)
        if is_best:
            torch.save(ckpt, best_pth)

    print(f"\nDone. Best val_acc={best_acc*100:.2f}%  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
