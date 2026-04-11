#!/usr/bin/env python3
"""
Train HRNet-W48 for field keypoint heatmap prediction.

Two-stage training:
  1. Pre-train on synthetic data (~5000 frames)
  2. Fine-tune on real annotated data (~240 frames)

Input: 960×540 images
Output: 240×135 heatmaps (106 channels, one per keypoint)
Loss: MSE on Gaussian heatmaps, averaged over visible keypoints only

Usage (on RunPod):
    # Stage 1: synthetic pre-training
    python train_hrnet_keypoints.py --dataset /workspace/dataset_keypoints/synthetic \
        --epochs 100 --lr 1e-3 --output /workspace/output_synthetic

    # Stage 2: fine-tune on real
    python train_hrnet_keypoints.py --dataset /workspace/dataset_keypoints \
        --resume /workspace/output_synthetic/best.pth \
        --epochs 200 --lr 1e-4 --backbone-lr-mult 0.1 \
        --output /workspace/output_real
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import cv2

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── Constants ────────────────────────────────────────────────────────────────

NUM_KEYPOINTS = 110  # 106 identity + 4 category
INPUT_H, INPUT_W = 540, 960
HEATMAP_H, HEATMAP_W = 135, 240  # 1/4 resolution
HEATMAP_SIGMA = 2.0  # Gaussian sigma in heatmap pixels
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Dataset ──────────────────────────────────────────────────────────────────

def _generate_heatmap(cx: float, cy: float, h: int, w: int, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx, cy)."""
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


class FieldKeypointDataset(Dataset):
    """COCO keypoints dataset for field keypoint detection."""

    def __init__(self, data_dir: str, augment: bool = True,
                 flip_mapping: dict | None = None):
        self.data_dir = data_dir
        self.augment = augment
        self.flip_mapping = flip_mapping

        ann_path = os.path.join(data_dir, "annotations.json")
        with open(ann_path) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = []
        for ann in coco["annotations"]:
            if ann["image_id"] in self.images:
                self.annotations.append(ann)

        self.img_dir = os.path.join(data_dir, "images")
        print(f"Loaded {len(self.annotations)} annotations from {ann_path}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]

        # Load image
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Parse keypoints: [x0, y0, v0, x1, y1, v1, ...]
        kp_flat = ann["keypoints"]
        keypoints = np.array(kp_flat, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        # keypoints[:, 0] = x, keypoints[:, 1] = y, keypoints[:, 2] = visibility

        # Resize image to input size
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        scale_x = INPUT_W / orig_w
        scale_y = INPUT_H / orig_h
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Augmentation
        if self.augment:
            img, keypoints = self._augment(img, keypoints)

        # Generate heatmaps
        heatmaps = np.zeros((NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W), dtype=np.float32)
        visibility = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        for ki in range(NUM_KEYPOINTS):
            v = keypoints[ki, 2]
            if v > 0:
                # Scale to heatmap resolution
                hm_x = keypoints[ki, 0] * HEATMAP_W / INPUT_W
                hm_y = keypoints[ki, 1] * HEATMAP_H / INPUT_H

                if 0 <= hm_x < HEATMAP_W and 0 <= hm_y < HEATMAP_H:
                    heatmaps[ki] = _generate_heatmap(
                        hm_x, hm_y, HEATMAP_H, HEATMAP_W, HEATMAP_SIGMA
                    )
                    visibility[ki] = 1.0

        # Normalize image
        img = img.astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return (
            torch.from_numpy(img),
            torch.from_numpy(heatmaps),
            torch.from_numpy(visibility),
        )

    def _augment(self, img: np.ndarray, keypoints: np.ndarray) -> tuple:
        """Apply augmentations. NO vertical flip."""
        h, w = img.shape[:2]

        # Horizontal flip with keypoint remapping
        if self.flip_mapping and np.random.random() < 0.5:
            img = cv2.flip(img, 1)  # horizontal flip
            keypoints[:, 0] = w - 1 - keypoints[:, 0]

            # Remap keypoint identities
            new_kp = keypoints.copy()
            for old_id, new_id in self.flip_mapping.items():
                new_kp[new_id] = keypoints[old_id]
            keypoints = new_kp

        # Random rotation (±10°)
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Transform keypoint coordinates
        for ki in range(NUM_KEYPOINTS):
            if keypoints[ki, 2] > 0:
                pt = np.array([keypoints[ki, 0], keypoints[ki, 1], 1.0])
                new_pt = M @ pt
                keypoints[ki, 0] = new_pt[0]
                keypoints[ki, 1] = new_pt[1]
                # Mark as invisible if rotated out of frame
                if not (0 <= new_pt[0] < w and 0 <= new_pt[1] < h):
                    keypoints[ki, 2] = 0

        # Random scale (0.8-1.2) via crop/pad
        scale = np.random.uniform(0.8, 1.2)
        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_scaled = cv2.resize(img, (new_w, new_h))
            keypoints[:, 0] *= scale
            keypoints[:, 1] *= scale

            # Center crop/pad back to original size
            if scale > 1.0:
                # Crop center
                x1 = (new_w - w) // 2
                y1 = (new_h - h) // 2
                img = img_scaled[y1:y1 + h, x1:x1 + w]
                keypoints[:, 0] -= x1
                keypoints[:, 1] -= y1
            else:
                # Pad
                img = np.zeros((h, w, 3), dtype=img_scaled.dtype)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                img[y1:y1 + new_h, x1:x1 + new_w] = img_scaled
                keypoints[:, 0] += x1
                keypoints[:, 1] += y1

            # Update visibility for out-of-frame keypoints
            for ki in range(NUM_KEYPOINTS):
                if keypoints[ki, 2] > 0:
                    if not (0 <= keypoints[ki, 0] < w and 0 <= keypoints[ki, 1] < h):
                        keypoints[ki, 2] = 0

        # Color jitter
        img = img.astype(np.float32)
        img *= np.random.uniform(0.8, 1.2)  # brightness
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Gaussian blur
        if np.random.random() < 0.3:
            sigma = np.random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (0, 0), sigma)

        return img, keypoints


# ── Model ────────────────────────────────────────────────────────────────────

class HRNetKeypointModel(nn.Module):
    """HRNet-W48 backbone + keypoint heatmap head."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        super().__init__()
        import timm

        # HRNet-W48 as feature extractor
        self.backbone = timm.create_model(
            "hrnet_w48",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,),  # only need highest resolution (1/4)
        )

        # Get the number of channels from the backbone
        # HRNet-W48 stage 0 output: 48 channels at 1/4 resolution
        backbone_channels = 48

        # Heatmap head: 1×1 conv
        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, num_keypoints, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, 540, 960) input images

        Returns:
            (B, 106, 135, 240) heatmaps
        """
        features = self.backbone(x)
        heatmaps = self.head(features[0])
        return heatmaps


# ── Training loop ────────────────────────────────────────────────────────────

def compute_loss(pred_heatmaps, gt_heatmaps, visibility):
    """MSE loss averaged over visible keypoints only.

    Args:
        pred_heatmaps: (B, K, H, W)
        gt_heatmaps: (B, K, H, W)
        visibility: (B, K) - 1.0 if visible, 0.0 if not
    """
    # Mask invisible keypoints
    mask = visibility.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    diff = (pred_heatmaps - gt_heatmaps) ** 2 * mask

    # Average over spatial dims, then over visible keypoints
    spatial_mean = diff.mean(dim=(-2, -1))  # (B, K)
    n_visible = visibility.sum(dim=1).clamp(min=1)  # (B,)
    loss = (spatial_mean.sum(dim=1) / n_visible).mean()  # scalar

    return loss


def compute_pck(pred_heatmaps, gt_heatmaps, visibility, threshold_px=10):
    """Percentage of Correct Keypoints at given pixel threshold.

    Args:
        pred_heatmaps: (B, K, H, W)
        gt_heatmaps: (B, K, H, W)
        visibility: (B, K)
        threshold_px: distance threshold in heatmap pixels
    """
    B, K, H, W = pred_heatmaps.shape

    # Get predicted and GT peak positions
    pred_flat = pred_heatmaps.view(B, K, -1)
    gt_flat = gt_heatmaps.view(B, K, -1)

    pred_idx = pred_flat.argmax(dim=2)
    gt_idx = gt_flat.argmax(dim=2)

    pred_x = (pred_idx % W).float()
    pred_y = (pred_idx // W).float()
    gt_x = (gt_idx % W).float()
    gt_y = (gt_idx // W).float()

    dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

    # Only count visible keypoints with non-zero GT
    gt_has_peak = gt_flat.max(dim=2).values > 0.1
    valid = (visibility > 0) & gt_has_peak

    if valid.sum() == 0:
        return 1.0

    correct = (dist < threshold_px) & valid
    pck = correct.sum().float() / valid.sum().float()
    return pck.item()


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, heatmaps, visibility) in enumerate(loader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        visibility = visibility.to(device)

        pred = model(images)

        # Resize pred if needed (should match, but safety check)
        if pred.shape[-2:] != heatmaps.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = compute_loss(pred, heatmaps, visibility)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch} batch {batch_idx + 1}/{len(loader)}: "
                  f"loss={loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_pck5 = 0
    total_pck10 = 0
    n_batches = 0

    for images, heatmaps, visibility in loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        visibility = visibility.to(device)

        pred = model(images)
        if pred.shape[-2:] != heatmaps.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = compute_loss(pred, heatmaps, visibility)
        pck5 = compute_pck(pred, heatmaps, visibility, threshold_px=5)
        pck10 = compute_pck(pred, heatmaps, visibility, threshold_px=10)

        total_loss += loss.item()
        total_pck5 += pck5
        total_pck10 += pck10
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_pck5 / n, total_pck10 / n


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HRNet field keypoint detector")
    parser.add_argument("--dataset", required=True, help="Dataset directory with annotations.json")
    parser.add_argument("--val-dataset", default=None, help="Separate validation dataset (default: split from train)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                        help="Learning rate multiplier for backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output", default="/workspace/output", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Don't use ImageNet pretrained backbone")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build flip mapping for augmentation
    # Import here since this runs on RunPod where keypoint_schema may not be in path
    flip_mapping = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.homography.keypoint_schema import FLIP_MAPPING
        flip_mapping = FLIP_MAPPING
        print("Loaded flip mapping for horizontal augmentation")
    except ImportError:
        print("Warning: could not load FLIP_MAPPING, horizontal flip disabled")

    # Load datasets
    train_ds = FieldKeypointDataset(args.dataset, augment=True, flip_mapping=flip_mapping)

    if args.val_dataset:
        val_ds = FieldKeypointDataset(args.val_dataset, augment=False)
    else:
        # Split train into train/val (80/20)
        n_val = max(1, len(train_ds) // 5)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # Disable augmentation on val split
        val_ds.dataset = FieldKeypointDataset(args.dataset, augment=False)
        print(f"Split: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Build model
    model = HRNetKeypointModel(
        num_keypoints=NUM_KEYPOINTS,
        pretrained=not args.no_pretrained,
    ).to(device)

    # Load checkpoint if resuming
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Resumed from {args.resume}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded weights from {args.resume}")

    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine annealing scheduler (after warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )

    # W&B logging
    if args.wandb and HAS_WANDB:
        wandb.init(project="field-keypoints", config=vars(args))
        wandb.watch(model, log_freq=100)

    # Training loop
    best_pck10 = 0.0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  LR: backbone={args.lr * args.backbone_lr_mult:.1e}, head={args.lr:.1e}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                if pg is optimizer.param_groups[0]:
                    pg["lr"] = args.lr * args.backbone_lr_mult * warmup_factor
                else:
                    pg["lr"] = args.lr * warmup_factor

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_pck5, val_pck10 = validate(model, val_loader, device)

        if epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        lr_head = optimizer.param_groups[1]["lr"]

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s): "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"PCK@5={val_pck5:.3f}, PCK@10={val_pck10:.3f}, lr={lr_head:.2e}")

        # Logging
        if args.wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pck5": val_pck5,
                "val_pck10": val_pck10,
                "lr": lr_head,
            })

        # Save best model
        if val_pck10 > best_pck10:
            best_pck10 = val_pck10
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pck10": val_pck10,
                "val_loss": val_loss,
            }, os.path.join(args.output, "best.pth"))
            print(f"  -> New best PCK@10: {val_pck10:.3f}")
        else:
            patience_counter += 1

        # Save last checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pck10": val_pck10,
                "val_loss": val_loss,
            }, os.path.join(args.output, "last.pth"))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_pck10": val_pck10,
    }, os.path.join(args.output, "last.pth"))

    print(f"\nDone. Best PCK@10: {best_pck10:.3f}")
    print(f"Weights saved to {args.output}/best.pth")

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
