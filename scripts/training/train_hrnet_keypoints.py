#!/usr/bin/env python3
"""
Train HRNet-W48 for field keypoint heatmap prediction.

Two-stage training:
  1. Pre-train on synthetic data (~5000 frames)
  2. Fine-tune on real annotated data (~240 frames)

Input: 960×540 images
Output: 256×448 heatmaps (2 channels: sideline, hash)
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

NUM_CHANNELS = 2      # sideline_intersection, hash_intersection
INPUT_H, INPUT_W = 512, 896
HEATMAP_H, HEATMAP_W = 256, 448  # 1/2 resolution (HRNet stage 0 output)
SIGMA_MAX = 6.0   # wide Gaussians early (easy to learn)
SIGMA_MIN = 1.0   # tight Gaussians late (forces precision)
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
    """Dataset for 2-channel field keypoint detection.

    Annotations use a simple JSON format:
      {"points": [{"x": px, "y": py, "channel": 0-1, "visible": true}, ...]}

    Each channel can have MULTIPLE peaks (one per visible instance of that type).
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = data_dir
        self.augment = augment
        self.sigma = SIGMA_MAX  # updated by training loop each epoch

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

        # Parse points: list of {x, y, channel, visible}
        points = ann["points"]

        # Resize image to input size
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        scale_x = INPUT_W / orig_w
        scale_y = INPUT_H / orig_h

        # Scale point coordinates
        scaled_points = []
        for p in points:
            if p["visible"]:
                scaled_points.append({
                    "x": p["x"] * scale_x,
                    "y": p["y"] * scale_y,
                    "channel": p["channel"],
                })

        # Augmentation
        if self.augment:
            img, scaled_points = self._augment(img, scaled_points)

        # Generate multi-peak heatmaps (2 channels)
        heatmaps = np.zeros((NUM_CHANNELS, HEATMAP_H, HEATMAP_W), dtype=np.float32)
        channel_has_peaks = np.zeros(NUM_CHANNELS, dtype=np.float32)

        for p in scaled_points:
            hm_x = p["x"] * HEATMAP_W / INPUT_W
            hm_y = p["y"] * HEATMAP_H / INPUT_H
            ch = p["channel"]

            if 0 <= hm_x < HEATMAP_W and 0 <= hm_y < HEATMAP_H:
                # Add peak to channel (use max so overlapping peaks don't exceed 1.0)
                peak = _generate_heatmap(hm_x, hm_y, HEATMAP_H, HEATMAP_W, self.sigma)
                heatmaps[ch] = np.maximum(heatmaps[ch], peak)
                channel_has_peaks[ch] = 1.0

        # Normalize image
        img = img.astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return (
            torch.from_numpy(img),
            torch.from_numpy(heatmaps),
            torch.from_numpy(channel_has_peaks),
        )

    def _augment(self, img: np.ndarray, points: list[dict]) -> tuple:
        """Apply augmentations. Returns (img, points)."""
        h, w = img.shape[:2]

        # Horizontal flip (no channel remapping needed — types are symmetric)
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            for p in points:
                p["x"] = w - 1 - p["x"]

        # Random rotation (±10°)
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        new_points = []
        for p in points:
            pt = np.array([p["x"], p["y"], 1.0])
            new_pt = M @ pt
            if 0 <= new_pt[0] < w and 0 <= new_pt[1] < h:
                new_points.append({"x": new_pt[0], "y": new_pt[1], "channel": p["channel"]})
        points = new_points

        # Random scale (0.8-1.2)
        scale = np.random.uniform(0.8, 1.2)
        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_scaled = cv2.resize(img, (new_w, new_h))
            for p in points:
                p["x"] *= scale
                p["y"] *= scale

            if scale > 1.0:
                x1 = (new_w - w) // 2
                y1 = (new_h - h) // 2
                img = img_scaled[y1:y1 + h, x1:x1 + w]
                for p in points:
                    p["x"] -= x1
                    p["y"] -= y1
            else:
                img = np.zeros((h, w, 3), dtype=img_scaled.dtype)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                img[y1:y1 + new_h, x1:x1 + new_w] = img_scaled
                for p in points:
                    p["x"] += x1
                    p["y"] += y1

            points = [p for p in points if 0 <= p["x"] < w and 0 <= p["y"] < h]

        # Color jitter
        img = img.astype(np.float32)
        img *= np.random.uniform(0.8, 1.2)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Gaussian blur
        if np.random.random() < 0.3:
            sigma = np.random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (0, 0), sigma)

        return img, points


# ── Model ────────────────────────────────────────────────────────────────────

class HRNetKeypointModel(nn.Module):
    """HRNet-W48 backbone + 2-channel heatmap head."""

    def __init__(self, num_channels: int = NUM_CHANNELS, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "hrnet_w48",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,),
        )

        backbone_channels = 64  # HRNet-W48 stage 0

        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, num_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (B, 3, H/2, W/2) heatmaps."""
        features = self.backbone(x)
        return self.head(features[0])


# ── Training loop ────────────────────────────────────────────────────────────

def compute_loss(pred_heatmaps, gt_heatmaps, channel_has_peaks, channel_weights=None):
    """CenterNet-style focal loss for multi-peak heatmaps.

    Adapted from CornerNet/CenterNet. Handles the sparse peak problem:
    - At GT peak locations (gt=1): penalize if prediction is low
    - At background (gt<1): penalize if prediction is high, but reduce
      penalty near GT peaks (where some activation is expected)

    Args:
        pred_heatmaps: (B, C, H, W) — raw model output (will be sigmoided)
        gt_heatmaps: (B, C, H, W) — Gaussian targets, peak=1.0
        channel_has_peaks: (B, C) — unused but kept for API compatibility
        channel_weights: optional (C,) tensor of per-channel loss multipliers.
            Used to upweight underperforming classes (e.g. sidelines).
    """
    # Sigmoid to constrain predictions to [0, 1]
    pred = torch.sigmoid(pred_heatmaps)

    # Clamp for numerical stability
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)

    # Separate peak and background pixels
    pos_mask = (gt_heatmaps >= 0.9).float()
    neg_mask = (gt_heatmaps < 0.9).float()

    # Focal weights
    alpha = 2.0
    beta = 2.0

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_weight = torch.pow(1 - gt_heatmaps, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask

    # Apply per-channel weights (broadcast over B, H, W).
    if channel_weights is not None:
        w = channel_weights.to(pred.device).view(1, -1, 1, 1)
        pos_loss = pos_loss * w
        neg_loss = neg_loss * w

    # Normalize by number of peaks
    n_peaks = pos_mask.sum().clamp(min=1)
    loss = (pos_loss.sum() + neg_loss.sum()) / n_peaks

    return loss


def compute_detection_score(pred_heatmaps, gt_heatmaps, channel_has_peaks, threshold=0.3):
    """Measure how well the model detects peaks.

    For each GT peak location, check if the predicted heatmap has a value
    above threshold within a radius. Returns (precision, recall).
    """
    from scipy import ndimage

    # Apply sigmoid since model outputs logits with focal loss
    pred_heatmaps = torch.sigmoid(pred_heatmaps)

    B, C, H, W = pred_heatmaps.shape
    total_gt_peaks = 0
    total_detected = 0
    total_pred_peaks = 0

    for b in range(B):
        for c in range(C):
            gt = gt_heatmaps[b, c].cpu().numpy()
            pred = pred_heatmaps[b, c].cpu().numpy()

            # Find GT peak locations (local maxima above 0.5)
            gt_peaks = []
            gt_binary = (gt > 0.5)
            if gt_binary.any():
                labeled, n_features = ndimage.label(gt_binary)
                for i in range(1, n_features + 1):
                    ys, xs = np.where(labeled == i)
                    cy, cx = ys.mean(), xs.mean()
                    gt_peaks.append((cx, cy))

            # Find predicted peaks (local maxima above threshold)
            pred_peaks = []
            pred_binary = (pred > threshold)
            if pred_binary.any():
                labeled, n_features = ndimage.label(pred_binary)
                for i in range(1, n_features + 1):
                    ys, xs = np.where(labeled == i)
                    cy, cx = ys.mean(), xs.mean()
                    pred_peaks.append((cx, cy))

            # Match: for each GT peak, is there a pred peak within 10px?
            for gx, gy in gt_peaks:
                total_gt_peaks += 1
                for px, py in pred_peaks:
                    if np.sqrt((gx - px)**2 + (gy - py)**2) < 10:
                        total_detected += 1
                        break

            total_pred_peaks += len(pred_peaks)

    recall = total_detected / max(total_gt_peaks, 1)
    precision = total_detected / max(total_pred_peaks, 1)
    return precision, recall


def train_epoch(model, loader, optimizer, device, epoch, channel_weights=None):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, heatmaps, channel_has_peaks) in enumerate(loader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        channel_has_peaks = channel_has_peaks.to(device)

        pred = model(images)

        # Resize pred if needed (should match, but safety check)
        if pred.shape[-2:] != heatmaps.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = compute_loss(pred, heatmaps, channel_has_peaks, channel_weights)

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
def validate(model, loader, device, compute_detection=False, channel_weights=None):
    """Validate model. Only computes expensive detection score when requested."""
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    n_batches = 0

    for images, heatmaps, channel_has_peaks in loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        channel_has_peaks = channel_has_peaks.to(device)

        pred = model(images)
        if pred.shape[-2:] != heatmaps.shape[-2:]:
            pred = nn.functional.interpolate(
                pred, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = compute_loss(pred, heatmaps, channel_has_peaks, channel_weights)
        total_loss += loss.item()

        if compute_detection:
            precision, recall = compute_detection_score(pred, heatmaps, channel_has_peaks)
            total_precision += precision
            total_recall += recall

        n_batches += 1

    n = max(n_batches, 1)
    if compute_detection:
        return total_loss / n, total_precision / n, total_recall / n
    else:
        return total_loss / n, None, None


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
    parser.add_argument("--sigma-max", type=float, default=None, help="Starting sigma (default: SIGMA_MAX)")
    parser.add_argument("--sigma-min", type=float, default=None, help="Ending sigma (default: SIGMA_MIN)")
    parser.add_argument("--channel-weights", type=str, default=None,
                        help="Comma-separated per-channel loss weights, e.g. '3,1' to upweight sidelines 3x. "
                             "Must match NUM_CHANNELS.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output", default="/workspace/output", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Don't use ImageNet pretrained backbone")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    train_ds = FieldKeypointDataset(args.dataset, augment=True)

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
        num_channels=NUM_CHANNELS,
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
    best_recall = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    # Parse channel weights (e.g. "3,1" upweights channel 0 / sidelines 3x)
    channel_weights_tensor = None
    if args.channel_weights:
        weights_list = [float(w) for w in args.channel_weights.split(",")]
        assert len(weights_list) == NUM_CHANNELS, \
            f"Got {len(weights_list)} channel weights, need {NUM_CHANNELS}"
        channel_weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=device)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  LR: backbone={args.lr * args.backbone_lr_mult:.1e}, head={args.lr:.1e}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Channels: {NUM_CHANNELS} (0=sideline, 1=hash)")
    if channel_weights_tensor is not None:
        print(f"  Channel weights: {channel_weights_tensor.tolist()}")
    print(f"  Sigma: {SIGMA_MAX} -> {SIGMA_MIN} (shrinking over training)")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Update sigma: linear decay for first 60% of epochs, then hold at min
        s_max = args.sigma_max if args.sigma_max is not None else SIGMA_MAX
        s_min = args.sigma_min if args.sigma_min is not None else SIGMA_MIN
        shrink_epochs = int(args.epochs * 0.6)
        if epoch <= shrink_epochs:
            progress = (epoch - 1) / max(shrink_epochs - 1, 1)
            current_sigma = s_max - (s_max - s_min) * progress
        else:
            current_sigma = s_min
        # Update on the underlying dataset (handles random_split wrapper)
        ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        ds.sigma = current_sigma
        val_base = val_ds.dataset if hasattr(val_ds, 'dataset') else val_ds
        val_base.sigma = current_sigma

        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                if pg is optimizer.param_groups[0]:
                    pg["lr"] = args.lr * args.backbone_lr_mult * warmup_factor
                else:
                    pg["lr"] = args.lr * warmup_factor

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch,
                                 channel_weights=channel_weights_tensor)

        # Compute expensive detection score every 10 epochs (or first/last)
        do_detection = (epoch % 10 == 0) or (epoch <= 2) or (epoch == args.epochs)
        val_loss, val_precision, val_recall = validate(
            model, val_loader, device, compute_detection=do_detection,
            channel_weights=channel_weights_tensor,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        lr_head = optimizer.param_groups[1]["lr"]

        if val_precision is not None:
            print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s): "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"precision={val_precision:.3f}, recall={val_recall:.3f}, "
                  f"sigma={current_sigma:.2f}, lr={lr_head:.2e}")
        else:
            print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s): "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"sigma={current_sigma:.2f}, lr={lr_head:.2e}")

        # Logging
        if args.wandb and HAS_WANDB:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_head,
            }
            if val_precision is not None:
                log_dict["val_precision"] = val_precision
                log_dict["val_recall"] = val_recall
            wandb.log(log_dict)

        # Save best model (optimize for recall when available, otherwise val_loss)
        save_best = False
        if val_recall is not None and val_recall > best_recall:
            best_recall = val_recall
            save_best = True
        elif val_recall is None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best = True

        if save_best:
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall": val_recall if val_recall is not None else best_recall,
                "val_precision": val_precision,
                "val_loss": val_loss,
            }, os.path.join(args.output, "best.pth"))
            if val_recall is not None:
                print(f"  -> New best recall: {val_recall:.3f} (precision: {val_precision:.3f})")
            else:
                print(f"  -> New best val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        # Save last checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall": val_recall,
                "val_loss": val_loss,
            }, os.path.join(args.output, "last.pth"))

        # Early stopping (disabled with --no-early-stop or --patience 0)
        if args.patience > 0 and not args.no_early_stop and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_recall": val_recall,
    }, os.path.join(args.output, "last.pth"))

    print(f"\nDone. Best recall: {best_recall:.3f}")
    print(f"Weights saved to {args.output}/best.pth")

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
