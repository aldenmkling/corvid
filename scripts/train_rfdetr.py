#!/usr/bin/env python3
"""
Train RF-DETR-Large for player detection on All-22 football film.

Designed to run on a RunPod instance with an RTX 5090 (32GB VRAM).

Setup on RunPod:
  1. Create a pod with RTX 5090 and PyTorch template
  2. Upload this script and dataset/ folder
  3. pip install rfdetr[train,loggers]
  4. python train_rfdetr.py

Dataset structure (COCO format from Label Studio export):
  dataset/
  ├── train/
  │   ├── _annotations.coco.json
  │   └── *.jpg
  └── valid/
      ├── _annotations.coco.json
      └── *.jpg

The script auto-detects the GPU and handles batch sizing.
If you get OOM, reduce --batch-size (halve it each time).
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR-Large for All-22 player detection")
    parser.add_argument("--dataset", default="dataset", help="Path to COCO dataset directory (default: dataset/)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU (default: 16, halve if OOM)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--resolution", type=int, default=1280, help="Input resolution, must be divisible by 64 (default: 1280)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--output", default="output", help="Output directory for checkpoints (default: output/)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--strategy", default="auto", help="Training strategy: auto, ddp (default: auto)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--project", default="all22-player-detect", help="W&B project name")
    args = parser.parse_args()

    # Validate resolution
    if args.resolution % 64 != 0:
        valid = [r for r in range(640, 1344, 64)]
        print(f"ERROR: Resolution must be divisible by 64. Valid options: {valid}")
        return

    # Validate dataset
    for split in ["train", "valid"]:
        split_dir = os.path.join(args.dataset, split)
        ann_file = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(ann_file):
            print(f"ERROR: Missing {ann_file}")
            print(f"Expected dataset structure:")
            print(f"  {args.dataset}/train/_annotations.coco.json")
            print(f"  {args.dataset}/train/*.jpg")
            print(f"  {args.dataset}/valid/_annotations.coco.json")
            print(f"  {args.dataset}/valid/*.jpg")
            return

    effective_batch = args.batch_size * args.devices * args.grad_accum
    print(f"{'='*60}")
    print(f"  RF-DETR-Large Training — All-22 Player Detection")
    print(f"{'='*60}")
    print(f"  Dataset:     {os.path.abspath(args.dataset)}")
    print(f"  Resolution:  {args.resolution}px")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size} × {args.devices} GPU(s) × {args.grad_accum} accum = {effective_batch} effective")
    print(f"  LR:          {args.lr}")
    print(f"  Output:      {args.output}")
    print(f"  Resume:      {args.resume or 'None (training from pretrained)'}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  W&B:         {'enabled' if args.wandb else 'disabled'}")
    print(f"{'='*60}\n")

    from rfdetr import RFDETRLarge

    # Initialize model (downloads pretrained weights if needed)
    model = RFDETRLarge(resolution=args.resolution)

    # Train
    model.train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output,
        resume=args.resume,
        devices=args.devices,
        strategy=args.strategy,
        wandb=args.wandb,
        project=args.project if args.wandb else None,
        multi_scale=True,
        num_workers=4,
        checkpoint_interval=10,
    )

    print(f"\nTraining complete!")
    print(f"Best weights: {args.output}/best.pt")
    print(f"Last weights: {args.output}/last.pt")


if __name__ == "__main__":
    main()
