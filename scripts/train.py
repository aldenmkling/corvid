#!/usr/bin/env python3
"""
Fine-tune YOLOv12 on annotated All-22 frames.

Usage:
  python scripts/train.py [--epochs 50] [--batch 8]
"""

import argparse
import os
import random
import shutil

from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
IMAGES_DIR = os.path.join(ANNOTATIONS_DIR, "images")
LABELS_DIR = os.path.join(ANNOTATIONS_DIR, "labels")


def setup_train_val_split(val_ratio=0.15, seed=42):
    """Split images/labels into train and val sets."""
    random.seed(seed)

    train_img = os.path.join(ANNOTATIONS_DIR, "train", "images")
    train_lbl = os.path.join(ANNOTATIONS_DIR, "train", "labels")
    val_img = os.path.join(ANNOTATIONS_DIR, "val", "images")
    val_lbl = os.path.join(ANNOTATIONS_DIR, "val", "labels")

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # Get all images that have labels
    images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith(".jpg")
        and os.path.exists(os.path.join(LABELS_DIR, f.replace(".jpg", ".txt")))
    ])

    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    val_set = set(images[:val_count])

    for img_name in images:
        lbl_name = img_name.replace(".jpg", ".txt")
        if img_name in val_set:
            shutil.copy2(os.path.join(IMAGES_DIR, img_name), os.path.join(val_img, img_name))
            shutil.copy2(os.path.join(LABELS_DIR, lbl_name), os.path.join(val_lbl, lbl_name))
        else:
            shutil.copy2(os.path.join(IMAGES_DIR, img_name), os.path.join(train_img, img_name))
            shutil.copy2(os.path.join(LABELS_DIR, lbl_name), os.path.join(train_lbl, lbl_name))

    print(f"Split: {len(images) - val_count} train, {val_count} val")
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolo12x.pt",
                        help="Base model to fine-tune from")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size")
    args = parser.parse_args()

    # Setup train/val split
    n_images = setup_train_val_split()
    if n_images == 0:
        print("No annotated images found! Run extract + annotate first.")
        return

    # Write dataset config with train/val paths
    dataset_yaml = os.path.join(ANNOTATIONS_DIR, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {ANNOTATIONS_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: player\n")

    # Fine-tune
    print(f"\nFine-tuning {args.model} on {n_images} images...")
    print(f"  epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")

    model = YOLO(args.model)
    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=os.path.join(PROJECT_ROOT, "output", "training"),
        name="all22_player_detect",
        exist_ok=True,
        # Freeze backbone for faster fine-tuning with limited data
        freeze=10,
        # Augmentation suitable for overhead sports footage
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0,       # no rotation — field is always oriented
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,      # no vertical flip — camera is always above
        mosaic=0.5,
    )

    print("\nTraining complete!")
    best = os.path.join(PROJECT_ROOT, "output", "training", "all22_player_detect", "weights", "best.pt")
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
