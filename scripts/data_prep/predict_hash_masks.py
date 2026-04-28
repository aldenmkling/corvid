#!/usr/bin/env python3
"""Run trained hash UNet on a set of frames; save predicted masks.

Source set is selected via --mode against `hash_triage_results.json`:
  fix    — frames the user marked needs-fixing  (default)
  good   — frames the user marked good  (sanity check)
  all    — every triaged frame
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--triage", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/triage/hash_triage_results.json"))
    ap.add_argument("--source-images", default=os.path.join(
        PROJECT_ROOT, "data/field_keypoints/train/images"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_ROOT, "data/hash_masks/predicted_round1"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", default="fix", choices=["fix", "good", "all"])
    ap.add_argument("--encoder", default="mit_b0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = smp.Unet(encoder_name=args.encoder, encoder_weights=None,
                      in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    with open(args.triage) as f:
        decisions = json.load(f)
    if args.mode == "all":
        targets = list(decisions.keys())
    else:
        targets = [k for k, v in decisions.items() if v == args.mode]
    print(f"  {len(targets)} '{args.mode}' frames")

    out_mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(out_mask_dir, exist_ok=True)

    n = 0
    with torch.no_grad():
        for frame in targets:
            src = os.path.join(args.source_images, frame)
            if not os.path.exists(src):
                continue
            img = cv2.imread(src)
            tensor = preprocess(img).to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob > args.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            stem = os.path.splitext(frame)[0]
            cv2.imwrite(os.path.join(out_mask_dir, stem + ".png"), mask)
            n += 1
    print(f"  predicted {n} masks → {out_mask_dir}/")


if __name__ == "__main__":
    main()
