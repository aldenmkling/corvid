"""Number-crop classifier — minimal inference-only loader.

The trained checkpoint `models/dsresnet10ww_round3_128x32/best.pth` was
produced by a deleted training script (`train_compare_classifiers.py`).
This module reconstructs the architecture from the state_dict's layer
shapes (depthwise-separable conv blocks + squeeze-excitation + linear
head) so the production pipeline can load the weights without depending
on the lost training file.

Exports:
- DSResNet10ww (model class)
- build_model(arch, dropout, num_classes) — factory
- make_backbone_logits_fn(ckpt, arch, device) — returns crops → logits fn
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from number_classifier_constants import (
    CLASSES, NUM_CLASSES, PIXEL_MEAN, PIXEL_STD,
)


class _SE(nn.Module):
    """Squeeze-and-excitation gate, reduction=4."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(1, ch // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class _DSBlock(nn.Module):
    """Depthwise-separable conv block: dw 3x3 → BN → act → pw 1x1 → BN → SE → act."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 use_se: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = _SE(out_ch) if use_se else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        if self.se is not None:
            x = self.se(x)
        # Residual when shapes align (stride=1 + in_ch==out_ch).
        if identity.shape == x.shape:
            x = x + identity
        return self.act(x)


class DSResNet10ww(nn.Module):
    """Depthwise-separable ResNet-10-style classifier with SE gates.

    Architecture reconstructed from state_dict shapes in
    models/dsresnet10ww_round3_128x32/best.pth:
      features.0: 1 → 24, stride 1, no SE
      features.1: 24 → 24, stride 2, SE(mid=6)
      features.2: 24 → 40, stride 1, SE(mid=10)
      features.3: 40 → 40, stride 2, SE(mid=10)
      features.4: 40 → 56, stride 1, SE(mid=14)
      features.5: 56 → 80, stride 2, SE(mid=20)
      features.6: 80 → 128, stride 1, SE(mid=32)
      head: AdaptiveAvgPool → Flatten → Dropout → Linear(128, 9)

    Input: (B, 1, H, W) — single-channel binary crop, default 32×128 (HxW).
    """
    def __init__(self, num_classes: int = 9, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            _DSBlock(1,   24, stride=1, use_se=False),
            _DSBlock(24,  24, stride=2, use_se=True),
            _DSBlock(24,  40, stride=1, use_se=True),
            _DSBlock(40,  40, stride=2, use_se=True),
            _DSBlock(40,  56, stride=1, use_se=True),
            _DSBlock(56,  80, stride=2, use_se=True),
            _DSBlock(80, 128, stride=1, use_se=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch: str, dropout: float = 0.0, num_classes: int = NUM_CLASSES):
    """Factory matching the deleted train_compare_classifiers.build_model
    signature. Currently only `dsresnet10ww` is wired up (it's our active
    crop classifier)."""
    if arch == "dsresnet10ww":
        return DSResNet10ww(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported crop classifier arch: {arch}")


def make_backbone_logits_fn(ckpt_path: str, arch: str, device):
    """Load the crop classifier and return a `crops → (N, num_classes) logits`
    callable. Mirrors the old train_scene_refiner.make_backbone_logits_fn API.

    Crops are uint8 single-channel arrays (typically 32×128); normalized via
    PIXEL_MEAN / PIXEL_STD before forward.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    classes = ck.get("classes", CLASSES)
    if list(classes) != CLASSES:
        idx_map = [classes.index(c) for c in CLASSES if c in classes]
        if len(idx_map) != NUM_CLASSES:
            raise ValueError(f"ckpt classes {classes} don't match {CLASSES}")
    else:
        idx_map = None
    model = build_model(arch, dropout=0.0, num_classes=len(classes))
    model.load_state_dict(state)
    model.to(device).eval()

    @torch.no_grad()
    def _fn(crops):
        if not crops:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        arr = np.stack([c.astype(np.float32) for c in crops], axis=0)
        arr = (arr / 255.0 - PIXEL_MEAN) / PIXEL_STD
        x = torch.from_numpy(arr).unsqueeze(1).to(device)
        logits = model(x).cpu().numpy().astype(np.float32)
        if idx_map is not None:
            logits = logits[:, idx_map]
        return logits

    return _fn
