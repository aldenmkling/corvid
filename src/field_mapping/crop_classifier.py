"""Number-crop classifier — classifies a 32×128 binary crop of a painted
yardline number into one of 9 painted classes ("10L"/"10R"/.../"50").

Architecture: depthwise-separable conv blocks (7) with squeeze-excitation
gates, AdaptiveAvgPool + Linear head. ~40K params. ResNet-style strides
(stride-2 at blocks 1/3/5). Reconstructed from state_dict shapes after the
original training script was lost in the 2026-05-14 cleanup; retrained
2026-05-14 to val_acc 96.76%.

Public API:
- CropClassifier (model class)
- load_crop_classifier(ckpt_path, device) → (crops → logits) function
- CLASSES — the 9-class label list, ordered to match the trained checkpoint
- PIXEL_MEAN / PIXEL_STD — input normalization constants
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── Crop preprocessing + label constants ─────────────────────────────────
INPUT_W = 128
INPUT_H = 32
PIXEL_MEAN = 0.456    # avg of 3-ch ImageNet pretrain mean (single-channel)
PIXEL_STD = 0.224

CLASSES = ["10L", "10R", "20L", "20R", "30L", "30R", "40L", "40R", "50"]
NUM_CLASSES = len(CLASSES)


# ── Model ────────────────────────────────────────────────────────────────
class _SqueezeExcite(nn.Module):
    """Channel-wise squeeze-and-excitation gate, reduction=4."""
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


class _DepthwiseBlock(nn.Module):
    """dw 3×3 → BN → ReLU → pw 1×1 → BN → SE? → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = _SqueezeExcite(out_ch) if use_se else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        if self.se is not None:
            x = self.se(x)
        return self.act(x)


class CropClassifier(nn.Module):
    """7 depthwise-separable blocks with SE gates → AvgPool → Linear(9).

    Input: (B, 1, 32, 128) — binary number crop, normalized.
    Output: (B, 9) — logits over CLASSES.

    Channel progression: 1 → 24 → 24 → 40 → 40 → 56 → 80 → 128.
    Stride-2 downsampling at blocks 1, 3, 5.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            _DepthwiseBlock(1,   24, stride=1, use_se=False),
            _DepthwiseBlock(24,  24, stride=2, use_se=True),
            _DepthwiseBlock(24,  40, stride=1, use_se=True),
            _DepthwiseBlock(40,  40, stride=2, use_se=True),
            _DepthwiseBlock(40,  56, stride=1, use_se=True),
            _DepthwiseBlock(56,  80, stride=2, use_se=True),
            _DepthwiseBlock(80, 128, stride=1, use_se=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ── Loader ───────────────────────────────────────────────────────────────
def load_crop_classifier(ckpt_path: str, device):
    """Load a CropClassifier checkpoint and return a `crops → (N, NUM_CLASSES)`
    logits callable. Crops are uint8 single-channel arrays (any size; resized
    to 32×128 by the caller); normalized to ImageNet-ish stats internally.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    classes = list(ck.get("classes", CLASSES))
    if classes != CLASSES:
        idx_map = [classes.index(c) for c in CLASSES if c in classes]
        if len(idx_map) != NUM_CLASSES:
            raise ValueError(f"ckpt classes {classes} don't match {CLASSES}")
    else:
        idx_map = None
    model = CropClassifier(num_classes=len(classes), dropout=0.0)
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


# ── Painted-order permutation ────────────────────────────────────────────
def make_painted_logits_fn(ckpt_path: str, arch: str, device):
    """Backward-compat wrapper: load crop classifier and return a function
    that emits logits in *painted order* (PAINTED_CLASSES from
    field_mapping.classes), not the trained CLASSES order. This is what
    the downstream NumberRefiner expects.
    """
    from .classes import PAINTED_CLASSES   # local import to avoid cycle at module-load

    base_fn = load_crop_classifier(ckpt_path, device)
    # arch is accepted for backward-compat with the old API; we only have
    # one arch (CropClassifier / DSResNet10ww).
    name_to_painted = {n: i for i, n in enumerate(PAINTED_CLASSES)}
    perm = np.array([name_to_painted[c] for c in CLASSES], dtype=int)

    def _fn(crops):
        logits = base_fn(crops)
        if logits.size == 0:
            return logits
        out = np.zeros_like(logits)
        out[:, perm] = logits
        return out

    return _fn
