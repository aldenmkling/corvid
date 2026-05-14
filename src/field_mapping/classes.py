"""NGS-x classification constants for the field-mapping pipeline.

The pipeline labels each detected token (yardline / sideline / hash / number)
with its NGS-x coordinate, quantized into 21 classes spaced 5 yards apart
(positions 10, 15, 20, ..., 110 — the goal lines and every yardline +
midpoints).
"""
import torch

# ── Field bounds (NGS yards) ─────────────────────────────────────────────
NGS_X_MAX = 120.0    # full field length, including endzones
NGS_Y_MAX = 53.33    # field width, sideline to sideline

# ── 21-class NGS-x quantization ──────────────────────────────────────────
NGS_X_CLASS_STEP = 5.0
NGS_X_CLASS_MIN = 10.0
NGS_X_CLASS_MAX = 110.0
N_NGS_X_CLASSES = int(round(
    (NGS_X_CLASS_MAX - NGS_X_CLASS_MIN) / NGS_X_CLASS_STEP)) + 1   # 21


def ngs_x_to_class(ngs_x_yards):
    """Quantize an NGS-x value (yards) into a class index in [0, 20]."""
    idx = torch.round((ngs_x_yards - NGS_X_CLASS_MIN) / NGS_X_CLASS_STEP)
    return idx.long().clamp(0, N_NGS_X_CLASSES - 1)


def make_class_to_ngs_x_norm(device):
    """Map class index → NGS_x normalized to [0, 1]. Used at inference for
    soft expected-value computation (pass-1 predictions feeding pass-2
    anchors).
    """
    classes = torch.arange(N_NGS_X_CLASSES, device=device, dtype=torch.float32)
    yards = NGS_X_CLASS_MIN + classes * NGS_X_CLASS_STEP
    return yards / NGS_X_MAX


# ── Painted-number → 21-class mapping ────────────────────────────────────
# Painted-number index order (left-to-right across the field):
#   PAINTED_CLASSES = ["10L", "20L", "30L", "40L", "50", "40R", "30R", "20R", "10R"]
# That maps to NGS-x positions (yards) 20, 30, 40, 50, 60, 70, 80, 90, 100,
# which are 21-class indices 2, 4, 6, 8, 10, 12, 14, 16, 18.
#
# Note: the crop classifier emits logits in its own CLASSES order
# ("10L", "10R", "20L", "20R", ...) — that gets permuted to PAINTED_CLASSES
# order inside crop_classifier.py before this mapping is applied.
PAINTED_TO_21 = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=torch.long)
PAINTED_CLASSES = ["10L", "20L", "30L", "40L", "50", "40R", "30R", "20R", "10R"]
