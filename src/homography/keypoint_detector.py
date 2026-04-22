"""
ML-based field keypoint detector using trained HRNet-W48.

Detects field features (sideline intersections, hash intersections) as
multi-peak heatmaps. Each channel may contain many peaks — one per visible
instance. Identity assignment (which yard line) is handled downstream by
the grid solver, not here.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from dataclasses import dataclass
from scipy import ndimage

from .keypoint_schema import NUM_CHANNELS, CHANNEL_NAMES


# ── Constants ────────────────────────────────────────────────────────────────

INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class KeypointDetection:
    """Result of field keypoint detection for a single frame.

    Contains all detected peaks across all channels. No identity assignment —
    just pixel locations, which channel they came from, and confidence.
    """
    pixel_xy: np.ndarray       # (K, 2) detected peak positions in original frame coords
    channel_ids: np.ndarray    # (K,) which heatmap channel each peak came from
    confidences: np.ndarray    # (K,) per-peak confidence (sigmoid value)


# ── Model definition (must match training) ───────────────────────────────────

class HRNetKeypointModel(nn.Module):
    """HRNet-W48 backbone + keypoint heatmap head."""

    def __init__(self, num_channels: int = NUM_CHANNELS):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "hrnet_w48",
            pretrained=False,
            features_only=True,
            out_indices=(0,),
        )

        backbone_channels = 64
        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, num_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features[0])


# ── Sub-pixel refinement ────────────────────────────────────────────────────

def _refine_peak(heatmap: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """DARK-style sub-pixel refinement using quadratic fit on 3x3 neighborhood."""
    h, w = heatmap.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return float(x), float(y)

    dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
    dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
    dxx = heatmap[y, x + 1] - 2 * heatmap[y, x] + heatmap[y, x - 1]
    dyy = heatmap[y + 1, x] - 2 * heatmap[y, x] + heatmap[y - 1, x]

    ox = -dx / dxx if abs(dxx) > 1e-6 else 0.0
    oy = -dy / dyy if abs(dyy) > 1e-6 else 0.0
    ox = float(np.clip(ox, -0.5, 0.5))
    oy = float(np.clip(oy, -0.5, 0.5))

    return float(x) + ox, float(y) + oy


def _extract_peaks(
    heatmap: np.ndarray,
    conf_thresh: float,
    min_distance: int = 5,
) -> list[tuple[float, float, float]]:
    """Extract multiple peaks from a single-channel heatmap.

    Uses connected-component labeling on thresholded heatmap, then finds
    the max within each component for sub-pixel refinement.

    Args:
        heatmap: (H, W) sigmoid-activated heatmap
        conf_thresh: minimum peak confidence
        min_distance: not used directly, but components smaller than this
                      squared are filtered as noise

    Returns:
        List of (refined_x, refined_y, confidence) tuples.
    """
    mask = heatmap >= conf_thresh
    if not mask.any():
        return []

    labels, num_components = ndimage.label(mask)
    peaks = []

    for comp_id in range(1, num_components + 1):
        comp_mask = labels == comp_id

        # Find max within this component
        comp_vals = heatmap * comp_mask
        peak_idx = comp_vals.argmax()
        peak_y = peak_idx // heatmap.shape[1]
        peak_x = peak_idx % heatmap.shape[1]
        peak_val = heatmap[peak_y, peak_x]

        ref_x, ref_y = _refine_peak(heatmap, peak_y, peak_x)
        peaks.append((ref_x, ref_y, float(peak_val)))

    return peaks


# ── Detector class ──────────────────────────────────────────────────────────

class FieldKeypointDetector:
    """Detect field keypoints using trained HRNet model.

    Usage:
        detector = FieldKeypointDetector("models/hrnet_last.pth")
        result = detector.detect(frame)
        # result.pixel_xy — (K, 2) peak locations in original frame coords
        # result.channel_ids — (K,) 0=sideline, 1=hash
        # result.confidences — (K,) sigmoid confidence
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.3,
    ):
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh

        self.model = HRNetKeypointModel(num_channels=NUM_CHANNELS)
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> KeypointDetection:
        """Detect field keypoints in a single BGR frame.

        Args:
            frame: (H, W, 3) BGR image

        Returns:
            KeypointDetection with all detected peaks above confidence threshold
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Forward pass + sigmoid
        logits = self.model(tensor)  # (1, 2, H_hm, W_hm)
        heatmaps = torch.sigmoid(logits[0]).cpu().numpy()  # (2, H_hm, W_hm)

        _, hm_h, hm_w = heatmaps.shape

        # Extract peaks from each channel
        pixel_xy_list = []
        channel_ids_list = []
        confidences_list = []

        for ch in range(NUM_CHANNELS):
            peaks = _extract_peaks(heatmaps[ch], self.conf_thresh)
            for ref_x, ref_y, conf in peaks:
                # Map from heatmap coords to original frame coords
                px = ref_x / hm_w * orig_w
                py = ref_y / hm_h * orig_h

                pixel_xy_list.append([px, py])
                channel_ids_list.append(ch)
                confidences_list.append(conf)

        if len(pixel_xy_list) == 0:
            return KeypointDetection(
                pixel_xy=np.array([]).reshape(0, 2),
                channel_ids=np.array([], dtype=np.int32),
                confidences=np.array([], dtype=np.float32),
            )

        return KeypointDetection(
            pixel_xy=np.array(pixel_xy_list, dtype=np.float64),
            channel_ids=np.array(channel_ids_list, dtype=np.int32),
            confidences=np.array(confidences_list, dtype=np.float32),
        )
