"""
ML-based field keypoint detector using trained HRNet-W48.

Detects semantically labeled field keypoints (yard line intersections,
painted numbers, end zone corners) from a single frame. Each keypoint
has a known real-world field coordinate, enabling direct homography
computation without temporal tracking.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from dataclasses import dataclass

from .keypoint_schema import (
    NUM_KEYPOINTS, FIELD_COORDS, KEYPOINT_NAMES, KEYPOINTS,
)


# ── Constants ────────────────────────────────────────────────────────────────

INPUT_H, INPUT_W = 540, 960
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class KeypointDetection:
    """Result of field keypoint detection for a single frame."""
    pixel_xy: np.ndarray       # (K, 2) detected keypoint pixel positions (original frame coords)
    keypoint_ids: np.ndarray   # (K,) integer IDs into the 106-keypoint schema
    confidences: np.ndarray    # (K,) per-keypoint confidence (heatmap peak value)
    field_xy: np.ndarray       # (K, 2) corresponding field coordinates from schema
    all_confidences: np.ndarray  # (106,) confidence for every keypoint (0 if not detected)


# ── Model definition (must match training) ───────────────────────────────────

class HRNetKeypointModel(nn.Module):
    """HRNet-W48 backbone + keypoint heatmap head."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "hrnet_w48",
            pretrained=False,
            features_only=True,
            out_indices=(0,),
        )

        backbone_channels = 48
        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, num_keypoints, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features[0])


# ── Sub-pixel refinement ────────────────────────────────────────────────────

def _refine_peak(heatmap: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """DARK-style sub-pixel refinement using quadratic fit on 3×3 neighborhood.

    Returns refined (x, y) coordinates.
    """
    h, w = heatmap.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return float(x), float(y)

    # Quadratic fit along each axis independently
    dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
    dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
    dxx = heatmap[y, x + 1] - 2 * heatmap[y, x] + heatmap[y, x - 1]
    dyy = heatmap[y + 1, x] - 2 * heatmap[y, x] + heatmap[y - 1, x]

    # Offset from peak
    if abs(dxx) > 1e-6:
        ox = -dx / dxx
        ox = np.clip(ox, -0.5, 0.5)
    else:
        ox = 0.0

    if abs(dyy) > 1e-6:
        oy = -dy / dyy
        oy = np.clip(oy, -0.5, 0.5)
    else:
        oy = 0.0

    return float(x) + ox, float(y) + oy


# ── Detector class ──────────────────────────────────────────────────────────

class FieldKeypointDetector:
    """Detect field keypoints using trained HRNet model.

    Usage:
        detector = FieldKeypointDetector("models/hrnet_best.pth")
        result = detector.detect(frame)
        # result.pixel_xy, result.field_xy, result.confidences
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.3,
    ):
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh

        # Load model
        self.model = HRNetKeypointModel(num_keypoints=NUM_KEYPOINTS)
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> KeypointDetection:
        """Detect field keypoints in a single BGR frame.

        Args:
            frame: (H, W, 3) BGR image

        Returns:
            KeypointDetection with detected keypoints above confidence threshold
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Forward pass
        heatmaps = self.model(tensor)  # (1, 106, H_hm, W_hm)
        heatmaps = heatmaps[0].cpu().numpy()  # (106, H_hm, W_hm)

        _, hm_h, hm_w = heatmaps.shape

        # Extract peaks and filter by confidence
        pixel_xy_list = []
        keypoint_ids_list = []
        confidences_list = []
        field_xy_list = []
        all_confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        for ki in range(NUM_KEYPOINTS):
            hm = heatmaps[ki]
            peak_val = hm.max()
            all_confidences[ki] = peak_val

            if peak_val < self.conf_thresh:
                continue

            # Find peak position
            peak_idx = hm.argmax()
            peak_y = peak_idx // hm_w
            peak_x = peak_idx % hm_w

            # Sub-pixel refinement
            ref_x, ref_y = _refine_peak(hm, peak_y, peak_x)

            # Map back to original frame coordinates
            px = ref_x / hm_w * orig_w
            py = ref_y / hm_h * orig_h

            pixel_xy_list.append([px, py])
            keypoint_ids_list.append(ki)
            confidences_list.append(peak_val)
            field_xy_list.append(FIELD_COORDS[ki])

        if len(pixel_xy_list) == 0:
            return KeypointDetection(
                pixel_xy=np.array([]).reshape(0, 2),
                keypoint_ids=np.array([], dtype=np.int32),
                confidences=np.array([], dtype=np.float32),
                field_xy=np.array([]).reshape(0, 2),
                all_confidences=all_confidences,
            )

        return KeypointDetection(
            pixel_xy=np.array(pixel_xy_list, dtype=np.float64),
            keypoint_ids=np.array(keypoint_ids_list, dtype=np.int32),
            confidences=np.array(confidences_list, dtype=np.float32),
            field_xy=np.array(field_xy_list, dtype=np.float64),
            all_confidences=all_confidences,
        )
