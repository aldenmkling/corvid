"""
Player detection module — wraps RF-DETR or YOLO for consistent interface.

Each detector returns a standardized Detections object per frame.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Detections:
    """Standardized detection output.

    All arrays have length N (number of detections in this frame).
    """
    xyxy: np.ndarray        # (N, 4) bounding boxes in x1, y1, x2, y2 format
    confidence: np.ndarray  # (N,) detection confidence scores
    class_id: np.ndarray    # (N,) class IDs (0 = player for our single-class model)

    def __len__(self):
        return len(self.xyxy)

    @property
    def foot_points(self) -> np.ndarray:
        """Ground-level position estimate for each player.

        Uses 95% of the way down the bounding box (horizontally centered).
        Not the true bottom — that can clip at the feet edge and be noisy.
        Not the centroid — that maps to the player's torso, which is above
        field level and causes homography errors due to parallax.

        Returns (N, 2) array of (x, y) pixel coordinates.
        """
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2                     # center x
        y_95 = self.xyxy[:, 1] + 0.95 * (self.xyxy[:, 3] - self.xyxy[:, 1])  # 95% down
        return np.column_stack([cx, y_95])


class YOLODetector:
    """Player detector using ultralytics YOLO."""

    def __init__(self, weights: str, device: str = "cpu", conf_thresh: float = 0.3):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> Detections:
        """Run detection on a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Detections object with all players found.
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf_thresh,
            imgsz=1280,
            verbose=False,
        )[0]

        boxes = results.boxes
        if len(boxes) == 0:
            return Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=np.int32),
            )

        return Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            confidence=boxes.conf.cpu().numpy().astype(np.float32),
            class_id=boxes.cls.cpu().numpy().astype(np.int32),
        )


class RFDETRDetector:
    """Player detector using RF-DETR."""

    def __init__(self, weights: str, device: str = "cpu", conf_thresh: float = 0.3,
                 resolution: int = 1280):
        from rfdetr import RFDETRLarge
        self.model = RFDETRLarge(pretrain_weights=weights, resolution=resolution)
        self.device = device
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> Detections:
        """Run detection on a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Detections object with all players found.
        """
        results = self.model.predict(frame, threshold=self.conf_thresh)

        if len(results.xyxy) == 0:
            return Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=np.int32),
            )

        return Detections(
            xyxy=np.array(results.xyxy, dtype=np.float32),
            confidence=np.array(results.confidence, dtype=np.float32),
            class_id=np.array(results.class_id, dtype=np.int32),
        )


def create_detector(weights: str, device: str = "cpu", conf_thresh: float = 0.3,
                    resolution: int = 1280):
    """Factory: auto-detect model type from weights file and return the right detector."""
    if weights.endswith(".pt"):
        # Could be YOLO or RF-DETR — try YOLO first (most .pt files are YOLO)
        # RF-DETR weights are typically .pth
        return YOLODetector(weights, device=device, conf_thresh=conf_thresh)
    elif weights.endswith(".pth"):
        return RFDETRDetector(weights, device=device, conf_thresh=conf_thresh,
                              resolution=resolution)
    else:
        raise ValueError(f"Unknown model format: {weights}")
