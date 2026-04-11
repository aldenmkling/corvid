"""
ML-based homography computation using HRNet field keypoint detector.

Replaces the classical pipeline (yard_lines.py + hash_marks.py + field_tracker.py)
with a single ML model that directly predicts absolute field landmarks.

Per-frame:
  1. FieldKeypointDetector predicts pixel positions + absolute identities
  2. Keypoint identities give us known field coordinates from the schema
  3. Feed pixel ↔ field correspondences to RANSAC homography solver

No temporal tracking needed for the basic pipeline. The FieldTracker can
optionally be used on top for identity propagation through zoomed-in frames.
"""

import numpy as np
import cv2
from typing import Callable

from .keypoint_detector import FieldKeypointDetector, KeypointDetection
from .compute_homography import compute_homography, HomographyResult


def create_ml_homography_fn(
    weights_path: str,
    device: str = "cuda",
    conf_thresh: float = 0.3,
    min_keypoints: int = 4,
) -> Callable[[np.ndarray], HomographyResult | None]:
    """Create a per-frame homography function using the HRNet keypoint detector.

    Returns a callable: frame -> HomographyResult | None

    This replaces the entire classical pipeline:
      - No yard_lines.py (Hough + clustering)
      - No hash_marks.py (edge filtering + dot pairs)
      - No field_tracker.py (optical flow tracking)
      - No yard line identification step

    The ML model directly outputs:
      1. Pixel positions of field landmarks
      2. Absolute identity of each landmark (which yard line, which intersection)
      3. Confidence per landmark

    We feed pixel↔field correspondences into compute_homography()
    which runs cv2.findHomography with RANSAC.

    Args:
        weights_path: path to trained HRNet checkpoint (.pth)
        device: "cuda" or "cpu"
        conf_thresh: minimum heatmap peak confidence to include a keypoint
        min_keypoints: minimum number of keypoints needed (4 = absolute minimum
                       for homography, 6+ recommended for RANSAC robustness)

    Returns:
        callable that takes a BGR frame and returns HomographyResult or None
    """
    detector = FieldKeypointDetector(weights_path, device, conf_thresh)

    def homography_fn(frame: np.ndarray) -> HomographyResult | None:
        result = detector.detect(frame)

        if len(result.pixel_xy) < min_keypoints:
            return None

        return compute_homography(
            pixel_pts=result.pixel_xy,
            field_pts=result.field_xy,
            ransac_threshold=5.0,
        )

    return homography_fn
