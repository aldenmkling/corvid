"""Color signature — chromatic feature for jersey identification.

Lives in player_tracking/ because it's used both during tracking (per-frame
association cost) and after (team classification). The same 24-dim feature
vector serves both: hue + saturation/value histogram on the chromatic
(non-grass, non-white, non-shadow) pixels of a detection box.

Public API:
- compute_color_signature(frame_bgr, xyxy) → (24,) array or None
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_color_signature(frame_bgr: np.ndarray,
                              xyxy: np.ndarray,
                              n_hue_bins: int = 12,
                              n_s_bins: int = 4,
                              n_v_bins: int = 3,
                              ) -> np.ndarray | None:
    """24-dim chromatic signature for a detection box.

    Layout: 12-bin hue histogram (sums to 1) concatenated with a flattened
    4×3 (S,V) joint histogram (sums to 1). Pipeline: clamp box to frame →
    crop full box → mask out grass / white / dark / glare via HSV
    thresholds → if ≥20 chromatic pixels remain, build the two
    histograms and concatenate.

    Used by both team_classifier (post-tracking team labels) and
    tracker (per-track color feature in association cost). Returns
    None if the crop has too few chromatic pixels — caller should
    skip the d_color term in that case.
    """
    h, w = frame_bgr.shape[:2]
    region = _box_crop(xyxy, h, w)
    if region is None:
        return None
    y1, y2, x1, x2 = region
    crop = frame_bgr[y1:y2, x1:x2]
    chrom = _chromatic_pixels(crop)
    if chrom is None or len(chrom) < 20:
        return None
    h_hist, _ = np.histogram(chrom[:, 0], bins=n_hue_bins, range=(0, 180))
    h_hist = h_hist.astype(np.float32)
    h_sum = float(h_hist.sum())
    h_hist = h_hist / max(h_sum, 1.0)
    sv_hist, _, _ = np.histogram2d(
        chrom[:, 1], chrom[:, 2],
        bins=[n_s_bins, n_v_bins], range=[[0, 256], [50, 220]])
    sv_hist = sv_hist.astype(np.float32).flatten()
    sv_sum = float(sv_hist.sum())
    sv_hist = sv_hist / max(sv_sum, 1.0)
    return np.concatenate([h_hist, sv_hist], axis=0)


def _box_crop(xyxy: np.ndarray, frame_h: int, frame_w: int) -> tuple[int, int, int, int] | None:
    """Clamp the detection xyxy to frame bounds. Returns (y1, y2, x1, x2)."""
    x1 = max(0, int(round(float(xyxy[0]))))
    y1 = max(0, int(round(float(xyxy[1]))))
    x2 = min(frame_w, int(round(float(xyxy[2]))))
    y2 = min(frame_h, int(round(float(xyxy[3]))))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return y1, y2, x1, x2


def _chromatic_pixels(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Extract chromatic pixels (drop grass / white / shadow / black).

    OpenCV HSV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255].

    Drop:
      - Grass: H ∈ [35, 85], S > 60     (greens)
      - White: S < 40,  V > 150           (jersey numbers, helmet stripes,
                                            white pants)
      - Dark / shadow: V < 50              (helmet shadows, shoes)
      - Washed / glare: S < 60, V > 220   (very bright, low-saturation)

    Keep: chromatic body color (S > 60, 50 < V < 220, not green).
    """
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    grass = (H >= 35) & (H <= 85) & (S > 60)
    white = (S < 40) & (V > 150)
    dark = V < 50
    glare = (S < 60) & (V > 220)
    keep = ~(grass | white | dark | glare) & (S > 60) & (V >= 50) & (V <= 220)
    return hsv[keep]   # (N, 3) array of chromatic HSV pixels


def _hue_histogram(chromatic_hsv: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """36-bin hue histogram, normalized to sum to 1."""
    if len(chromatic_hsv) < 5:    # too few pixels — return zeros
        return np.zeros(n_bins, dtype=np.float32)
    h = chromatic_hsv[:, 0]
    hist, _ = np.histogram(h, bins=n_bins, range=(0, 180))
    s = hist.sum()
    if s == 0:
        return np.zeros(n_bins, dtype=np.float32)
    return (hist / s).astype(np.float32)


def _jersey_region(xyxy: np.ndarray, frame_h: int, frame_w: int,
                     y_top_frac: float = 0.15, y_bot_frac: float = 0.50,
                     x_inset_frac: float = 0.20) -> tuple[int, int, int, int] | None:
    """Crop the jersey region from an actual detection box.

    Standing player box layout (rough):
      y ∈ [0,   0.15]: helmet   ← skip
      y ∈ [0.15, 0.50]: jersey body / pads + arms  ← KEEP
      y ∈ [0.50, 1.00]: pants / legs / shoes     ← skip

    Also crops 20% off each side horizontally to skip arms swung out.

    Returns (y1, y2, x1, x2) clipped to frame bounds, or None if degenerate.
    """
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 8:
        return None
    jx1 = int(round(x1 + x_inset_frac * bw))
    jx2 = int(round(x2 - x_inset_frac * bw))
    jy1 = int(round(y1 + y_top_frac * bh))
    jy2 = int(round(y1 + y_bot_frac * bh))
    jx1 = max(0, jx1); jx2 = min(frame_w, jx2)
    jy1 = max(0, jy1); jy2 = min(frame_h, jy2)
    if jx2 - jx1 < 3 or jy2 - jy1 < 3:
        return None
    return jy1, jy2, jx1, jx2
