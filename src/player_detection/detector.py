"""
Player detection module — wraps RF-DETR or YOLO for consistent interface.

Each detector returns a standardized Detections object per frame. Also
supports a per-clip on-disk cache so tracker iteration doesn't re-pay
the ~13 min/clip RF-DETR cost on Apple Silicon. See `cache_detections`
and `get_or_build_detection_cache` at the bottom of this file.
"""

import json
import os
import time

import cv2
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


# ── Per-clip detection cache ──────────────────────────────────────────────
#
# Stores all detections for a clip in a single .npz next to the clip's
# game/play identity. First run is slow (full detector pass); subsequent
# loads are essentially free, so tracker iteration on the same clip can
# go from ~13 min/run to ~2 sec/run.

DEFAULT_CACHE_DIR = "output/detection_cache"


def detection_cache_path(clip_path: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Cache file for a clip. Uses <game>_<play>_<filename>.npz so cache is
    addressed by (game, play) when clip lives in videos/clips/<game>/<play>/."""
    norm = os.path.abspath(clip_path)
    parts = norm.split(os.sep)
    if "clips" in parts:
        idx = parts.index("clips")
        tag_parts = parts[idx + 1:]
    else:
        tag_parts = [os.path.splitext(os.path.basename(clip_path))[0]]
    tag = "_".join(p.replace(".mp4", "") for p in tag_parts)
    return os.path.join(cache_dir, f"{tag}.npz")


def save_detection_cache(cache_path: str, dets_per_frame: list,
                          metadata: dict):
    """Save per-frame detections + metadata to a single .npz.

    Layout: contiguous arrays + per-frame offsets so we can slice
    detections[i] = arr[offsets[i]:offsets[i+1]] in O(1).
    """
    n_frames = len(dets_per_frame)
    if n_frames == 0:
        offsets = np.array([0], dtype=np.int64)
        xyxy = np.empty((0, 4), dtype=np.float32)
        conf = np.empty(0, dtype=np.float32)
        cls = np.empty(0, dtype=np.int32)
    else:
        offsets = np.empty(n_frames + 1, dtype=np.int64)
        offsets[0] = 0
        for i, d in enumerate(dets_per_frame):
            offsets[i + 1] = offsets[i] + len(d)
        xyxy = (np.concatenate([d.xyxy for d in dets_per_frame], axis=0)
                if offsets[-1] > 0 else np.empty((0, 4), dtype=np.float32))
        conf = (np.concatenate([d.confidence for d in dets_per_frame])
                if offsets[-1] > 0 else np.empty(0, dtype=np.float32))
        cls = (np.concatenate([d.class_id for d in dets_per_frame])
                if offsets[-1] > 0 else np.empty(0, dtype=np.int32))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path,
              xyxy=xyxy.astype(np.float32),
              conf=conf.astype(np.float32),
              cls=cls.astype(np.int32),
              offsets=offsets,
              meta=np.array(json.dumps(metadata)))


def load_detection_cache(cache_path: str) -> tuple[list, dict]:
    """Load per-frame detections + metadata from a cache file.

    Returns (list[Detections], metadata_dict). Each Detections has the
    same shape as if it had come from a live detector this frame.
    """
    data = np.load(cache_path, allow_pickle=False)
    offsets = data["offsets"]
    xyxy_all = data["xyxy"]
    conf_all = data["conf"]
    cls_all = data["cls"]
    meta = json.loads(str(data["meta"]))
    n_frames = len(offsets) - 1
    dets = []
    for i in range(n_frames):
        s, e = int(offsets[i]), int(offsets[i + 1])
        dets.append(Detections(
            xyxy=xyxy_all[s:e].copy(),
            confidence=conf_all[s:e].copy(),
            class_id=cls_all[s:e].copy(),
        ))
    return dets, meta


def cache_detections(clip_path: str,
                       weights: str,
                       device: str = "mps",
                       conf_thresh: float = 0.3,
                       resolution: int = 1280,
                       cache_dir: str = DEFAULT_CACHE_DIR,
                       verbose: bool = True) -> tuple[list, dict, str]:
    """Run the detector across every frame of a clip, save to cache.

    Returns (dets_per_frame, metadata, cache_path).
    """
    detector = create_detector(weights, device=device, conf_thresh=conf_thresh,
                                 resolution=resolution)
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"  caching detections for {os.path.basename(clip_path)} "
              f"({n_total} frames @ {fps:.1f} fps) ...")
    dets_per_frame = []
    t0 = time.time()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets_per_frame.append(detector.detect(frame))
        fi += 1
        if verbose and fi % 30 == 0:
            elapsed = time.time() - t0
            eta = (n_total - fi) * (elapsed / fi)
            print(f"    frame {fi}/{n_total}  "
                  f"{elapsed:.0f}s elapsed  ~{eta:.0f}s left")
    cap.release()
    elapsed = time.time() - t0
    metadata = {
        "clip_path": os.path.relpath(clip_path) if not os.path.isabs(clip_path)
                      else clip_path,
        "weights": os.path.basename(weights),
        "device": device,
        "conf_thresh": conf_thresh,
        "resolution": resolution,
        "n_frames": len(dets_per_frame),
        "fps": float(fps),
        "build_time_s": float(elapsed),
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_path = detection_cache_path(clip_path, cache_dir=cache_dir)
    save_detection_cache(cache_path, dets_per_frame, metadata)
    if verbose:
        n_det = sum(len(d) for d in dets_per_frame)
        print(f"  cached {len(dets_per_frame)} frames "
              f"({n_det} detections, {elapsed:.1f}s) -> {cache_path}")
    return dets_per_frame, metadata, cache_path


def get_or_build_detection_cache(clip_path: str,
                                    weights: str,
                                    device: str = "mps",
                                    conf_thresh: float = 0.3,
                                    resolution: int = 1280,
                                    cache_dir: str = DEFAULT_CACHE_DIR,
                                    force_rebuild: bool = False,
                                    verbose: bool = True) -> list:
    """Return per-frame detections for a clip. Build + cache on first call,
    load from cache on subsequent calls.

    If a cache exists but its metadata (weights / conf_thresh / resolution)
    differs from the requested params, rebuild it. Set `force_rebuild=True`
    to always rebuild.
    """
    cache_path = detection_cache_path(clip_path, cache_dir=cache_dir)
    if os.path.exists(cache_path) and not force_rebuild:
        try:
            dets, meta = load_detection_cache(cache_path)
        except Exception as e:
            if verbose:
                print(f"  cache at {cache_path} is corrupt ({e}); rebuilding")
        else:
            requested = {"weights": os.path.basename(weights),
                          "conf_thresh": float(conf_thresh),
                          "resolution": int(resolution)}
            cached = {"weights": meta.get("weights"),
                       "conf_thresh": float(meta.get("conf_thresh", 0)),
                       "resolution": int(meta.get("resolution", 0))}
            if cached == requested:
                if verbose:
                    n_det = sum(len(d) for d in dets)
                    print(f"  loaded {len(dets)} frames ({n_det} detections) "
                          f"from cache {os.path.basename(cache_path)}")
                return dets
            elif verbose:
                print(f"  cache at {cache_path} was built with different "
                      f"params (cached={cached}, requested={requested}); "
                      f"rebuilding")
    dets, _meta, _path = cache_detections(
        clip_path, weights=weights, device=device, conf_thresh=conf_thresh,
        resolution=resolution, cache_dir=cache_dir, verbose=verbose)
    return dets
