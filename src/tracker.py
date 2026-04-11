"""
Multi-object tracking module — wraps BoT-SORT for player tracking.

Takes per-frame Detections and produces tracked player trajectories
with stable IDs maintained across frames within a single play.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .detector import Detections


@dataclass
class TrackedPlayer:
    """Single player detection with tracking info."""
    track_id: int
    xyxy: np.ndarray          # (4,) bounding box
    confidence: float         # detection confidence
    foot_point: np.ndarray    # (2,) bottom-center pixel coordinate


@dataclass
class TrackingResult:
    """All tracked players in a single frame."""
    frame_idx: int
    players: list[TrackedPlayer]

    def __len__(self):
        return len(self.players)

    @property
    def track_ids(self) -> list[int]:
        return [p.track_id for p in self.players]

    @property
    def foot_points(self) -> np.ndarray:
        """(N, 2) array of foot points for all tracked players."""
        if not self.players:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([p.foot_point for p in self.players], dtype=np.float32)

    @property
    def confidences(self) -> np.ndarray:
        if not self.players:
            return np.empty(0, dtype=np.float32)
        return np.array([p.confidence for p in self.players], dtype=np.float32)


@dataclass
class TrajectoryPoint:
    """Single point in a player's trajectory."""
    frame_idx: int
    pixel_xy: np.ndarray      # (2,) foot point in pixel space
    field_xy: np.ndarray | None = None  # (2,) position in field coords (after homography)
    confidence: float = 0.0
    interrupted: bool = False  # True if tracker lost confidence here


@dataclass
class PlayerTrajectory:
    """Full trajectory of one tracked player across a play."""
    track_id: int
    points: list[TrajectoryPoint] = field(default_factory=list)

    @property
    def frame_indices(self) -> list[int]:
        return [p.frame_idx for p in self.points]

    @property
    def field_coords(self) -> np.ndarray:
        """(T, 2) array of field coordinates (NaN where interrupted or unmapped)."""
        coords = []
        for p in self.points:
            if p.field_xy is not None and not p.interrupted:
                coords.append(p.field_xy)
            else:
                coords.append([np.nan, np.nan])
        return np.array(coords, dtype=np.float64)


class PlayerTracker:
    """BoT-SORT based multi-object tracker for NFL players.

    Wraps boxmot's BotSort with settings tuned for All-22 football:
    - 30fps video with ~22 tracked objects
    - Camera panning (ECC motion compensation)
    - Appearance-based re-ID for crossing routes
    - Configurable confidence gating
    """

    def __init__(
        self,
        device: str = "cpu",
        reid_weights: str = "osnet_x0_25_msmt17.pt",
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,         # frames to keep lost tracks (~1s at 30fps)
        match_thresh: float = 0.8,
        appearance_thresh: float = 0.25,
        with_reid: bool = True,
        cmc_method: str = "ecc",        # camera motion compensation
        frame_rate: int = 30,
        confidence_gate: float = 0.3,   # below this, mark trajectory as interrupted
    ):
        from boxmot import BotSort

        self.tracker = BotSort(
            reid_weights=Path(reid_weights),
            device=device,
            half=False,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            appearance_thresh=appearance_thresh,
            with_reid=with_reid,
            cmc_method=cmc_method,
            frame_rate=frame_rate,
        )
        self.confidence_gate = confidence_gate
        self.frame_idx = 0
        self.trajectories: dict[int, PlayerTrajectory] = {}

    def reset(self):
        """Reset tracker state between plays."""
        # Re-initialize by creating a fresh tracker with same params
        # boxmot doesn't have a clean reset, so we store init params
        self.frame_idx = 0
        self.trajectories = {}
        # The tracker itself needs to be re-instantiated for a clean slate
        # This is handled by creating a new PlayerTracker per play

    def update(self, detections: Detections, frame: np.ndarray) -> TrackingResult:
        """Process one frame of detections through the tracker.

        Args:
            detections: Player detections for this frame.
            frame: BGR image (needed for ReID feature extraction).

        Returns:
            TrackingResult with tracked players and stable IDs.
        """
        if len(detections) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.hstack([
                detections.xyxy,
                detections.confidence[:, None],
                detections.class_id[:, None].astype(np.float32),
            ])  # (N, 6)

        # BoT-SORT update: returns (M, 8) = x1,y1,x2,y2, track_id, conf, cls, det_idx
        tracks = self.tracker.update(dets, frame)

        players = []
        for t in tracks:
            x1, y1, x2, y2 = t[0:4]
            track_id = int(t[4])
            conf = float(t[5])

            xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
            # 95% down the box, horizontally centered — matches detector.foot_points
            foot_point = np.array([(x1 + x2) / 2, y1 + 0.95 * (y2 - y1)], dtype=np.float32)

            player = TrackedPlayer(
                track_id=track_id,
                xyxy=xyxy,
                confidence=conf,
                foot_point=foot_point,
            )
            players.append(player)

            # Update trajectory
            interrupted = conf < self.confidence_gate
            point = TrajectoryPoint(
                frame_idx=self.frame_idx,
                pixel_xy=foot_point.copy(),
                confidence=conf,
                interrupted=interrupted,
            )

            if track_id not in self.trajectories:
                self.trajectories[track_id] = PlayerTrajectory(track_id=track_id)
            self.trajectories[track_id].points.append(point)

        result = TrackingResult(frame_idx=self.frame_idx, players=players)
        self.frame_idx += 1
        return result

    def get_trajectories(self) -> dict[int, PlayerTrajectory]:
        """Return all accumulated trajectories after processing a play."""
        return self.trajectories
