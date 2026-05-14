"""
Multi-object tracking module — multi-cue field-coord tracker.

Each player is tracked in NGS field coordinates (yards) by a constant-velocity
Kalman filter. Detections are projected from distorted pixel space into the
field via undistortPoints + per-frame homography H. Association combines two
cues:

  1. Mahalanobis distance in field coords (gated by χ²-2DOF 99% = 9.21).
  2. 1 - expansion-IoU between the track's last image-space box and the
     detection's xyxy (Deep HM-SORT style; expansion_factor enlarges both
     boxes outward from their centers before computing IoU).

Both cues are normalized to [0, 1] and combined via harmonic mean. Hungarian
assignment runs twice — a strict first round (expansion=0.3, threshold=0.7)
and a relaxed second round on what's left (expansion=0.6, threshold=0.85).

Tracks live in three states:
  - 'live':       matched in the most recent frame
  - 'lost':       not matched recently, still being predicted, eligible for
                  primary re-match within max_lost_frames
  - 'graveyard':  aged out of lost; eligible only for graveyard
                  re-association via direct extrapolation, gated by max
                  player speed × elapsed time. Resurrected tracks keep
                  their original ID.

Layer 1 (Kalman) + Layer 2 (multi-cue cost) + Layer 3 (iterated assoc) +
Layer 5 (confidence gate). Layer 4 (team color) is in src/team_classifier.py.
"""

import numpy as np
from dataclasses import dataclass, field

import cv2
from scipy.optimize import linear_sum_assignment

from src.player_detection.detector import Detections


# ── Public dataclasses (callers depend on these) ──────────────────────────


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
    pixel_xy: np.ndarray      # (2,) foot point in pixel space (NaN if predicted)
    field_xy: np.ndarray | None = None  # (2,) position in field coords (after homography)
    confidence: float = 0.0
    interrupted: bool = False  # True if tracker lost confidence here
    xyxy: np.ndarray | None = None  # (4,) image-space detection box (None if predicted)


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


# ── Helpers ───────────────────────────────────────────────────────────────


def _project_to_field(pts_pixel: np.ndarray, H: np.ndarray,
                      K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """Project pixel foot points (in DISTORTED image space) into NGS field
    coords using the rectify pipeline's H (which expects undistorted-image
    input).

    Steps:
      1. Undistort foot points via cv2.undistortPoints (if K is given and k1
         is non-trivial; otherwise skip).
      2. Apply H homogeneously: (x_field, y_field) = H · (x_u, y_u, 1).

    Returns (N, 2) field coords (yards).
    """
    if pts_pixel.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if K is not None and dist is not None and abs(float(dist[0])) > 1e-6:
        pts = pts_pixel.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    else:
        und = pts_pixel.astype(np.float64)
    homo = np.column_stack([und, np.ones(len(und))])
    proj = (H @ homo.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return proj[:, :2] / w


def _expansion_iou(box_a: np.ndarray, box_b: np.ndarray,
                   expansion_factor: float) -> float:
    """IoU between two xyxy boxes after both are dilated outward from their
    centers by (1 + expansion_factor). Returns 0.0 if boxes don't overlap.

    Used as a permissive image-space cue for partial occlusions.
    """
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    s = 1.0 + float(expansion_factor)
    ax1e, ax2e = acx - aw * 0.5 * s, acx + aw * 0.5 * s
    ay1e, ay2e = acy - ah * 0.5 * s, acy + ah * 0.5 * s
    bx1e, bx2e = bcx - bw * 0.5 * s, bcx + bw * 0.5 * s
    by1e, by2e = bcy - bh * 0.5 * s, bcy + bh * 0.5 * s
    ix1 = max(ax1e, bx1e)
    iy1 = max(ay1e, by1e)
    ix2 = min(ax2e, bx2e)
    iy2 = min(ay2e, by2e)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = (ax2e - ax1e) * (ay2e - ay1e)
    b_area = (bx2e - bx1e) * (by2e - by1e)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


# ── Track ─────────────────────────────────────────────────────────────────


class _KalmanTrack:
    """Constant-velocity Kalman track in field coords (yd, yd/s).

    State: x = [px, py, vx, vy] (yards, yards, yd/s, yd/s)
    Measurement: z = [px, py]   (yards)

    Adds image-space box memory + a 'state' field ('live'|'lost'|'graveyard')
    so the host PlayerTracker can drive the multi-cue association + grave-
    yard re-association logic.
    """

    __slots__ = (
        "track_id", "x", "P", "F", "Q", "H_obs", "R",
        "frames_since_update", "n_observations",
        "born_frame", "last_meas_frame",
        "last_image_box", "last_image_height_px",
        "state",
        "color_sig",   # 24-dim chromatic signature, EMA-updated
    )

    def __init__(self, track_id: int, init_xy: np.ndarray,
                 dt: float, process_noise: float, measurement_noise: float,
                 born_frame: int):
        self.track_id = track_id
        self.x = np.array([init_xy[0], init_xy[1], 0.0, 0.0], dtype=np.float64)
        # Initial cov: trust position, distrust velocity.
        self.P = np.diag([measurement_noise**2,
                          measurement_noise**2,
                          25.0, 25.0]).astype(np.float64)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        G = np.array([[dt * dt / 2.0, 0],
                      [0, dt * dt / 2.0],
                      [dt, 0],
                      [0, dt]], dtype=np.float64)
        self.Q = (G @ G.T) * (process_noise ** 2)
        self.H_obs = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * (measurement_noise ** 2)
        self.frames_since_update = 0
        self.n_observations = 1
        self.born_frame = born_frame
        self.last_meas_frame = born_frame
        self.last_image_box: np.ndarray | None = None
        self.last_image_height_px: float = 0.0
        self.state: str = "live"
        self.color_sig: np.ndarray | None = None   # set on first measurement

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.frames_since_update += 1

    def innovation_cov(self) -> np.ndarray:
        return self.H_obs @ self.P @ self.H_obs.T + self.R

    def update_meas(self, z: np.ndarray, frame_idx: int):
        """Standard Kalman update with measurement z = [px, py]."""
        S = self.innovation_cov()
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        y = z - self.H_obs @ self.x
        K = self.P @ self.H_obs.T @ S_inv
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H_obs) @ self.P
        self.frames_since_update = 0
        self.n_observations += 1
        self.last_meas_frame = frame_idx

    def predict_n(self, n_steps: int) -> np.ndarray:
        """Return predicted (x, y) after n_steps additional predicts, without
        mutating self. Used for graveyard re-association reachability."""
        if n_steps <= 0:
            return self.x[:2].copy()
        x = self.x.copy()
        for _ in range(int(n_steps)):
            x = self.F @ x
        return x[:2].copy()

    @property
    def field_xy(self) -> np.ndarray:
        return self.x[:2].copy()


# ── Tracker ───────────────────────────────────────────────────────────────


class PlayerTracker:
    """Multi-cue field-coord Kalman tracker.

    Args:
        device: kept for API compatibility (unused — Kalman is CPU-only).
        process_noise_yd: stddev of physical acceleration noise, yd/s²
            (default 8.0 — covers a sprinter's ~6 yd/s² peak with margin).
        measurement_noise_yd: stddev of detection noise in field coords
            (default 0.5 yd).
        max_lost_frames: number of unmatched frames before a 'live' track
            is moved to 'lost', and the same again before 'lost' becomes
            'graveyard' (default 8 → ~0.27s @ 30fps).
        max_graveyard_frames: drop graveyard tracks older than this many
            frames since their last measurement (default 90 = 3s @ 30fps).
        max_player_speed_yd_s: ceiling used to gate graveyard re-association
            (default 10 yd/s ≈ 9.1 m/s, well above an NFL sprint).
        chi_square_gate: χ² threshold (2-DOF) for the Mahalanobis gate
            (default 9.21 = 99th percentile).
        match_thresh: 1st-round Hungarian acceptance threshold on the
            harmonic-mean cost (default 0.7).
        match_thresh_relaxed: 2nd-round Hungarian threshold (default 0.85).
        expansion_factor: 1st-round IoU dilation (default 0.3).
        expansion_factor_relaxed: 2nd-round IoU dilation (default 0.6).
        graveyard_match_thresh_yd: yards cap for graveyard re-association
            cost (default 8.0).
        new_track_min_dist_yd: orphan detections within this many yards of
            an existing live/lost track are suppressed (don't seed a new
            track) — this is the duplicate-detection guard for pile-ups
            where RF-DETR sometimes returns >1 box per player. Default 1.5.
        confidence_gate: detections below this confidence still update the
            track but the trajectory point is marked interrupted=True
            (default 0.3).
        frame_rate: source video fps; sets dt for the Kalman F matrix.

    Backward-compat kwargs (silently absorbed):
        gating_distance_yd, max_age_frames, reid_weights, track_buffer,
        track_high_thresh, track_low_thresh, new_track_thresh,
        appearance_thresh, with_reid, cmc_method, **kwargs.
    """

    def __init__(
        self,
        device: str = "mps",
        process_noise_yd: float = 8.0,
        measurement_noise_yd: float = 0.5,
        max_lost_frames: int = 8,
        max_graveyard_frames: int = 90,
        max_player_speed_yd_s: float = 10.0,
        chi_square_gate: float = 9.21,
        match_thresh: float = 1.0,
        match_thresh_relaxed: float = 1.5,
        expansion_factor: float = 0.3,
        expansion_factor_relaxed: float = 0.6,
        graveyard_match_thresh_yd: float = 8.0,
        new_track_min_dist_yd: float = 1.5,
        confidence_gate: float = 0.3,
        color_ema_alpha: float = 0.999,
        color_min_obs_for_use: int = 3,
        color_skip_update_overlap_iou: float = 0.15,
        frame_rate: int = 30,
        # Legacy kwargs accepted but unused (kept for backward compat):
        gating_distance_yd=None,
        max_age_frames=None,
        reid_weights=None,
        track_buffer=None,
        track_high_thresh=None,
        track_low_thresh=None,
        new_track_thresh=None,
        appearance_thresh=None,
        with_reid=None,
        cmc_method=None,
        **kwargs,
    ):
        self.device = device
        self.process_noise = float(process_noise_yd)
        self.measurement_noise = float(measurement_noise_yd)
        self.max_lost_frames = int(max_lost_frames)
        self.max_graveyard_frames = int(max_graveyard_frames)
        self.max_player_speed = float(max_player_speed_yd_s)
        self.chi_gate = float(chi_square_gate)
        self.match_thresh = float(match_thresh)
        self.match_thresh_relaxed = float(match_thresh_relaxed)
        self.expansion_factor = float(expansion_factor)
        self.expansion_factor_relaxed = float(expansion_factor_relaxed)
        self.graveyard_match_thresh = float(graveyard_match_thresh_yd)
        self.new_track_min_dist = float(new_track_min_dist_yd)
        self.confidence_gate = float(confidence_gate)
        self.color_ema_alpha = float(color_ema_alpha)
        self.color_min_obs_for_use = int(color_min_obs_for_use)
        self.color_skip_update_overlap_iou = float(color_skip_update_overlap_iou)
        self.frame_rate = int(frame_rate)
        self.dt = 1.0 / float(frame_rate)

        # Active pool: 'live' + 'lost' tracks. Graveyard is a separate list.
        self._tracks: list[_KalmanTrack] = []
        self._graveyard: list[_KalmanTrack] = []
        self._next_id: int = 1
        self.frame_idx: int = 0
        self.trajectories: dict[int, PlayerTrajectory] = {}

    # ── Public API ─────────────────────────────────────────────────────
    def update(self, detections: Detections, frame: np.ndarray,
               H: np.ndarray | None = None,
               K: np.ndarray | None = None,
               dist: np.ndarray | None = None) -> TrackingResult:
        """Process one frame: predict, associate (multi-round), update.

        Args:
            detections: per-frame Detections from the detector.
            frame: BGR image (kept for API parity; unused at this layer).
            H: (3,3) homography mapping undistorted image -> NGS field. If
                None, no measurement update happens — all current tracks
                predict forward and are recorded as interrupted.
            K: (3,3) camera intrinsic for undistortPoints (or None).
            dist: (5,) distortion coefficients matching K (or None).

        Returns:
            TrackingResult containing the players that received a measurement
            update this frame. Predicted-only tracks are NOT included in the
            result, but ARE appended to their trajectory with interrupted=True.
        """
        # 1) Predict every active track forward by one step.
        for tr in self._tracks:
            tr.predict()
        for tr in self._graveyard:
            tr.predict()

        # 2) Skip association if H or detections missing.
        if H is None or len(detections) == 0:
            self._record_predicted_only()
            self._age_off_unmatched()
            self.frame_idx += 1
            return TrackingResult(frame_idx=self.frame_idx - 1, players=[])

        # 3) Compute foot points and project to field coords.
        foot_px = detections.foot_points.astype(np.float64)
        det_field = _project_to_field(foot_px, H, K, dist)

        good = np.isfinite(det_field).all(axis=1)
        if not good.all():
            foot_px = foot_px[good]
            det_field = det_field[good]
            kept_idx = np.where(good)[0]
        else:
            kept_idx = np.arange(len(foot_px))

        # Detection xyxys (in distorted image space) — kept aligned with
        # foot_px / det_field via kept_idx.
        det_xyxy = detections.xyxy[kept_idx].astype(np.float64)
        det_conf = detections.confidence[kept_idx].astype(np.float64)

        # Per-detection chromatic signatures (24-dim). None if too few
        # chromatic pixels to be reliable. Used as the third arm of the
        # 3-way harmonic-mean association cost in _associate.
        from .color_signature import compute_color_signature
        det_color_sigs: list[np.ndarray | None] = [
            compute_color_signature(frame, det_xyxy[i])
            for i in range(len(det_xyxy))
        ]

        n_dets = len(det_field)
        n_tracks = len(self._tracks)

        matched_pairs: list[tuple[int, int]] = []   # (track_idx, det_idx)
        unmatched_tracks = list(range(n_tracks))
        unmatched_dets = list(range(n_dets))

        # 4) First-round association (strict).
        if n_tracks > 0 and n_dets > 0:
            mp1, ut1, ud1 = self._associate(
                self._tracks, det_field, det_xyxy, det_color_sigs,
                track_idxs=unmatched_tracks,
                det_idxs=unmatched_dets,
                expansion=self.expansion_factor,
                threshold=self.match_thresh,
            )
            matched_pairs.extend(mp1)
            unmatched_tracks = ut1
            unmatched_dets = ud1

        # 5) Second-round association (relaxed).
        if unmatched_tracks and unmatched_dets:
            mp2, ut2, ud2 = self._associate(
                self._tracks, det_field, det_xyxy, det_color_sigs,
                track_idxs=unmatched_tracks,
                det_idxs=unmatched_dets,
                expansion=self.expansion_factor_relaxed,
                threshold=self.match_thresh_relaxed,
            )
            matched_pairs.extend(mp2)
            unmatched_tracks = ut2
            unmatched_dets = ud2

        # 6) Graveyard re-association on remaining unmatched detections.
        resurrected_pairs: list[tuple[_KalmanTrack, int]] = []
        if unmatched_dets and self._graveyard:
            resurrected_pairs, unmatched_dets = self._graveyard_associate(
                det_field, unmatched_dets,
            )

        # 7) Apply matches: update Kalman state + record trajectory point.
        result_players: list[TrackedPlayer] = []
        matched_track_set: set[int] = set()

        for ti, di in matched_pairs:
            tr = self._tracks[ti]
            tr.update_meas(det_field[di], frame_idx=self.frame_idx)
            tr.last_image_box = det_xyxy[di].copy()
            tr.last_image_height_px = float(det_xyxy[di, 3] - det_xyxy[di, 1])
            tr.state = "live"
            matched_track_set.add(ti)
            # Color update: skip if this detection's box overlaps another
            # active track's last box too much — chromatic pixels would
            # be contaminated by the other player's body.
            ambiguous = self._det_overlaps_other_tracks(
                det_xyxy[di], skip_track_idx=ti)
            if not ambiguous:
                self._update_color_sig(tr, det_color_sigs[di])
            conf = float(det_conf[di])
            foot = foot_px[di].astype(np.float32)
            xyxy = det_xyxy[di].astype(np.float32)
            tp = TrackedPlayer(track_id=tr.track_id, xyxy=xyxy,
                               confidence=conf, foot_point=foot)
            result_players.append(tp)
            interrupted = conf < self.confidence_gate
            self._append_traj(tr.track_id, frame_idx=self.frame_idx,
                              pixel_xy=foot.astype(np.float64),
                              field_xy=tr.field_xy, confidence=conf,
                              interrupted=interrupted, xyxy=xyxy)

        # 7b) Resurrect graveyard tracks: move from graveyard back to active,
        #     keep ID, update state.
        for tr, di in resurrected_pairs:
            self._graveyard.remove(tr)
            tr.update_meas(det_field[di], frame_idx=self.frame_idx)
            tr.last_image_box = det_xyxy[di].copy()
            tr.last_image_height_px = float(det_xyxy[di, 3] - det_xyxy[di, 1])
            tr.state = "lost"  # resurrected — give it a frame to confirm
            tr.frames_since_update = 0
            # Color: only update if not contaminated by overlap with
            # another active track. (skip_track_idx=-1 since this is a
            # resurrected track not yet in self._tracks.)
            if not self._det_overlaps_other_tracks(det_xyxy[di], skip_track_idx=-1):
                self._update_color_sig(tr, det_color_sigs[di])
            self._tracks.append(tr)
            conf = float(det_conf[di])
            foot = foot_px[di].astype(np.float32)
            xyxy = det_xyxy[di].astype(np.float32)
            tp = TrackedPlayer(track_id=tr.track_id, xyxy=xyxy,
                               confidence=conf, foot_point=foot)
            result_players.append(tp)
            interrupted = conf < self.confidence_gate
            self._append_traj(tr.track_id, frame_idx=self.frame_idx,
                              pixel_xy=foot.astype(np.float64),
                              field_xy=tr.field_xy, confidence=conf,
                              interrupted=interrupted, xyxy=xyxy)

        # 8) Create new tracks for still-unmatched detections.
        #    Duplicate-detection guard: skip orphan dets that fall within
        #    new_track_min_dist of any existing live/lost track's predicted
        #    position. RF-DETR occasionally returns 2 boxes for the same
        #    player in pile-ups; we don't want to seed a new ID for those.
        existing_xy = (np.array([tr.x[:2] for tr in self._tracks])
                       if self._tracks else np.empty((0, 2)))
        for di in unmatched_dets:
            if len(existing_xy) > 0:
                dists = np.linalg.norm(existing_xy - det_field[di], axis=1)
                if dists.min() < self.new_track_min_dist:
                    continue  # treat as duplicate, don't seed
            tr = _KalmanTrack(
                track_id=self._next_id,
                init_xy=det_field[di],
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                born_frame=self.frame_idx,
            )
            tr.last_image_box = det_xyxy[di].copy()
            tr.last_image_height_px = float(det_xyxy[di, 3] - det_xyxy[di, 1])
            if det_color_sigs[di] is not None:
                tr.color_sig = det_color_sigs[di].copy()
            self._tracks.append(tr)
            self._next_id += 1
            conf = float(det_conf[di])
            foot = foot_px[di].astype(np.float32)
            xyxy = det_xyxy[di].astype(np.float32)
            tp = TrackedPlayer(track_id=tr.track_id, xyxy=xyxy,
                               confidence=conf, foot_point=foot)
            result_players.append(tp)
            interrupted = conf < self.confidence_gate
            self._append_traj(tr.track_id, frame_idx=self.frame_idx,
                              pixel_xy=foot.astype(np.float64),
                              field_xy=tr.field_xy, confidence=conf,
                              interrupted=interrupted, xyxy=xyxy)

        # 9) Predicted-only tracks (the ones in _tracks that didn't match
        #    AND weren't just created/resurrected this frame): record an
        #    interrupted point.
        live_or_lost_unmatched = [ti for ti in range(n_tracks)
                                  if ti not in matched_track_set]
        for ti in live_or_lost_unmatched:
            tr = self._tracks[ti]
            self._append_traj(tr.track_id, frame_idx=self.frame_idx,
                              pixel_xy=np.array([np.nan, np.nan], dtype=np.float64),
                              field_xy=tr.field_xy, confidence=0.0,
                              interrupted=True)

        # 10) State transitions + age-off.
        self._age_off_unmatched()

        result = TrackingResult(frame_idx=self.frame_idx, players=result_players)
        self.frame_idx += 1
        return result

    def get_trajectories(self) -> dict[int, PlayerTrajectory]:
        """Return all accumulated trajectories after processing a play."""
        return self.trajectories

    # ── Internals ──────────────────────────────────────────────────────
    def _associate(
        self,
        tracks: list[_KalmanTrack],
        det_field: np.ndarray,
        det_xyxy: np.ndarray,
        det_color_sigs: list,
        track_idxs: list[int],
        det_idxs: list[int],
        expansion: float,
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """One round of multi-cue Hungarian association.

        Cost = harmonic_mean(d_field_norm, 1 - expansion_iou, d_color),
        falling back to 2-way harmonic mean (field + iou) when either
        the track has no color signature yet OR the detection has too
        few chromatic pixels to compute one.

        Color cost: d_color = 1 - cosine_similarity(track_sig, det_sig),
        in [0, 1] (clamped). Track signatures are EMA-updated after
        association; tracks need >= color_min_obs_for_use measurements
        before their signature is trusted enough to use in cost.

        Pairs are accepted if cost < threshold.

        Args:
            tracks: list of all tracks (we index into it via track_idxs).
            det_field: (M, 2) projected detection field coords for ALL dets.
            det_xyxy: (M, 4) image-space xyxy for ALL dets.
            det_color_sigs: list[ndarray | None], len M, signatures for
                each detection (None if insufficient chromatic pixels).
            track_idxs: subset of track indices to consider.
            det_idxs: subset of detection indices to consider.
            expansion: IoU dilation factor.
            threshold: cost threshold for accepting a match.

        Returns:
            (matched_pairs, unmatched_track_idxs, unmatched_det_idxs)
            where matched_pairs is [(track_idx, det_idx), ...] in the
            ORIGINAL index spaces.
        """
        nT = len(track_idxs)
        nD = len(det_idxs)
        if nT == 0 or nD == 0:
            return [], list(track_idxs), list(det_idxs)

        BIG = 1e6
        cost = np.full((nT, nD), BIG, dtype=np.float64)

        for i_local, ti in enumerate(track_idxs):
            tr = tracks[ti]
            S = tr.innovation_cov()
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            pred = tr.x[:2]
            for j_local, dj in enumerate(det_idxs):
                d = det_field[dj] - pred
                mahal_sq = float(d @ S_inv @ d)
                if mahal_sq > self.chi_gate:
                    continue  # out of gate, leave at BIG
                d_field_norm = mahal_sq / self.chi_gate  # in [0, 1]

                if tr.last_image_box is not None:
                    iou = _expansion_iou(tr.last_image_box, det_xyxy[dj],
                                         expansion_factor=expansion)
                else:
                    iou = 0.0
                d_iou = 1.0 - iou  # in [0, 1]

                # d_color: cosine distance between track running signature
                # and this detection's signature. Skip if either is missing
                # or track is too new to have a stable signature.
                d_color = None
                if (tr.color_sig is not None
                        and tr.n_observations >= self.color_min_obs_for_use
                        and det_color_sigs[dj] is not None):
                    a = tr.color_sig
                    b = det_color_sigs[dj]
                    na = float(np.linalg.norm(a))
                    nb = float(np.linalg.norm(b))
                    if na > 1e-9 and nb > 1e-9:
                        cos = float(np.dot(a, b) / (na * nb))
                        d_color = max(0.0, min(1.0, 1.0 - cos))

                # Weighted sum aggregator. Each signal contributes
                # proportionally — unlike harmonic mean which is
                # dominated by the smallest input. d_field (Mahalanobis
                # / chi-gate) is typically very small for well-tracked
                # objects (0.005-0.05); under HM that drowns out d_iou
                # and d_color. With weighted sum, IoU stability and
                # color identity get to weigh in. Weights:
                #   w_field=0.5  — Kalman position is a soft prior
                #   w_iou=1.0    — image-space continuity is strong
                #   w_color=1.0  — team identity is strong
                if d_color is not None:
                    cost_val = (0.5 * d_field_norm
                                + 1.0 * d_iou
                                + 1.0 * d_color)
                else:
                    cost_val = 0.5 * d_field_norm + 1.0 * d_iou
                cost[i_local, j_local] = cost_val

        row_idx, col_idx = linear_sum_assignment(cost)
        matched: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= threshold or cost[r, c] >= BIG:
                continue
            ti = track_idxs[r]
            dj = det_idxs[c]
            matched.append((ti, dj))
            used_t.add(r)
            used_d.add(c)

        unmatched_t = [track_idxs[i] for i in range(nT) if i not in used_t]
        unmatched_d = [det_idxs[j] for j in range(nD) if j not in used_d]
        return matched, unmatched_t, unmatched_d

    def _graveyard_associate(
        self,
        det_field: np.ndarray,
        unmatched_dets: list[int],
    ) -> tuple[list[tuple[_KalmanTrack, int]], list[int]]:
        """Try to reattach orphan detections to graveyard tracks.

        For each graveyard track, extrapolate where it could be NOW
        (predict_n by frames-since-last-measurement). Cost is Euclidean
        field distance + linear time penalty. Reachability is gated by
        max_player_speed × elapsed time.

        Greedy assignment: iterate detections in input order, pick best
        graveyard track for each (no two detections re-link to the same
        graveyard track).
        """
        if not unmatched_dets or not self._graveyard:
            return [], list(unmatched_dets)

        time_penalty = 5.0  # yards, scaled by frames_since_seen / max_graveyard
        used_grave: set[int] = set()
        resurrected: list[tuple[_KalmanTrack, int]] = []
        still_unmatched: list[int] = []

        for dj in unmatched_dets:
            best_idx = -1
            best_cost = float("inf")
            for gi, tr in enumerate(self._graveyard):
                if gi in used_grave:
                    continue
                frames_since = self.frame_idx - tr.last_meas_frame
                if frames_since <= 0:
                    continue
                if frames_since > self.max_graveyard_frames:
                    continue
                # Track has already been advanced via predict() this frame,
                # so its current x already represents prediction at this
                # frame. Compare directly.
                pred_xy = tr.x[:2]
                d = det_field[dj] - pred_xy
                dist_yd = float(np.sqrt(d @ d))
                # Reachability gate.
                elapsed_s = frames_since * self.dt
                max_reach = self.max_player_speed * elapsed_s
                if dist_yd > max_reach:
                    continue
                cost = dist_yd + time_penalty * (frames_since / max(1, self.max_graveyard_frames))
                if cost < best_cost:
                    best_cost = cost
                    best_idx = gi
            if best_idx >= 0 and best_cost < self.graveyard_match_thresh:
                used_grave.add(best_idx)
                resurrected.append((self._graveyard[best_idx], dj))
            else:
                still_unmatched.append(dj)

        return resurrected, still_unmatched

    def _det_overlaps_other_tracks(self, det_xyxy: np.ndarray,
                                       skip_track_idx: int) -> bool:
        """True if this detection's box overlaps another active track's
        last_image_box with IoU > color_skip_update_overlap_iou. Used
        to skip color signature updates on contaminated detections."""
        for ti, tr in enumerate(self._tracks):
            if ti == skip_track_idx:
                continue
            if tr.last_image_box is None:
                continue
            iou = _expansion_iou(det_xyxy, tr.last_image_box,
                                  expansion_factor=0.0)
            if iou > self.color_skip_update_overlap_iou:
                return True
        return False

    def _update_color_sig(self, track: _KalmanTrack,
                            new_sig: np.ndarray | None):
        """EMA-update a track's running color signature with a new
        per-frame measurement. No-op if new_sig is None (det had too few
        chromatic pixels). Initializes from the first observation."""
        if new_sig is None:
            return
        if track.color_sig is None:
            track.color_sig = new_sig.astype(np.float32).copy()
            return
        a = self.color_ema_alpha
        track.color_sig = (a * track.color_sig
                            + (1.0 - a) * new_sig.astype(np.float32))

    def _append_traj(self, track_id: int, frame_idx: int,
                     pixel_xy: np.ndarray, field_xy: np.ndarray,
                     confidence: float, interrupted: bool,
                     xyxy: np.ndarray | None = None):
        if track_id not in self.trajectories:
            self.trajectories[track_id] = PlayerTrajectory(track_id=track_id)
        self.trajectories[track_id].points.append(TrajectoryPoint(
            frame_idx=frame_idx,
            pixel_xy=pixel_xy.copy(),
            field_xy=field_xy.copy(),
            confidence=confidence,
            interrupted=interrupted,
            xyxy=(xyxy.copy() if xyxy is not None else None),
        ))

    def _record_predicted_only(self):
        """Append an interrupted point to every active track (no detections
        / no H this frame)."""
        for tr in self._tracks:
            self._append_traj(tr.track_id, frame_idx=self.frame_idx,
                              pixel_xy=np.array([np.nan, np.nan], dtype=np.float64),
                              field_xy=tr.field_xy, confidence=0.0,
                              interrupted=True)

    def _age_off_unmatched(self):
        """Move stale tracks through the live -> lost -> graveyard -> drop
        pipeline. Called once per frame after associations apply."""
        new_active: list[_KalmanTrack] = []
        for tr in self._tracks:
            if tr.frames_since_update == 0:
                # matched this frame
                new_active.append(tr)
                continue
            if tr.frames_since_update <= self.max_lost_frames:
                # still in primary pool, just demote 'live' -> 'lost'
                if tr.state == "live":
                    tr.state = "lost"
                new_active.append(tr)
            else:
                # too long unmatched → graveyard
                tr.state = "graveyard"
                self._graveyard.append(tr)
        self._tracks = new_active

        # Drop graveyard entries past max_graveyard_frames since last meas.
        self._graveyard = [
            tr for tr in self._graveyard
            if (self.frame_idx - tr.last_meas_frame) <= self.max_graveyard_frames
        ]
