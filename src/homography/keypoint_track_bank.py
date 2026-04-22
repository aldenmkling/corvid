"""Field-space keypoint track bank.

Tracks yard-line keypoints (hash intersections, sideline intersections) across
frames by their stable FIELD position. Yard lines are stationary in world
coordinates — only the camera moves — so each keypoint's field (x_yd, y_yd) is
a fixed identity we can latch onto.

Used by HomographyTracker to:
  1. Validate a candidate H_cur by checking that tracked keypoints project to
     their known field positions within tolerance. Catches cases where the
     grid solver misidentifies a frame's yard lines (which produces a
     self-consistent-but-shifted H).
  2. Coast tracked keypoints through frames HRNet missed them in, so the
     correspondence set stays dense and H_cur stays smooth.
  3. Reject new detections that contradict a well-established track (optional
     stricter mode).

Policy notes:
  * Tracks are asymmetric: new detections are trusted immediately (create a
    new track if no match). A track is only used to REJECT a detection when
    two detections of the same kind land in the same field slot with
    contradictory pixel positions. Sparse sideline detections survive this.
  * Track identity = (kind, yard_slot_index). Two detections share a track iff
    they both snap to the same yard-line column + same hash row (or sideline).
  * Field position is exactly determined by identity (no EMA over field pos
    needed) — snap-to-grid is the single source of truth. What we track over
    time is: observation count, most recent pixel position, and age.

Usage:
    bank = KeypointTrackBank()
    for frame in clip:
        correspondences = ...  # list of (pixel_u, field, kind)
        bank.observe(correspondences, frame_idx=i)
        valid, info = bank.validate_h(H_candidate, correspondences)
        if not valid:
            # fall back to carry
            ...
        coasted = bank.coast_unobserved(H_prev, frame_idx=i)  # optional fill-in
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .apply_homography import pixel_to_field, field_to_pixel
from .field_model import (
    HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH, YARD_LINE_POSITIONS,
)


# ── Identity helpers ─────────────────────────────────────────────────────────

HASH_KINDS = ("far_hash", "near_hash")
SIDELINE_KINDS = ("sideline_near", "sideline_far")
ALL_KINDS = HASH_KINDS + SIDELINE_KINDS

_YARD_LINES = np.asarray(YARD_LINE_POSITIONS, dtype=np.float64)
_Y_BY_KIND = {
    "far_hash": HASH_Y_FAR,
    "near_hash": HASH_Y_NEAR,
    "sideline_near": 0.0,
    "sideline_far": FIELD_WIDTH,
}


def kind_from_field(field_xy, channel):
    """Infer kind from a field (x, y) and the HRNet channel (0=sideline, 1=hash)."""
    fy = float(field_xy[1])
    if channel == 1:
        return "far_hash" if abs(fy - HASH_Y_FAR) < abs(fy - HASH_Y_NEAR) else "near_hash"
    return "sideline_far" if fy > FIELD_WIDTH * 0.5 else "sideline_near"


def snap_to_yard_slot(field_x):
    """Return (yard_slot_index, snapped_field_x, snap_dist_yd)."""
    fx = float(field_x)
    idx = int(np.argmin(np.abs(_YARD_LINES - fx)))
    snapped = float(_YARD_LINES[idx])
    return idx, snapped, abs(fx - snapped)


def field_pos_for(kind, yard_slot_idx):
    """Canonical field position for a (kind, yard-slot) identity."""
    return np.array(
        [float(_YARD_LINES[yard_slot_idx]), _Y_BY_KIND[kind]],
        dtype=np.float64,
    )


# ── Track dataclass ─────────────────────────────────────────────────────────

@dataclass
class KeypointTrack:
    track_id: int
    kind: str                           # one of ALL_KINDS
    yard_slot: int                      # index into YARD_LINE_POSITIONS
    field_pos: np.ndarray               # canonical (x_yd, y_yd), constant
    first_seen: int
    last_seen: int
    n_observations: int
    last_pixel_u: np.ndarray            # most recent undistorted pixel
    # Recent observations for diagnostics + future smoothing
    pixel_history: list = field(default_factory=list)  # [(frame_idx, pixel_u)]


# ── Main bank ────────────────────────────────────────────────────────────────

class KeypointTrackBank:
    """Field-space keypoint tracker.

    Args:
        max_age_frames: drop tracks not seen for this many frames.
        min_obs_for_trust: a track needs this many observations before it's
            used to VALIDATE new homographies (prevents a single bad detection
            from propagating).
        h_validate_tol_yd: during validate_h, a trusted track's observed
            pixel must project (via H_cur) to within this of the track's
            canonical field position.
        h_validate_bad_frac: reject H if more than this fraction of trusted
            tracks fail the per-track residual test.
        coast_tol_yd: when proposing a coasted correspondence via H_prev,
            we accept the coast only if the track was observed recently enough
            (≤ max_coast_frames).
        max_coast_frames: upper bound on how many frames a track can coast.
    """

    def __init__(
        self,
        max_age_frames: int = 8,
        min_obs_for_trust: int = 2,
        h_validate_tol_yd: float = 1.0,
        h_validate_bad_frac: float = 0.34,
        max_coast_frames: int = 4,
    ):
        self.max_age = max_age_frames
        self.min_obs_for_trust = min_obs_for_trust
        self.h_tol = h_validate_tol_yd
        self.h_bad_frac = h_validate_bad_frac
        self.max_coast = max_coast_frames

        self.tracks: dict[tuple, KeypointTrack] = {}   # (kind, slot) -> track
        self._next_id = 0

    # ── Observation ──

    def observe(self, correspondences: list[dict], frame_idx: int) -> list[int]:
        """Record each correspondence against the track bank.

        correspondences: list of dicts with keys
            pixel_u : (2,) undistorted pixel
            field   : (2,) snapped field position
            kind    : str, one of ALL_KINDS (optional — we'll infer if absent)
            channel : int 0/1 (used only if kind is missing)

        Returns list of track_ids aligned with correspondences.
        """
        ids = []
        for c in correspondences:
            pixel_u = np.asarray(c["pixel_u"], dtype=np.float64)
            field_xy = np.asarray(c["field"], dtype=np.float64)
            kind = c.get("kind")
            if kind is None:
                kind = kind_from_field(field_xy, c.get("channel", 1))
            slot_idx, _, _ = snap_to_yard_slot(field_xy[0])
            key = (kind, slot_idx)

            t = self.tracks.get(key)
            if t is None:
                t = KeypointTrack(
                    track_id=self._next_id,
                    kind=kind,
                    yard_slot=slot_idx,
                    field_pos=field_pos_for(kind, slot_idx),
                    first_seen=frame_idx,
                    last_seen=frame_idx,
                    n_observations=1,
                    last_pixel_u=pixel_u,
                    pixel_history=[(frame_idx, pixel_u.copy())],
                )
                self._next_id += 1
                self.tracks[key] = t
            else:
                t.last_seen = frame_idx
                t.n_observations += 1
                t.last_pixel_u = pixel_u
                t.pixel_history.append((frame_idx, pixel_u.copy()))
                # bound history length
                if len(t.pixel_history) > 20:
                    t.pixel_history = t.pixel_history[-20:]
            ids.append(t.track_id)
        return ids

    def prune(self, frame_idx: int):
        """Drop tracks not seen for max_age frames."""
        self.tracks = {
            k: t for k, t in self.tracks.items()
            if frame_idx - t.last_seen <= self.max_age
        }

    # ── H validation ──

    def validate_h(
        self,
        H_cur: np.ndarray,
        correspondences: list[dict],
        frame_idx: int,
    ) -> tuple[bool, dict]:
        """Check whether H_cur is consistent with established tracks.

        For each correspondence whose identity (kind, slot) has an existing
        TRUSTED track, project the correspondence's pixel via H_cur to field,
        and measure residual to the track's canonical field position.

        If too many trusted tracks fail the residual check → H_cur is bad.

        Returns (is_valid, diagnostics_dict).
        """
        trusted = {
            k: t for k, t in self.tracks.items()
            if t.n_observations >= self.min_obs_for_trust
        }
        if not trusted or not correspondences:
            return True, {"n_trusted_checked": 0}

        residuals = []
        for c in correspondences:
            pixel_u = np.asarray(c["pixel_u"], dtype=np.float64)
            field_xy = np.asarray(c["field"], dtype=np.float64)
            kind = c.get("kind") or kind_from_field(field_xy, c.get("channel", 1))
            slot_idx, _, _ = snap_to_yard_slot(field_xy[0])
            t = trusted.get((kind, slot_idx))
            if t is None:
                continue
            # Project current pixel via H_cur to field, compare to canonical
            fxy_via_cur = pixel_to_field(
                pixel_u.reshape(1, 2), H_cur,
            )[0]
            residuals.append(float(np.linalg.norm(fxy_via_cur - t.field_pos)))

        if not residuals:
            return True, {"n_trusted_checked": 0}

        residuals = np.asarray(residuals)
        n_bad = int(np.sum(residuals > self.h_tol))
        bad_frac = n_bad / len(residuals)
        median_res = float(np.median(residuals))
        mean_res = float(np.mean(residuals))
        is_valid = bad_frac <= self.h_bad_frac and median_res <= self.h_tol

        return is_valid, {
            "n_trusted_checked": len(residuals),
            "n_bad": n_bad,
            "bad_frac": bad_frac,
            "median_residual_yd": median_res,
            "mean_residual_yd": mean_res,
        }

    # ── Coasting ──

    def coast_unobserved(
        self,
        H_prev: np.ndarray,
        frame_idx: int,
        observed_keys: Optional[set] = None,
    ) -> list[dict]:
        """For each recently-observed track NOT in observed_keys, synthesize a
        correspondence using H_prev to predict the track's current pixel.

        Returns list of dicts {pixel_u, field, kind, source='coasted', track_id}.
        These are intended to be merged with the current frame's real
        correspondences before computing H_cur.
        """
        H_inv = np.linalg.inv(H_prev)
        out = []
        for key, t in self.tracks.items():
            if observed_keys is not None and key in observed_keys:
                continue
            if t.n_observations < self.min_obs_for_trust:
                continue
            if frame_idx - t.last_seen > self.max_coast:
                continue
            pixel = field_to_pixel(
                t.field_pos.reshape(1, 2), H_inv,
            )[0]
            out.append({
                "pixel_u": pixel.copy(),
                "field": t.field_pos.copy(),
                "kind": t.kind,
                "source": "coasted",
                "track_id": t.track_id,
                "coast_age": frame_idx - t.last_seen,
            })
        return out

    # ── Utility ──

    def observed_keys_from(self, correspondences):
        """Helper: return set of (kind, slot) keys covered by a list of
        correspondences."""
        keys = set()
        for c in correspondences:
            field_xy = np.asarray(c["field"], dtype=np.float64)
            kind = c.get("kind") or kind_from_field(field_xy, c.get("channel", 1))
            slot_idx, _, _ = snap_to_yard_slot(field_xy[0])
            keys.add((kind, slot_idx))
        return keys

    def __len__(self):
        return len(self.tracks)

    def stats(self) -> dict:
        return {
            "n_tracks": len(self.tracks),
            "n_trusted": sum(
                1 for t in self.tracks.values()
                if t.n_observations >= self.min_obs_for_trust
            ),
            "kinds_counts": {
                kind: sum(1 for t in self.tracks.values() if t.kind == kind)
                for kind in ALL_KINDS
            },
        }
