"""
Track field keypoints (hash-yardline intersections) across frames.

Provides identity-consistent keypoints for homography computation throughout
a play. Detects yard lines + hashes every frame, then matches new detections
to previously tracked keypoints using optical flow predictions.

Pipeline per frame:
  1. Detect yard lines and hash marks (fresh detection every frame)
  2. Track previous keypoints forward via sparse optical flow (LK)
  3. Match new detections to flow-predicted positions (Hungarian assignment)
  4. Merge: prefer fresh detections, keep flow-predicted points for brief gaps
  5. Retire keypoints that haven't been detected for too long

Handles camera panning naturally: new keypoints enter on one side of the
frame while old ones exit the other. Each keypoint maintains a consistent
ID tied to its relative grid position.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field as dataclass_field

from .yard_lines import detect_yard_lines, YardLineResult
from .hash_marks import detect_hashes, HashResult


# ── Configuration ─────────────────────────────────────────────────────────

MATCH_DIST_PX = 25.0         # max pixel distance to match a detection to a tracked point
MAX_UNDETECTED_FRAMES = 10   # retire a keypoint after this many frames without detection
LK_WIN_SIZE = (21, 21)       # Lucas-Kanade window size
LK_MAX_LEVEL = 3             # pyramid levels for LK


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class TrackedKeypoint:
    """A single tracked hash-yardline intersection."""
    id: int
    pixel_xy: np.ndarray           # current pixel position (2,)
    grid_pos: int                  # yard line grid position (relative, 0-based)
    hash_type: str                 # "far" or "near"
    last_detected_frame: int       # last frame where this was freshly detected
    is_detected: bool              # True if position came from detection this frame
    confidence: float              # 1.0 if detected, decays when flow-only

    @property
    def grid_key(self) -> tuple[int, str]:
        """Unique key: (grid_position, hash_type)."""
        return (self.grid_pos, self.hash_type)


@dataclass
class FieldTrackingResult:
    """Result of field keypoint tracking for a single frame."""
    keypoints: list[TrackedKeypoint]
    yl_result: YardLineResult | None
    hash_result: HashResult | None
    n_detected: int                # keypoints from fresh detection
    n_flow_only: int               # keypoints from optical flow only
    n_new: int                     # newly appeared keypoints this frame
    n_retired: int                 # keypoints dropped this frame


# ── Keypoint extraction from detections ───────────────────────────────────

def _extract_keypoints_from_detection(
    yl_result: YardLineResult,
    hash_result: HashResult,
) -> list[tuple[np.ndarray, int, str]]:
    """Extract (pixel_xy, grid_pos, hash_type) from detection results.

    Each keypoint is a hash-yardline intersection computed from the yard
    line geometry and the parametric t-value of the hash mark.
    """
    keypoints = []

    for line_idx, entry in hash_result.hashes.items():
        if line_idx >= len(yl_result.lines):
            continue

        line = yl_result.lines[line_idx]
        grid_pos = yl_result.grid[line_idx]
        x_top, y_top, x_bot, y_bot = line
        ldx = x_bot - x_top
        ldy = y_bot - y_top

        for hash_type in ("far", "near"):
            if hash_type not in entry:
                continue
            t = entry[hash_type]
            px = x_top + t * ldx
            py = y_top + t * ldy
            keypoints.append((np.array([px, py]), grid_pos, hash_type))

    return keypoints


# ── Optical flow ──────────────────────────────────────────────────────────

def _track_with_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track points from prev frame to curr frame using Lucas-Kanade.

    Args:
        prev_gray: previous grayscale frame
        curr_gray: current grayscale frame
        points: (N, 2) float32 array of points to track

    Returns:
        (new_points, status) where status[i] = 1 if tracking succeeded
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=np.uint8)

    pts = points.reshape(-1, 1, 2).astype(np.float32)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Back-track to validate: track the result backwards and check consistency
    if new_pts is not None and len(new_pts) > 0:
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_pts, None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_pts is not None:
            # Forward-backward consistency check
            fb_error = np.sqrt(np.sum((pts - back_pts) ** 2, axis=2)).ravel()
            status = status.ravel() & back_status.ravel() & (fb_error < 2.0).astype(np.uint8)

    new_pts = new_pts.reshape(-1, 2) if new_pts is not None else np.array([]).reshape(0, 2)
    status = status.ravel() if status is not None else np.array([], dtype=np.uint8)

    return new_pts, status


# ── Matching ──────────────────────────────────────────────────────────────

def _match_detections_to_tracked(
    detected: list[tuple[np.ndarray, int, str]],
    tracked_predictions: dict[int, np.ndarray],
    tracked_keypoints: dict[int, TrackedKeypoint],
) -> tuple[dict[int, int], list[int]]:
    """Match detected keypoints to flow-predicted tracked keypoints.

    Uses grid_key (grid_pos, hash_type) as the primary match criterion,
    with pixel distance as a sanity check.

    Returns:
        (matches, unmatched_det_indices)
        matches: {tracked_id: det_index}
        unmatched_det_indices: list of detection indices with no match
    """
    matches: dict[int, int] = {}
    matched_det: set[int] = set()

    # First pass: match by grid_key (exact semantic match)
    for track_id, kp in tracked_keypoints.items():
        pred_pos = tracked_predictions.get(track_id)
        best_di = None
        best_dist = float('inf')

        for di, (det_pos, det_grid, det_hash) in enumerate(detected):
            if di in matched_det:
                continue
            # Must be same hash type
            if det_hash != kp.hash_type:
                continue

            # Pixel distance check against flow prediction (if available)
            # or last known position
            ref_pos = pred_pos if pred_pos is not None else kp.pixel_xy
            dist = np.linalg.norm(det_pos - ref_pos)

            if dist < best_dist and dist < MATCH_DIST_PX:
                best_dist = dist
                best_di = di

        if best_di is not None:
            matches[track_id] = best_di
            matched_det.add(best_di)

    unmatched = [i for i in range(len(detected)) if i not in matched_det]
    return matches, unmatched


# ── Main tracker class ────────────────────────────────────────────────────

class FieldTracker:
    """Track field keypoints across frames of a play.

    Usage:
        tracker = FieldTracker()
        for frame in frames:
            result = tracker.update(frame, frame_idx)
            # result.keypoints has identity-consistent keypoints
    """

    def __init__(self):
        self._keypoints: dict[int, TrackedKeypoint] = {}
        self._next_id: int = 0
        self._prev_gray: np.ndarray | None = None
        self._frame_idx: int = -1
        # Track the grid offset so we can maintain consistent grid IDs
        # even as the camera pans and the relative grid shifts
        self._grid_offset: int = 0
        self._prev_yl_result: YardLineResult | None = None

    def _allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _resolve_grid_offset(self, yl_result: YardLineResult) -> int:
        """Compute the grid offset to maintain consistent grid IDs across pans.

        When the camera pans, the yard line detector assigns grid positions
        starting from 0 for the leftmost detected line. But the actual yard
        lines have shifted. We use optical flow on yard line x-intercepts to
        figure out the offset.
        """
        if self._prev_yl_result is None or not self._keypoints:
            return 0

        # Find the grid offset that best aligns new detections with existing
        # tracked keypoints. Use the tracked keypoints' grid positions as
        # reference.
        existing_grids = {kp.grid_pos for kp in self._keypoints.values()}
        if not existing_grids:
            return self._grid_offset

        best_offset = self._grid_offset
        best_count = 0

        # Try offsets in a reasonable range
        new_grids = set(yl_result.grid)
        for test_offset in range(self._grid_offset - 5, self._grid_offset + 6):
            shifted = {g + test_offset for g in new_grids}
            overlap = len(shifted & existing_grids)
            if overlap > best_count:
                best_count = overlap
                best_offset = test_offset

        return best_offset

    def update(self, frame: np.ndarray, frame_idx: int) -> FieldTrackingResult:
        """Process a new frame: detect, track, match, merge.

        Args:
            frame: BGR frame
            frame_idx: frame index (for aging/retirement)

        Returns:
            FieldTrackingResult with identity-consistent keypoints
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Detect yard lines and hashes ──────────────────────
        yl_result = detect_yard_lines(frame)
        hash_result = None
        detected_kps: list[tuple[np.ndarray, int, str]] = []

        if yl_result is not None:
            hash_result = detect_hashes(
                frame, yl_result,
                gray=yl_result.gray,
                canny=yl_result.canny,
            )
            if hash_result is not None:
                # Resolve grid offset for pan consistency
                grid_offset = self._resolve_grid_offset(yl_result)
                self._grid_offset = grid_offset

                raw_kps = _extract_keypoints_from_detection(yl_result, hash_result)
                # Apply grid offset so grid positions are globally consistent
                detected_kps = [
                    (pos, gp + grid_offset, ht)
                    for pos, gp, ht in raw_kps
                ]

        # ── Step 2: Track previous keypoints with optical flow ────────
        tracked_predictions: dict[int, np.ndarray] = {}

        if self._prev_gray is not None and self._keypoints:
            track_ids = list(self._keypoints.keys())
            prev_points = np.array([self._keypoints[tid].pixel_xy for tid in track_ids])

            new_points, status = _track_with_flow(self._prev_gray, gray, prev_points)

            for i, tid in enumerate(track_ids):
                if status[i]:
                    predicted = new_points[i]
                    # Only keep predictions that are still in frame
                    if 0 <= predicted[0] < w and 0 <= predicted[1] < h:
                        tracked_predictions[tid] = predicted

        # ── Step 3: Match detections to tracked keypoints ─────────────
        matches, unmatched_det = _match_detections_to_tracked(
            detected_kps, tracked_predictions, self._keypoints,
        )

        # ── Step 4: Update matched keypoints with fresh detections ────
        n_detected = 0
        n_flow_only = 0

        for track_id, det_idx in matches.items():
            det_pos, det_grid, det_hash = detected_kps[det_idx]
            kp = self._keypoints[track_id]
            kp.pixel_xy = det_pos
            kp.grid_pos = det_grid
            kp.last_detected_frame = frame_idx
            kp.is_detected = True
            kp.confidence = 1.0
            n_detected += 1

        # ── Step 5: Update unmatched tracked points with flow ─────────
        unmatched_tracked = set(self._keypoints.keys()) - set(matches.keys())
        for track_id in unmatched_tracked:
            if track_id in tracked_predictions:
                kp = self._keypoints[track_id]
                kp.pixel_xy = tracked_predictions[track_id]
                kp.is_detected = False
                # Decay confidence
                frames_since = frame_idx - kp.last_detected_frame
                kp.confidence = max(0.0, 1.0 - frames_since / MAX_UNDETECTED_FRAMES)
                n_flow_only += 1

        # ── Step 6: Add new keypoints (unmatched detections) ──────────
        n_new = 0
        for det_idx in unmatched_det:
            det_pos, det_grid, det_hash = detected_kps[det_idx]

            # Check we don't already have a keypoint at this grid position
            existing_keys = {kp.grid_key for kp in self._keypoints.values()}
            new_key = (det_grid, det_hash)
            if new_key in existing_keys:
                # Duplicate grid key — the match failed but we already have
                # this point. Update the existing one instead.
                for kp in self._keypoints.values():
                    if kp.grid_key == new_key:
                        kp.pixel_xy = det_pos
                        kp.last_detected_frame = frame_idx
                        kp.is_detected = True
                        kp.confidence = 1.0
                        break
                continue

            new_id = self._allocate_id()
            self._keypoints[new_id] = TrackedKeypoint(
                id=new_id,
                pixel_xy=det_pos,
                grid_pos=det_grid,
                hash_type=det_hash,
                last_detected_frame=frame_idx,
                is_detected=True,
                confidence=1.0,
            )
            n_new += 1
            n_detected += 1

        # ── Step 7: Retire stale keypoints ────────────────────────────
        n_retired = 0
        to_remove = []
        for track_id, kp in self._keypoints.items():
            frames_since = frame_idx - kp.last_detected_frame
            # Remove if: too old, or flow lost it (not in predictions and not detected)
            if frames_since > MAX_UNDETECTED_FRAMES:
                to_remove.append(track_id)
            elif (track_id not in matches
                  and track_id not in tracked_predictions
                  and kp.last_detected_frame != frame_idx):
                # Flow lost this point and detection didn't find it
                # (but don't retire points that were just created this frame)
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._keypoints[track_id]
            n_retired += 1

        # ── Store state for next frame ────────────────────────────────
        self._prev_gray = gray
        self._frame_idx = frame_idx
        self._prev_yl_result = yl_result

        return FieldTrackingResult(
            keypoints=list(self._keypoints.values()),
            yl_result=yl_result,
            hash_result=hash_result,
            n_detected=n_detected,
            n_flow_only=n_flow_only,
            n_new=n_new,
            n_retired=n_retired,
        )

    def get_homography_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current keypoints as pixel/grid arrays for homography computation.

        Returns (pixel_pts, grid_info) where:
            pixel_pts: (N, 2) pixel coordinates
            grid_info: list of (grid_pos, hash_type, confidence) per point

        Only includes points with confidence > 0.
        """
        if not self._keypoints:
            return np.array([]).reshape(0, 2), []

        pixel_pts = []
        grid_info = []

        for kp in self._keypoints.values():
            if kp.confidence > 0:
                pixel_pts.append(kp.pixel_xy)
                grid_info.append((kp.grid_pos, kp.hash_type, kp.confidence))

        if not pixel_pts:
            return np.array([]).reshape(0, 2), []

        return np.array(pixel_pts), grid_info

    def reset(self):
        """Reset tracker state (call between plays)."""
        self._keypoints.clear()
        self._next_id = 0
        self._prev_gray = None
        self._frame_idx = -1
        self._grid_offset = 0
        self._prev_yl_result = None
