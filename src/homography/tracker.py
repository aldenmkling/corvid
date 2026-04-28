"""
Homography tracker with graceful degradation.

Default pipeline (using grid_solver_v2):
  1. UNet detects yard-line and sideline pixel masks per frame.
     HRNet-W18 detects hash-intersection keypoints.
  2. Grid solver groups UNet pixels into yard-line objects, fits polynomials,
     attaches hashes, and intersects yardlines × sidelines for sideline keypoints.
  3. Plumb-line calibration solves radial distortion once from a well-populated
     frame (usually the first), then reused for the rest of the clip.
  4. Identity assignment: on bootstrap, user-provided anchor; on subsequent
     frames, project detections using the previous H and snap to nearest valid
     field grid point.
  5. Homography selection per frame:
       - FULL: ≥4 identified correspondences → cv2.findHomography(RANSAC)
       - DELTA: 2–3 correspondences → 4-DOF similarity from previous→current
                pixels, apply as H_cur = H_prev @ inv(S)
       - CARRY: <2 correspondences → reuse H_prev unchanged
  6. Caller receives (H, H_inv, method, diagnostics) per frame.

Invariants:
  * Distortion is solved once and never changes within a clip.
  * H is always in UNDISTORTED pixel space — consumers should undistort
    their own pixel points (or the whole frame) before applying H.
  * Field coords use NGS convention (x: 0–120 yards, y: 0–53.33 yards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .apply_homography import pixel_to_field, field_to_pixel
from .distortion import CameraIntrinsics, undistort_points
from .field_model import (
    FIELD_LENGTH, FIELD_WIDTH, HASH_Y_NEAR, HASH_Y_FAR,
    YARD_LINE_POSITIONS,
)
from .keypoint_track_bank import KeypointTrackBank
from .grid_solver_v2 import (
    Yardline, GridSolverResult,
    solve_grid, run_unet, run_hash_w18,
    yardlines_to_correspondences, calibrate_distortion_from_result,
    UNET_YARD_THRESH, UNET_SIDE_THRESH, HASH_THRESH,
)


# ── Constants ──────────────────────────────────────────────────────────────

# Bounds for snap-to-nearest identity assignment
IDENTITY_SNAP_MAX_DIST_YD = 1.5   # reject if snap distance > this
# If the full-H solution disagrees with H_prev by more than this on
# the current correspondences, reject it as a bad identity and fall back
# to delta / carry.
FULL_H_SANITY_MAX_YD = 2.0

# Similarity-delta safety bounds
DELTA_MAX_SCALE_CHANGE = 1.8       # per frame, e.g. zoom doubles at most
DELTA_MAX_ROTATION_DEG = 3.0       # cap per-frame rotation
DELTA_MAX_TRANSLATION_PX = 400     # cap per-frame translation

# Valid sideline y options (near = 0, far = FIELD_WIDTH)
SIDELINE_Y_CHOICES = np.array([0.0, FIELD_WIDTH])
# Valid hash y options (near = 23.58, far = 29.75)
HASH_Y_CHOICES = np.array([HASH_Y_NEAR, HASH_Y_FAR])

# Per-keypoint metadata: (attribute on Yardline, kind string, channel, field_y)
#   channel: 0=sideline, 1=hash (used by snap-to-grid)
_KEYPOINT_SLOTS = [
    ("near_hash",     "near_hash",     1, HASH_Y_NEAR),
    ("far_hash",      "far_hash",      1, HASH_Y_FAR),
    ("near_sideline", "sideline_near", 0, 0.0),
    ("far_sideline",  "sideline_far",  0, FIELD_WIDTH),
]


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Output of HomographyTracker.process_frame()."""
    H: np.ndarray                       # pixel → field, 3×3
    H_inv: np.ndarray                   # field → pixel, 3×3
    method: str                         # "full", "delta", "carry"
    n_correspondences: int              # keypoints used this frame
    pixel_pts_u: np.ndarray | None      # (K, 2) undistorted, used correspondences
    field_pts: np.ndarray | None        # (K, 2) field coords of same
    pixel_reproj_error_mean: float      # on the used correspondences
    field_reproj_error_mean: float      # same, in yards
    delta_scale: float | None = None    # zoom factor if delta used
    delta_rotation_deg: float | None = None
    delta_translation_px: tuple | None = None
    frame_idx: int = 0
    diagnostics: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────

def _snap_to_field_grid(field_xy, channel):
    """Given a projected field coordinate and channel, snap to nearest valid
    grid intersection. Returns (field_xy_snapped, snap_dist_yd).

    Channel 0 = sideline → y ∈ {0, FIELD_WIDTH}
    Channel 1 = hash → y ∈ {HASH_Y_NEAR, HASH_Y_FAR}
    x is snapped to nearest yard-line in YARD_LINE_POSITIONS.
    """
    fx, fy = float(field_xy[0]), float(field_xy[1])
    yl = np.asarray(YARD_LINE_POSITIONS, dtype=np.float64)
    x_idx = int(np.argmin(np.abs(yl - fx)))
    fx_snap = float(yl[x_idx])
    y_choices = SIDELINE_Y_CHOICES if channel == 0 else HASH_Y_CHOICES
    y_idx = int(np.argmin(np.abs(y_choices - fy)))
    fy_snap = float(y_choices[y_idx])
    dist = float(np.hypot(fx - fx_snap, fy - fy_snap))
    return np.array([fx_snap, fy_snap]), dist


def _iter_yardline_keypoints(yl: Yardline):
    """Yield (pixel_xy, kind_str, channel, expected_field_y) for each
    detected keypoint on this yardline (skips None slots)."""
    for attr, kind, channel, fy in _KEYPOINT_SLOTS:
        pt = getattr(yl, attr, None)
        if pt is not None:
            yield np.asarray(pt, dtype=np.float64), kind, channel, fy


# ── Main tracker class ─────────────────────────────────────────────────────

class HomographyTracker:
    """Tracks a per-frame homography through a clip using grid_solver_v2.

    Usage:
        tracker = HomographyTracker(
            "models/unet_line_round2_best.pth",
            "models/hrnet_w18_hash_round1_best.pth",
        )
        result0 = tracker.process_frame(frame0, anchor_ngs_x=35.0)  # bootstrap
        for frame in frames[1:]:
            result = tracker.process_frame(frame)
    """

    def __init__(
        self,
        unet_weights: str,
        hash_weights: str,
        device: str = "mps",
        hash_conf_thresh: float = HASH_THRESH,
        unet_yard_thresh: float = UNET_YARD_THRESH,
        unet_side_thresh: float = UNET_SIDE_THRESH,
        identity_snap_max_yd: float = IDENTITY_SNAP_MAX_DIST_YD,
        delta_max_rotation_deg: float = DELTA_MAX_ROTATION_DEG,
        delta_max_translation_px: float = DELTA_MAX_TRANSLATION_PX,
        delta_max_scale_change: float = DELTA_MAX_SCALE_CHANGE,
        use_track_bank: bool = True,
        track_bank_coast: bool = False,     # default off: coasting introduces
                                             # a feedback loop where
                                             # H_prev-drift → coasted points
                                             # pull H_cur toward the drift →
                                             # runaway slanting. Validation is
                                             # the useful half of the bank.
        use_gpu_vp: bool = False,            # superseded by CC grouping; kept
                                             # as opt-in for back-compat.
        grouping_mode: str = "cc",           # "cc" (CC + collinearity merge,
                                             # ~7× faster, no degenerate cases)
                                             # or "vp" (legacy VP-search).
        recompute_distortion_each_frame: bool = False,
                                             # False (default): solve at
                                             # bootstrap and cache. Stable
                                             # source-panel undistortion + no
                                             # frame-to-frame H jitter from k1
                                             # variation. True: recompute per
                                             # frame for zoom-tracking, but
                                             # introduces visible source-panel
                                             # jitter (each frame undistorted
                                             # with its own k1/k2).
        linearize: bool = False,             # Default off: linearize works
                                             # cleanly on some clips but
                                             # introduces grid-pos off-by-1
                                             # errors on others (likely from
                                             # non-uniform perspective-induced
                                             # yardline spacing fooling the
                                             # `unit = median(diffs)` estimate
                                             # in assign_grid_positions). Keep
                                             # as opt-in pending diagnostic.
    ):
        self.device = device
        self.unet_weights = unet_weights
        self.hash_weights = hash_weights
        self.hash_thresh = hash_conf_thresh
        self.unet_yard_thresh = unet_yard_thresh
        self.unet_side_thresh = unet_side_thresh
        self.identity_snap_max_yd = identity_snap_max_yd
        self.delta_max_rotation_deg = delta_max_rotation_deg
        self.delta_max_translation_px = delta_max_translation_px
        self.delta_max_scale_change = delta_max_scale_change
        self.use_track_bank = use_track_bank
        self.track_bank_coast = track_bank_coast
        self.track_bank = KeypointTrackBank() if use_track_bank else None
        self.use_gpu_vp = use_gpu_vp
        self.grouping_mode = grouping_mode
        self.recompute_distortion_each_frame = recompute_distortion_each_frame
        self.linearize = linearize

        # Per-clip state
        self.H_prev: Optional[np.ndarray] = None
        self.H_inv_prev: Optional[np.ndarray] = None
        self.prev_pixel_pts_u: Optional[np.ndarray] = None
        self.prev_field_pts: Optional[np.ndarray] = None
        self.frame_shape: Optional[tuple] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.frame_count: int = 0
        self.vp_prev: Optional[tuple] = None  # warm-start anchor for GPU VP

    # ── Detection ────────────────────────────────────────────────────────

    def _detect(self, frame):
        """Run UNet + W18 + grid solver. Returns dict with masks, hashes,
        and the GridSolverResult."""
        h, w = frame.shape[:2]
        yard_mask, side_mask = run_unet(
            frame, self.unet_weights, device=self.device,
            yard_thresh=self.unet_yard_thresh, side_thresh=self.unet_side_thresh,
        )
        hash_pxs, hash_confs = run_hash_w18(
            frame, self.hash_weights, device=self.device,
            conf_thresh=self.hash_thresh,
        )
        result = solve_grid(
            yard_mask, side_mask, hash_pxs,
            hash_confs=hash_confs, frame_shape=(h, w),
            use_gpu_vp=self.use_gpu_vp, vp_init=self.vp_prev,
            vp_device=self.device, grouping_mode=self.grouping_mode,
            linearize=self.linearize,
            # If we already have cached intrinsics from a prior frame, pass
            # them as override so solve_grid skips its internal per-frame
            # distortion calibration. Calibration runs ONCE on the bootstrap
            # frame; every other frame reuses k1/k2.
            intrinsics_override=(self.intrinsics
                                  if self.linearize and self.intrinsics is not None
                                  else None),
        )
        # Cache VP for warm-starting next frame's GPU search.
        if result.vp is not None:
            self.vp_prev = result.vp
        return {
            "yard_mask": yard_mask,
            "side_mask": side_mask,
            "hash_pxs": hash_pxs,
            "hash_confs": hash_confs,
            "result": result,
        }

    # ── Distortion ───────────────────────────────────────────────────────

    def _compute_distortion(self, result: GridSolverResult, frame_shape):
        """Solve k1, k2 from plumb-line fit using this frame's line groups."""
        h, w = frame_shape
        focal_guess = float(max(h, w))
        k1, k2 = calibrate_distortion_from_result(
            result, frame_shape=frame_shape, focal_length_guess=focal_guess,
        )
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            k1, k2 = 0.0, 0.0  # sanity-reject runaway values
        return k1, k2, focal_guess

    # ── Identity assignment by projection + snap ─────────────────────────

    def _identify_via_projection(self, result: GridSolverResult):
        """Use previous H to identify each yardline's keypoints' field coords.

        Iterates over `result.yardlines`. For each detected keypoint (near_hash,
        far_hash, near_sideline, far_sideline), projects through H_prev, snaps
        to nearest grid intersection, and accepts if within tolerance.

        Returns (pixel_pts, field_pts, kinds, labels) — kinds are the track-bank
        kind strings, labels are human-readable per-correspondence tags.
        """
        if self.H_prev is None:
            return (np.zeros((0, 2)), np.zeros((0, 2)), [], [])

        pixel_pts: list = []
        field_pts: list = []
        kinds: list[str] = []
        labels: list[str] = []

        # If solve_grid linearized the result, the keypoints are already in
        # undistorted space — don't undistort again.
        already_undistorted = bool(getattr(result, "is_linearized", False))

        for yl in result.yardlines:
            gp_tag = "?" if yl.grid_pos is None else str(yl.grid_pos)
            for pix, kind, channel, _expected_fy in _iter_yardline_keypoints(yl):
                if already_undistorted:
                    pix_u = pix.astype(np.float64)
                else:
                    pix_u = undistort_points(pix.reshape(1, 2), self.intrinsics)[0]
                fxy_proj = pixel_to_field(pix_u.reshape(1, 2), self.H_prev)[0]
                snapped, dist = _snap_to_field_grid(fxy_proj, channel)
                if dist > self.identity_snap_max_yd:
                    continue
                pixel_pts.append(pix.tolist())
                field_pts.append(snapped.tolist())
                kinds.append(kind)
                labels.append(f"g{gp_tag}_{kind}")

        return (
            np.array(pixel_pts) if pixel_pts else np.zeros((0, 2)),
            np.array(field_pts) if field_pts else np.zeros((0, 2)),
            kinds,
            labels,
        )

    def _bootstrap_correspondences(self, result: GridSolverResult, anchor_ngs_x: float):
        """Build correspondences for the bootstrap frame using anchor_ngs_x.

        Returns (pixel_pts, field_pts, kinds, labels). Mirrors the projection
        path's return so process_frame can branch uniformly afterward.
        """
        pixel_pts, field_pts, labels = yardlines_to_correspondences(
            result, anchor_ngs_x,
        )
        # Re-derive kinds from labels suffix (yardlines_to_correspondences
        # encodes the kind in the label, e.g. "g3_near_hash").
        kind_lookup = {
            "near_hash": "near_hash", "far_hash": "far_hash",
            "near_side": "sideline_near", "far_side": "sideline_far",
        }
        kinds = []
        for lbl in labels:
            for suffix, kind in kind_lookup.items():
                if lbl.endswith(suffix):
                    kinds.append(kind)
                    break
            else:
                kinds.append("unknown")
        return pixel_pts, field_pts, kinds, labels

    # ── Homography solve modes ──────────────────────────────────────────

    def _solve_full_homography(self, pixel_pts_u, field_pts):
        """cv2.findHomography with RANSAC. Returns (H, H_inv, mask, inliers)."""
        if len(pixel_pts_u) < 4:
            return None
        H, mask = cv2.findHomography(
            pixel_pts_u.astype(np.float64),
            field_pts.astype(np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
        )
        if H is None:
            return None
        H_inv = np.linalg.inv(H)
        inliers = int(mask.sum()) if mask is not None else len(pixel_pts_u)
        return H, H_inv, mask, inliers

    def _solve_similarity_delta(self, pixel_pts_u, field_pts):
        """Fit similarity that maps previous pixel positions of these same
        field points → current pixel positions, then H_cur = H_prev @ inv(S).
        """
        if self.H_prev is None or len(pixel_pts_u) < 2:
            return None

        # Where did these field points live in the previous frame?
        prev_pixel_pts = field_to_pixel(field_pts, self.H_inv_prev)

        M, _ = cv2.estimateAffinePartial2D(
            prev_pixel_pts.astype(np.float64),
            pixel_pts_u.astype(np.float64),
            method=cv2.LMEDS,
        )
        if M is None:
            return None

        # Extract scale / rotation / translation
        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        tx, ty = float(M[0, 2]), float(M[1, 2])

        # Safety: reject implausible transforms
        if (scale > self.delta_max_scale_change
                or scale < 1.0 / self.delta_max_scale_change):
            return None
        if abs(rotation_deg) > self.delta_max_rotation_deg:
            sign = 1 if rotation_deg > 0 else -1
            rotation_deg = sign * self.delta_max_rotation_deg
            c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
            M = np.array([
                [scale * c, -scale * s, tx],
                [scale * s,  scale * c, ty],
            ])
        if np.hypot(tx, ty) > self.delta_max_translation_px:
            return None

        S = np.vstack([M, [0, 0, 1]])
        H_cur = self.H_prev @ np.linalg.inv(S)
        H_inv_cur = np.linalg.inv(H_cur)
        return H_cur, H_inv_cur, S, {
            "scale": scale,
            "rotation_deg": rotation_deg,
            "translation_px": (tx, ty),
        }

    # ── Main entry ───────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        anchor_ngs_x: Optional[float] = None,
    ) -> FrameResult:
        """Process the next frame in a clip.

        Args:
            frame: BGR image.
            anchor_ngs_x: NGS x of the leftmost detected yard line (grid_pos 0).
                Required on the first frame. Optional on subsequent frames —
                if provided, forces a fresh full-homography calibration and
                resets the identity assignment.

        Returns:
            FrameResult dataclass.
        """
        if frame is None:
            raise ValueError("frame is None")

        self.frame_count += 1
        h, w = frame.shape[:2]

        # Run UNet + W18 + grid solver
        det = self._detect(frame)
        result: GridSolverResult = det["result"]

        # Bootstrap or re-anchor path: requires anchor_ngs_x
        is_bootstrap = self.H_prev is None or anchor_ngs_x is not None
        if is_bootstrap and anchor_ngs_x is None:
            raise ValueError(
                "First frame (or re-anchor) requires anchor_ngs_x — "
                "the NGS x coordinate of grid_pos 0 (leftmost yard line)."
            )

        # Solve / update distortion. With solve_grid's linearize=True (default),
        # the result already carries calibrated intrinsics + has applied them
        # to all keypoints — just reuse those. Otherwise compute via plumb-line.
        if getattr(result, "is_linearized", False) and result.intrinsics is not None:
            self.intrinsics = result.intrinsics
            self.frame_shape = (h, w)
        elif self.intrinsics is None or self.recompute_distortion_each_frame:
            k1, k2, focal_guess = self._compute_distortion(result, (h, w))
            self.intrinsics = CameraIntrinsics(
                fx=focal_guess, fy=focal_guess, cx=w / 2.0, cy=h / 2.0,
                k1=k1, k2=k2,
            )
            self.frame_shape = (h, w)

        # Build identified correspondences
        if is_bootstrap:
            pixel_pts, field_pts, kinds, labels = self._bootstrap_correspondences(
                result, anchor_ngs_x,
            )
        else:
            pixel_pts, field_pts, kinds, labels = self._identify_via_projection(result)

        # If the result was linearized, keypoints are ALREADY in undistorted
        # space — skip the redundant undistort_points call.
        if getattr(result, "is_linearized", False):
            pixel_pts_u = (pixel_pts.astype(np.float64)
                           if len(pixel_pts) else np.zeros((0, 2)))
        else:
            pixel_pts_u = (undistort_points(pixel_pts, self.intrinsics)
                           if len(pixel_pts) else np.zeros((0, 2)))

        # Build correspondence dicts (used by track bank).
        current_corr = [
            {"pixel_u": pixel_pts_u[i], "field": field_pts[i],
             "kind": kinds[i] if i < len(kinds) else "unknown"}
            for i in range(len(pixel_pts_u))
        ]

        # Optionally coast unobserved tracks into the correspondence set.
        coasted = []
        if (self.track_bank is not None and self.track_bank_coast
                and self.H_prev is not None and not is_bootstrap):
            observed_keys = self.track_bank.observed_keys_from(current_corr)
            coasted = self.track_bank.coast_unobserved(
                self.H_prev, frame_idx=self.frame_count,
                observed_keys=observed_keys,
            )

        combined_corr = current_corr + coasted
        combined_pixel_u = (
            np.asarray([c["pixel_u"] for c in combined_corr], dtype=np.float64)
            if combined_corr else np.zeros((0, 2))
        )
        combined_field = (
            np.asarray([c["field"] for c in combined_corr], dtype=np.float64)
            if combined_corr else np.zeros((0, 2))
        )
        n = len(combined_pixel_u)

        # Decide method
        full = None
        validated_by_bank = None
        bank_diag = None
        n_real = len(current_corr)
        if n >= 4 and n_real >= 4:
            full = self._solve_full_homography(combined_pixel_u, combined_field)
            # Sanity check: on non-bootstrap frames, verify the new H doesn't
            # disagree hugely with H_prev.
            if full is not None and not is_bootstrap and self.H_prev is not None:
                H_candidate = full[0]
                field_projected = pixel_to_field(combined_pixel_u, H_candidate)
                field_prev = pixel_to_field(combined_pixel_u, self.H_prev)
                divergence = float(np.mean(np.linalg.norm(
                    field_projected - field_prev, axis=1)))
                if divergence > FULL_H_SANITY_MAX_YD:
                    full = None  # fall through to delta/carry

            # Track-bank validation: check that existing trusted tracks agree
            # with the candidate H. Catches shifted-grid misidentifications.
            if full is not None and self.track_bank is not None and not is_bootstrap:
                H_candidate = full[0]
                valid, diag = self.track_bank.validate_h(
                    H_candidate, current_corr, frame_idx=self.frame_count,
                )
                bank_diag = diag
                validated_by_bank = valid
                if not valid:
                    full = None

        if full is not None:
            H, H_inv, mask, inliers = full
            method = "full"
            delta_info = None
            used_pixel = pixel_pts_u
            used_field = field_pts
        elif n_real >= 2 and self.H_prev is not None:
            delta = self._solve_similarity_delta(pixel_pts_u, field_pts)
            if delta is not None:
                H, H_inv, S, delta_info = delta
                method = "delta"
                used_pixel = pixel_pts_u
                used_field = field_pts
            elif self.H_prev is not None:
                H = self.H_prev
                H_inv = self.H_inv_prev
                method = "carry"
                delta_info = None
                used_pixel = np.zeros((0, 2))
                used_field = np.zeros((0, 2))
            else:
                raise RuntimeError("Cannot compute H on first frame with <4 points")
        elif self.H_prev is not None:
            H = self.H_prev
            H_inv = self.H_inv_prev
            method = "carry"
            delta_info = None
            used_pixel = np.zeros((0, 2))
            used_field = np.zeros((0, 2))
        else:
            raise RuntimeError(
                f"First frame only has {n} correspondences; need ≥4 to bootstrap."
            )

        # Errors on the points we actually used this frame
        if len(used_field) > 0:
            projected = field_to_pixel(used_field, H_inv)
            pix_err = float(np.mean(np.linalg.norm(
                projected - used_pixel, axis=1)))
            field_proj = pixel_to_field(used_pixel, H)
            fld_err = float(np.mean(np.linalg.norm(field_proj - used_field, axis=1)))
        else:
            pix_err = float("nan")
            fld_err = float("nan")

        # Update state
        self.H_prev = H
        self.H_inv_prev = H_inv
        self.prev_pixel_pts_u = used_pixel
        self.prev_field_pts = used_field

        # Update track bank with this frame's REAL observations (not coasted).
        track_bank_stats = None
        if self.track_bank is not None and method in ("full", "delta"):
            self.track_bank.observe(current_corr, frame_idx=self.frame_count)
            self.track_bank.prune(frame_idx=self.frame_count)
            track_bank_stats = self.track_bank.stats()

        return FrameResult(
            H=H,
            H_inv=H_inv,
            method=method,
            n_correspondences=n,
            pixel_pts_u=used_pixel,
            field_pts=used_field,
            pixel_reproj_error_mean=pix_err,
            field_reproj_error_mean=fld_err,
            delta_scale=delta_info["scale"] if delta_info else None,
            delta_rotation_deg=delta_info["rotation_deg"] if delta_info else None,
            delta_translation_px=delta_info["translation_px"] if delta_info else None,
            frame_idx=self.frame_count,
            diagnostics={
                "n_hash_detections": len(det["hash_pxs"]),
                "n_yardlines": len(result.yardlines),
                "n_sidelines": int(result.far_sideline is not None)
                                + int(result.near_sideline is not None),
                "labels_used": labels if is_bootstrap else [""] * n,
                "n_current_corr": len(current_corr),
                "n_coasted_corr": len(coasted),
                "track_bank": track_bank_stats,
                "track_bank_validated": validated_by_bank,
                "track_bank_validation": bank_diag,
            },
        )
