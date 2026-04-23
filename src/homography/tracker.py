"""
Homography tracker with graceful degradation.

Default pipeline:
  1. HRNet detects hash/sideline keypoints per frame.
  2. Grid solver pairs hashes → yard-line groups with relative grid positions.
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

import os
import sys
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
from .keypoint_detector import FieldKeypointDetector
from .keypoint_track_bank import (
    KeypointTrackBank, kind_from_field, snap_to_yard_slot,
)
from .grid_solver import (
    split_hash_rows, pair_hashes, find_sideline_on_yard_line,
    assign_grid_positions, compute_hash_pca, _row_coord,
    yardline_tilt_slope_from_pairs, calibrate_distortion_from_lines,
)


# ── Constants ──────────────────────────────────────────────────────────────

HASH_CONF_THRESH = 0.40
SIDELINE_CONF_THRESH = 0.30

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

def _extract_peaks(heatmap, threshold, orig_shape):
    """Return (N,2) pixel peaks (in original frame coords) and their confidences."""
    from scipy import ndimage
    from .keypoint_detector import _refine_peak

    orig_h, orig_w = orig_shape
    hm_h, hm_w = heatmap.shape
    mask = heatmap >= threshold
    if not mask.any():
        return np.zeros((0, 2)), np.zeros(0)
    labels, n = ndimage.label(mask)
    pxs, confs = [], []
    for comp_id in range(1, n + 1):
        comp_mask = labels == comp_id
        vals = heatmap * comp_mask
        peak_idx = vals.argmax()
        py, px_h = peak_idx // hm_w, peak_idx % hm_w
        confs.append(float(heatmap[py, px_h]))
        ref_x, ref_y = _refine_peak(heatmap, py, px_h)
        pxs.append([ref_x / hm_w * orig_w, ref_y / hm_h * orig_h])
    return np.array(pxs), np.array(confs)


def _build_yard_line_groups(hash_pxs, sideline_pxs, sideline_confs):
    """Pair hashes, attach sidelines, assign grid positions."""
    far_hashes, near_hashes = split_hash_rows(hash_pxs)
    pairs, unpaired_far, unpaired_near, _, _ = pair_hashes(
        far_hashes, near_hashes,
    )

    groups = []
    used_sideline = set()
    for fh, nh in pairs:
        fh = np.asarray(fh)
        nh = np.asarray(nh)
        sl_idx, _ = find_sideline_on_yard_line(
            nh, fh, sideline_pxs, max_perp_distance=12,
        )
        sideline_pt = None
        sideline_conf = None
        if sl_idx is not None and sl_idx not in used_sideline:
            used_sideline.add(sl_idx)
            sideline_pt = sideline_pxs[sl_idx].tolist()
            sideline_conf = float(sideline_confs[sl_idx])
        groups.append({
            "far_hash": fh.tolist(),
            "near_hash": nh.tolist(),
            "sideline": sideline_pt,
            "sideline_conf": sideline_conf,
            "singleton": False,
        })
    for fh in unpaired_far:
        groups.append({
            "far_hash": np.asarray(fh).tolist(),
            "near_hash": None,
            "sideline": None,
            "sideline_conf": None,
            "singleton": True,
        })
    for nh in unpaired_near:
        groups.append({
            "far_hash": None,
            "near_hash": np.asarray(nh).tolist(),
            "sideline": None,
            "sideline_conf": None,
            "singleton": True,
        })

    # Note: singleton sidelines intentionally not emitted (unreliable).
    assign_grid_positions(groups)
    return groups, used_sideline


def _groups_to_correspondences(groups, base_ngs_x, frame_shape=None):
    """Convert yard-line groups into pixel/field arrays using a known anchor.

    Emits:
      - Paired groups (both hashes + any matched sideline).
      - Singleton hashes that fit the grid spacing (grid_fit_ok).
      - Singleton sidelines that fit the grid spacing, classified near/far
        by image-half (requires frame_shape).
    """
    pixel_pts = []
    field_pts = []
    labels = []
    img_mid_y = frame_shape[0] / 2.0 if frame_shape is not None else None

    for g in groups:
        gp = g.get("grid_pos")
        if gp is None:
            continue
        fx = base_ngs_x + gp * 5

        if g.get("singleton"):
            if not g.get("grid_fit_ok", False):
                continue
            if g["far_hash"] is not None:
                pixel_pts.append(g["far_hash"])
                field_pts.append([fx, HASH_Y_FAR])
                labels.append(f"g{gp}_far_s")
            elif g["near_hash"] is not None:
                pixel_pts.append(g["near_hash"])
                field_pts.append([fx, HASH_Y_NEAR])
                labels.append(f"g{gp}_near_s")
            elif g["sideline"] is not None and img_mid_y is not None:
                sl = g["sideline"]
                field_y = FIELD_WIDTH if sl[1] < img_mid_y else 0.0
                pixel_pts.append(sl)
                field_pts.append([fx, field_y])
                tag = "side_s_far" if field_y > 0 else "side_s_near"
                labels.append(f"g{gp}_{tag}")
            continue

        # Paired group
        if g["near_hash"] is not None:
            pixel_pts.append(g["near_hash"])
            field_pts.append([fx, HASH_Y_NEAR])
            labels.append(f"g{gp}_near")
        if g["far_hash"] is not None:
            pixel_pts.append(g["far_hash"])
            field_pts.append([fx, HASH_Y_FAR])
            labels.append(f"g{gp}_far")
        if g["sideline"] is not None:
            pixel_pts.append(g["sideline"])
            field_pts.append([fx, FIELD_WIDTH])
            labels.append(f"g{gp}_side")
    return (np.array(pixel_pts) if pixel_pts else np.zeros((0, 2)),
            np.array(field_pts) if field_pts else np.zeros((0, 2)),
            labels)


def _groups_flat_detections(groups):
    """Flatten groups into (pixel, channel) pairs for identity-via-projection.

    Channel: 0 = sideline, 1 = hash.
    """
    out = []
    for g in groups:
        for key in ("near_hash", "far_hash"):
            if g.get(key) is not None:
                out.append((np.asarray(g[key], dtype=np.float64), 1))
        if g.get("sideline") is not None:
            out.append((np.asarray(g["sideline"], dtype=np.float64), 0))
    return out


def _snap_to_field_grid(field_xy, channel):
    """Given a projected field coordinate and channel, snap to nearest valid
    grid intersection. Returns (field_xy_snapped, snap_dist_yd) or
    (None, large) if off-grid.

    Channel 0 = sideline → y ∈ {0, FIELD_WIDTH}
    Channel 1 = hash → y ∈ {HASH_Y_NEAR, HASH_Y_FAR}
    x is snapped to nearest yard-line in YARD_LINE_POSITIONS.
    """
    fx, fy = float(field_xy[0]), float(field_xy[1])
    # Snap x
    yl = np.asarray(YARD_LINE_POSITIONS, dtype=np.float64)
    x_idx = int(np.argmin(np.abs(yl - fx)))
    fx_snap = float(yl[x_idx])
    # Snap y
    y_choices = SIDELINE_Y_CHOICES if channel == 0 else HASH_Y_CHOICES
    y_idx = int(np.argmin(np.abs(y_choices - fy)))
    fy_snap = float(y_choices[y_idx])
    dist = float(np.hypot(fx - fx_snap, fy - fy_snap))
    return np.array([fx_snap, fy_snap]), dist


# ── Main tracker class ─────────────────────────────────────────────────────

class HomographyTracker:
    """Tracks a per-frame homography through a clip.

    Usage:
        tracker = HomographyTracker("models/hrnet_finetuned_last.pth")
        result0 = tracker.process_frame(frame0, anchor_ngs_x=35.0)  # bootstrap
        for frame in frames[1:]:
            result = tracker.process_frame(frame)
            # result.H, result.method, result.n_correspondences ...
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        hash_conf_thresh: float = HASH_CONF_THRESH,
        sideline_conf_thresh: float = SIDELINE_CONF_THRESH,
        identity_snap_max_yd: float = IDENTITY_SNAP_MAX_DIST_YD,
        delta_max_rotation_deg: float = DELTA_MAX_ROTATION_DEG,
        delta_max_translation_px: float = DELTA_MAX_TRANSLATION_PX,
        delta_max_scale_change: float = DELTA_MAX_SCALE_CHANGE,
        use_track_bank: bool = True,
        track_bank_coast: bool = True,
    ):
        self.device = device
        self.hash_thresh = hash_conf_thresh
        self.sideline_thresh = sideline_conf_thresh
        self.identity_snap_max_yd = identity_snap_max_yd
        self.delta_max_rotation_deg = delta_max_rotation_deg
        self.delta_max_translation_px = delta_max_translation_px
        self.delta_max_scale_change = delta_max_scale_change
        self.use_track_bank = use_track_bank
        self.track_bank_coast = track_bank_coast
        self.track_bank = KeypointTrackBank() if use_track_bank else None

        # Build a bare detector; we need raw heatmaps, so we use the model directly.
        self._detector = FieldKeypointDetector(
            weights_path, device=device, conf_thresh=0.1,  # permissive, we re-threshold
        )
        self._model = self._detector.model

        # Per-clip state
        self.H_prev: Optional[np.ndarray] = None
        self.H_inv_prev: Optional[np.ndarray] = None
        self.prev_pixel_pts_u: Optional[np.ndarray] = None
        self.prev_field_pts: Optional[np.ndarray] = None
        self.frame_shape: Optional[tuple] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.frame_count: int = 0

    # ── Inference ─────────────────────────────────────────────────────────

    def _run_hrnet(self, frame):
        """Return 2-channel sigmoided heatmaps (channel 0 = sideline, 1 = hash)."""
        import torch
        INPUT_H, INPUT_W = 512, 896
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self._detector.device)

        with torch.no_grad():
            logits = self._model(tensor)
            heatmaps = torch.sigmoid(logits[0]).cpu().numpy()
        return heatmaps

    def _detect(self, frame):
        """Run HRNet and extract paired yard-line groups."""
        heatmaps = self._run_hrnet(frame)
        h, w = frame.shape[:2]
        sideline_pxs, sideline_confs = _extract_peaks(
            heatmaps[0], self.sideline_thresh, (h, w),
        )
        hash_pxs, hash_confs = _extract_peaks(
            heatmaps[1], self.hash_thresh, (h, w),
        )
        groups, used_sideline = _build_yard_line_groups(
            hash_pxs, sideline_pxs, sideline_confs,
        )
        return {
            "heatmaps": heatmaps,
            "sideline_pxs": sideline_pxs,
            "sideline_confs": sideline_confs,
            "hash_pxs": hash_pxs,
            "hash_confs": hash_confs,
            "groups": groups,
        }

    # ── Distortion ───────────────────────────────────────────────────────

    def _compute_distortion(self, groups, frame_shape):
        """Solve k1, k2 from plumb-line fit using this frame's line groups."""
        side_row = [g["sideline"] for g in groups if g.get("sideline") is not None]
        far_row = [g["far_hash"] for g in groups if g.get("far_hash") is not None]
        near_row = [g["near_hash"] for g in groups if g.get("near_hash") is not None]
        line_sets = [
            np.asarray(x) for x in (side_row, far_row, near_row) if len(x) >= 3
        ]
        h, w = frame_shape
        focal_guess = float(max(h, w))
        if not line_sets:
            return 0.0, 0.0, focal_guess
        k1, k2 = calibrate_distortion_from_lines(line_sets, (h, w), focal_guess)
        if abs(k1) > 1.0 or abs(k2) > 1.0:
            k1, k2 = 0.0, 0.0  # sanity-reject runaway values
        return k1, k2, focal_guess

    # ── Identity assignment by projection + snap ─────────────────────────

    def _identify_via_projection(self, groups):
        """Use previous H to identify the current frame's groups' field coords.

        Returns (pixel_pts, field_pts, labels) arrays.
        Only includes detections whose projected field coords snap to a valid
        grid point within identity_snap_max_yd.
        """
        if self.H_prev is None:
            return np.zeros((0, 2)), np.zeros((0, 2)), []

        pixel_pts = []
        field_pts = []
        labels = []

        for g_idx, g in enumerate(groups):
            # Consider each of the up-to-three keypoints the group has
            candidates = []
            if g.get("near_hash") is not None:
                candidates.append((np.asarray(g["near_hash"]), 1, "near"))
            if g.get("far_hash") is not None:
                candidates.append((np.asarray(g["far_hash"]), 1, "far"))
            if g.get("sideline") is not None:
                candidates.append((np.asarray(g["sideline"]), 0, "side"))

            for pix, ch, tag in candidates:
                # Distortion-correct the pixel first
                pix_u = undistort_points(pix.reshape(1, 2),
                                          self.intrinsics)[0]
                fxy_proj = pixel_to_field(pix_u.reshape(1, 2),
                                            self.H_prev)[0]
                snapped, dist = _snap_to_field_grid(fxy_proj, ch)
                if snapped is None or dist > self.identity_snap_max_yd:
                    continue
                pixel_pts.append(pix.tolist())
                field_pts.append(snapped.tolist())
                labels.append(f"g{g.get('grid_pos', '?')}_{tag}")

        return (np.array(pixel_pts) if pixel_pts else np.zeros((0, 2)),
                np.array(field_pts) if field_pts else np.zeros((0, 2)),
                labels)

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
            # Clamp rotation to the cap and re-compose
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
                resets the tracker state.

        Returns:
            FrameResult dataclass.
        """
        if frame is None:
            raise ValueError("frame is None")

        self.frame_count += 1
        h, w = frame.shape[:2]

        # Run HRNet + grid solver
        det = self._detect(frame)
        groups = det["groups"]

        # Bootstrap or re-anchor path: requires anchor_ngs_x
        is_bootstrap = self.H_prev is None or anchor_ngs_x is not None
        if is_bootstrap and anchor_ngs_x is None:
            raise ValueError(
                "First frame (or re-anchor) requires anchor_ngs_x — "
                "the NGS x coordinate of grid_pos 0 (leftmost yard line)."
            )

        # Solve / update distortion (only on bootstrap, reused afterwards)
        if self.intrinsics is None:
            k1, k2, focal_guess = self._compute_distortion(groups, (h, w))
            self.intrinsics = CameraIntrinsics(
                fx=focal_guess, fy=focal_guess, cx=w / 2.0, cy=h / 2.0,
                k1=k1, k2=k2,
            )
            self.frame_shape = (h, w)

        # Build identified correspondences
        if is_bootstrap:
            pixel_pts, field_pts, labels = _groups_to_correspondences(
                groups, anchor_ngs_x, frame_shape=(h, w),
            )
            pixel_pts_u = (undistort_points(pixel_pts, self.intrinsics)
                           if len(pixel_pts) else np.zeros((0, 2)))
        else:
            pixel_pts, field_pts, labels = self._identify_via_projection(groups)
            pixel_pts_u = (undistort_points(pixel_pts, self.intrinsics)
                           if len(pixel_pts) else np.zeros((0, 2)))

        # Build correspondence dicts (used by track bank). Infer kind from
        # the snapped field y — sidelines sit at 0 or FIELD_WIDTH, hashes at
        # the hash-row y values.
        def _infer_kind(fy):
            if fy < 1.0:
                return "sideline_near"
            if fy > FIELD_WIDTH - 1.0:
                return "sideline_far"
            return "far_hash" if abs(fy - HASH_Y_FAR) < abs(fy - HASH_Y_NEAR) else "near_hash"

        current_corr = [
            {"pixel_u": pixel_pts_u[i], "field": field_pts[i],
             "kind": _infer_kind(float(field_pts[i][1]))}
            for i in range(len(pixel_pts_u))
        ]

        # Optionally coast unobserved tracks into the correspondence set.
        # This both (a) densifies sparse-detection frames and (b) creates an
        # opportunity for the track bank to reject inconsistent fresh detections.
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
        # Require at least 4 REAL current-frame observations before doing full.
        # If we only have coasted points, the solve is trivial (they all
        # came from H_prev) — should be treated as carry, not full.
        n_real = len(current_corr)
        if n >= 4 and n_real >= 4:
            full = self._solve_full_homography(combined_pixel_u, combined_field)
            # Sanity check: on non-bootstrap frames, verify the new H doesn't
            # disagree hugely with H_prev. If it does, the identity assignment
            # was probably wrong (camera moved too far between frames).
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
                    full = None  # track bank says this H is inconsistent

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
        # Only update on successful full/delta; carry-mode means we don't
        # trust the current frame's identities.
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
                "n_sideline_detections": len(det["sideline_pxs"]),
                "n_hash_detections": len(det["hash_pxs"]),
                "n_yard_line_groups": len(groups),
                "labels_used": labels if is_bootstrap else [""] * n,
                "n_current_corr": len(current_corr),
                "n_coasted_corr": len(coasted),
                "track_bank": track_bank_stats,
                "track_bank_validated": validated_by_bank,
                "track_bank_validation": bank_diag,
            },
        )
