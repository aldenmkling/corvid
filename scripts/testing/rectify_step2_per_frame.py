#!/usr/bin/env python3
"""Step 2 of the new rectify workflow: per-frame homography on a clip
using the unified UNet + line-fit hash detection.

Pipeline per frame:
  1. Unified UNet → yard / side / hash masks.
  2. Bootstrap: on frame 0, calibrate k1 from line pixels (cached
     thereafter). Build the YardlineTracker initial state.
  3. Undistort yardline + sideline groups, fit linear forms.
  4. YardlineTracker assigns g-index per yardline (identity only — no
     parameter smoothing).
  5. Hash mask pixels → undistort → sequential RANSAC (2 lines).
     Single-line failsafe handled by HashRowTracker (matches the
     detected line to the previous frame's near or far).
  6. Build correspondences: yardline × hash-row intersection +
     sideline × yardline intersection. Drop anything outside
     NGS x ∈ [10, 110].
  7. cv2.findHomography RANSAC → H per frame.
  8. Render: undistorted frame + projected field grid overlay.

Output: an MP4 with the overlay drawn over each frame.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/testing"))

from src.homography.grid_solver_v2 import (
    group_yardline_pixels_cc,
)
from src.homography.distortion import CameraIntrinsics, undistort_points
from src.homography.field_model import HASH_Y_NEAR, HASH_Y_FAR, FIELD_WIDTH

import subprocess
import shutil
import segmentation_models_pytorch as smp
from rebuild_full_clip_viz import YardlineTracker, group_sideline_pixels_cc as group_sideline_pixels
from rebuild_step4_hashes_v2 import (
    total_mse, ransac_line,
    YARD_THRESH, SIDE_THRESH, HASH_THRESH, MAX_HASH_PIXELS,
)
from rebuild_step8_homography import HomographyTrackerLite, detect_lost

# Two-specialist setup.
LINE_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_line_stage2_last.pth")
HASH_WEIGHTS = os.path.join(PROJECT_ROOT, "models/unet_hash_round3_last.pth")
UNET_INPUT_H, UNET_INPUT_W = 512, 896
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(frame_bgr: np.ndarray, grayscale: bool):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (UNET_INPUT_W, UNET_INPUT_H))
    if grayscale:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([g, g, g], axis=-1)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


_MODEL_CACHE = {}


def _load_smp_unet(weights: str, classes: int, device: torch.device):
    key = (weights, classes, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = smp.Unet(encoder_name="mit_b0", encoder_weights=None,
                  in_channels=3, classes=classes, activation=None)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.to(device).eval()
    _MODEL_CACHE[key] = m
    return m


@torch.no_grad()
def run_specialists(frame: np.ndarray, line_weights: str, hash_weights: str,
                     device_str: str = "mps"):
    """Two forward passes: line UNet (grayscale, 2ch) + hash UNet (RGB, 1ch).
    Returns (yard_mask, side_mask, hash_mask) as binary masks at frame resolution.
    """
    device = torch.device(device_str)
    line_model = _load_smp_unet(line_weights, classes=2, device=device)
    hash_model = _load_smp_unet(hash_weights, classes=1, device=device)
    h0, w0 = frame.shape[:2]

    # Line: grayscale-replicated input, 2-channel output
    t_line = _preprocess(frame, grayscale=True).to(device)
    p_line = torch.sigmoid(line_model(t_line))[0].cpu().numpy()
    yard = (p_line[0] > YARD_THRESH).astype(np.uint8)
    side = (p_line[1] > SIDE_THRESH).astype(np.uint8)

    # Hash: RGB input, 1-channel output
    t_hash = _preprocess(frame, grayscale=False).to(device)
    p_hash = torch.sigmoid(hash_model(t_hash))[0, 0].cpu().numpy()
    hash_ = (p_hash > HASH_THRESH).astype(np.uint8)

    yard = cv2.resize(yard, (w0, h0), interpolation=cv2.INTER_NEAREST)
    side = cv2.resize(side, (w0, h0), interpolation=cv2.INTER_NEAREST)
    hash_ = cv2.resize(hash_, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return yard, side, hash_


# ── Field-coordinate constants ─────────────────────────────────────────────
G0_NGS_X = 20.0          # leftmost yardline (g=0) → NGS x in yards (user input)
YD_PER_GRID = 5.0
NGS_X_LEFT_GOAL = 10.0   # left goal line in NGS coords
NGS_X_RIGHT_GOAL = 110.0
G_MIN = int((NGS_X_LEFT_GOAL - G0_NGS_X) / YD_PER_GRID)    # = -2
G_MAX = int((NGS_X_RIGHT_GOAL - G0_NGS_X) / YD_PER_GRID)   # = +18

# Hash row tracker thresholds (in pixels at image-x = w/2)
HASH_MATCH_TOL_PX = 80.0
HASH_MIN_ROW_SEP_PX = 30.0
RANSAC_REPROJ_PX = 4.0

# Rectified canvas — display spans the whole field of play including
# endzones (NGS 0-120). OOB filter for correspondences stays at NGS 10-110
# (only the actual yardlines).
DISPLAY_X_LEFT = 0.0
DISPLAY_X_RIGHT = 120.0
PX_PER_YARD = 10
RECT_W = int((DISPLAY_X_RIGHT - DISPLAY_X_LEFT) * PX_PER_YARD)   # 1200
RECT_H = int(FIELD_WIDTH * PX_PER_YARD)                            # 533


def build_rectify_warp(H_img_to_field):
    """Compose H (img → NGS yards) with translate+scale to produce a
    direct warp matrix from undistorted image → rectified canvas pixels.
    """
    T = np.array([[1, 0, -DISPLAY_X_LEFT],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float64)
    S = np.array([[PX_PER_YARD, 0, 0],
                  [0, PX_PER_YARD, 0],
                  [0, 0,           1]], dtype=np.float64)
    return S @ T @ H_img_to_field


def render_rectified_frame(frame_u, H):
    """Warp undistorted frame into the rectified field canvas."""
    canvas = np.zeros((RECT_H, RECT_W, 3), dtype=np.uint8)
    if H is None:
        cv2.putText(canvas, "no H", (RECT_W // 2 - 30, RECT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        return canvas
    Hw = build_rectify_warp(H)
    canvas = cv2.warpPerspective(frame_u, Hw, (RECT_W, RECT_H))
    # Overlay: yardlines + sidelines + hash rows on the canvas as ground
    # truth (so it's obvious if rectification is off). Yardlines only at
    # NGS 10-110 (the playing field). Goallines highlighted thicker.
    for x_yd in range(int(NGS_X_LEFT_GOAL), int(NGS_X_RIGHT_GOAL) + 1, 5):
        x_px = int((x_yd - DISPLAY_X_LEFT) * PX_PER_YARD)
        thick = 2 if x_yd in (NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL) else 1
        cv2.line(canvas, (x_px, 0), (x_px, RECT_H - 1),
                 (0, 255, 0), thick, cv2.LINE_AA)
    # Endzone back-of-line markers (faint, dotted style via short dashes)
    for x_yd in (0.0, 120.0):
        x_px = int((x_yd - DISPLAY_X_LEFT) * PX_PER_YARD)
        if 0 <= x_px < RECT_W:
            cv2.line(canvas, (x_px, 0), (x_px, RECT_H - 1),
                     (120, 200, 120), 1, cv2.LINE_AA)
    for y_yd, color in [(0, (220, 220, 0)),
                          (FIELD_WIDTH, (220, 220, 0)),
                          (HASH_Y_NEAR, (255, 80, 80)),
                          (HASH_Y_FAR, (60, 60, 255))]:
        y_px = int(y_yd * PX_PER_YARD)
        cv2.line(canvas, (0, y_px), (RECT_W - 1, y_px),
                 color, 1, cv2.LINE_AA)
    # Flip vertically so NGS y=0 (near sideline) is at the BOTTOM of the
    # rectified canvas, matching the broadcast view orientation.
    canvas = cv2.flip(canvas, 0)
    return canvas


def stack_frames(top, bot):
    """Resize bot to top's width, stack vertically. Returns combined."""
    if bot.shape[1] != top.shape[1]:
        scale = top.shape[1] / bot.shape[1]
        bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * scale)))
    return np.vstack([top, bot])


# ── HashRowTracker (identity only, with single-line failsafe) ─────────────
class HashRowTracker:
    """Stores last frame's (m, c) for `far` and `near` rows; uses them to
    label new detections. No parameter smoothing — only identity carry.
    """

    def __init__(self, image_w: int,
                 match_tol_px: float = HASH_MATCH_TOL_PX,
                 min_row_sep_px: float = HASH_MIN_ROW_SEP_PX):
        self.w = image_w
        self.match_tol = match_tol_px
        self.min_sep = min_row_sep_px
        self.far = None      # (m, c) or None
        self.near = None

    def _y_at_center(self, line):
        m, c = line
        return m * (self.w / 2.0) + c

    def observe(self, lines: list[tuple[float, float]]):
        """lines = list of (m, c) from sequential RANSAC, ordered desc by
        inlier count.

        Returns dict: {'far': (m,c)|None, 'near': (m,c)|None, 'state': str}.
        Only freshly-detected lines are returned this frame; we don't
        carry the missing row's previous (m, c) into the correspondences,
        since the camera may have moved.
        """
        if not lines:
            return {'far': None, 'near': None, 'state': 'no_lines'}

        if len(lines) == 1:
            new = lines[0]
            y_new = self._y_at_center(new)
            if self.far is None and self.near is None:
                # No tracker memory yet — ambiguous, skip this frame's hashes.
                return {'far': None, 'near': None, 'state': 'bootstrap_single'}
            # Match to the closer of (far, near)
            best_label, best_d = None, float('inf')
            for label, prev in (('far', self.far), ('near', self.near)):
                if prev is None: continue
                d = abs(y_new - self._y_at_center(prev))
                if d < best_d:
                    best_d, best_label = d, label
            if best_d > self.match_tol:
                return {'far': None, 'near': None, 'state': 'single_no_match'}
            # Accept and update only the matched row.
            if best_label == 'far':
                self.far = new
                return {'far': new, 'near': None, 'state': 'single_far'}
            else:
                self.near = new
                return {'far': None, 'near': new, 'state': 'single_near'}

        # ≥ 2 lines: take top two, then assign by y-at-center.
        l1, l2 = lines[0], lines[1]
        y1, y2 = self._y_at_center(l1), self._y_at_center(l2)
        if abs(y1 - y2) < self.min_sep:
            # Too close — both fit the same row. Fall back to single.
            return self.observe([l1])
        far, near = (l1, l2) if y1 < y2 else (l2, l1)

        # Continuity check vs tracker.
        if self.far is not None and self.near is not None:
            d_far = abs(self._y_at_center(far) - self._y_at_center(self.far))
            d_near = abs(self._y_at_center(near) - self._y_at_center(self.near))
            if max(d_far, d_near) > self.match_tol:
                # Possible camera cut — accept new labels but flag.
                state = 'both_after_cut'
            else:
                state = 'both'
        else:
            state = 'both_bootstrap'

        self.far = far
        self.near = near
        return {'far': far, 'near': near, 'state': state}


# ── Per-frame helpers ─────────────────────────────────────────────────────
def fit_yardline_undistorted(pixels: np.ndarray, intr: CameraIntrinsics):
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    ys, xs = pts_u[:, 1], pts_u[:, 0]
    b, a = np.polyfit(ys, xs, 1)
    return {'a': float(a), 'b': float(b),
            'ymin': float(ys.min()), 'ymax': float(ys.max())}


def fit_sideline_undistorted(pixels: np.ndarray, intr: CameraIntrinsics):
    pts_u = undistort_points(pixels.astype(np.float64), intr)
    xs, ys = pts_u[:, 0], pts_u[:, 1]
    b, a = np.polyfit(xs, ys, 1)        # y = a + b·x
    return {'a': float(a), 'b': float(b),
            'xmin': float(xs.min()), 'xmax': float(xs.max())}


def detect_hash_rows(hash_mask: np.ndarray, intr: CameraIntrinsics,
                     min_pixels: int = 30):
    """Returns list of (m, c) — up to 2 lines from sequential RANSAC."""
    ys, xs = np.where(hash_mask > 0)
    if len(xs) < min_pixels:
        return []
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) > MAX_HASH_PIXELS:
        idx = np.random.RandomState(0).choice(len(pts),
                                                MAX_HASH_PIXELS, replace=False)
        pts = pts[idx]
    pts_u = undistort_points(pts, intr)
    out = []
    m1, c1, in1 = ransac_line(pts_u, inlier_dist=2.0, min_inliers=20)
    if m1 is None:
        return []
    out.append((m1, c1))
    rem = pts_u[~in1]
    if len(rem) >= 20:
        m2, c2, in2 = ransac_line(rem, inlier_dist=2.0, min_inliers=20)
        if m2 is not None:
            out.append((m2, c2))
    return out


def yardline_x_at_y(fit, y):
    return fit['a'] + fit['b'] * y


def sideline_y_at_x(fit, x):
    return fit['a'] + fit['b'] * x


def line_intersect_yardline_row(yl_fit, row):
    """yl: x = a + b·y;  row: y = m·x + c.  Returns (x, y) or None."""
    a, b = yl_fit['a'], yl_fit['b']
    m, c = row
    denom = 1.0 - b * m
    if abs(denom) < 1e-6:
        return None
    y = (c + a * m) / denom
    x = a + b * y
    return float(x), float(y)


def line_intersect_yardline_sideline(yl_fit, sl_fit):
    """yl: x = a + b·y;  sl: y = a + b·x.
    Substitute: x = a_yl + b_yl·(a_sl + b_sl·x) → x(1 − b_yl·b_sl) = a_yl + b_yl·a_sl.
    """
    a_yl, b_yl = yl_fit['a'], yl_fit['b']
    a_sl, b_sl = sl_fit['a'], sl_fit['b']
    denom = 1.0 - b_yl * b_sl
    if abs(denom) < 1e-6:
        return None
    x = (a_yl + b_yl * a_sl) / denom
    y = a_sl + b_sl * x
    return float(x), float(y)


# ── Render: project field grid into image ──────────────────────────────────
def project_field_grid(canvas, H_field_to_img, w, h):
    """H_field_to_img maps (x_field, y_field) → (x_img, y_img). Draw yard
    lines, sidelines, hash rows in field coords and project."""
    def proj(pts_field):
        homo = np.column_stack([pts_field, np.ones(len(pts_field))])
        out = (H_field_to_img @ homo.T).T
        out = out[:, :2] / out[:, 2:3]
        return out

    # Yard lines x = 10..110 every 5 yd
    for x_f in np.arange(NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL + 0.1, YD_PER_GRID):
        field = np.array([[x_f, 0], [x_f, FIELD_WIDTH]])
        img_pts = proj(field).astype(np.int32)
        cv2.line(canvas, tuple(img_pts[0]), tuple(img_pts[1]),
                 (200, 200, 200), 1, cv2.LINE_AA)

    # Sidelines (y=0 and y=FIELD_WIDTH for x in [10, 110])
    for y_f in (0, FIELD_WIDTH):
        field = np.array([[NGS_X_LEFT_GOAL, y_f], [NGS_X_RIGHT_GOAL, y_f]])
        img_pts = proj(field).astype(np.int32)
        cv2.line(canvas, tuple(img_pts[0]), tuple(img_pts[1]),
                 (220, 220, 0), 2, cv2.LINE_AA)

    # Hash rows (y=NEAR / FAR for x in [10, 110])
    for y_f, color in [(HASH_Y_NEAR, (255, 80, 80)),
                         (HASH_Y_FAR, (60, 60, 255))]:
        field = np.array([[NGS_X_LEFT_GOAL, y_f], [NGS_X_RIGHT_GOAL, y_f]])
        img_pts = proj(field).astype(np.int32)
        cv2.line(canvas, tuple(img_pts[0]), tuple(img_pts[1]),
                 color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(
        PROJECT_ROOT, "videos/clips/2019102712/play_046/sideline.mp4"))
    ap.add_argument("--g0-ngs-x", type=float, default=G0_NGS_X)
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output/rebuild/rectify_step2_overlay.mp4"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    g0_x = args.g0_ngs_x
    g_min = int((NGS_X_LEFT_GOAL - g0_x) / YD_PER_GRID)
    g_max = int((NGS_X_RIGHT_GOAL - g0_x) / YD_PER_GRID)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"  failed to open {args.clip}"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    ok, frame0 = cap.read()
    if not ok:
        print(f"  empty clip"); return
    h, w = frame0.shape[:2]
    focal = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    print(f"  clip: {os.path.relpath(args.clip, PROJECT_ROOT)}  "
          f"{w}x{h}  {n_frames} frames  fps={fps:.1f}")

    # ── Bootstrap on frame 0 ────────────────────────────────────────────
    yard0, side0, hash0 = run_specialists(
        frame0, LINE_WEIGHTS, HASH_WEIGHTS, args.device)
    yl0 = group_yardline_pixels_cc(yard0)
    sl0 = group_sideline_pixels(side0)
    line_pts = [g.pixels for g in yl0] + [g.pixels for g in sl0]
    line_kinds = ["yardline"] * len(yl0) + ["sideline"] * len(sl0)
    line_pts_sub = [p[::max(1, len(p) // 50)] for p in line_pts]
    res = minimize_scalar(
        lambda k1: total_mse(line_pts_sub, line_kinds,
                              CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy,
                                                k1=float(k1), k2=0.0)),
        bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-4},
    )
    k1 = float(res.x)
    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, k1=k1, k2=0.0)
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0, 0, 0], dtype=np.float64)
    print(f"  bootstrap k1 = {k1:+.4f}")

    yl_tracker = YardlineTracker(g_min=g_min, g_max=g_max, frame_h=h)
    hash_tracker = HashRowTracker(image_w=w)
    h_tracker = HomographyTrackerLite()
    methods = []
    method_counts = {"full": 0, "delta": 0, "carry": 0, "none": 0}

    fits_yl0 = [fit_yardline_undistorted(g.pixels, intr) for g in yl0]
    yl_tracker.init_from(fits_yl0, cy)

    # ── PASS 1: compute everything per frame (no rendering) ────────────
    print("  pass 1: computing per-frame H ...")
    frame_meta = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_done = 0

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n_done >= args.max_frames):
            break

        yard, side, hash_ = run_specialists(
            frame, LINE_WEIGHTS, HASH_WEIGHTS, args.device)
        yl = group_yardline_pixels_cc(yard)
        sl = group_sideline_pixels(side)
        if abs(k1) > 1e-6:
            yard_u_m = cv2.undistort(yard, K, dist)
            side_u_m = cv2.undistort(side, K, dist)
            hash_u_m = cv2.undistort(hash_, K, dist)
        else:
            yard_u_m, side_u_m, hash_u_m = yard, side, hash_

        fits_yl = [fit_yardline_undistorted(g.pixels, intr) for g in yl]
        fits_sl = [fit_sideline_undistorted(g.pixels, intr) for g in sl]

        if n_done == 0:
            init = yl_tracker.init_from(fits_yl, cy)
            if init is None:
                fits_kept, g_index = [], np.array([], dtype=int)
            else:
                fits_kept, g_index, _ = init
        else:
            fits_kept, g_index, _, _ = yl_tracker.update(fits_yl, cy)

        rows_raw = detect_hash_rows(hash_, intr)
        rows = hash_tracker.observe(rows_raw)

        # Build correspondences
        corrs = []
        for i, fit in enumerate(fits_kept):
            g = int(g_index[i])
            x_ngs = g0_x + YD_PER_GRID * g
            if not (NGS_X_LEFT_GOAL <= x_ngs <= NGS_X_RIGHT_GOAL):
                continue
            for label, row in (("far", rows['far']), ("near", rows['near'])):
                if row is None: continue
                pt = line_intersect_yardline_row(fit, row)
                if pt is None: continue
                px, py = pt
                if not (0 <= px < w and 0 <= py < h): continue
                y_field = HASH_Y_FAR if label == "far" else HASH_Y_NEAR
                corrs.append({
                    "pixel_u": np.array([px, py], dtype=np.float64),
                    "field": np.array([x_ngs, y_field], dtype=np.float64),
                    "kind": f"{label}_hash",
                    "label": f"{label}_hash@g{g:+d}",
                })
        for sf in fits_sl:
            for i, fit in enumerate(fits_kept):
                g = int(g_index[i])
                x_ngs = g0_x + YD_PER_GRID * g
                if not (NGS_X_LEFT_GOAL <= x_ngs <= NGS_X_RIGHT_GOAL):
                    continue
                pt = line_intersect_yardline_sideline(fit, sf)
                if pt is None: continue
                px, py = pt
                if not (0 <= px < w and 0 <= py < h): continue
                y_at_center = sideline_y_at_x(sf, w / 2)
                y_field = 0.0 if y_at_center > h / 2 else FIELD_WIDTH
                sl_label = "near" if y_field == 0.0 else "far"
                corrs.append({
                    "pixel_u": np.array([px, py], dtype=np.float64),
                    "field": np.array([x_ngs, y_field], dtype=np.float64),
                    "kind": f"sideline_{sl_label}",
                    "label": f"{sl_label}sl×g{g:+d}",
                })

        r = h_tracker.update(corrs, frame_idx=n_done)
        method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1
        methods.append(r["method"])

        frame_meta.append({
            "yard_bytes": cv2.imencode(".png", yard_u_m)[1].tobytes(),
            "side_bytes": cv2.imencode(".png", side_u_m)[1].tobytes(),
            "hash_bytes": cv2.imencode(".png", hash_u_m)[1].tobytes(),
            "fits_kept": fits_kept,
            "g_index": g_index,
            "fits_sl": fits_sl,
            "rows": rows,
            "corrs": corrs,
            "H_raw": r["H"],
            "method": r["method"],
            "n_inliers": r["n_inliers"] or 0,
            "n_corrs": r["n_corrs"],
            "hash_state": rows.get("state", "?"),
        })
        n_done += 1
        if n_done % 30 == 0:
            print(f"  pass1 frame {n_done}/{n_frames}  "
                  f"full={method_counts.get('full', 0)}  "
                  f"delta={method_counts.get('delta', 0)}  "
                  f"carry={method_counts.get('carry', 0)}")
    cap.release()

    # ── Detect lost + smooth Hs ─────────────────────────────────────────
    lost_from = detect_lost(methods, min_sustained_loss=3)
    valid_until = lost_from if lost_from is not None else n_done
    if lost_from is not None:
        print(f"  clip LOST from frame {lost_from} "
              f"({n_done - lost_from} frames; \u22653 consecutive carries)")

    # Smooth H matrices over the valid range (Savitzky-Golay).
    for mm in frame_meta:
        mm["H"] = mm["H_raw"]
    first_ok = next((i for i, mm in enumerate(frame_meta)
                      if mm["H_raw"] is not None), None)
    if first_ok is not None and first_ok < valid_until:
        Hs_in = [frame_meta[i]["H_raw"] for i in range(first_ok, valid_until)]
        from rebuild_step8_homography import smooth_hs
        Hs_out = smooth_hs(Hs_in, window=7, poly=2)
        for k_off, hs in enumerate(Hs_out):
            frame_meta[first_ok + k_off]["H"] = hs
        print(f"  smoothed Hs over frames {first_ok}-{valid_until-1} "
              f"(SG window=7, poly=2)")

    # ── PASS 2: render with smoothed Hs ─────────────────────────────────
    print("  pass 2: rendering ...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    bot_h = int(RECT_H * (w / RECT_W))
    out_w, out_h = w, h + bot_h
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))
    print(f"  output: {out_w}x{out_h}  (top {w}x{h}  bot {w}x{bot_h})")

    cap2 = cv2.VideoCapture(args.clip)
    n_written = 0
    n_h_ok = 0
    sum_inlier_frac = 0.0
    for fi, mm in enumerate(frame_meta):
        if fi >= valid_until:
            break
        ok, frame = cap2.read()
        if not ok: break
        frame_u = cv2.undistort(frame, K, dist) if abs(k1) > 1e-6 else frame.copy()
        yard_u_m = cv2.imdecode(np.frombuffer(mm["yard_bytes"], np.uint8), 0)
        side_u_m = cv2.imdecode(np.frombuffer(mm["side_bytes"], np.uint8), 0)
        hash_u_m = cv2.imdecode(np.frombuffer(mm["hash_bytes"], np.uint8), 0)
        H = mm["H"]
        method = mm["method"]
        n_inliers = mm["n_inliers"]
        if method == "full" and H is not None:
            n_h_ok += 1
            sum_inlier_frac += n_inliers / max(mm["n_corrs"], 1)

        # Top canvas: dim source + masks + fits
        canvas = (frame_u * 0.25).astype(np.uint8)
        ov = canvas.copy()
        ov[yard_u_m > 0] = (60, 60, 230)
        ov[side_u_m > 0] = (60, 230, 60)
        ov[hash_u_m > 0] = (230, 60, 60)
        canvas = cv2.addWeighted(ov, 0.7, canvas, 0.3, 0)

        for i, fit in enumerate(mm["fits_kept"]):
            a, b = fit['a'], fit['b']
            ymin, ymax = fit['ymin'], fit['ymax']
            ys_l = np.linspace(ymin, ymax, 200)
            xs_l = a + b * ys_l
            g = int(mm["g_index"][i])
            x_ngs = g0_x + YD_PER_GRID * g
            in_field = NGS_X_LEFT_GOAL <= x_ngs <= NGS_X_RIGHT_GOAL
            col = (200, 200, 200) if in_field else (80, 80, 80)
            cv2.polylines(canvas,
                          [np.stack([xs_l, ys_l], axis=1).astype(np.int32)],
                          False, col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"g{g:+d}",
                        (int(a + b * ymin) + 4, int(ymin) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        for sf in mm["fits_sl"]:
            a, b = sf['a'], sf['b']
            xmin, xmax = sf['xmin'], sf['xmax']
            xs_l = np.linspace(xmin, xmax, 200)
            ys_l = a + b * xs_l
            cv2.polylines(canvas,
                          [np.stack([xs_l, ys_l], axis=1).astype(np.int32)],
                          False, (220, 220, 0), 2, cv2.LINE_AA)
        for label, row in (("far", mm["rows"]['far']),
                             ("near", mm["rows"]['near'])):
            if row is None: continue
            mr, cr = row
            xs_l = np.linspace(0, w - 1, 200)
            ys_l = mr * xs_l + cr
            col = (60, 60, 255) if label == "far" else (255, 80, 80)
            cv2.polylines(canvas,
                          [np.stack([xs_l, ys_l], axis=1).astype(np.int32)],
                          False, col, 1, cv2.LINE_AA)
        if H is not None:
            try:
                project_field_grid(canvas, np.linalg.inv(H), w, h)
            except np.linalg.LinAlgError:
                pass
        for c in mm["corrs"]:
            px_pt = c["pixel_u"]
            cv2.circle(canvas,
                        (int(round(px_pt[0])), int(round(px_pt[1]))),
                        3, (60, 220, 60), -1)
        method_color = {"full":  (100, 255, 100),
                         "delta": (0, 200, 255),
                         "carry": (60, 60, 255),
                         "none":  (180, 180, 180)}.get(method, (255, 255, 255))
        cv2.putText(canvas,
                    f"frame {fi:4d}/{n_done}  {method}  "
                    f"yl={len(mm['fits_kept'])}  corrs={mm['n_corrs']}  "
                    f"inliers={n_inliers}  rows={mm['hash_state']}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    method_color, 2, cv2.LINE_AA)

        rectified = render_rectified_frame(frame_u, H)
        out_frame = stack_frames(canvas, rectified)
        writer.write(out_frame)
        n_written += 1
        if n_written % 30 == 0:
            print(f"  pass2 frame {n_written}/{valid_until}")
    cap2.release()
    writer.release()

    print(f"  done: pass1 {n_done} frames, pass2 wrote {n_written}")
    print(f"  methods: full={method_counts.get('full', 0)}  "
          f"delta={method_counts.get('delta', 0)}  "
          f"carry={method_counts.get('carry', 0)}  "
          f"none={method_counts.get('none', 0)}")
    print(f"  full-H frames: {n_h_ok}  "
          f"avg inlier frac {sum_inlier_frac/max(n_h_ok,1):.2f}")
    print(f"  out: {args.out}")


if __name__ == "__main__":
    main()
