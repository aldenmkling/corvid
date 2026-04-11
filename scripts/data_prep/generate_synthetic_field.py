#!/usr/bin/env python3
"""
Generate synthetic training data for the HRNet field keypoint detector.

Creates a high-resolution top-down field template with all NFL markings,
then warps it through random camera poses to produce 1280×720 training
frames with perfect keypoint annotations in COCO keypoints format.

Camera model:
  - Fixed elevated position on the near sideline, ~midfield
  - Pure rotation (pan) to follow play, no lateral translation
  - Zoom variation (FOV changes)
  - ~15% of frames have sharper oblique angles (end zone plays)

Usage:
    python scripts/generate_synthetic_field.py --num-frames 5000 --output data/field_keypoints/synthetic
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.homography.keypoint_schema import (
    KEYPOINTS, NUM_KEYPOINTS, FIELD_COORDS, KEYPOINT_NAMES,
    KEYPOINTS_BY_TYPE, NUM_IDENTITY_KEYPOINTS,
    get_visible_keypoints,
)
from src.homography.field_model import (
    FIELD_LENGTH, FIELD_WIDTH, GOAL_LINE_LEFT, GOAL_LINE_RIGHT,
    YARD_LINE_POSITIONS, TEN_YARD_POSITIONS,
    HASH_Y_NEAR, HASH_Y_FAR, NUMBER_Y_NEAR, NUMBER_Y_FAR,
)


# ── Field template constants ────────────────────────────────────────────────

# Template resolution: ~40 pixels per yard
PX_PER_YARD = 40
TEMPLATE_W = int(FIELD_LENGTH * PX_PER_YARD)   # 4800
TEMPLATE_H = int(FIELD_WIDTH * PX_PER_YARD)    # 2133

# Line widths in pixels (4 inches = 1/9 yard)
LINE_W = max(2, int(PX_PER_YARD / 9))  # ~4px
# Sidelines: 6 feet wide = 2 yards. At 40px/yard = 80px, scale down but keep prominent
SIDELINE_W = 32  # clearly thicker than yard lines (~4px), includes back of endzone

# Hash mark dimensions: 2 feet long = 2/3 yard, 4 inches wide
HASH_LEN = max(4, int(PX_PER_YARD * 2 / 3))  # ~27px
HASH_W = LINE_W

# Number dimensions: 6 feet tall = 2 yards
NUMBER_H = int(PX_PER_YARD * 2)  # ~80px

# Output frame size
FRAME_W = 1280
FRAME_H = 720


# ── Field template rendering ────────────────────────────────────────────────

def _yard_to_px(x_yard: float, y_yard: float) -> tuple[int, int]:
    """Convert field yards to template pixel coordinates."""
    px = int(x_yard * PX_PER_YARD)
    py = int(y_yard * PX_PER_YARD)
    return px, py


def _paste_canvas(img: np.ndarray, canvas: np.ndarray, cx: int, cy: int):
    """Paste a single-channel canvas onto img centered at (cx, cy)."""
    ch, cw = canvas.shape[:2]
    y1 = cy - ch // 2
    x1 = cx - cw // 2
    y2, x2 = y1 + ch, x1 + cw

    sy1 = max(0, -y1)
    sx1 = max(0, -x1)
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img.shape[0], y2)
    x2 = min(img.shape[1], x2)
    sy2 = sy1 + (y2 - y1)
    sx2 = sx1 + (x2 - x1)

    if y2 > y1 and x2 > x1:
        mask = canvas[sy1:sy2, sx1:sx2]
        img[y1:y2, x1:x2] = np.where(mask[..., None] > 0, 255, img[y1:y2, x1:x2])


# Load font once at module level
# Prefer Clarendon Bold if available, fall back to Georgia Bold
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FIELD_FONT_SIZE = 115  # pixels — renders slightly under 2 yards, less heavy than bold
_FIELD_FONT = None
for font_path in [
    os.path.join(_PROJECT_ROOT, "fonts", "Clarendon Regular.otf"),
    os.path.join(_PROJECT_ROOT, "fonts", "Clarendon Bold.otf"),
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
]:
    try:
        _FIELD_FONT = ImageFont.truetype(font_path, _FIELD_FONT_SIZE)
        break
    except (OSError, IOError):
        continue
if _FIELD_FONT is None:
    _FIELD_FONT = ImageFont.load_default()


def _render_digit_pil(digit: str) -> np.ndarray:
    """Render a single digit using PIL with the field font. Returns grayscale canvas."""
    # Render with generous padding, then crop to tight bbox
    canvas_size = _FIELD_FONT_SIZE + 20
    pil_img = Image.new("L", (canvas_size, canvas_size * 2), 0)
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), digit, fill=255, font=_FIELD_FONT)
    arr = np.array(pil_img)
    # Tight crop
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _draw_number(img: np.ndarray, number: int, cx: int, cy: int,
                 side: str = "near"):
    """Draw a field number straddling the yard line at cx.

    Each digit is rendered with PIL (Georgia Bold), placed evenly on either
    side of the yard line with a consistent gap from the line.

    Near-side numbers: vertical flip so they face upward (toward y=0).
    Far-side numbers: horizontal flip so readable from far sideline.
    """
    digits = str(number)
    gap = int(PX_PER_YARD * 0.35)  # gap between each digit's inner edge and the yard line

    # Render each digit
    digit_imgs = [_render_digit_pil(d) for d in digits]

    # For far side: the number is horizontally flipped, which means
    # the digit ORDER must also reverse. "50" viewed from the far sideline
    # has "5" on the right and "0" on the left (from near-side perspective).
    if side == "far" and len(digit_imgs) == 2:
        digit_imgs = digit_imgs[::-1]

    for i, dimg in enumerate(digit_imgs):
        dh, dw = dimg.shape[:2]
        if len(digit_imgs) == 2:
            if i == 0:
                digit_cx = cx - gap - dw // 2  # left of yard line
            else:
                digit_cx = cx + gap + dw // 2  # right of yard line
        else:
            digit_cx = cx

        c = dimg.copy()
        if side == "near":
            c = cv2.flip(c, 0)
        elif side == "far":
            c = cv2.flip(c, 1)

        _paste_canvas(img, c, digit_cx, cy)


def _draw_arrow(img: np.ndarray, cx: int, cy: int, pointing_left: bool,
                size: int = 18):
    """Draw a directional arrow (triangle) near the top of a number.

    Sized to be clearly visible but not huge — roughly 1/3 the height of a number.
    """
    half = size // 2
    if pointing_left:
        pts = np.array([
            [cx - size, cy],
            [cx + half, cy - half],
            [cx + half, cy + half],
        ])
    else:
        pts = np.array([
            [cx + size, cy],
            [cx - half, cy - half],
            [cx - half, cy + half],
        ])
    cv2.fillPoly(img, [pts], (255, 255, 255))


def render_field_template(rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a top-down NFL field template image.

    Args:
        rng: random generator for endzone color/text variation.
             If None, uses defaults.

    Returns BGR image of shape (TEMPLATE_H, TEMPLATE_W, 3).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Base: green grass — vary the base tone per template
    img = np.zeros((TEMPLATE_H, TEMPLATE_W, 3), dtype=np.uint8)
    base_green = np.array([
        35 + rng.integers(-5, 10),   # B
        110 + rng.integers(-15, 20),  # G
        40 + rng.integers(-5, 10),   # R
    ], dtype=np.uint8)

    # Line paint color — not always pure white, can be slightly off-white
    white_val = 240 + rng.integers(0, 16)  # 240-255
    white = (int(white_val), int(white_val), int(white_val))

    # Alternating stripe pattern (mow lines every 5 yards)
    for i, x in enumerate(YARD_LINE_POSITIONS):
        x_px = int(x * PX_PER_YARD)
        if i + 1 < len(YARD_LINE_POSITIONS):
            next_x_px = int(YARD_LINE_POSITIONS[i + 1] * PX_PER_YARD)
        else:
            next_x_px = TEMPLATE_W

        if i % 2 == 0:
            stripe_color = base_green
        else:
            stripe_color = base_green + np.array([5, 10, 5], dtype=np.uint8)

        img[:, x_px:next_x_px] = stripe_color

    # Fill end zones — same color both sides, randomly sampled
    left_goal_px = int(GOAL_LINE_LEFT * PX_PER_YARD)
    right_goal_px = int(GOAL_LINE_RIGHT * PX_PER_YARD)

    # Random endzone color (team color, saturated)
    ez_hue = rng.integers(0, 180)
    ez_bgr = cv2.cvtColor(np.array([[[ez_hue, 200, 80]]], dtype=np.uint8),
                           cv2.COLOR_HSV2BGR)[0, 0]
    ez_color = tuple(int(c) for c in ez_bgr)
    img[:, :left_goal_px] = ez_color
    img[:, right_goal_px:] = ez_color

    # Random team name (5-7 uppercase letters)
    name_len = rng.integers(5, 8)
    ez_text = "".join(chr(c) for c in rng.integers(65, 91, size=name_len))

    ez_font = cv2.FONT_HERSHEY_DUPLEX
    ez_scale = 5.0
    ez_thick = 16
    (tw, th), _ = cv2.getTextSize(ez_text, ez_font, ez_scale, ez_thick)

    # Left endzone: rotated 90° CW, reads across the endzone from near sideline
    ez_canvas = np.zeros((th + 20, tw + 20), dtype=np.uint8)
    cv2.putText(ez_canvas, ez_text, (10, th + 10), ez_font, ez_scale, 255, ez_thick)
    ez_rotated = cv2.rotate(ez_canvas, cv2.ROTATE_90_CLOCKWISE)
    _paste_canvas(img, ez_rotated, int(5 * PX_PER_YARD), TEMPLATE_H // 2)

    # Right endzone: rotated opposite direction (reads backwards from near side)
    ez_rotated_r = cv2.rotate(ez_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _paste_canvas(img, ez_rotated_r, int(115 * PX_PER_YARD), TEMPLATE_H // 2)

    # Midfield logo: large colored oval between the hashes at midfield
    # Drawn BEFORE lines so yard lines render on top
    mid_px = int(60 * PX_PER_YARD)
    logo_color = tuple(int(min(255, c + 40)) for c in ez_bgr)
    logo_rx = int(PX_PER_YARD * 4)  # ~4 yards wide radius
    logo_ry = int(PX_PER_YARD * 3)  # ~3 yards tall radius
    cv2.ellipse(img, (mid_px, TEMPLATE_H // 2),
                (logo_rx, logo_ry), 0, 0, 360, logo_color, -1)

    # ── Boundary lines: thick white border around entire field ──────
    # Sidelines (top and bottom)
    cv2.line(img, (0, SIDELINE_W // 2), (TEMPLATE_W, SIDELINE_W // 2),
             white, SIDELINE_W)
    cv2.line(img, (0, TEMPLATE_H - SIDELINE_W // 2),
             (TEMPLATE_W, TEMPLATE_H - SIDELINE_W // 2), white, SIDELINE_W)
    # End lines (left and right) — same thickness as sidelines
    cv2.line(img, (SIDELINE_W // 2, 0), (SIDELINE_W // 2, TEMPLATE_H),
             white, SIDELINE_W)
    cv2.line(img, (TEMPLATE_W - SIDELINE_W // 2, 0),
             (TEMPLATE_W - SIDELINE_W // 2, TEMPLATE_H), white, SIDELINE_W)

    # ── Goal lines ───────────────────────────────────────────────────
    cv2.line(img, (left_goal_px, 0), (left_goal_px, TEMPLATE_H), white, LINE_W + 2)
    cv2.line(img, (right_goal_px, 0), (right_goal_px, TEMPLATE_H), white, LINE_W + 2)

    # ── Yard lines (every 5 yards) ──────────────────────────────────
    for x in YARD_LINE_POSITIONS:
        if x == GOAL_LINE_LEFT or x == GOAL_LINE_RIGHT:
            continue
        px_x = int(x * PX_PER_YARD)
        cv2.line(img, (px_x, 0), (px_x, TEMPLATE_H), white, LINE_W)

    # ── Hash marks ──────────────────────────────────────────────────
    near_hash_py = int(HASH_Y_NEAR * PX_PER_YARD)
    far_hash_py = int(HASH_Y_FAR * PX_PER_YARD)

    # Hash spacing: 18.5 feet = 6.167 yards apart (inside edge to inside edge)
    # The crosses at 5-yard lines sit on the INSIDE edges of the two hash rows
    # Near hash crosses: at near_hash_py (inside edge of near hash row)
    # Far hash crosses: at far_hash_py (inside edge of far hash row)

    # At 5-yard lines: horizontal cross marks (parallel to sidelines)
    # These sit on the yard lines at the hash mark y-positions
    for x in YARD_LINE_POSITIONS:
        px_x = int(x * PX_PER_YARD)
        # Near hash cross: horizontal dash on the inside (field-side) of near hash
        cv2.line(img, (px_x - HASH_LEN // 2, near_hash_py),
                 (px_x + HASH_LEN // 2, near_hash_py), white, HASH_W)
        # Far hash cross: horizontal dash on the inside (field-side) of far hash
        cv2.line(img, (px_x - HASH_LEN // 2, far_hash_py),
                 (px_x + HASH_LEN // 2, far_hash_py), white, HASH_W)

    # Between 5-yard lines at 1-yard intervals: vertical dashes
    # (perpendicular to sidelines) at both hash positions
    for i in range(len(YARD_LINE_POSITIONS) - 1):
        x_start = YARD_LINE_POSITIONS[i]
        for yd in range(1, 5):
            x = x_start + yd
            if x > GOAL_LINE_RIGHT:
                break
            px_x = int(x * PX_PER_YARD)
            # Near hash - vertical dash centered on hash y
            cv2.line(img, (px_x, near_hash_py - HASH_LEN // 4),
                     (px_x, near_hash_py + HASH_LEN // 4), white, max(2, HASH_W // 2))
            # Far hash - vertical dash
            cv2.line(img, (px_x, far_hash_py - HASH_LEN // 4),
                     (px_x, far_hash_py + HASH_LEN // 4), white, max(2, HASH_W // 2))

    # ── 1-yard sideline tick marks ──────────────────────────────────
    # Short marks along each sideline at every yard
    tick_len = max(3, int(PX_PER_YARD * 0.3))  # ~12px
    for yard in range(int(GOAL_LINE_LEFT), int(GOAL_LINE_RIGHT) + 1):
        px_x = int(yard * PX_PER_YARD)
        # Near sideline tick (pointing inward from sideline)
        cv2.line(img, (px_x, SIDELINE_W), (px_x, SIDELINE_W + tick_len),
                 white, max(1, LINE_W // 2))
        # Far sideline tick
        cv2.line(img, (px_x, TEMPLATE_H - SIDELINE_W),
                 (px_x, TEMPLATE_H - SIDELINE_W - tick_len),
                 white, max(1, LINE_W // 2))

    # ── Painted numbers ─────────────────────────────────────────────
    near_num_py = int(NUMBER_Y_NEAR * PX_PER_YARD)
    far_num_py = int(NUMBER_Y_FAR * PX_PER_YARD)

    from src.homography.field_model import ngs_x_to_field_number

    # Draw all numbers first
    for x in TEN_YARD_POSITIONS:
        px_x = int(x * PX_PER_YARD)
        num = ngs_x_to_field_number(x)
        _draw_number(img, num, px_x, near_num_py, side="near")
        _draw_number(img, num, px_x, far_num_py, side="far")

    # Draw arrows AFTER numbers so they aren't covered
    for x in TEN_YARD_POSITIONS:
        px_x = int(x * PX_PER_YARD)
        num = ngs_x_to_field_number(x)
        if num < 50:
            is_left_half = (x <= 60)
            arrow_x_offset = int(PX_PER_YARD * 2.8)  # must clear the widest digit + gap
            if is_left_half:
                arrow_x = px_x - arrow_x_offset
            else:
                arrow_x = px_x + arrow_x_offset

            # Arrow y: toward midfield (the "top" of the number)
            near_arrow_y = near_num_py + int(PX_PER_YARD * 0.4)
            far_arrow_y = far_num_py - int(PX_PER_YARD * 0.4)

            _draw_arrow(img, arrow_x, near_arrow_y,
                        pointing_left=is_left_half, size=18)
            _draw_arrow(img, arrow_x, far_arrow_y,
                        pointing_left=is_left_half, size=18)

    return img


# ── Camera model ─────────────────────────────────────────────────────────────

def _camera_to_homography(
    cam_x: float,
    cam_y: float,
    cam_z: float,
    target_x: float,
    target_y: float,
    focal_length: float,
) -> np.ndarray:
    """Compute homography from field template pixels to camera image pixels.

    Uses a look-at camera model. The field lies in the z=0 plane.
    Camera is at (cam_x, cam_y, cam_z) looking at (target_x, target_y, 0).

    World coordinate system:
      x: along the field (0=left end line, 120=right end line)
      y: across the field (0=near sideline, 53.33=far sideline)
      z: up

    The camera is positioned behind the near sideline (y < 0), elevated (z > 0),
    looking across the field. In the output image:
      - Near sideline appears at the bottom
      - Far sideline appears near the top
      - Yard lines run roughly vertically

    Args:
        cam_x: camera x position (yards along field)
        cam_y: camera y position (yards, negative = behind near sideline)
        cam_z: camera height (yards above field)
        target_x: x coordinate of the point the camera is looking at
        target_y: y coordinate of the look-at point (typically mid-field ~27)
        focal_length: in pixels (controls zoom/FOV)

    Returns:
        3x3 homography matrix mapping field template pixels to output pixels
    """
    cam = np.array([cam_x, cam_y, cam_z])
    target = np.array([target_x, target_y, 0.0])

    # Look-at construction
    forward = target - cam
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])

    # Right vector (points rightward in image when camera faces the field)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Down vector (points downward in image)
    down = np.cross(forward, right)
    # No need to normalize — forward and right are already orthonormal

    # Rotation matrix: maps world vectors to camera coordinates
    # Camera coords: x=right, y=down, z=forward
    R = np.array([right, down, forward])

    # Camera intrinsics
    K = np.array([
        [focal_length, 0, FRAME_W / 2],
        [0, focal_length, FRAME_H / 2],
        [0, 0, 1],
    ])

    # For ground plane points (z=0), projection P = K @ [R | -R@cam]
    # reduces to H = K @ [r1, r2, -R@cam] applied to [X; Y; 1]
    t_cam = -R @ cam

    H_world = K @ np.column_stack([R[:, 0], R[:, 1], t_cam])

    # Scale from template pixels to yards
    S = np.diag([1.0 / PX_PER_YARD, 1.0 / PX_PER_YARD, 1.0])
    H = H_world @ S

    return H


def sample_camera_pose(rng: np.random.Generator, endzone_shot: bool = False) -> dict:
    """Sample a random camera pose from realistic All-22 distribution.

    Args:
        rng: numpy random generator
        endzone_shot: if True, sample a sharper angle toward an end zone

    Returns:
        dict with camera parameters for _camera_to_homography()
    """
    # Camera position: near sideline, roughly midfield, elevated
    cam_x = 60.0 + rng.uniform(-10, 10)  # near midfield, ±10 yards
    cam_y = -8.0 + rng.uniform(-2, 2)    # behind near sideline, ~8 yards back
    cam_z = 22.0 + rng.uniform(-3, 3)    # ~57-75 feet up (steeper downward look)

    # Focal length first (needed to determine target_y variation)
    # Range covers wide pre-snap (~1100) to tight end-of-play zoom (~2500)
    if endzone_shot:
        focal_length = rng.uniform(1200, 2500)
    else:
        focal_length = rng.uniform(1100, 2200)

    # Target x: where along the field the camera is looking
    if endzone_shot:
        target_x = rng.choice([15.0, 105.0]) + rng.uniform(-5, 5)
    else:
        target_x = rng.uniform(15, 105)

    # Target y: generally mid-field, biased slightly toward far sideline
    # so wide views show a bit more past the far sideline than near.
    # Tighter zoom allows more y variation (can focus near either sideline).
    # Must always target somewhere on the field.
    base_y = FIELD_WIDTH * 0.55  # ~29 yards, slightly past center toward far sideline
    if focal_length > 1400:
        jitter = rng.uniform(-12, 12)
    else:
        jitter = rng.uniform(-5, 5)
    target_y = np.clip(base_y + jitter, 3.0, FIELD_WIDTH - 3.0)

    return {
        "cam_x": cam_x,
        "cam_y": cam_y,
        "cam_z": cam_z,
        "target_x": target_x,
        "target_y": target_y,
        "focal_length": focal_length,
    }


# ── Domain randomization ────────────────────────────────────────────────────

def apply_domain_randomization(
    frame: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply random augmentations to make synthetic frames more realistic."""
    h, w = frame.shape[:2]

    # ── Out-of-field areas ────────────────────────────────────────
    # Detect the near-black sentinel border value (1,1,1) from warpPerspective.
    # This can't be confused with field grass or any field marking.
    border_mask = np.all(frame <= 3, axis=2)
    if border_mask.any():
        # Fill all border with green (apron area)
        frame[border_mask] = np.array([35, 100, 40], dtype=np.uint8)

    # ── Lighting variation (day vs dome/night) ─────────────────
    # Day: bright, slightly blue/cool cast
    # Dome/night: dimmer, warmer yellow/orange cast, higher contrast
    frame_f = frame.astype(np.float32)
    brightness = rng.uniform(0.6, 1.3)  # wide range: dim dome to bright day
    frame_f *= brightness

    # Color temperature shift: warm (dome) vs cool (day)
    temp_shift = rng.uniform(-15, 15)  # negative=cool/blue, positive=warm/yellow
    frame_f[:, :, 0] -= temp_shift  # B channel
    frame_f[:, :, 2] += temp_shift  # R channel
    frame = np.clip(frame_f, 0, 255).astype(np.uint8)

    # ── Grass color variation (per-frame hue/sat jitter) ─────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] += rng.uniform(-8, 8)
    hsv[:, :, 1] *= rng.uniform(0.85, 1.15)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ── Player-like occlusions (more realistic sizing) ───────────
    # NFL players from All-22 overhead: roughly 15-25px wide, 25-45px tall
    # Clustered near the middle of the field (line of scrimmage area)
    n_players = rng.integers(15, 30)
    # Cluster center: roughly middle 60% of frame
    cluster_cx = w // 2 + rng.integers(-w // 4, w // 4)
    cluster_cy = h // 2 + rng.integers(-h // 6, h // 6)

    for _ in range(n_players):
        pw = rng.integers(12, 28)
        ph = rng.integers(20, 45)
        # Players cluster around the action with some spread
        px = int(cluster_cx + rng.normal(0, w * 0.15))
        py = int(cluster_cy + rng.normal(0, h * 0.12))
        px = np.clip(px, 0, w - pw)
        py = np.clip(py, 0, h - ph)

        # Player jersey colors: mix of team colors (darker/lighter)
        if rng.random() < 0.5:
            color = tuple(int(c) for c in rng.integers(20, 80, size=3))  # dark
        else:
            color = tuple(int(c) for c in rng.integers(150, 255, size=3))  # light/white

        cv2.rectangle(frame, (int(px), int(py)),
                      (int(px) + pw, int(py) + ph), color, -1)

    # ── Sideline/stadium occlusions ─────────────────────────────────
    # Use the saved border_mask (from before color jitter) to ensure we
    # ONLY place occlusions outside the field, never on it.

    if border_mask.any():
        # Distance from field edge into the border area
        dist_map = cv2.distanceTransform(
            border_mask.astype(np.uint8) * 255, cv2.DIST_L2, 5
        )

        border_ys, border_xs = np.where(border_mask)
        if len(border_ys) > 0:
            dists = dist_map[border_ys, border_xs]

            # Skip pixels too close to the boundary (>30px from field edge)
            valid = dists > 30
            if valid.any():
                valid_ys = border_ys[valid]
                valid_xs = border_xs[valid]
                valid_dists = dists[valid]

                # Probability increases with distance (denser further out)
                probs = valid_dists / valid_dists.max()
                probs = probs ** 0.7  # moderate density curve
                probs = probs / probs.sum()

                n_occlusions = min(int(len(valid_ys) * 0.04), 4000)
                indices = rng.choice(len(valid_ys), size=n_occlusions,
                                     replace=False, p=probs)

                for idx in indices:
                    oy, ox = int(valid_ys[idx]), int(valid_xs[idx])
                    d = valid_dists[idx]

                    # Size scales with distance: larger further from field
                    base_size = max(4, int(4 + d * 0.2))
                    ow = rng.integers(base_size, base_size + 10)
                    oh = rng.integers(base_size, int(base_size * 2.5))

                    # Color: dark near field, grey/muted further out
                    if d < 40:
                        color = tuple(int(c) for c in rng.integers(20, 120, size=3))
                    else:
                        v = int(rng.integers(30, 100))
                        jitter = rng.integers(-15, 15, size=3)
                        color = tuple(max(0, min(255, v + int(j))) for j in jitter)

                    # Ensure occlusion stays within border area
                    ox2 = min(w, ox + ow)
                    oy2 = min(h, oy + oh)
                    cv2.rectangle(frame, (ox, oy), (ox2, oy2), color, -1)

    # ── Gaussian blur (broadcast compression) ────────────────────
    if rng.random() < 0.5:
        sigma = rng.uniform(0.3, 1.5)
        frame = cv2.GaussianBlur(frame, (0, 0), sigma)

    # ── Gaussian noise ───────────────────────────────────────────
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 8), frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


# ── Synthetic frame generation ───────────────────────────────────────────────

def generate_frame(
    template: np.ndarray,
    rng: np.random.Generator,
    endzone_shot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one synthetic training frame.

    Returns:
        (frame, pixel_coords, visibility)
        frame: (720, 1280, 3) BGR image
        pixel_coords: (110, 2) pixel positions of all keypoints
        visibility: (110,) int array, 0=not visible, 2=visible
    """
    pose = sample_camera_pose(rng, endzone_shot=endzone_shot)
    H = _camera_to_homography(**pose)

    # Warp template to camera view
    frame = cv2.warpPerspective(template, H, (FRAME_W, FRAME_H),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(1, 1, 1))  # near-black sentinel

    # Transform keypoint field coordinates to pixel coordinates
    field_px = FIELD_COORDS * PX_PER_YARD  # (110, 2)

    ones = np.ones((NUM_KEYPOINTS, 1))
    pts_h = np.hstack([field_px, ones])  # (110, 3)
    projected = (H @ pts_h.T).T  # (110, 3)
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    pixel_coords = projected[:, :2]  # (110, 2)

    # Determine visibility for identity keypoints (0-105)
    visible = get_visible_keypoints(pixel_coords, FRAME_W, FRAME_H, margin=5.0)
    visibility = np.where(visible, 2, 0).astype(np.int32)

    # Category keypoints (106-109): visible if ANY identity keypoint of that
    # type is visible. Position set to first visible identity keypoint of type.
    # (Training script generates multi-peak heatmaps from identity keypoints.)
    category_type_map = {
        106: "near_sideline",
        107: "near_hash",
        108: "far_hash",
        109: "far_sideline",
    }
    for cat_id, cat_type in category_type_map.items():
        # Find visible identity keypoints of this type
        type_kps = KEYPOINTS_BY_TYPE.get(cat_type, [])
        for kp in type_kps:
            kid = kp["id"]
            if kid < NUM_IDENTITY_KEYPOINTS and visibility[kid] > 0:
                pixel_coords[cat_id] = pixel_coords[kid]
                visibility[cat_id] = 2
                break  # just need one position for COCO format

    # Apply domain randomization
    frame = apply_domain_randomization(frame, rng)

    return frame, pixel_coords, visibility


# ── COCO export ──────────────────────────────────────────────────────────────

def save_coco_dataset(
    output_dir: str,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prefix: str = "synthetic",
):
    """Save frames and annotations in COCO keypoints format."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    images = []
    annotations = []

    for idx, (frame, pixel_coords, visibility) in enumerate(frames):
        fname = f"{prefix}_{idx:05d}.jpg"
        fpath = os.path.join(img_dir, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        images.append({
            "id": idx,
            "file_name": fname,
            "width": FRAME_W,
            "height": FRAME_H,
        })

        # Build keypoints array: [x0, y0, v0, x1, y1, v1, ...]
        kp_flat = []
        n_visible = 0
        for ki in range(NUM_KEYPOINTS):
            x, y = pixel_coords[ki]
            v = int(visibility[ki])
            if v == 0:
                kp_flat.extend([0.0, 0.0, 0])
            else:
                kp_flat.extend([float(x), float(y), v])
                n_visible += 1

        annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "keypoints": kp_flat,
            "num_keypoints": n_visible,
            "bbox": [0, 0, FRAME_W, FRAME_H],
            "area": FRAME_W * FRAME_H,
            "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "field",
            "supercategory": "field",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [],
        }],
    }

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"Saved {len(frames)} frames to {img_dir}")
    print(f"Annotations: {ann_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic field training data")
    parser.add_argument("--num-frames", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/field_keypoints/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endzone-pct", type=float, default=0.15,
                        help="Fraction of frames with end zone angles")
    parser.add_argument("--preview", action="store_true",
                        help="Generate 10 frames and save preview, don't make full dataset")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    rng = np.random.default_rng(args.seed)

    os.makedirs(output_dir, exist_ok=True)

    # Render multiple templates for variation in endzone color/text,
    # grass tone, and line paint white
    n_templates = max(1, args.num_frames // 250)
    if args.preview:
        n_templates = 3  # enough to see variation
    print(f"Rendering {n_templates} field templates for variation...")
    templates = []
    for ti in range(n_templates):
        t = render_field_template(np.random.default_rng(args.seed + ti))
        templates.append(t)
    template_path = os.path.join(output_dir, "field_template.png")
    cv2.imwrite(template_path, templates[0])
    print(f"Template 0 saved: {template_path} ({templates[0].shape[1]}x{templates[0].shape[0]})")

    if args.preview:
        print("\nGenerating 10 preview frames...")
        preview_dir = os.path.join(output_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)

        for i in range(10):
            endzone = (i >= 8)  # last 2 are endzone shots
            template = templates[i % len(templates)]
            frame, pixel_coords, visibility = generate_frame(template, rng, endzone)

            # Draw keypoints on frame for visualization
            viz = frame.copy()
            for ki in range(NUM_KEYPOINTS):
                if visibility[ki] > 0:
                    x, y = int(pixel_coords[ki, 0]), int(pixel_coords[ki, 1])
                    kp = KEYPOINTS[ki]
                    # Color by type
                    colors = {
                        "near_sideline": (0, 0, 255),
                        "near_hash": (0, 255, 0),
                        "far_hash": (255, 0, 0),
                        "far_sideline": (0, 255, 255),
                        "near_number": (255, 0, 255),
                        "far_number": (255, 128, 255),
                        "endzone_corner": (0, 165, 255),
                    }
                    color = colors.get(kp["type"], (255, 255, 255))
                    cv2.circle(viz, (x, y), 5, color, -1)
                    cv2.circle(viz, (x, y), 5, (255, 255, 255), 1)

                    # Label with short name
                    label = kp["name"][:8]
                    cv2.putText(viz, label, (x + 6, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            n_vis = int(np.sum(visibility > 0))
            cv2.putText(viz, f"Frame {i} | {n_vis} keypoints | {'ENDZONE' if endzone else 'normal'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(preview_dir, f"preview_{i:02d}.jpg"), viz)

        print(f"Previews saved to {preview_dir}")
        return

    print(f"\nGenerating {args.num_frames} synthetic frames...")
    frames = []
    for i in range(args.num_frames):
        endzone = rng.random() < args.endzone_pct
        template = templates[i % len(templates)]
        frame, pixel_coords, visibility = generate_frame(template, rng, endzone)
        frames.append((frame, pixel_coords, visibility))

        if (i + 1) % 500 == 0:
            n_vis = int(np.mean([np.sum(v > 0) for _, _, v in frames[-500:]]))
            print(f"  {i + 1}/{args.num_frames} frames ({n_vis:.0f} avg visible keypoints)")

    save_coco_dataset(output_dir, frames)
    print("Done!")


if __name__ == "__main__":
    main()
