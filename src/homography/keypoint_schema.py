"""
Field keypoint schema for HRNet-based homography.

The model outputs 2 heatmap channels, each detecting ALL instances of a
feature type:
  0: sideline_intersection — where any yard line crosses either sideline
  1: hash_intersection — any hash mark cross (near or far)

Each channel has MULTIPLE peaks per frame (one per visible instance).
Identity (which yard line, which side) is resolved downstream by:
  - Grid spacing analysis (regular 5-yard intervals)
  - Temporal tracking

Number detection is handled separately (not part of the heatmap model).

The schema also defines all individual field points for ground truth
generation and homography computation.
"""

import numpy as np
from .field_model import (
    YARD_LINE_POSITIONS,
    FIELD_WIDTH,
    GOAL_LINE_LEFT,
    GOAL_LINE_RIGHT,
    HASH_Y_NEAR,
    HASH_Y_FAR,
)


# ── Model output channels ──────────────────────────────────────────────────

NUM_CHANNELS = 2  # model output heatmap channels

CHANNEL_SIDELINE = 0     # yard line × sideline intersections (both near and far)
CHANNEL_HASH = 1         # hash mark crosses (both near and far)

CHANNEL_NAMES = ["sideline_intersection", "hash_intersection"]


# ── Field point definitions ─────────────────────────────────────────────────
# These define every identifiable point on the field with its real-world
# coordinates and which heatmap channel it belongs to.

# Intersection types: (type_name, y_coordinate, channel_id)
INTERSECTION_TYPES = [
    ("near_sideline", 0.0, CHANNEL_SIDELINE),
    ("near_hash", HASH_Y_NEAR, CHANNEL_HASH),
    ("far_hash", HASH_Y_FAR, CHANNEL_HASH),
    ("far_sideline", FIELD_WIDTH, CHANNEL_SIDELINE),
]

def _build_field_points() -> list[dict]:
    """Build the complete list of field points.

    Each point has: name, field_xy, type, channel, yard_line_x.
    """
    points = []

    # Yard-line intersections (no hashes on goal lines)
    for x in YARD_LINE_POSITIONS:
        is_goal_line = (x == GOAL_LINE_LEFT or x == GOAL_LINE_RIGHT)
        for type_name, y, channel in INTERSECTION_TYPES:
            if is_goal_line and "hash" in type_name:
                continue
            points.append({
                "name": f"{int(x)}_{type_name}",
                "field_xy": (float(x), float(y)),
                "type": type_name,
                "channel": channel,
                "yard_line_x": float(x),
            })

    return points


# ── Module-level exports ────────────────────────────────────────────────────

FIELD_POINTS: list[dict] = _build_field_points()
NUM_FIELD_POINTS = len(FIELD_POINTS)

# Field coordinates array for all points
FIELD_COORDS: np.ndarray = np.array(
    [p["field_xy"] for p in FIELD_POINTS], dtype=np.float64
)

# Channel assignment for each field point
POINT_CHANNELS: np.ndarray = np.array(
    [p["channel"] for p in FIELD_POINTS], dtype=np.int32
)

# Group field points by channel
POINTS_BY_CHANNEL: dict[int, list[dict]] = {}
for p in FIELD_POINTS:
    POINTS_BY_CHANNEL.setdefault(p["channel"], []).append(p)

# Group by type
POINTS_BY_TYPE: dict[str, list[dict]] = {}
for p in FIELD_POINTS:
    POINTS_BY_TYPE.setdefault(p["type"], []).append(p)

# Lookup
NAME_TO_POINT: dict[str, dict] = {p["name"]: p for p in FIELD_POINTS}


# ── Utility functions ───────────────────────────────────────────────────────

def get_visible_points(
    pixel_coords: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin: float = 0.0,
) -> np.ndarray:
    """Return boolean mask of field points within frame bounds.

    Args:
        pixel_coords: (N, 2) pixel positions of field points
        frame_w: frame width in pixels
        frame_h: frame height in pixels
        margin: pixels of margin inside frame edge

    Returns:
        (N,) boolean array
    """
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]
    return (
        (x >= margin) & (x < frame_w - margin) &
        (y >= margin) & (y < frame_h - margin)
    )
