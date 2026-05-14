"""
NFL field geometry constants and reference point generation.

Coordinate system matches NGS convention:
  x: 0–120 yards (end line to end line, 10 yards of end zone each side)
  y: 0–53.33 yards (sideline to sideline)

The "near" sideline (y=0) is the one closest to the broadcast camera.
"""

import numpy as np

# ── Field dimensions (yards) ────────────────────────────────────────────────

FIELD_LENGTH = 120.0        # end line to end line (including end zones)
FIELD_WIDTH = 53.33         # sideline to sideline (160 feet)
END_ZONE_DEPTH = 10.0       # each end zone is 10 yards deep
FIELD_OF_PLAY = 100.0       # goal line to goal line

# Goal lines
GOAL_LINE_LEFT = 10.0       # x coordinate of left goal line
GOAL_LINE_RIGHT = 110.0     # x coordinate of right goal line

# ── Yard line positions ─────────────────────────────────────────────────────

# Yard lines every 5 yards from goal line to goal line
# In NGS x-coords: 10, 15, 20, ..., 105, 110
YARD_LINE_POSITIONS = [GOAL_LINE_LEFT + i * 5 for i in range(21)]  # 10 to 110

# 10-yard interval lines (where numbers are painted)
# NGS x: 20, 30, 40, 50, 60, 70, 80, 90, 100
TEN_YARD_POSITIONS = [GOAL_LINE_LEFT + i * 10 for i in range(1, 10)]

# Mapping from NGS x-coordinate to the painted number on the field
# The field shows yards from nearest goal line (10, 20, 30, 40, 50)
def ngs_x_to_field_number(x: float) -> int:
    """Convert NGS x-coordinate to the number painted on the field."""
    yards_from_left_goal = x - GOAL_LINE_LEFT
    yards_from_right_goal = GOAL_LINE_RIGHT - x
    return int(min(yards_from_left_goal, yards_from_right_goal))


# ── Hash marks ───────────────────────────────────────────────────────────────

# NFL hash marks are 70 feet 9 inches from each sideline
# 70.75 feet = 23.583 yards from each sideline
HASH_Y_NEAR = 23.583       # hash mark line closer to near sideline (y=0)
HASH_Y_FAR = FIELD_WIDTH - 23.583  # = 29.750 yards

HASH_SPACING = HASH_Y_FAR - HASH_Y_NEAR  # 6.167 yards (18 feet 6 inches)

# ── Painted numbers ─────────────────────────────────────────────────────────

# NFL spec: bottom of numbers 12 yards from sideline. Numbers are 2 yards tall,
# so the center is at 13 yards from the sideline.
NUMBER_Y_NEAR = 13.0       # center of near-side numbers
NUMBER_Y_FAR = FIELD_WIDTH - 13.0  # center of far-side numbers


# ── Reference point generation ──────────────────────────────────────────────

def get_yard_line_endpoints(x: float) -> tuple[np.ndarray, np.ndarray]:
    """Get the two sideline endpoints of a yard line in field coordinates.

    Returns (near_sideline_point, far_sideline_point) as (x, y) arrays.
    """
    return np.array([x, 0.0]), np.array([x, FIELD_WIDTH])


def get_yard_line_reference_points(x: float) -> list[np.ndarray]:
    """Get all identifiable reference points along a yard line.

    Returns points at: near sideline, near hash, far hash, far sideline.
    """
    return [
        np.array([x, 0.0]),           # near sideline
        np.array([x, HASH_Y_NEAR]),   # near hash
        np.array([x, HASH_Y_FAR]),    # far hash
        np.array([x, FIELD_WIDTH]),   # far sideline
    ]


def get_all_reference_points(
    x_min: float = GOAL_LINE_LEFT,
    x_max: float = GOAL_LINE_RIGHT,
) -> list[tuple[np.ndarray, str]]:
    """Get all reference points on the field within a yard range.

    Returns list of (field_coord, label) tuples.
    """
    points = []
    for x in YARD_LINE_POSITIONS:
        if x < x_min or x > x_max:
            continue
        yard_num = ngs_x_to_field_number(x)
        is_five = (x - GOAL_LINE_LEFT) % 10 == 5  # 5-yard line (no number)
        suffix = f"_{yard_num}yd" if not is_five else f"_{yard_num}+5yd"

        points.append((np.array([x, 0.0]), f"near_sideline{suffix}"))
        points.append((np.array([x, FIELD_WIDTH]), f"far_sideline{suffix}"))
        points.append((np.array([x, HASH_Y_NEAR]), f"near_hash{suffix}"))
        points.append((np.array([x, HASH_Y_FAR]), f"far_hash{suffix}"))

    return points


def get_number_positions() -> list[tuple[np.ndarray, int]]:
    """Get positions of painted numbers on the field.

    Returns list of (field_coord, displayed_number) tuples.
    Only the near-sideline numbers (more visible from broadcast camera).
    """
    positions = []
    for x in TEN_YARD_POSITIONS:
        num = ngs_x_to_field_number(x)
        if num > 0:  # skip goal lines
            positions.append((np.array([x, NUMBER_Y_NEAR]), num))
            positions.append((np.array([x, NUMBER_Y_FAR]), num))
    return positions


# ── Painted-number tangent points (inside edges) ────────────────────────────
# Where the painted yardline number's "inside" (toward field center) edge sits
# in NGS-y coordinates. Used as keypoint tangents in field_mapping/keypoints.py.
NUMBER_Y_NEAR = 13.0   # near-sideline-side painted number outer edge
NUMBER_Y_FAR = 40.33   # far-sideline-side painted number outer edge
NGS_Y_NEAR_INSIDE = NUMBER_Y_NEAR + 1.0   # 14.0 — inside edge of near number
NGS_Y_FAR_INSIDE = NUMBER_Y_FAR - 1.0     # 39.33 — inside edge of far number
