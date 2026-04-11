"""
Field keypoint schema for HRNet-based homography.

Defines 106 semantically labeled keypoints on an NFL football field.
Each keypoint has a unique integer ID and known real-world field coordinates
in the NGS coordinate system (x: 0-120, y: 0-53.33).

Keypoint types:
  - Yard-line intersections (84): 21 yard lines × 4 types
    (near_sideline, near_hash, far_hash, far_sideline)
  - Painted numbers (18): 9 positions × 2 sides (near, far)
  - End zone corners (4): back-of-endzone end line corners

The model outputs one heatmap channel per keypoint. On any given frame,
most channels will be zero (keypoint not visible). This is expected and
mirrors how pose estimation handles occluded joints.
"""

import numpy as np
from .field_model import (
    YARD_LINE_POSITIONS,
    TEN_YARD_POSITIONS,
    FIELD_WIDTH,
    FIELD_LENGTH,
    HASH_Y_NEAR,
    HASH_Y_FAR,
    NUMBER_Y_NEAR,
    NUMBER_Y_FAR,
)


# ── Schema constants ────────────────────────────────────────────────────────

# 106 identity-specific keypoints + 4 generic category keypoints = 110 total
NUM_KEYPOINTS = 110
NUM_IDENTITY_KEYPOINTS = 106  # IDs 0-105: specific yard line identity
NUM_CATEGORY_KEYPOINTS = 4   # IDs 106-109: generic type (fires for ANY yard line)

# Intersection types and their y-coordinates
INTERSECTION_TYPES = [
    ("near_sideline", 0.0),
    ("near_hash", HASH_Y_NEAR),
    ("far_hash", HASH_Y_FAR),
    ("far_sideline", FIELD_WIDTH),
]

# Number types and their y-coordinates
NUMBER_TYPES = [
    ("near_number", NUMBER_Y_NEAR),
    ("far_number", NUMBER_Y_FAR),
]


# ── Build keypoint list ─────────────────────────────────────────────────────

def _build_keypoints() -> list[dict]:
    """Build the full 106-keypoint schema."""
    keypoints = []
    kp_id = 0

    # 84 yard-line intersection keypoints (IDs 0-83)
    for yl_idx, x in enumerate(YARD_LINE_POSITIONS):
        for type_idx, (type_name, y) in enumerate(INTERSECTION_TYPES):
            keypoints.append({
                "id": kp_id,
                "name": f"{int(x)}_{type_name}",
                "field_xy": (float(x), float(y)),
                "type": type_name,
                "yard_line_x": float(x),
            })
            kp_id += 1

    # 18 painted number keypoints (IDs 84-101)
    for num_idx, x in enumerate(TEN_YARD_POSITIONS):
        for side_idx, (type_name, y) in enumerate(NUMBER_TYPES):
            keypoints.append({
                "id": kp_id,
                "name": f"{int(x)}_{type_name}",
                "field_xy": (float(x), float(y)),
                "type": type_name,
                "yard_line_x": float(x),
            })
            kp_id += 1

    # 4 end zone corner keypoints (IDs 102-105)
    endzone_corners = [
        ("left_endline_near", 0.0, 0.0),
        ("left_endline_far", 0.0, FIELD_WIDTH),
        ("right_endline_near", FIELD_LENGTH, 0.0),
        ("right_endline_far", FIELD_LENGTH, FIELD_WIDTH),
    ]
    for name, x, y in endzone_corners:
        keypoints.append({
            "id": kp_id,
            "name": name,
            "field_xy": (float(x), float(y)),
            "type": "endzone_corner",
            "yard_line_x": float(x),
        })
        kp_id += 1

    # 4 generic category keypoints (IDs 106-109)
    # These fire for ANY yard line of the given type — used when the model
    # can detect an intersection but can't tell which yard line it is
    # (e.g., tight zoom, no numbers visible). The tracker resolves identity.
    # field_xy is set to (0, y) as a placeholder — the actual x comes from
    # the detected position, not the schema.
    category_types = [
        ("any_near_sideline", 0.0),
        ("any_near_hash", HASH_Y_NEAR),
        ("any_far_hash", HASH_Y_FAR),
        ("any_far_sideline", FIELD_WIDTH),
    ]
    for name, y in category_types:
        keypoints.append({
            "id": kp_id,
            "name": name,
            "field_xy": (0.0, float(y)),  # x is unknown, resolved by tracker
            "type": "category",
            "yard_line_x": 0.0,
        })
        kp_id += 1

    assert len(keypoints) == NUM_KEYPOINTS
    return keypoints


# ── Module-level exports ────────────────────────────────────────────────────

KEYPOINTS: list[dict] = _build_keypoints()

# Ordered name list (index = keypoint ID)
KEYPOINT_NAMES: list[str] = [kp["name"] for kp in KEYPOINTS]

# (110, 2) array of field coordinates, indexed by keypoint ID
# Note: category keypoints (106-109) have x=0 placeholder — actual x resolved by tracker
FIELD_COORDS: np.ndarray = np.array([kp["field_xy"] for kp in KEYPOINTS], dtype=np.float64)

# Lookup dicts
ID_BY_NAME: dict[str, int] = {kp["name"]: kp["id"] for kp in KEYPOINTS}
NAME_BY_ID: dict[int, str] = {kp["id"]: kp["name"] for kp in KEYPOINTS}

# Group keypoints by type
KEYPOINTS_BY_TYPE: dict[str, list[dict]] = {}
for kp in KEYPOINTS:
    KEYPOINTS_BY_TYPE.setdefault(kp["type"], []).append(kp)


# ── Horizontal flip mapping ────────────────────────────────────────────────

def _build_flip_mapping() -> dict[int, int]:
    """Build keypoint ID remapping for horizontal flip augmentation.

    Horizontal flip swaps left and right halves of the field:
    NGS x ↔ (FIELD_LENGTH - x). near/far stays the same.

    Returns dict mapping old_id → new_id after flip.
    """
    name_to_id = ID_BY_NAME
    mapping = {}

    for kp in KEYPOINTS:
        kp_id = kp["id"]
        x, y = kp["field_xy"]
        flipped_x = FIELD_LENGTH - x
        kp_type = kp["type"]

        if kp_type == "endzone_corner":
            # Swap left ↔ right endzone corners
            name = kp["name"]
            if "left" in name:
                flipped_name = name.replace("left", "right")
            else:
                flipped_name = name.replace("right", "left")
            mapping[kp_id] = name_to_id[flipped_name]
        else:
            # Swap x coordinate, keep type
            flipped_name = f"{int(flipped_x)}_{kp_type}"
            if flipped_name in name_to_id:
                mapping[kp_id] = name_to_id[flipped_name]
            else:
                # No corresponding keypoint (shouldn't happen with symmetric schema)
                mapping[kp_id] = kp_id

    return mapping


FLIP_MAPPING: dict[int, int] = _build_flip_mapping()


# ── Utility functions ───────────────────────────────────────────────────────

def get_visible_keypoints(
    pixel_coords: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin: float = 0.0,
) -> np.ndarray:
    """Return boolean mask of keypoints within frame bounds.

    Args:
        pixel_coords: (106, 2) pixel positions of all keypoints
        frame_w: frame width in pixels
        frame_h: frame height in pixels
        margin: pixels of margin inside frame edge

    Returns:
        (106,) boolean array, True if keypoint is within frame
    """
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]
    return (
        (x >= margin) & (x < frame_w - margin) &
        (y >= margin) & (y < frame_h - margin)
    )
