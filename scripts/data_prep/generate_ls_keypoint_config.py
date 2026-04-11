#!/usr/bin/env python3
"""
Generate Label Studio project configuration for field keypoint annotation.

Creates:
  1. Label Studio XML template with 106 keypoint labels
  2. Prints setup instructions

Usage:
    python scripts/generate_ls_keypoint_config.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.homography.keypoint_schema import KEYPOINTS, KEYPOINTS_BY_TYPE, NUM_KEYPOINTS

# Color mapping by keypoint type
TYPE_COLORS = {
    "near_sideline": "#FF4444",    # red
    "near_hash": "#44FF44",        # green
    "far_hash": "#4488FF",         # blue
    "far_sideline": "#FFFF44",     # yellow
    "near_number": "#FF44FF",      # magenta
    "far_number": "#CC88FF",       # light purple
    "endzone_corner": "#FF8800",   # orange
}


def generate_xml() -> str:
    """Generate Label Studio XML template."""
    lines = [
        '<View>',
        '  <Header value="Field Keypoint Annotation - click to place keypoints at yard line intersections, numbers, and end zone corners"/>',
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>',
        '  <KeyPointLabels name="keypoint" toName="image" smart="true" strokeWidth="2" opacity="0.9">',
    ]

    # Group labels by type for visual organization
    type_order = [
        "near_sideline", "near_hash", "far_hash", "far_sideline",
        "near_number", "far_number", "endzone_corner",
    ]

    for kp_type in type_order:
        color = TYPE_COLORS[kp_type]
        kps = KEYPOINTS_BY_TYPE.get(kp_type, [])
        for kp in sorted(kps, key=lambda k: k["id"]):
            lines.append(
                f'    <Label value="{kp["name"]}" background="{color}"/>'
            )

    lines.extend([
        '  </KeyPointLabels>',
        '</View>',
    ])

    return "\n".join(lines)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml = generate_xml()

    # Save XML config
    config_path = os.path.join(project_root, "data", "field_keypoints", "labeling_config.xml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(xml)

    print(f"Label Studio config saved to: {config_path}")
    print(f"Total keypoint labels: {NUM_KEYPOINTS}")
    print()
    print("Keypoint types and colors:")
    for kp_type, color in TYPE_COLORS.items():
        count = len(KEYPOINTS_BY_TYPE.get(kp_type, []))
        print(f"  {color} {kp_type}: {count} keypoints")
    print()
    print("Setup instructions:")
    print("1. Start Label Studio:")
    print(f'   cd "{project_root}" && nohup env LOCAL_FILES_SERVING_ENABLED=true '
          f'LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/data/field_keypoints/annotation_images" '
          f'.venv-labelstudio/bin/label-studio > /tmp/label-studio.log 2>&1 &')
    print()
    print("2. Create a new project in Label Studio")
    print(f"3. In Settings > Labeling Interface, paste contents of:")
    print(f"   {config_path}")
    print()
    print("4. Import images from data/field_keypoints/annotation_images/images/")
    print()
    print("5. Annotation workflow per frame:")
    print("   a. Identify visible yard lines by reading painted numbers")
    print("   b. Click to place keypoints at intersections (hash × yard line, sideline × yard line)")
    print("   c. Place number keypoints at center of each painted number")
    print("   d. Place end zone corners if visible")
    print()
    print("6. Export as JSON when done")


if __name__ == "__main__":
    main()
