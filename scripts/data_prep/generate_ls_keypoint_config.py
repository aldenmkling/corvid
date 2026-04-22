#!/usr/bin/env python3
"""
Generate Label Studio project configuration for field keypoint annotation.

2 labels matching the 2 HRNet output channels:
  - sideline_intersection (red) — yard line × either sideline
  - hash_intersection (green) — yard line × either hash mark row

Usage:
    python scripts/data_prep/generate_ls_keypoint_config.py
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = [
    ("sideline_intersection", "#FF4444"),  # red
    ("hash_intersection", "#44FF44"),      # green
]


def generate_xml() -> str:
    lines = [
        '<View>',
        '  <Header value="Field Keypoint Annotation — place points at yard line intersections"/>',
        '  <Text name="instructions" value="Select a label, then click the intersection. '
        'sideline_intersection = where a yard line meets either sideline (top or bottom). '
        'hash_intersection = where a yard line crosses a hash mark row (near or far). '
        'Only annotate points you can clearly identify."/>',
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>',
        '  <KeyPointLabels name="keypoint" toName="image" smart="true" strokeWidth="2" opacity="0.9">',
    ]

    for label, color in LABELS:
        lines.append(f'    <Label value="{label}" background="{color}"/>')

    lines.extend([
        '  </KeyPointLabels>',
        '</View>',
    ])

    return "\n".join(lines)


def main():
    xml = generate_xml()

    config_path = os.path.join(PROJECT_ROOT, "data", "field_keypoints", "labeling_config.xml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(xml)

    print(f"Saved: {config_path}")
    for label, color in LABELS:
        print(f"  {color} {label}")


if __name__ == "__main__":
    main()
