"""Inject per-clip camera intrinsics (K, dist) from the per-clip stage files
into the training manifest at top level.

The pool stage files (`output/dense_field_pool/stage/<game_play>.json`)
each contain an `intrinsics` dict (K, dist). The flattened
`training_manifest.json` dropped them. The dense regression trainer needs
them to undistort source pixels before applying H — without this the GT
is silently wrong by 0.1-0.4 yd avg per frame, up to ~3 yd at corners.

Output: rewrites `training_manifest.json` to add an
`intrinsics_by_clip: { "<game>/<play>/sideline.mp4": {K, dist}, ... }` map
at the top level. Backs up the original to
`training_manifest.json.pre_intrinsics_backup`.
"""
import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        PROJECT_ROOT, "data/h_pool_and_intrinsics.json"))
    ap.add_argument("--stage-dir", default=os.path.join(
        PROJECT_ROOT, "output/dense_field_pool/stage"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Loading manifest: {args.manifest}")
    m = json.load(open(args.manifest))
    entries = m["entries"]
    print(f"  {len(entries)} entries")

    # Find unique clip paths
    clips = sorted({e["clip"] for e in entries})
    print(f"  {len(clips)} unique clips")

    intrinsics_by_clip = {}
    n_missing = 0
    for clip in clips:
        # clip is like "2024090802/play_113/sideline.mp4"
        # stage file is like "2024090802_play_113.json"
        parts = clip.split("/")
        tag = f"{parts[0]}_{parts[1]}"
        stage_path = os.path.join(args.stage_dir, f"{tag}.json")
        if not os.path.exists(stage_path):
            print(f"  MISSING stage file: {stage_path}")
            n_missing += 1
            continue
        s = json.load(open(stage_path))
        intr = s.get("intrinsics")
        if intr is None:
            print(f"  no intrinsics in {stage_path}")
            n_missing += 1
            continue
        intrinsics_by_clip[clip] = {
            "K": intr["K"],
            "dist": intr["dist"],
        }

    print(f"\n{len(intrinsics_by_clip)}/{len(clips)} clips have intrinsics; "
          f"{n_missing} missing")
    if n_missing > 0:
        # Show one missing-clip entry count to gauge impact
        missing_clips = set(clips) - set(intrinsics_by_clip.keys())
        n_affected = sum(1 for e in entries if e["clip"] in missing_clips)
        print(f"  → {n_affected} entries affected by missing intrinsics")

    if args.dry_run:
        print("\n[dry-run] not modifying manifest")
        return

    backup = args.manifest + ".pre_intrinsics_backup"
    if not os.path.exists(backup):
        shutil.copy2(args.manifest, backup)
        print(f"Backup: {backup}")
    else:
        print(f"Backup already exists: {backup}")

    m["intrinsics_by_clip"] = intrinsics_by_clip
    with open(args.manifest, "w") as f:
        json.dump(m, f)
    print(f"Wrote manifest with intrinsics_by_clip: {args.manifest}")


if __name__ == "__main__":
    main()
