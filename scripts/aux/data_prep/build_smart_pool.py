"""Smart manifest builder for the dense field pool with clip-diversity-
aware subsampling.

Usage modes:

1. **Build a manifest of UNDECIDED entries to review** (strata 1+2+3):
       python scripts/data_prep/build_smart_pool.py \\
           --strata 1 2 3 --target-per-stratum 1000

2. **Build a final TRAINING manifest from Y'd entries** (after all Y/N is done):
       python scripts/data_prep/build_smart_pool.py \\
           --strata 0 1 2 3 4 --y-only --target-per-stratum 500 \\
           --out data/h_pool_and_intrinsics.json

In both modes, sampling within each stratum maximizes clip diversity:
  - First round: pick 1 entry per distinct clip
  - Second round: pick 2nd from each clip with available entries
  - ...continue until target met
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data_prep.build_dense_field_training_pool import (
    thumbnail_filename, STRATA_EDGES, N_STRATA,
)


def load_pool(pool_dir):
    """Read all stage files and build a flat candidate list."""
    stage_dir = os.path.join(pool_dir, "stage")
    out = []
    import glob
    for stage_file in sorted(glob.glob(os.path.join(stage_dir, "*.json"))):
        with open(stage_file) as f:
            data = json.load(f)
        if data.get("status") != "ok":
            continue
        clip_rel = data["clip"]
        for e in data["entries"]:
            e2 = dict(e)
            e2["clip"] = clip_rel
            e2["thumbnail"] = thumbnail_filename(clip_rel, e["frame_idx"])
            e2["id"] = e2["thumbnail"][:-len(".jpg")]
            out.append(e2)
    return out


def clip_diverse_sample(candidates, target, rng):
    """Round-robin sample across distinct clips:
      Round 1: pick (at most) one entry per clip
      Round 2: pick a second from each clip with available entries
      ... etc, until we reach target.

    Within each clip's candidate list, frames are picked in random order
    each round. Returns up to target entries.
    """
    by_clip = defaultdict(list)
    for c in candidates:
        by_clip[c["clip"]].append(c)
    # Shuffle each clip's list once so per-clip pick order is deterministic-random
    for clip in by_clip:
        rng.shuffle(by_clip[clip])
    clip_cursors = {clip: 0 for clip in by_clip}
    clip_keys = sorted(by_clip.keys())

    picked = []
    rng.shuffle(clip_keys)
    while len(picked) < target:
        before = len(picked)
        for clip in clip_keys:
            if len(picked) >= target:
                break
            cur = clip_cursors[clip]
            if cur >= len(by_clip[clip]):
                continue
            picked.append(by_clip[clip][cur])
            clip_cursors[clip] += 1
        if len(picked) == before:
            # No clip has any more candidates — exhausted
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=os.path.join(PROJECT_ROOT, "output/dense_field_pool"))
    ap.add_argument("--out", default=None,
                    help="Output manifest path. Default: <pool-dir>/manifest.json "
                         "(will overwrite the active mosaic UI manifest).")
    ap.add_argument("--strata", nargs="+", type=int, required=True,
                    help="Stratum indices to include (0..4).")
    ap.add_argument("--target-per-stratum", type=int, required=True)
    ap.add_argument("--y-only", action="store_true",
                    help="Only consider entries that already have a 'y' decision.")
    ap.add_argument("--n-only", action="store_true",
                    help="Only consider entries with a 'n' decision (debug).")
    ap.add_argument("--exclude-decided", action="store_true",
                    help="Skip entries that have any decision (Y or N).")
    ap.add_argument("--require-thumb", action="store_true",
                    help="Skip entries without a local thumbnail.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.y_only and args.n_only:
        sys.exit("--y-only and --n-only are mutually exclusive")
    if args.y_only and args.exclude_decided:
        sys.exit("--y-only and --exclude-decided are mutually exclusive")

    rng = random.Random(args.seed)
    pool = load_pool(args.pool_dir)
    print(f"Loaded {len(pool)} candidates from {args.pool_dir}/stage/")

    # Decision filter
    decisions_path = os.path.join(args.pool_dir, "decisions.json")
    decisions = json.load(open(decisions_path)) if os.path.exists(decisions_path) else {}

    keep_strata = set(args.strata)
    frames_dir = os.path.join(args.pool_dir, "frames")

    final_entries = []
    print(f"\nSampling {args.target_per_stratum} per stratum from "
          f"{sorted(keep_strata)}:")
    for s in sorted(keep_strata):
        cands = [c for c in pool if c["stratum"] == s]
        v = decisions.get
        if args.y_only:
            cands = [c for c in cands if v(c["id"]) == "y"]
        elif args.n_only:
            cands = [c for c in cands if v(c["id"]) == "n"]
        elif args.exclude_decided:
            cands = [c for c in cands if c["id"] not in decisions]
        if args.require_thumb:
            cands = [c for c in cands
                     if os.path.exists(os.path.join(frames_dir, c["thumbnail"]))]
        n_clips_avail = len(set(c["clip"] for c in cands))
        picks = clip_diverse_sample(cands, args.target_per_stratum, rng)
        n_clips_picked = len(set(c["clip"] for c in picks))
        avg_per_clip = len(picks) / max(1, n_clips_picked)
        print(f"  stratum {s}: {len(cands)} candidates ({n_clips_avail} clips) → "
              f"sampled {len(picks)} ({n_clips_picked} clips, "
              f"{avg_per_clip:.2f}/clip)")
        final_entries.extend(picks)

    rng.shuffle(final_entries)

    out_path = args.out or os.path.join(args.pool_dir, "manifest.json")
    manifest = {
        "n_pool": len(final_entries),
        "is_partial": True,
        "strata_included": sorted(keep_strata),
        "target_per_stratum": args.target_per_stratum,
        "y_only": args.y_only,
        "exclude_decided": args.exclude_decided,
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "entries": final_entries,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {out_path}")
    print(f"  {len(final_entries)} entries total")


if __name__ == "__main__":
    main()
