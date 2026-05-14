"""Wrapper that runs farm_pseudo_labels.py on every unique clip in the
training manifest.

Reads the manifest, extracts unique clip paths, and iterates by clip.
Uses os.system to keep state isolated per clip — if a clip crashes, the
others are unaffected.

Resume-friendly: skips clips whose .npz already exists in --out-dir.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/h_pool_and_intrinsics.json")
    ap.add_argument("--clips-dir", default="videos/clips")
    ap.add_argument("--out-dir", default="data/pseudo_labels")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--game-filter", default=None,
                       help="if given, only process clips whose path "
                            "contains this substring (e.g. '2024091501')")
    ap.add_argument("--exclude-games", nargs="+",
                       default=["2024090802", "2024100601", "2024122201"],
                       help="game IDs to skip — defaults to val + 2 holdouts "
                            "to avoid training-data leakage")
    ap.add_argument("--max-clips", type=int, default=None,
                       help="cap the number of clips to process this run")
    ap.add_argument("--worker-id", type=int, default=0,
                       help="partition id for parallel workers (0..N-1)")
    ap.add_argument("--num-workers", type=int, default=1,
                       help="total worker count; each worker only takes "
                            "clips where sorted_index %% num_workers == worker_id")
    ap.add_argument("--dry-run", action="store_true",
                       help="just print what would be run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(args.manifest))
    clips = sorted({e["clip"] for e in manifest["entries"]})

    if args.game_filter:
        clips = [c for c in clips if args.game_filter in c]
    for g in args.exclude_games:
        clips = [c for c in clips if g not in c]

    print(f"Manifest covers {len(clips)} unique clips after filters "
            f"(excluded games: {args.exclude_games})")

    todo = []
    for idx, clip_rel in enumerate(clips):
        # Partition by sorted index for stable, collision-free sharding
        # across parallel workers. Workers race only on skip-if-exists
        # for clips OTHER workers already finished — never on the same
        # clip simultaneously.
        if args.num_workers > 1 and (idx % args.num_workers) != args.worker_id:
            continue
        clip_path = os.path.join(args.clips_dir, clip_rel)
        clip_id = clip_rel.replace("/", "_").replace(".mp4", "")
        out_path = os.path.join(args.out_dir, f"{clip_id}.npz")
        if os.path.exists(out_path):
            continue
        if not os.path.exists(clip_path):
            # Don't spam when the clip simply hasn't been uploaded yet.
            continue
        todo.append(clip_path)

    print(f"[worker {args.worker_id}/{args.num_workers}] "
            f"{len(todo)} clips to process")
    if args.max_clips:
        todo = todo[:args.max_clips]
        print(f"capped at {len(todo)} for this run")

    if args.dry_run:
        for c in todo[:10]:
            print(f"  would run: {c}")
        if len(todo) > 10:
            print(f"  ... + {len(todo) - 10} more")
        return

    farm_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "farm_pseudo_labels.py")

    t_total = time.time()
    for i, clip_path in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {clip_path}")
        cmd = [
            sys.executable, "-u", farm_script,
            "--clip", clip_path,
            "--clips-dir", args.clips_dir,
            "--manifest", args.manifest,
            "--out-dir", args.out_dir,
            "--device", args.device,
        ]
        env = os.environ.copy()
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            subprocess.run(cmd, env=env, check=False)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        dt = time.time() - t0
        print(f"  clip done in {dt:.0f}s   "
                f"running total: {(time.time() - t_total)/60:.1f} min")

    print(f"\nAll done. Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
