"""Build a pseudo-labeled training dataset for the unified mask model.

Per game (10 eligible, 4 NGS-test clips excluded), randomly sample N clips
and M frames per clip. For each frame, run all four existing specialists
(line UNet → yard + side, hash UNet, number UNet) to produce 4-channel
pseudo-mask labels. Save (rgb, masks_4ch) pairs as .npz.

Output structure:
  <out-dir>/raw/<game>_<play>_<frameidx>.npz   — one per frame
  <out-dir>/dataset_manifest.json               — list of all entries

The user QC's this in a mosaic UI, then a separate script splits the
Y'd subset into train/val and writes the actual training tensors.

Usage:
    python scripts/data_prep/build_unified_mask_dataset.py \\
        --frames-per-game 250 --device mps
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography import painted_numbers
from src.homography.rectify import (
    LINE_WEIGHTS, HASH_WEIGHTS, NUMBER_WEIGHTS,
    run_specialists,
)


EXCLUDED_GAMES = {"2019111007"}     # corrupt source
EXCLUDED_CLIP_TAGS = {              # NGS validation clips — keep clean
    "2019092204/play_065",
    "2019102712/play_011",
    "2019102712/play_046",
    "2019102712/play_118",
}


def discover_games(clips_root):
    games = []
    for d in sorted(glob.glob(os.path.join(clips_root, "*"))):
        if not os.path.isdir(d):
            continue
        game = os.path.basename(d)
        if game in EXCLUDED_GAMES:
            continue
        if not glob.glob(os.path.join(d, "play_*", "sideline.mp4")):
            continue
        games.append(game)
    return games


def list_clips(clips_root, game):
    """All sideline.mp4 paths for a game, with NGS-test clips excluded."""
    out = []
    for fn in sorted(glob.glob(os.path.join(clips_root, game, "play_*", "sideline.mp4"))):
        rel = os.path.relpath(fn, clips_root)
        tag = "/".join(rel.split(os.sep)[:2])
        if tag in EXCLUDED_CLIP_TAGS:
            continue
        out.append(fn)
    return out


def sample_frames_for_clip(clip_path, n_frames, rng, min_spacing=60):
    """Pick n_frames frame indices spread across the clip with at least
    min_spacing frames between them. Returns list of int frame indices."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return []
    if n_frames * min_spacing > total:
        # not enough room; pick evenly spaced
        if n_frames >= total:
            return list(range(total))
        step = max(1, total // (n_frames + 1))
        return [step * (i + 1) for i in range(n_frames)]
    # Random spacing within budget
    picks = sorted(rng.sample(range(0, total - 1), min(n_frames * 5, total)))
    out = []
    last = -10**9
    for p in picks:
        if p - last >= min_spacing:
            out.append(p)
            last = p
            if len(out) >= n_frames:
                break
    return out


def extract_and_label(clip_path, frame_indices, device):
    """Open clip, seek to each frame, extract RGB + run all 4 specialists
    to produce (yard, side, hash, number) binary masks.

    Returns list of dicts: {frame_idx, rgb (HxWx3 uint8), masks (HxWx4 uint8)}.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []
    out = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        # Line UNet → yard + side (returns binary 0/1, but stored as uint8)
        yard, side, hash_ = run_specialists(
            frame, LINE_WEIGHTS, HASH_WEIGHTS, device)
        # Number UNet returns a binary 0/255 mask at frame resolution
        num = painted_numbers.predict_mask(frame, NUMBER_WEIGHTS, device)
        # Normalize to {0, 1} per channel
        masks = np.stack([
            (yard > 0).astype(np.uint8),
            (side > 0).astype(np.uint8),
            (hash_ > 0).astype(np.uint8),
            (num > 0).astype(np.uint8),
        ], axis=-1)
        out.append({
            "frame_idx": int(fi),
            "rgb": frame,            # BGR uint8 (cv2 native)
            "masks": masks,
        })
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data/unified_masks/round1"))
    ap.add_argument("--frames-per-game", type=int, default=250)
    ap.add_argument("--clips-per-game", type=int, default=50,
                    help="Number of distinct clips to sample from per game.")
    ap.add_argument("--frames-per-clip", type=int, default=None,
                    help="Frames per sampled clip. Default = frames-per-game / clips-per-game.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--jpeg-thumb-quality", type=int, default=88)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.frames_per_clip is None:
        args.frames_per_clip = max(1, args.frames_per_game // args.clips_per_game)

    raw_dir = os.path.join(args.out_dir, "raw")
    thumb_dir = os.path.join(args.out_dir, "thumbs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)

    games = discover_games(args.clips_root)
    if args.max_games:
        games = games[: args.max_games]
    print(f"Found {len(games)} eligible games")
    print(f"Sampling target: {args.frames_per_game} frames/game = "
          f"{args.frames_per_game * len(games)} total")
    print(f"  ({args.clips_per_game} clips/game × ~{args.frames_per_clip} frames/clip)")

    manifest_entries = []
    t_start = time.time()

    for gi, game in enumerate(games, 1):
        clips = list_clips(args.clips_root, game)
        if not clips:
            print(f"  [{gi}/{len(games)}] {game}: no eligible clips, skip")
            continue
        # Sample clips
        n_clips = min(args.clips_per_game, len(clips))
        sampled_clips = rng.sample(clips, n_clips)
        # Determine per-clip frame count to hit the target
        target = args.frames_per_game
        per_clip = max(1, target // n_clips)

        print(f"\n[{gi}/{len(games)}] {game}: sampling {n_clips} clips "
              f"× {per_clip} frames = ~{n_clips * per_clip}")
        n_done_game = 0
        for ci, clip in enumerate(sampled_clips, 1):
            frame_indices = sample_frames_for_clip(clip, per_clip, rng)
            if not frame_indices:
                continue
            entries = extract_and_label(clip, frame_indices, args.device)
            for e in entries:
                clip_rel = os.path.relpath(clip, args.clips_root)
                tag = "_".join(clip_rel.split(os.sep)[:2])
                fid = f"{tag}_{e['frame_idx']:04d}"
                npz_path = os.path.join(raw_dir, f"{fid}.npz")
                np.savez_compressed(npz_path,
                                    rgb=e["rgb"], masks=e["masks"])
                # Lightweight thumbnail with overlay for QC UI
                # Channel colors (BGR): yard=red, side=green, hash=blue, number=yellow
                thumb = e["rgb"].copy()
                color_yard = np.array([60, 60, 230], dtype=np.uint8)
                color_side = np.array([60, 230, 60], dtype=np.uint8)
                color_hash = np.array([230, 60, 60], dtype=np.uint8)
                color_num  = np.array([60, 230, 230], dtype=np.uint8)
                masks = e["masks"]
                ov = thumb.copy()
                ov[masks[..., 0] > 0] = color_yard
                ov[masks[..., 1] > 0] = color_side
                ov[masks[..., 2] > 0] = color_hash
                ov[masks[..., 3] > 0] = color_num
                thumb = cv2.addWeighted(ov, 0.55, thumb, 0.45, 0)
                cv2.imwrite(os.path.join(thumb_dir, f"{fid}.jpg"), thumb,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_thumb_quality])
                manifest_entries.append({
                    "id": fid, "clip": clip_rel,
                    "frame_idx": e["frame_idx"], "game": game,
                })
                n_done_game += 1
            if ci % 10 == 0 or ci == len(sampled_clips):
                elapsed = time.time() - t_start
                print(f"  [{ci}/{len(sampled_clips)}] {n_done_game} frames so far "
                      f"({elapsed:.0f}s elapsed)", flush=True)

    manifest = {
        "n_total": len(manifest_entries),
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_per_game": args.frames_per_game,
        "clips_per_game": args.clips_per_game,
        "seed": args.seed,
        "entries": manifest_entries,
    }
    manifest_path = os.path.join(args.out_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")
    print(f"  {len(manifest_entries)} frames in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
