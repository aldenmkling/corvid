"""Generate the training-pool candidates for the dense field-coord regression
model.

Pipeline per clip:
  1. Run rectify (compute_homographies) → per-frame H + metadata.
  2. Sample frames at 1Hz (every 30 source frames).
  3. Keep frames where:
       method == "full"
       inlier_frac >= 0.95
       clip-level quality_flag.questionable == False
  4. Compute per-frame stratum from mean(NGS_x of corrs):
       0: [10, 30)   1: [30, 50)   2: [50, 70)   3: [70, 90)   4: [90, 110]
  5. Append (clip, frame_idx, stratum, H, n_corrs, inlier_frac, mean_ngs_x)
     to a per-clip stage file.

Across all clips:
  6. Aggregate stage files → per-stratum pools.
  7. Random subsample each pool to `frames_per_stratum` (default 1200) so the
     final pool is balanced by camera-position bucket.
  8. For each surviving entry, extract the source frame from the MP4 + render
     a projected-grid overlay using the saved H. Save both as JPEGs into
     <out_dir>/frames/.
  9. Write <out_dir>/manifest.json with the final pool.

Resume-friendly: per-clip stage files mean a crash mid-batch can be resumed
without recomputing the homography for completed clips.

Usage:
    python scripts/data_prep/build_dense_field_training_pool.py \\
        --out-dir output/dense_field_pool \\
        --device cuda                          # mps for laptop, cuda on pod
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.homography.rectify import (
    compute_homography_at_sampled_frames,
    NGS_X_LEFT_GOAL, NGS_X_RIGHT_GOAL,
    render_review_rectified,
)


# Clips that must NOT enter the training pool (NGS validation set).
EXCLUDED_CLIP_TAGS = {
    "2019092204/play_065",
    "2019102712/play_011",
    "2019102712/play_046",
    "2019102712/play_118",
}

# Games with corrupted source video.
EXCLUDED_GAMES = {
    "2019111007",
}

STRATA_EDGES = [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]   # 5 buckets
N_STRATA = len(STRATA_EDGES) - 1


def list_clips(clips_root):
    """Return all sideline.mp4 paths under videos/clips/<game>/<play>/."""
    pat = os.path.join(clips_root, "*", "play_*", "sideline.mp4")
    out = []
    for path in sorted(glob.glob(pat)):
        rel = os.path.relpath(path, clips_root)
        # <game>/<play>/sideline.mp4 → <game>/<play>
        tag = "/".join(rel.split(os.sep)[:2])
        game = rel.split(os.sep)[0]
        if game in EXCLUDED_GAMES:
            continue
        if tag in EXCLUDED_CLIP_TAGS:
            continue
        out.append(path)
    return out


def stratum_for_x(mean_x):
    """Return the stratum index 0..N_STRATA-1 for a mean NGS_x value, or -1
    if outside [10, 110]."""
    if mean_x < STRATA_EDGES[0] or mean_x > STRATA_EDGES[-1]:
        return -1
    for i in range(N_STRATA):
        if mean_x < STRATA_EDGES[i + 1]:
            return i
    return N_STRATA - 1   # exactly at right edge


def stage_clip(clip_path, device, sample_stride, inlier_frac_min, verbose=True):
    """Solve H at sample_stride-spaced frames, filter, return entries +
    per-clip camera intrinsics (K, dist) needed to undistort frames at
    thumbnail-render time."""
    res = compute_homography_at_sampled_frames(
        video_path=clip_path, sample_stride=sample_stride,
        device=device, verbose=False,
    )
    if res is None:
        return None, "compute_homography_at_sampled_frames returned None", None
    samples = res["samples"]
    entries = []
    for s in samples:
        if s["inlier_frac"] < inlier_frac_min:
            continue
        stratum = stratum_for_x(s["mean_ngs_x"])
        if stratum < 0:
            continue
        entries.append({
            "frame_idx": int(s["frame_idx"]),
            "stratum": int(stratum),
            "n_corrs": int(s["n_corrs"]),
            "inlier_frac": float(s["inlier_frac"]),
            "mean_ngs_x": float(s["mean_ngs_x"]),
            "ngs_x_range": float(s["ngs_x_range"]),
            "H": [list(map(float, row)) for row in s["H"].tolist()],
        })
    intrinsics = {
        "K": [list(map(float, row)) for row in res["K"].tolist()],
        "dist": list(map(float, res["dist"].tolist())),
    }
    return entries, "ok", intrinsics


def thumbnail_filename(clip_rel, frame_idx):
    """Deterministic filename per (clip, frame). Stable across runs so the
    per-game RunPod flow can render once and the local aggregator can
    reference the same files without renumbering. Format:
    <game>_<play>_<frameidx>.jpg
    """
    parts = clip_rel.split(os.sep)[:2]      # <game>/<play>
    tag = "_".join(parts)
    return f"{tag}_{int(frame_idx):04d}.jpg"


def extract_thumbnails(clip_path, entries, K, dist, out_frames_dir,
                          jpeg_quality=88):
    """Open the clip once, seek to each entry's frame_idx, render the
    annotated rectified canvas and write to a deterministic filename."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return 0
    written = 0
    for e in entries:
        thumb_path = os.path.join(out_frames_dir, e["thumbnail"])
        if os.path.exists(thumb_path):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(e["frame_idx"]))
        ok, frame = cap.read()
        if not ok:
            continue
        H = np.array(e["H"], dtype=np.float64)
        canvas = render_review_rectified(frame, H, K, dist)
        cv2.imwrite(thumb_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=os.path.join(PROJECT_ROOT, "videos/clips"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "output/dense_field_pool"))
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--sample-stride", type=int, default=30,
                    help="Frame stride for 1Hz sampling at 30fps.")
    ap.add_argument("--inlier-frac-min", type=float, default=0.95)
    ap.add_argument("--frames-per-stratum", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-clips", type=int, default=None,
                    help="For debug: stop after this many clips.")
    ap.add_argument("--game", default=None,
                    help="Only process clips from this game id (e.g., "
                         "2019092204). Used by the per-game RunPod runner.")
    ap.add_argument("--skip-thumbnails", action="store_true",
                    help="Just write manifest, don't render JPEGs.")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    stage_dir = os.path.join(args.out_dir, "stage")
    os.makedirs(stage_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    clips = list_clips(args.clips_root)
    if args.game:
        clips = [c for c in clips if f"/{args.game}/" in c]
        print(f"Filtered to game {args.game}: {len(clips)} clips")
    if args.max_clips:
        clips = clips[: args.max_clips]
    print(f"Discovered {len(clips)} eligible clips "
          f"(excluded {len(EXCLUDED_CLIP_TAGS)} NGS-test + "
          f"{len(EXCLUDED_GAMES)} broken games)")

    # ── PASS 1: stage each clip's candidates ───────────────────────────────
    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    for i, clip in enumerate(clips, 1):
        rel = os.path.relpath(clip, args.clips_root)
        tag = "_".join(rel.split(os.sep)[:2])
        stage_path = os.path.join(stage_dir, f"{tag}.json")
        if os.path.exists(stage_path):
            n_skipped += 1
            continue
        try:
            entries, status, intrinsics = stage_clip(
                clip, device=args.device, sample_stride=args.sample_stride,
                inlier_frac_min=args.inlier_frac_min, verbose=False)
        except Exception as e:
            print(f"  [{i}/{len(clips)}] {tag}: FAILED ({type(e).__name__}: {e})", flush=True)
            n_failed += 1
            with open(stage_path + ".error", "w") as f:
                json.dump({"error": str(e), "type": type(e).__name__}, f)
            continue
        if entries is None:
            print(f"  [{i}/{len(clips)}] {tag}: SKIP ({status})", flush=True)
            with open(stage_path, "w") as f:
                json.dump({"clip": rel, "tag": tag, "status": status,
                           "entries": [], "intrinsics": None}, f)
            n_failed += 1
            continue
        with open(stage_path, "w") as f:
            json.dump({"clip": rel, "tag": tag, "status": status,
                       "entries": entries, "intrinsics": intrinsics}, f)
        n_done += 1
        if i % 20 == 0 or i == len(clips):
            elapsed = time.time() - t0
            eta = elapsed / max(1, i) * (len(clips) - i)
            print(f"  [{i}/{len(clips)}] tag={tag}  kept={len(entries)}  "
                  f"({elapsed:.0f}s elapsed, eta {eta:.0f}s)", flush=True)
    print(f"\nStaging done: {n_done} new, {n_skipped} resumed, {n_failed} failed.")

    # ── PASS 2: aggregate into per-stratum buckets ─────────────────────────
    buckets = defaultdict(list)
    intrinsics_by_clip = {}    # rel_clip_path → {K, dist}
    for stage_file in sorted(glob.glob(os.path.join(stage_dir, "*.json"))):
        with open(stage_file) as f:
            data = json.load(f)
        clip_rel = data["clip"]
        intr = data.get("intrinsics")
        if intr is not None:
            intrinsics_by_clip[clip_rel] = intr
        for e in data["entries"]:
            e2 = dict(e)
            e2["clip"] = clip_rel
            e2["thumbnail"] = thumbnail_filename(clip_rel, e["frame_idx"])
            buckets[e["stratum"]].append(e2)
    total_candidates = sum(len(v) for v in buckets.values())
    print(f"\nAggregated {total_candidates} candidates across {N_STRATA} strata:")
    for s in range(N_STRATA):
        print(f"  stratum {s} [NGS_x {STRATA_EDGES[s]:.0f}-{STRATA_EDGES[s+1]:.0f}): "
              f"{len(buckets[s])} candidates")

    # ── PASS 3: subsample each stratum ─────────────────────────────────────
    pool = []
    for s in range(N_STRATA):
        cands = buckets[s]
        if len(cands) > args.frames_per_stratum:
            picks = random.sample(cands, args.frames_per_stratum)
        else:
            picks = cands
            print(f"  WARN: stratum {s} has only {len(cands)} < target "
                  f"{args.frames_per_stratum}")
        pool.extend(picks)
    random.shuffle(pool)
    # id = thumbnail filename (without .jpg) so the mosaic UI's <id>.jpg
    # references the deterministic on-disk thumbnail.
    for e in pool:
        e["id"] = e["thumbnail"][:-len(".jpg")]
    print(f"\nFinal pool: {len(pool)} frames.")

    # ── PASS 4: extract + render thumbnails ────────────────────────────────
    if not args.skip_thumbnails:
        # Group by clip so we open each MP4 once.
        by_clip = defaultdict(list)
        for e in pool:
            by_clip[e["clip"]].append(e)
        print(f"Extracting thumbnails for {len(pool)} frames "
              f"across {len(by_clip)} clips...")
        t1 = time.time()
        n_written = 0
        for ci, (clip_rel, entries) in enumerate(by_clip.items(), 1):
            entries.sort(key=lambda e: e["frame_idx"])
            clip_full = os.path.join(args.clips_root, clip_rel)
            intr = intrinsics_by_clip.get(clip_rel)
            if intr is None:
                K, dist = None, None
            else:
                K = np.array(intr["K"], dtype=np.float64)
                dist = np.array(intr["dist"], dtype=np.float64)
            wrote = extract_thumbnails(
                clip_full, entries, K, dist, frames_dir)
            n_written += wrote
            if ci % 50 == 0 or ci == len(by_clip):
                print(f"  [{ci}/{len(by_clip)}] {n_written} thumbnails  "
                      f"({time.time()-t1:.0f}s)", flush=True)
        print(f"Wrote {n_written} thumbnails to {frames_dir}")

    # ── PASS 5: save manifest ──────────────────────────────────────────────
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "n_pool": len(pool),
            "n_strata": N_STRATA,
            "strata_edges": STRATA_EDGES,
            "frames_per_stratum_target": args.frames_per_stratum,
            "sample_stride": args.sample_stride,
            "inlier_frac_min": args.inlier_frac_min,
            "seed": args.seed,
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": pool,
        }, f, indent=2)
    print(f"\nManifest -> {manifest_path}")
    print(f"Total wall clock: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
