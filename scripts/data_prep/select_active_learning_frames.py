#!/usr/bin/env python3
"""
Active-learning frame selector for HRNet keypoint retraining.

Builds a large pool of candidate frames from existing play clips, runs the
current HRNet on each, scores them by uncertainty + desired signals, then
weighted-samples a target number with temporal dedup to produce the next
annotation batch.

Scoring (each term roughly 0-0.2, so score roughly 0-0.6):
  score = hash_uncertainty + 2.0 * sideline_uncertainty

  hash_uncertainty      — mean of (0.5 - conf).clip(0) over hash peaks >= 0.3.
                          Peaks in 0.3-0.5 range score up to 0.2; >0.5 is 0.
  sideline_uncertainty  — same formula, over sideline peaks. Weighted 2x
                          because sidelines are our weaker channel.

Selection: weighted stochastic sample (probability = score + 0.1) without
replacement, oversampled 3x, greedy temporal dedup within each play clip,
stop at target count.

The script has two phases that can run separately:

  --phase extract     Extract candidate frames from clip videos to JPEGs.
                      Run this locally where the MP4 clips live. Produces
                      <output-dir>/candidates/*.jpg and candidates_manifest.json.

  --phase score       Load candidate JPEGs, run HRNet, score, and select.
                      Can run locally on CPU (slow) or on a pod with CUDA.
                      Reads <output-dir>/candidates/ and writes
                      <output-dir>/images/ (selected) + scores.csv + manifest.json.

  --phase full        Do both phases (default). Only useful if clips and GPU
                      are on the same machine.

Typical Option-C workflow:
  # Locally
  python scripts/data_prep/select_active_learning_frames.py --phase extract \\
    --output-dir data/field_keypoints/al_round1

  # Upload data/field_keypoints/al_round1/candidates/ to pod, then on pod:
  python select_active_learning_frames.py --phase score --device cuda \\
    --weights /workspace/hrnet_finetuned_last.pth \\
    --output-dir /workspace/al_round1 \\
    --n-select 300

  # Download /workspace/al_round1/images/ + manifest.json + scores.csv

The output directory will contain:
  candidates/*.jpg           — all extracted candidate frames (phase extract)
  candidates_manifest.json   — metadata for each candidate
  images/*.jpg               — selected frames (phase score)
  manifest.json              — metadata for selected frames with scores
  scores.csv                 — scores for ALL scored candidates (for inspection)
"""

import argparse
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# When running locally we can import from src/. On the pod we make the
# script self-contained by falling back to inline definitions.
try:
    from src.homography.keypoint_detector import HRNetKeypointModel, _refine_peak
except ImportError:
    import timm
    import torch.nn as nn

    def _refine_peak(heatmap, y, x):
        h, w = heatmap.shape
        if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
            return float(x), float(y)
        dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
        dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
        dxx = heatmap[y, x + 1] - 2 * heatmap[y, x] + heatmap[y, x - 1]
        dyy = heatmap[y + 1, x] - 2 * heatmap[y, x] + heatmap[y - 1, x]
        ox = float(np.clip(-dx / dxx, -0.5, 0.5)) if abs(dxx) > 1e-6 else 0.0
        oy = float(np.clip(-dy / dyy, -0.5, 0.5)) if abs(dyy) > 1e-6 else 0.0
        return float(x) + ox, float(y) + oy

    class HRNetKeypointModel(nn.Module):
        def __init__(self, num_channels=2):
            super().__init__()
            self.backbone = timm.create_model(
                "hrnet_w48", pretrained=False,
                features_only=True, out_indices=(0,),
            )
            self.head = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_channels, 1, bias=True),
            )

        def forward(self, x):
            return self.head(self.backbone(x)[0])


# ── Constants ────────────────────────────────────────────────────────────
INPUT_H, INPUT_W = 512, 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CHANNELS = 2
CH_SIDELINE = 0
CH_HASH = 1

# Scoring thresholds
PEAK_THRESH_FOR_COUNT = 0.3      # min confidence to count as a detection
LOW_CONF_WINDOW = (0.3, 0.5)     # range where sideline bonus is awarded

# Sampling per play clip (fraction of clip duration)
TIME_POINTS = [0.05, 0.25, 0.5, 0.75, 0.9]

# Games reserved as evaluation holdouts — never sample for training.
HOLDOUT_GAMES = {"2024100601", "2024122201"}


# ── HRNet inference ──────────────────────────────────────────────────────

def load_model(weights_path, device):
    model = HRNetKeypointModel(num_channels=NUM_CHANNELS)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def extract_peak_confidences(heatmap, threshold):
    """Return a list of peak confidences from a single heatmap."""
    mask = heatmap >= threshold
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    peaks = []
    for comp_id in range(1, n + 1):
        comp_mask = labels == comp_id
        vals = heatmap * comp_mask
        peaks.append(float(vals.max()))
    return peaks


# ── Scoring ──────────────────────────────────────────────────────────────

def _channel_uncertainty(confs):
    """Mean of (0.5 - conf).clip(0) over peaks above PEAK_THRESH_FOR_COUNT.

    A peak at 0.3 contributes 0.2; at 0.5+ contributes 0. Frames with no
    detections score 0 (indeterminate — could be no-target or blind-miss).
    """
    if not confs:
        return 0.0
    return float(np.mean([max(0.0, 0.5 - c) for c in confs]))


def score_frame(heatmaps):
    """Compute active-learning score for a single frame.

    score = hash_uncertainty + 2.0 * sideline_uncertainty

    Sideline weighted higher because sidelines are the weaker channel —
    any ambiguity there is worth prioritizing for annotation.
    """
    sideline_confs = extract_peak_confidences(heatmaps[CH_SIDELINE], PEAK_THRESH_FOR_COUNT)
    hash_confs = extract_peak_confidences(heatmaps[CH_HASH], PEAK_THRESH_FOR_COUNT)

    hash_uncertainty = _channel_uncertainty(hash_confs)
    sideline_uncertainty = _channel_uncertainty(sideline_confs)
    score = hash_uncertainty + 2.0 * sideline_uncertainty

    return {
        "score": score,
        "hash_uncertainty": hash_uncertainty,
        "sideline_uncertainty": sideline_uncertainty,
        "n_hash": len(hash_confs),
        "n_sideline": len(sideline_confs),
    }


# ── Frame pool construction ──────────────────────────────────────────────

def build_frame_pool(clips_dir, game_ids, exclude_filenames, time_points=TIME_POINTS):
    """Enumerate candidate frames from clip manifests, excluding already-annotated."""
    candidates = []
    for game_id in game_ids:
        game_dir = os.path.join(clips_dir, game_id)
        manifest_path = os.path.join(game_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"  Skipping {game_id}: no manifest")
            continue
        with open(manifest_path) as f:
            m = json.load(f)

        for play in m.get("plays", []):
            play_num = play["play_num"]
            play_dir = os.path.join(game_dir, f"play_{play_num:03d}")
            sideline_path = os.path.join(play_dir, "sideline.mp4")
            if not os.path.exists(sideline_path):
                continue

            for pct in time_points:
                pct_int = int(pct * 100)
                fname = f"{game_id}_play_{play_num:03d}_sideline_{pct_int:02d}.jpg"
                if fname in exclude_filenames:
                    continue
                candidates.append({
                    "file_name": fname,
                    "game_id": game_id,
                    "play_num": play_num,
                    "play_dir": play_dir,
                    "video_path": sideline_path,
                    "time_pct": pct,
                })

    return candidates


def extract_frame_from_video(video_path, pct):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pct)))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── Sampling ─────────────────────────────────────────────────────────────

def weighted_sample_with_dedup(candidates, scores, n_select,
                                 oversample_factor=3, time_pct_min_gap=0.1,
                                 seed=42):
    """Weighted stochastic sample with per-play temporal dedup.

    Oversample at `oversample_factor * n_select` without replacement, then go
    through them in score order, skipping any candidate too close in time to
    an already-accepted candidate from the same play. Stops at n_select.
    """
    rng = np.random.default_rng(seed)

    scores_arr = np.asarray(scores, dtype=np.float64)
    weights = scores_arr + 0.1
    probs = weights / weights.sum()

    target_oversample = min(len(candidates), oversample_factor * n_select)

    # Sample indices
    candidate_indices = rng.choice(
        len(candidates), size=target_oversample,
        replace=False, p=probs,
    )

    # Sort by score descending
    score_order = np.argsort(-scores_arr[candidate_indices])
    ordered_indices = candidate_indices[score_order]

    # Greedy temporal dedup per play
    accepted_indices = []
    accepted_by_play = {}  # (game_id, play_num) -> list of time_pct

    for idx in ordered_indices:
        c = candidates[idx]
        key = (c["game_id"], c["play_num"])
        accepted_pcts = accepted_by_play.get(key, [])
        if any(abs(c["time_pct"] - p) < time_pct_min_gap for p in accepted_pcts):
            continue
        accepted_indices.append(int(idx))
        accepted_by_play.setdefault(key, []).append(c["time_pct"])
        if len(accepted_indices) >= n_select:
            break

    # If we ran out of oversample before reaching n_select, draw more
    if len(accepted_indices) < n_select and len(candidates) > target_oversample:
        remaining = n_select - len(accepted_indices)
        already = set(accepted_indices) | set(int(i) for i in candidate_indices)
        pool = [i for i in range(len(candidates)) if i not in already]
        if pool:
            extra_probs = probs[pool]
            extra_probs = extra_probs / extra_probs.sum()
            extra = rng.choice(pool, size=min(len(pool), remaining * 3),
                                replace=False, p=extra_probs)
            extra_order = np.argsort(-scores_arr[extra])
            for idx in extra[extra_order]:
                c = candidates[idx]
                key = (c["game_id"], c["play_num"])
                accepted_pcts = accepted_by_play.get(key, [])
                if any(abs(c["time_pct"] - p) < time_pct_min_gap for p in accepted_pcts):
                    continue
                accepted_indices.append(int(idx))
                accepted_by_play.setdefault(key, []).append(c["time_pct"])
                if len(accepted_indices) >= n_select:
                    break

    return accepted_indices


# ── Phase: extract candidate frames from videos ──────────────────────────

def run_extract(args):
    """Extract candidate frames from clips to disk. Must run where clips exist."""
    os.makedirs(args.output_dir, exist_ok=True)
    candidates_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    # Load annotated filenames to exclude (pool across all manifests)
    exclude = set()
    for path in args.annotated_manifests:
        if not os.path.exists(path):
            print(f"  Manifest not found (skipping): {path}")
            continue
        with open(path) as f:
            for entry in json.load(f):
                exclude.add(entry["file_name"])
        print(f"  Loaded exclusions from {path}")
    print(f"Excluding {len(exclude)} already-annotated frames")

    # Discover games
    if args.game_ids:
        game_ids = args.game_ids
    else:
        all_dirs = sorted(d for d in os.listdir(args.clips_dir)
                          if os.path.isdir(os.path.join(args.clips_dir, d)))
        game_ids = [g for g in all_dirs if g not in HOLDOUT_GAMES]
        excluded = [g for g in all_dirs if g in HOLDOUT_GAMES]
        if excluded:
            print(f"Excluding holdout games: {excluded}")
    print(f"Sampling from {len(game_ids)} games: {game_ids}")

    candidates = build_frame_pool(args.clips_dir, game_ids, exclude)
    print(f"Total candidate frames: {len(candidates)}")

    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        random.shuffle(candidates)
        candidates = candidates[:args.max_candidates]
        print(f"Capped to {len(candidates)} for dry run")

    # Extract each candidate
    t_start = __import__("time").time()
    last_print = t_start
    extracted = []
    for i, c in enumerate(candidates):
        out_path = os.path.join(candidates_dir, c["file_name"])
        if os.path.exists(out_path):
            # already extracted (resume-friendly)
            extracted.append(c)
            continue
        frame = extract_frame_from_video(c["video_path"], c["time_pct"])
        if frame is None:
            continue
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted.append(c)

        now = __import__("time").time()
        if now - last_print > 10 or (i + 1) == len(candidates):
            rate = (i + 1) / (now - t_start)
            eta = (len(candidates) - i - 1) / max(rate, 1e-6)
            print(f"  [{i + 1}/{len(candidates)}] rate={rate:.1f}/s eta={eta:.0f}s")
            last_print = now

    # Write candidates manifest (relative paths only — no video_path on pod)
    manifest = []
    for c in extracted:
        manifest.append({
            "file_name": c["file_name"],
            "game_id": c["game_id"],
            "play_num": c["play_num"],
            "time_pct": c["time_pct"],
        })

    manifest_path = os.path.join(args.output_dir, "candidates_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExtracted {len(extracted)} frames to {candidates_dir}")
    print(f"Wrote {manifest_path}")


# ── Phase: score extracted frames and select top-N ───────────────────────

def run_score(args):
    """Score pre-extracted candidate frames and select an active-learning batch."""
    candidates_dir = os.path.join(args.output_dir, "candidates")
    manifest_path = os.path.join(args.output_dir, "candidates_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found. Run --phase extract first.")
        sys.exit(1)

    with open(manifest_path) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidates from {manifest_path}")

    images_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loading model from {args.weights}")
    model = load_model(args.weights, device)

    scores = []
    details = []
    t_start = __import__("time").time()
    last_print = t_start

    for i, cand in enumerate(candidates):
        img_path = os.path.join(candidates_dir, cand["file_name"])
        frame = cv2.imread(img_path)
        if frame is None:
            scores.append(0.0)
            details.append({"score": 0.0, "failed": True})
            continue

        with torch.no_grad():
            tensor = preprocess(frame).to(device)
            logits = model(tensor)
            heatmaps = torch.sigmoid(logits[0]).cpu().numpy()

        result = score_frame(heatmaps)
        scores.append(result["score"])
        details.append(result)

        now = __import__("time").time()
        if now - last_print > 10 or (i + 1) == len(candidates):
            rate = (i + 1) / (now - t_start)
            eta = (len(candidates) - i - 1) / max(rate, 1e-6)
            print(f"  [{i + 1}/{len(candidates)}] rate={rate:.1f}/s eta={eta:.0f}s")
            last_print = now

    scores = np.array(scores)
    print(f"\nScore stats: min={scores.min():.3f}, max={scores.max():.3f}, "
          f"mean={scores.mean():.3f}, median={np.median(scores):.3f}")

    # Write full scores CSV
    scores_csv = os.path.join(args.output_dir, "scores.csv")
    with open(scores_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "game_id", "play_num", "time_pct", "score",
            "hash_uncertainty", "sideline_uncertainty",
            "n_hash", "n_sideline",
        ])
        for c, d in zip(candidates, details):
            writer.writerow([
                c["file_name"], c["game_id"], c["play_num"], c["time_pct"],
                round(d.get("score", 0.0), 4),
                round(d.get("hash_uncertainty", 0.0), 4),
                round(d.get("sideline_uncertainty", 0.0), 4),
                d.get("n_hash", 0),
                d.get("n_sideline", 0),
            ])
    print(f"Wrote {scores_csv}")

    # Weighted sample + dedup
    n_select = min(args.n_select, len(candidates))
    selected_indices = weighted_sample_with_dedup(
        candidates, scores, n_select, seed=args.seed,
    )
    print(f"Selected {len(selected_indices)} frames (target {n_select})")

    # Copy selected candidates to images/
    import shutil
    out_manifest = []
    for rank, idx in enumerate(selected_indices):
        c = candidates[idx]
        src = os.path.join(candidates_dir, c["file_name"])
        dst = os.path.join(images_out_dir, c["file_name"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        out_manifest.append({
            "file_name": c["file_name"],
            "game_id": c["game_id"],
            "play_num": c["play_num"],
            "time_pct": c["time_pct"],
            "score": float(scores[idx]),
            "selection_rank": rank,
            **{k: v for k, v in details[idx].items() if k != "score"},
        })

    manifest_out = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_out, "w") as f:
        json.dump(out_manifest, f, indent=2)
    print(f"Wrote {manifest_out}")

    per_game = {}
    for entry in out_manifest:
        per_game[entry["game_id"]] = per_game.get(entry["game_id"], 0) + 1
    print("\nSelected per game:")
    for gid, n in sorted(per_game.items()):
        print(f"  {gid}: {n}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="full", choices=["extract", "score", "full"],
                        help="'extract' pulls frames from videos to candidates/; "
                             "'score' runs HRNet on those JPEGs and selects; "
                             "'full' does both (requires videos + GPU on same host).")
    parser.add_argument("--weights", default=os.path.join(PROJECT_ROOT, "models", "hrnet_finetuned_last.pth"))
    parser.add_argument("--clips-dir", default=os.path.join(PROJECT_ROOT, "videos", "clips"))
    parser.add_argument("--annotated-manifests", nargs="+",
                        default=[
                            os.path.join(PROJECT_ROOT, "data", "field_keypoints",
                                         "manifest.json"),
                            os.path.join(PROJECT_ROOT, "data", "field_keypoints",
                                         "al_round1", "manifest.json"),
                        ],
                        help="Frames listed in any of these manifests are excluded.")
    parser.add_argument("--output-dir",
                        default=os.path.join(PROJECT_ROOT, "data", "field_keypoints", "al_round1"))
    parser.add_argument("--n-select", type=int, default=300)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Cap on total scored frames (0 = no cap). For dry runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game-ids", nargs="+", default=None,
                        help="Optional subset of game ids to sample from.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.phase in ("extract", "full"):
        run_extract(args)
    if args.phase in ("score", "full"):
        run_score(args)


if __name__ == "__main__":
    main()
