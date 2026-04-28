#!/usr/bin/env python3
"""UNet active-learning + diversity sampler for line-detection training.

Workflow (3 phases — same shape as `select_active_learning_frames.py`):

  1. EXTRACT — sample N_pool random frames across non-holdout clips, save
     them as JPEGs. Cheap (fps≈30 with a few I/O hits).

  2. SCORE — for each candidate, run current UNet, compute:
       - uncertainty_score:  mean of (1 − max(p_yard, p_side)) over pixels
                             where max(p_yard, p_side) > LIKELY_LINE_THRESH.
                             High = UNet is unsure where the lines are.
       - sideline_vis_score: fraction of pixels predicted as sideline at
                             confidence > SIDELINE_CONF. Captures "is the
                             sideline actually visible in this frame?"
       - composite = α·uncertainty + β·sideline_vis  (α, β tunable).

     We deliberately also keep some "easy/normal" frames in the final pool —
     the prediction quality is now BETTER than the original 300-frame
     ground-truth labels, so the new training set should not be combined
     with the old. To avoid the pool being all-weird-cases, we add a small
     epsilon to every frame's selection probability.

  3. SELECT — weighted-random sample with prob ∝ (composite + ε), without
     replacement, with greedy temporal dedup per play (no two frames within
     `min_clip_gap` of the same play). Outputs a manifest + the selected
     JPEGs in `<out>/images/`.

Usage:
  Phase 1 (local extract):
    python scripts/data_prep/select_unet_al_frames.py --phase extract \\
        --out data/line_detection/al_round3 --n-pool 6000

  Phase 2 (local or pod, MPS/CUDA):
    python scripts/data_prep/select_unet_al_frames.py --phase score \\
        --out data/line_detection/al_round3 --n-select 300 \\
        --device mps --weights models/unet_line_round2_best.pth

The two phases are split because EXTRACT is I/O-bound and SCORE benefits
from GPU. You can run extract locally, then score on RunPod if needed.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CLIPS_DIR = os.path.join(PROJECT_ROOT, "videos", "clips")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "unet_line_round2_best.pth")
HOLDOUT_GAMES = {"2024100601", "2024122201"}    # never sample for training

# UNet score thresholds — frame-level signal extraction.
LIKELY_LINE_THRESH = 0.30                       # treat pixels with max-prob > this as "near a line"
SIDELINE_CONF = 0.30                            # threshold for "is sideline visible here"
SIDELINE_VIS_PIXEL_FLOOR = 50                   # ignore frames with < this many sideline pixels
                                                 # → treats them as "0" instead of noise.

# Selection weights (composite = α·uncertainty + β·sideline_vis + ε)
DEFAULT_ALPHA = 1.0                             # uncertainty weight
DEFAULT_BETA = 2.0                              # sideline visibility weight (we want these!)
DEFAULT_EPSILON = 0.10                          # baseline prob — keeps "normal" frames in the pool


# ───────────────────────────── Phase 1: extract ─────────────────────────────

def enumerate_clips(clips_dir, angles=("sideline",)):
    """List (game, play, angle, mp4) for every play clip, holdouts excluded."""
    clips = []
    for game in sorted(os.listdir(clips_dir)):
        if game in HOLDOUT_GAMES:
            continue
        game_dir = os.path.join(clips_dir, game)
        if not os.path.isdir(game_dir):
            continue
        for play in sorted(os.listdir(game_dir)):
            play_dir = os.path.join(game_dir, play)
            if not os.path.isdir(play_dir) or not play.startswith("play_"):
                continue
            for angle in angles:
                mp4 = os.path.join(play_dir, f"{angle}.mp4")
                if os.path.exists(mp4):
                    clips.append((game, play, angle, mp4))
    return clips


def grab_random_frame(mp4_path, rng):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None, -1, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return None, -1, 0
    idx = rng.randrange(max(1, n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, -1, n
    return frame, idx, n


def phase_extract(args):
    rng = random.Random(args.seed)
    clips = enumerate_clips(args.clips_dir, angles=tuple(args.angles))
    if not clips:
        print(f"  no clips found under {args.clips_dir}"); return
    print(f"  enumerated {len(clips)} clips, sampling {args.n_pool} frames")

    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    manifest_rows = []
    t0 = time.time()
    for i in range(args.n_pool):
        # Sample uniformly across clips so big games don't dominate.
        game, play, angle, mp4 = rng.choice(clips)
        frame, frame_idx, total = grab_random_frame(mp4, rng)
        if frame is None:
            continue
        clip_frac = (frame_idx / max(total - 1, 1)) if total > 1 else 0.0
        fname = f"{game}_{play}_{angle}_f{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(cand_dir, fname), frame)
        manifest_rows.append({
            "filename": fname, "game": game, "play": play, "angle": angle,
            "frame_idx": int(frame_idx), "clip_total": int(total),
            "clip_frac": float(clip_frac),
        })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{args.n_pool}]  {elapsed:.1f}s elapsed")

    manifest_path = os.path.join(args.out, "candidates_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_rows, f, indent=2)
    print(f"  wrote {len(manifest_rows)} candidates → {cand_dir}")
    print(f"  manifest: {manifest_path}")


# ────────────────────────── Phase 2: score + select ──────────────────────────

def load_unet(weights_path, device):
    import torch
    import segmentation_models_pytorch as smp
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=2,
                      activation=None)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def run_unet_probs(model, frame, device):
    """Return (probs_yard, probs_side) at UNet's native output resolution
    (no resize — we score on the raw probabilities)."""
    import torch
    INPUT_H, INPUT_W = 512, 896
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normed, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs[0], probs[1]


def score_frame(probs_yard, probs_side):
    """Per-frame uncertainty + sideline-visibility scores.

    uncertainty_score: among pixels where max(p_yard, p_side) > LIKELY_LINE_THRESH,
      the mean of (1 − max). Bounded by 1−LIKELY_LINE_THRESH (=0.7). Captures
      "the model thinks lines are here but isn't sure". 0 if no likely-line
      pixels found (UNet thinks the frame has no lines at all — probably a
      segment cut or replay graphic; not useful for training).

    sideline_vis_score: fraction of pixels with p_side > SIDELINE_CONF.
      Bounded by 1.0 in theory; in practice <0.05 even on sideline-heavy frames.
      Multiplied by 20 below to scale to the same magnitude as uncertainty.
    """
    max_p = np.maximum(probs_yard, probs_side)
    likely = max_p > LIKELY_LINE_THRESH
    if likely.sum() == 0:
        unc = 0.0
    else:
        unc = float((1.0 - max_p[likely]).mean())

    side_pix = int((probs_side > SIDELINE_CONF).sum())
    if side_pix < SIDELINE_VIS_PIXEL_FLOOR:
        side_vis = 0.0
    else:
        side_vis = float(side_pix) / float(probs_side.size)
    return unc, side_vis


def phase_score(args):
    import torch
    cand_dir = os.path.join(args.out, "candidates")
    if not os.path.isdir(cand_dir):
        print(f"  no candidates dir: {cand_dir}  (run --phase extract first)")
        return
    manifest_path = os.path.join(args.out, "candidates_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  scoring {len(manifest)} candidates with UNet on {args.device}")

    device = torch.device(args.device)
    model = load_unet(args.weights, device)

    rows = []
    t0 = time.time()
    for i, row in enumerate(manifest):
        path = os.path.join(cand_dir, row["filename"])
        frame = cv2.imread(path)
        if frame is None:
            continue
        probs_yard, probs_side = run_unet_probs(model, frame, device)
        unc, side_vis = score_frame(probs_yard, probs_side)
        composite = args.alpha * unc + args.beta * side_vis
        rows.append({**row,
                     "uncertainty": unc, "sideline_vis": side_vis,
                     "composite": composite})
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{len(manifest)}]  {rate:.1f} fps  "
                  f"{elapsed:.0f}s elapsed")

    # Save full scores CSV (so user can inspect/threshold differently if desired).
    scores_csv = os.path.join(args.out, "scores.csv")
    with open(scores_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "game", "play", "angle",
                                            "frame_idx", "clip_total", "clip_frac",
                                            "uncertainty", "sideline_vis", "composite"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  scores written → {scores_csv}")

    # Selection: weighted random with epsilon, dedup by play.
    selected = weighted_random_select(rows, args.n_select, args.seed,
                                       args.epsilon, args.oversample,
                                       args.min_clip_gap)
    print(f"  selected {len(selected)} frames")

    out_images_dir = os.path.join(args.out, "images")
    os.makedirs(out_images_dir, exist_ok=True)
    for r in selected:
        src = os.path.join(cand_dir, r["filename"])
        dst = os.path.join(out_images_dir, r["filename"])
        if not os.path.exists(dst):
            # copy via re-encode (simplest, no PIL/shutil dance)
            frame = cv2.imread(src)
            cv2.imwrite(dst, frame)

    # Manifest of selected frames.
    sel_manifest = os.path.join(args.out, "manifest.json")
    with open(sel_manifest, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  selected manifest → {sel_manifest}")

    # Label Studio import JSON (mirrors the existing keypoint AL workflow).
    ls_import = []
    for r in selected:
        ls_import.append({
            "data": {"image": f"al_round3/images/{r['filename']}"},
            "meta": {k: r[k] for k in
                     ("game", "play", "angle", "frame_idx",
                      "uncertainty", "sideline_vis", "composite")},
        })
    ls_path = os.path.join(args.out, "ls_import.json")
    with open(ls_path, "w") as f:
        json.dump(ls_import, f, indent=2)
    print(f"  Label Studio import → {ls_path}")


def phase_reselect(args):
    """Reuse scores.csv from a prior `score` run, just re-do the weighted-random
    selection with the current α/β/ε. Useful for tuning the mix without
    re-running UNet inference (~8 min saved)."""
    scores_csv = os.path.join(args.out, "scores.csv")
    cand_dir = os.path.join(args.out, "candidates")
    if not os.path.exists(scores_csv):
        print(f"  no scores.csv at {scores_csv}; run --phase score first")
        return
    rows = []
    with open(scores_csv) as f:
        for r in csv.DictReader(f):
            r["clip_frac"] = float(r.get("clip_frac", 0.0) or 0.0)
            r["uncertainty"] = float(r["uncertainty"])
            r["sideline_vis"] = float(r["sideline_vis"])
            # Recompute composite with the CURRENT alpha/beta (CSV has whatever
            # the prior run used).
            r["composite"] = args.alpha * r["uncertainty"] + args.beta * r["sideline_vis"]
            rows.append(r)
    print(f"  reselecting from {len(rows)} scored candidates "
          f"(α={args.alpha} β={args.beta} ε={args.epsilon})")
    selected = weighted_random_select(rows, args.n_select, args.seed,
                                       args.epsilon, args.oversample,
                                       args.min_clip_gap)
    print(f"  selected {len(selected)} frames")

    out_images_dir = os.path.join(args.out, "images")
    # Wipe old image selection (since we're replacing it).
    if os.path.isdir(out_images_dir):
        for f in os.listdir(out_images_dir):
            os.remove(os.path.join(out_images_dir, f))
    else:
        os.makedirs(out_images_dir, exist_ok=True)
    for r in selected:
        src = os.path.join(cand_dir, r["filename"])
        dst = os.path.join(out_images_dir, r["filename"])
        frame = cv2.imread(src)
        if frame is not None:
            cv2.imwrite(dst, frame)

    sel_manifest = os.path.join(args.out, "manifest.json")
    with open(sel_manifest, "w") as f:
        json.dump(selected, f, indent=2)
    ls_import = []
    base = os.path.basename(os.path.normpath(args.out))
    for r in selected:
        ls_import.append({
            "data": {"image": f"/data/local-files/?d={base}/images/{r['filename']}"},
            "meta": {k: r[k] for k in
                     ("game", "play", "angle", "frame_idx",
                      "uncertainty", "sideline_vis", "composite")},
        })
    ls_path = os.path.join(args.out, "ls_import.json")
    with open(ls_path, "w") as f:
        json.dump(ls_import, f, indent=2)
    print(f"  selected manifest → {sel_manifest}")
    print(f"  Label Studio import → {ls_path}")


def weighted_random_select(rows, n_target, seed, epsilon, oversample, min_clip_gap):
    """Weighted-random sample with per-play temporal dedup.

    `prob = composite + epsilon`. The +ε guarantees every frame has nonzero
    chance, so the final pool isn't 100% high-uncertainty/sideline-only —
    we keep some normal frames in the set (per user direction: "randomly
    select so we get some normal frames in there").
    """
    rng = random.Random(seed)
    weights = np.array([max(r["composite"], 0.0) + epsilon for r in rows],
                       dtype=np.float64)
    weights /= weights.sum()
    n_to_draw = min(len(rows), oversample * n_target)
    drawn_idx = list(rng.choices(range(len(rows)), weights=weights, k=n_to_draw))
    # Order drawn by random walk; do greedy temporal dedup per play.
    seen = []
    seen_by_play: dict[tuple[str, str], list[float]] = {}
    for i in drawn_idx:
        if len(seen) >= n_target:
            break
        r = rows[i]
        key = (r["game"], r["play"])
        clip_frac = r.get("clip_frac", 0.0)
        if key in seen_by_play:
            if any(abs(clip_frac - prev) < min_clip_gap for prev in seen_by_play[key]):
                continue
        seen_by_play.setdefault(key, []).append(clip_frac)
        seen.append(r)
    return seen


# ───────────────────────── arg parsing + main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["extract", "score", "reselect"], required=True,
                    help="extract: pull random frames. score: run UNet + select. "
                         "reselect: reuse scores.csv from a prior `score` run "
                         "and just re-do the weighted-random selection (cheap).")
    ap.add_argument("--out", required=True,
                    help="Output dir (will contain candidates/, scores.csv, images/, manifest.json)")
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR)
    ap.add_argument("--angles", nargs="+", default=["sideline"])
    ap.add_argument("--n-pool", type=int, default=6000,
                    help="Phase 1: how many candidate frames to extract")
    ap.add_argument("--n-select", type=int, default=300,
                    help="Phase 2: how many frames to keep after scoring + sampling")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Weight on uncertainty term")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA,
                    help="Weight on sideline-visibility term")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                    help="Floor probability — keeps normal frames in pool")
    ap.add_argument("--oversample", type=int, default=3,
                    help="Draw oversample × n_select frames before dedup")
    ap.add_argument("--min-clip-gap", type=float, default=0.10,
                    help="Min temporal separation (clip-fraction) between two "
                         "frames from the same play")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.phase == "extract":
        phase_extract(args)
    elif args.phase == "score":
        phase_score(args)
    else:
        phase_reselect(args)


if __name__ == "__main__":
    main()
