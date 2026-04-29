# NFL Player Tracking Pipeline

## Project Goal
Build a computer vision pipeline that extracts player tracking data from NFL
All-22 film (YouTube source, 720p, 30fps). The extracted tracking data should
mimic NGS-style output: per-frame x/y field coordinates for all 22 players.
The primary use case is building tape-based athletic profiles for players
(speed, acceleration, turn radius, change-of-direction ability) by combining
CV-derived tracking with existing highlight play analysis.

## Accuracy Requirements
The target metrics (acceleration, turn radius, COD) are second-derivative
quantities, so position errors compound. Specific requirements:
- Position accuracy: sub-yard RMSE target (ideally <0.5 yd)
- Temporal resolution: 30fps source; trajectory smoothing (Kalman or spline)
  before computing velocity/accel derivatives
- Homography precision: tight enough that field mapping errors don't
  dominate the accel signal
- Validate derivatives (speed, accel) against NGS ground truth, not just raw
  position

## Available Data
- **9 games** segmented (~1650 plays total, all normalized to 30fps).
  Sideline-wide and endzone-tight angles for each.
- 2019 (3 games): Wk 3 BAL@KC (`2019092204`), Wk 8 GB@KC (`2019102712`),
  Wk 10 KC@TEN (`2019111007` — dropped, source corrupted).
- 2024 (6 games), with 2 marked as **holdout (never sample)**: Texans@Bills
  (`2024100601`) and Broncos@Chargers (`2024122201`).
- NGS highlight TSVs for 7 plays at 10 Hz, in
  `~/Personal Research/ngs_highlights-master/play_data/`.

## Camera Setup
- **Sideline wide**: good for lateral movement, east-west tracking.
- **Endzone tight**: good for depth, north-south tracking. (Not yet
  integrated.)
- Two-view fusion eventually needed to resolve both axes accurately.

## Pipeline Architecture (current state)

### 1. Video acquisition — DONE
yt-dlp from YouTube; clips segmented by play via the YOLO-cls view classifier
(`scripts/data_prep/segment_plays.py`, weights `models/view_classifier.pt`).

### 2. Player detection — DONE (RF-DETR-Large)
RF-DETR-Large fine-tuned on 967 hand-annotated frames across 8 games
(826 train / 141 val, stratified by game).
- Best EMA mAP50:95 = 0.863, mAP50 = 0.989, F1 = 0.981.
- Weights: `models/rfdetr_best_ema.pth` (use this), `models/rfdetr_best_regular.pth` (backup).
- Inference: 14.4 fps on RTX 5090, 19-23 players/frame.
- Detection threshold: **0.3** — handle noise downstream in the tracker, not by raising threshold.
- (YOLO is *only* used for the view classifier; player detection is RF-DETR.)

### 3. Field mapping / homography — ACTIVE (production pipeline shipped 2026-04-28)
**Canonical entry point**: `scripts/testing/rectify_step2_per_frame.py`.

Two-pass design:
- **Pass 1**: per-frame H + metadata. Uses two specialist UNets:
  - **Line UNet** (`models/unet_line_stage2_best.pth` / `_last.pth`): mit_b0
    grayscale, 2-ch (yard, side). Trained sequentially — older
    `data/line_detection/train` (256 frames) → fine-tune on
    `data/line_detection/al_round3` (240 frames). Best mean F1 = **0.876**.
  - **Hash UNet** (`models/unet_hash_round3_last.pth`): mit_b0 RGB, 1-ch.
    Trained on `data/hash_masks/round3` (650 frames, auto-cleaned via
    FP-drop + GT-pill seeding). Val keypoint F1 = **0.953** (vs HRNet-W18's
    0.871). Threshold-flat across 0.10-0.80.
  - Sequential RANSAC fits two hash row lines through the undistorted hash
    mask pixels. `HashRowTracker` preserves near/far identity across frames
    with a single-line failsafe.
  - `YardlineTracker` preserves g-index identity. `HomographyTrackerLite`
    (full / delta / carry + `KeypointTrackBank` validation) computes H per
    frame.
- After pass 1: `detect_lost` (≥3 consecutive carry frames) sets the cutoff;
  Savitzky-Golay smoothing (window=7, poly=2) on the H trajectory.
- **Pass 2**: re-render with smoothed H. Stops at lost cutoff.

Output: stacked video — top panel shows source + colored masks + line fits +
projected grid + correspondence dots + method-color HUD; bottom panel shows
rectified canvas spanning **NGS 0-120** (whole field with endzones), flipped so
NGS y=0 (near sideline) is at the bottom.

**Three import dependencies still live in `scripts/testing/rebuild_*.py`** —
small refactor TODO to move into `src/`:
- `rebuild_full_clip_viz.py` → `YardlineTracker`, `group_sideline_pixels_cc`
- `rebuild_step4_hashes_v2.py` → `total_mse`, `ransac_line`, thresholds
- `rebuild_step8_homography.py` → `HomographyTrackerLite`, `solve_h`,
  `smooth_hs`, `detect_lost`

**Known issues**:
1. **Back-of-endzone misalignment** when only 2 yardlines visible
   (back-of-endzone + goalline). YardlineTracker's median-spacing assumes
   5y unit but gets 10y → all subsequent g indices off by 5y. Affects
   plays like 2019092204/play_023 and 2024090801/play_032.
2. **Yard-line numbers not auto-detected** — currently requires manual
   `--g0-ngs-x` anchor per clip. Blocking batch processing of all 1650
   clips.

### 4. Player tracking — NOT STARTED
Plan: BoT-SORT (appearance-based re-ID over ByteTrack), gated by jersey
color/number features. Identity only needs to hold *within a play* (since
each clip starts at the snap), not across plays.

### 5. Jersey number detection — NOT STARTED
Per-player ID at the snap so the tracker can label players. Model TBD —
likely a small RF-DETR or crop classifier focused on bib digits.

## Post-Processing (planned)
- Trajectory smoothing (Kalman or cubic spline) on raw x/y before
  differentiating.
- Velocity = 1st derivative; accel = 2nd; turn radius from instantaneous
  curvature.
- Validate all derived metrics against NGS ground truth.

## Key Design Decisions
- **Identity scope**: each play stands alone — re-identify all 22 players at
  the snap; no need to maintain identity across plays.
- **Confidence gating**: any time the tracker can't reliably maintain
  identity (low re-ID, occlusion, pile-up, player leaves frame), mark the
  trajectory as interrupted rather than emit bad positions.
- **Offline pipeline**: not a live tracker. For decisions like "when did
  tracking fail" or "which frames are outliers", use both past AND future
  frames (e.g., walk backward from the clip end, not predictively from the
  start).
- **Two-camera fusion** (later): combine sideline (E-W) and endzone (N-S)
  weighted by foreshortening per player.

## Validation Approach
Compare CV-derived tracking against NGS ground truth for the 7 NGS-labeled
highlight plays:
- Position RMSE (yards)
- Speed correlation + RMSE (yd/s)
- Acceleration correlation + RMSE (yd/s²)
- Turn radius on specific route breaks

## Where We Are vs. What's Next

**Done**:
1. Video acquisition + play segmentation (~1650 plays).
2. Player detection (RF-DETR, F1=0.981).
3. Homography pipeline producing rectified video for individual clips, with
   manual NGS-x anchor.

**Immediate next step (homography)**:
- **Yard-line number detection** to remove the manual anchor. Without this
  we can't batch-process all clips.

**After number detection**:
- Player tracking (BoT-SORT or similar) — uses RF-DETR boxes + appearance
  features.
- **NGS validation** on the 7 labeled plays — project tracker output through
  homography into NGS coords; measure position + derivative RMSE.

**Later**:
- Jersey number detection for per-player ID at snap.
- Trajectory smoothing (Kalman/spline) before derivatives.
- Two-camera fusion (sideline + endzone).
- Fix back-of-endzone yardline-tracker failure mode.

## Tech Stack
- Python 3.10+
- torch (GPU + MPS)
- segmentation-models-pytorch (mit_b0 UNets)
- opencv-python, numpy, pandas, scipy
- albumentations (train-time augmentation)
- matplotlib/seaborn for viz
- ultralytics (only for the view classifier — not for player detection)

## Commands
- Single-clip rectify: `python scripts/testing/rectify_step2_per_frame.py
  --clip <path/to/sideline.mp4> --g0-ngs-x <NGS-x of leftmost yardline>
  --out <path>`
- Identify g0 on first frame: `python
  scripts/testing/rectify_step1_label_g0.py --clip <path> --out <jpg>`
- Train line UNet: `python scripts/runpod/launch_runpod.py --training-type
  unet --encoder mit_b0 --grayscale ...`
- Train hash UNet: `python scripts/runpod/launch_runpod.py --training-type
  unet-hash ...`

## File Structure
- `src/homography/` — production homography modules (`field_model.py`,
  `distortion.py`, `apply_homography.py`, `tracker.py`,
  `keypoint_track_bank.py`). Note: `grid_solver_v2.py` and
  `keypoint_detector.py` remain on disk but are no longer the production
  path; `rectify_step2_per_frame.py` supersedes them.
- `scripts/training/` — `train_rfdetr.py`, `train_unet_lines.py` (line
  UNet), `train_unet_hash.py` (hash UNet), `train_hrnet_keypoints.py`
  (HRNet — only kept for backward compatibility, not used in production).
- `scripts/data_prep/` — clip segmentation, frame extraction, AL frame
  selection, hash mask generation (`build_hash_round3_dataset.py` is the
  current builder), Label-Studio config generation.
- `scripts/runpod/` — pod lifecycle (`launch_runpod.py` supports
  `--training-type {rfdetr,hrnet,unet,unet-hash,unet-unified}`).
- `scripts/testing/` — `rectify_step{1,2}*.py` (production), plus three
  `rebuild_*.py` files retained only because rectify_step2 imports from
  them.
- `data/annotations/` — RF-DETR player labels (git-tracked, small).
- `data/player_detection/` — RF-DETR COCO dataset (gitignored).
- `data/field_keypoints/` — keypoint annotations + AL rounds (892 frames,
  713 train / 179 val).
- `data/line_detection/` — hand-annotated line masks: `train/`, `valid/`,
  plus `al_round3/` (the source of stage 2's near-sideline data).
- `data/hash_masks/round3/` — auto-cleaned hash mask training set
  (650 frames).
- `data/view_classifier/` — sideline/endzone classifier data.
- `videos/games/` — raw MP4s (gitignored).
- `videos/clips/` — segmented play clips at 30fps (gitignored except
  manifests).
- `models/` — `unet_line_stage2_{best,last}.pth`, `unet_hash_round3_last.pth`,
  `rfdetr_best_{ema,regular}.pth`, `view_classifier.pt`.
- `priors/` — research notes, data sources.
- `.env` — RunPod API key (NEVER edit).

## Critical Rules
- **NEVER launch RunPod / training / Label Studio scripts without explicit
  user permission.**
- **Don't chain steps** — finish a step, report, wait for direction.
- **Always detach long-running processes** with
  `nohup cmd > logfile 2>&1 &`.
- **NEVER edit `.env`** — has the user's RunPod API key.
- **Offline pipeline** — use future-frame info too, not just past.
