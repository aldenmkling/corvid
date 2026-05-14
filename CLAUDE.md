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
  — **HIT** on play_065 (NGS validation): median per-player position RMSE
  = 0.44 yd, speed correlation = 0.91.
- Temporal resolution: 30 fps source. Trajectory smoothing (Sav-Gol)
  on field-coord positions before differentiating.
- Homography precision: tight enough that field-mapping errors don't
  dominate the accel signal. LOO-residual H filter + polynomial bridging
  enforces this within ~0.025 yd per-frame on clean stretches.
- Validate derivatives (speed, accel) against NGS ground truth, not just
  raw position.

## Available Data
- **9 games** segmented (~1650 plays total, all normalized to 30 fps).
  Sideline-wide and endzone-tight angles for each.
- 2019 (3 games): Wk 3 BAL@KC (`2019092204`), Wk 8 GB@KC (`2019102712`),
  Wk 10 KC@TEN (`2019111007` — dropped, source corrupted).
- 2024 (6 games), with 2 marked as **holdout (never sample)**: Texans@Bills
  (`2024100601`) and Broncos@Chargers (`2024122201`).
- NGS highlight TSVs for the 4 in-scope plays at 10 Hz, in
  `~/Desktop/Personal Research/ngs_highlights-master/play_data/`. The
  confirmed NFL playId → clip mapping is documented in memory
  (`project_ngs_validation_clips.md`).

## Camera Setup
- **Sideline wide**: good for lateral movement, east-west tracking.
- **Endzone tight**: good for depth, north-south tracking. (Not yet
  integrated.)
- Two-view fusion eventually needed to resolve both axes accurately.

## Pipeline Architecture (current state, 2026-05-14)

The end-to-end clip → NGS-style tracking flow is:

```
clip → UNet (4-ch mask) → tokenizer → encoder (phase1) →
  ┌─ RFB (phase2, with number-crop classifier)
  └─ v10c cross-attn (phase3) → per-token NGS-x labels →
keypoints → solve H per frame → LOO filter + polynomial bridge →
SG smooth Hs → RF-DETR per-frame → custom field-coord Kalman tracker →
per-track SG smoothing → team-classify (color PCA + median split) →
per-frame x/y in NGS yards per player
```

### 1. Video acquisition — DONE
yt-dlp from YouTube; clips segmented by play via the YOLO-cls view
classifier (`scripts/aux/data_prep/segment_plays.py`, weights
`models/view_classifier.pt`).

### 2. Player detection — DONE (RF-DETR-Large)
RF-DETR-Large fine-tuned on 967 hand-annotated frames across 8 games
(826 train / 141 val, stratified by game).
- Best EMA mAP50:95 = 0.863, mAP50 = 0.989, F1 = 0.981.
- Weights: `models/rfdetr_best_ema.pth` (use this),
  `models/rfdetr_best_regular.pth` (backup).
- Inference: 14.4 fps on RTX 5090, 19-23 players/frame.
- Detection threshold: **0.3** — handle noise downstream in the tracker,
  not by raising the threshold.

### 3. Field mapping / homography — DONE (unified UNet + 3-phase token classifier)
Replaced the older 3-specialist (line + hash + number) UNet path. Lives
entirely in `src/field_mapping/`. Current production:
- **One 4-channel UNet** (`models/unet_unified_v8_yardside_recover/best.pth`)
  emits per-pixel scores for yard / side / hash / number masks.
- **Tokenizer** (`src/field_mapping/tokenizer.py`) converts the mask into
  per-region tokens.
- **Three-phase token classifier:**
  - **Phase 1: TokenEncoder** (`src/field_mapping/encoder.py`, weights
    `models/token_only_v10_phase1_pseudo/best.pth`) — 4-layer transformer
    encoder, attends across all tokens to build contextual features.
  - **Phase 2: NumberRefiner** (`src/field_mapping/number_refiner.py`,
    weights `models/rf_b_phase2_pseudo/best.pth`) — takes the encoder
    features for number tokens + per-crop logits from the CropClassifier
    (`models/dsresnet10ww_round3_128x32/best.pth`) and refines each
    number token's NGS-x label.
  - **Phase 3: TokenLabeler** (`src/field_mapping/token_labeler.py`,
    weights `models/v10c_phase3_pseudo/best.pth`) — cross-attention head;
    uses phase-2-resolved number anchors to label every token's NGS-x.
    99.66% val accuracy.
- **Keypoint extraction** (`src/field_mapping/keypoints.py`) emits
  image↔NGS correspondences from the labeled tokens.
- **Per-frame H solver** (`src/field_mapping/homography.py::HomographyTrackerLite`)
  fits H per frame with full/delta/carry fallback + bank validation.
- **LOO filter + polynomial bridge** (added 2026-05-13;
  `loo_filter_and_replace` in `h_tracker.py`): each frame's raw H is
  scored against a degree-2 polynomial fit through 3 frames on each side
  (excluding self). Flag frames where `loo > 0.20 yd` OR `rmse > 0.30 yd`,
  then *replace* flagged frames with a polynomial fit through the **3
  clean (non-flagged) frames on each side of the bad-run gap** — so
  contaminated neighbors don't poison the fit. Median LOO residual on
  clean frames is ~0.025 yd; outliers stand out at 50× median. Catches
  the back-of-endzone failure mode the old `rmse > 0.30 OR temp_div > 1.0`
  thresholds were blind to. Long red-runs (≥5 consecutive flagged
  frames) get a "BAD RUN" overlay in viz — bridged in output but flagged
  for review and likely clip-disqualification at data-collection time.
- **Sav-Gol smoothing** (window=7, poly=2) on the cleaned H trajectory.

### 4. Player tracking — DONE (custom field-coord Kalman)
`src/player_tracking/tracker.py::PlayerTracker`. Tracks in NGS yards (not pixels), so
camera motion is solved upstream by H. Per-track Kalman state
`[x, y, vx, vy]` with constant-velocity dynamics, dt=1/30 s.

Multi-cue association each frame: weighted-sum cost
`0.5·d_field + 1.0·d_iou + 1.0·d_color` over two Hungarian rounds (strict +
relaxed), then graveyard re-association on orphans:
- **d_field**: Mahalanobis squared distance / chi-square gate (9.21)
- **d_iou**: 1 − expansion-IoU(track.last_box, det.xyxy)
- **d_color**: 1 − cosine(track.color_sig, det.color_sig). Per-track
  24-dim chromatic signature (12-bin hue + 4×3 SV joint hist on
  chromatic-pixel-masked detection box), EMA α=0.999, skipped during
  box-overlap windows. Falls back to 2-cue when track has < 3
  observations.

Post-tracking team classification via
`src/player_tracking/team_classifier.py::classify_teams_color_pca` —
per-track HSV signature, baseline-subtract, PCA → PC1, median-split.
Forces 11/11 by construction (NFL formation prior).

### 5. NGS validation — DONE on all 4 in-scope plays
`scripts/aux/compare/compare_ngs.py` runs the pipeline +
loads the NGS TSV + sweeps a snap-frame offset (NGS `ball_snap` event
marks t=0, our clip starts ~120 frames before snap) + Hungarian
position-only match (22 NGS × N tracks) + scores position / speed /
accel.

All 4 in-scope plays (2026-05-14):

| Clip | Snap (frames / s) | Pos RMSE med/mean (yd) | Speed corr (med) |
|---|---|---|---|
| 2019092204/play_065 | 114 / 3.80 | 0.437 / 0.884 | 0.907 |
| 2019102712/play_011 | 303 / 10.10 | 0.422 / 0.713 | 0.914 |
| 2019102712/play_046 | 120 / 4.00 | 0.472 / 0.619 | 0.904 |
| 2019102712/play_118 | 108 / 3.60 | 0.765 / 0.963 | 0.743 |

3/4 plays hit the sub-yard position median target. Accel correlation is
poor across the board (0.18-0.26) — needs stronger position smoothing
before differentiating twice. Outputs at `output/ngs_compare/`.

### 6. Jersey number detection — NOT STARTED
Per-player ID at the snap so tracker output can be tagged with player
IDs (e.g. "13", "10"). Model TBD — likely a small classifier on
detection-box crops.

## Post-Processing
- Trajectory smoothing: per-track Sav-Gol (window=9, poly=2) on the
  field-coord positions before differentiating. Implemented in the viz +
  comparison scripts; production pipeline runs it inline.
- Velocity = first finite-diff derivative; accel = second. Noisy as
  expected. Stronger smoothing (Kalman) is the next step if we want
  reliable accel signal.

## Key Design Decisions
- **Identity scope**: each play stands alone — re-identify all 22 players
  at the snap; no need to maintain identity across plays.
- **Confidence gating**: tracker marks trajectories `interrupted=True`
  when it loses confidence (predicted-only frames, low-conf detections)
  rather than emit bad positions.
- **Offline pipeline**: not a live tracker. Decisions like "when did
  tracking fail" use whole-clip + future info (e.g. find last-stable-frame
  by walking backward from the clip end).
- **Two-camera fusion** (later): combine sideline (E-W) and endzone (N-S)
  weighted by foreshortening per player.

## Validation Approach
Compare CV-derived tracking against NGS ground truth for the 4 confirmed
NGS-labeled plays (`project_ngs_validation_clips.md`):
- Position RMSE (yards) — **hit on play_065**
- Speed correlation + RMSE (yd/s) — **hit on play_065**
- Acceleration correlation + RMSE (yd/s²) — noisy, needs better smoothing
- Turn radius on specific route breaks — future work

## Where We Are vs. What's Next

**Done:**
1. Clip segmentation (~1650 plays).
2. Player detection (RF-DETR, F1=0.981).
3. Homography (unified UNet + 3-phase token classifier, auto-anchor,
   LOO red-flag filter).
4. Player tracking (custom field-coord Kalman, multi-cue association).
5. Team classification (color PCA + median split, 11/11 forced).
6. NGS validation on all 4 in-scope plays (sub-yard position on 3/4,
   0.74–0.91 speed correlation).
7. Major refactor (2026-05-14): src/ reorganized into clearly-named
   pipeline modules (`field_mapping/`, `player_detection/`,
   `player_tracking/`); each file named for its pipeline role. New
   `src/pipeline.py` end-to-end orchestrator.

**Next:**
1. **Trajectory smoothing pass** before differentiating, to recover
   useful accel correlations.
2. **Batch-process the remaining ~1650 clips** — straightforward once
   validation cleared. Need a runner that captures per-clip quality
   flags + writes per-play tracking CSVs.
3. **Retrain crop classifier on the larger phase-2 pseudo-label set**
   (per `project_future_improvements.md`).
4. **Jersey number detection** at snap → labeled trajectories.
5. **Two-camera fusion** (sideline + endzone).

## Tech Stack
- Python 3.10+
- torch (GPU + MPS for local inference; CUDA on RunPod for any pipeline
  pass that needs the speed)
- segmentation-models-pytorch (mit_b0 UNet)
- opencv-python, numpy, pandas, scipy, scikit-learn
- rfdetr (player detector)
- albumentations (train-time augmentation)
- matplotlib for viz

## Commands

End-to-end production pipeline:

- **Full pipeline → tracking CSV**: `python -m src.pipeline --clip
  videos/clips/<game>/<play>/sideline.mp4 --out <output.csv>
  --device cuda|mps|cpu`. Loads all 5 stages, runs the clip, emits
  per-frame per-track NGS-yard CSV (frame_idx, track_id, x_yd, y_yd,
  team, in_bad_run).

Aux entry points:

- **4-panel clip overview viz**: `bash
  scripts/aux/runpod/four_panel_clip_overview_runpod.sh`. Locally:
  `python scripts/aux/viz/four_panel_clip_overview.py --device mps`.
- **NGS comparison** (single clip): `bash
  scripts/aux/runpod/compare_ngs_runpod.sh`. Configurable via the
  script's `--clip`, `--ngs-tsv`, `--snap-center` args.
- **3-clip NGS comparison batch**: `bash
  scripts/aux/runpod/compare_ngs_3plays_runpod.sh`.
- **LOO filter diagnostics**: `bash
  scripts/aux/runpod/h_residual_report_runpod.sh` (single clip)
  or `bash scripts/aux/runpod/compare_h_filters_runpod.sh` (old vs new
  filter across multiple clips).
- **Train crop classifier**: `bash
  scripts/aux/runpod/train_crop_classifier_runpod.sh`.
- **RunPod training (other models)**: `python
  scripts/aux/runpod/launch_runpod.py --training-type
  {rfdetr,number-classifier}`. The token-encoder / number-refiner /
  token-labeler / UNet trainers run via their own scripts in
  `scripts/aux/training/` (not wired into launch_runpod.py).

## File Structure

**Production code lives in `src/` and is fully importable.** Every file
is named for its role in the pipeline. `scripts/aux/` contains every
training script, viz, comparison, diagnostic, and runpod launcher.

### `src/` (production)

```
src/
├── pipeline.py                  end-to-end orchestrator: clip → CSV
├── field_mapping/               stage 1 — per-frame homography
│   ├── pipeline.py              FieldMappingPipeline class
│   ├── unet.py                  (none yet — UNet inlined in pipeline.py
│   │                             via segmentation_models_pytorch)
│   ├── tokenizer.py             mask → per-region tokens
│   ├── encoder.py               TokenEncoder (phase 1)
│   ├── crop_classifier.py       CropClassifier (DSResNet10ww)
│   ├── number_refiner.py        NumberRefiner (phase 2)
│   ├── token_labeler.py         TokenLabeler (phase 3, cross-attn)
│   ├── keypoints.py             tokens → image↔NGS correspondences
│   ├── homography.py            H solver + LOO + bridging + smoothing
│   ├── keypoint_bank.py         bank validation for H candidates
│   ├── apply_homography.py      pixel ↔ field projections
│   ├── field_model.py           NFL field constants (FIELD_LENGTH, etc.)
│   └── classes.py               NGS-x quantization, PAINTED_TO_21 etc.
├── player_detection/            stage 2 — bounding boxes
│   └── detector.py              RF-DETR wrapper + detection cache
└── player_tracking/             stages 3+4+5
    ├── tracker.py               field-coord Kalman + multi-cue assoc
    ├── color_signature.py       24-dim chromatic feature
    ├── team_classifier.py       color PCA + median split, select_long_tracks
    └── trajectory_smoothing.py  per-track Sav-Gol
```

### `scripts/aux/`

```
scripts/aux/
├── viz/four_panel_clip_overview.py    1920×1080 4-panel viz
├── compare/compare_ngs.py             NGS comparison (single clip)
├── diagnostics/
│   ├── h_residual_report.py           single-clip LOO residual stats
│   └── compare_h_filters.py           old-vs-new filter confusion
├── runpod/                            RunPod lifecycle + per-task .sh runners
│   ├── launch_runpod.py
│   ├── four_panel_clip_overview_runpod.sh
│   ├── compare_ngs_runpod.sh / compare_ngs_3plays_runpod.sh
│   ├── h_residual_report_runpod.sh / compare_h_filters_runpod.sh
│   ├── train_crop_classifier_runpod.sh
│   └── requirements_farm_runpod.txt
├── training/                          training scripts for the 8 models
│   ├── train_encoder.py               (was train_phase1_pseudo)
│   ├── train_number_refiner.py        (was train_phase2_pseudo)
│   ├── train_token_labeler.py         (was train_phase3_pseudo)
│   ├── train_unet.py                  (was train_unified_mask)
│   ├── train_crop_classifier.py       DSResNet10ww (the production one)
│   ├── train_crop_classifier_mit.py   MitClassifier variant (alternate)
│   └── train_player_detector.py       (was train_rfdetr)
└── data_prep/                         scripts that built kept training data
    ├── segment_plays.py, convert_clips_30fps.py,
    │   inject_intrinsics_into_manifest.py,
    ├── build_smart_pool.py, build_dense_field_training_pool.py,
    ├── build_unified_mask_dataset.py, build_qc_unified_mask_dataset.py,
    ├── farm_pseudo_labels.py, farm_pseudo_labels_all_clips.py,
    ├── qc_pseudo_labels.py, extract_crop_logits.py,
    └── build_round3_dataset.py
```

### Data
- `data/h_pool_and_intrinsics.json` — combined: (1) hand-verified
  2500-sample H training pool (`entries`), (2) per-clip camera
  intrinsics for 1344 clips (`intrinsics_by_clip`). Every active script
  loads this for `K + dist` lookups.
- `data/annotations/` — RF-DETR player labels (git-tracked, small).
- `data/player_detection/` — RF-DETR COCO dataset (gitignored).
- `data/field_keypoints/` — hand-labeled keypoint annotations
  (892 frames). Hand-labeled, irreplaceable.
- `data/line_detection/{train,valid,al_round3}/` — hand-labeled line
  annotations. Irreplaceable.
- `data/unified_masks/` — unified UNet training data (gitignored,
  regeneratable via `build_unified_mask_dataset.py`).
- `data/pseudo_labels/`, `data/pseudo_labels_crops/` — phase 1/2/3
  training data (gitignored, regeneratable via `farm_pseudo_labels.py`).
- `data/number_classifier/round1/` — number crop classifier training
  data (gitignored, regeneratable).
- `data/view_classifier/` — sideline/endzone classifier data.
- `data/ngs/2019_KC_2019092204_1643.tsv` — NGS TSV for play_065
  (validation ground truth).
- `data/phase1_blacklist.json`, `data/phase1_failures*.json`,
  `data/hard_train_samples.json`, `data/pseudo_labels_qc.json` — phase
  1/2/3 training metadata (blacklists, QC decisions, hard examples).
- `data/hash_excluded_frames.txt` — list of bad-GT hash frames to skip
  during dataset rebuilds (human-curated, hand-maintained).

### Models (all in `models/`, 8 production)
- `unet_unified_v8_yardside_recover/best.pth` — 4-ch UNet (yard / side /
  hash / num masks).
- `token_only_v10_phase1_pseudo/best.pth` — phase 1 encoder.
- `rf_b_phase2_pseudo/best.pth` — phase 2 number refiner.
- `v10c_phase3_pseudo/best.pth` — phase 3 cross-attention head.
- `dsresnet10ww_round3_128x32/best.pth` — number crop classifier.
- `rfdetr_best_ema.pth` — player detector. (`rfdetr_best_regular.pth`
  is the backup.)
- `view_classifier.pt` — sideline/endzone (used by `segment_plays.py`).

### Output
- `output/4panel_viz/*.mp4` — 4-panel comparison viz per clip
  (source + tracker + rectified + dots).
- `output/ngs_compare/{play_065_compare.mp4, play_065_per_player.csv,
  play_065_summary.txt}` — NGS validation outputs.

## Critical Rules
- **NEVER launch RunPod / training / pipeline / Label Studio scripts
  without explicit user permission.**
- **Don't chain steps** — finish a step, report, wait for direction.
- **Always detach long-running processes** with
  `nohup cmd > logfile 2>&1 &`. Use `--min-upload N` on
  `launch_runpod.py` (~25 Mbps is comfortable) to avoid the slow boxes.
- **NEVER edit `.env`** — has the user's RunPod API key.
- **Offline pipeline** — use future-frame info too, not just past.
- **`data/` and `models/` are gitignored.** Deleting from them is
  permanent — no git history to resurrect from. Always check with the
  user before deletion.
- **Hand-labeled data is irreplaceable** — `data/field_keypoints/`,
  `data/line_detection/`, `data/annotations/`, and the 2500 verified
  entries in `data/h_pool_and_intrinsics.json` represent work that
  can't be regenerated.
