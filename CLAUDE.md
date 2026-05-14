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

### 3. Field mapping / homography — DONE (unified UNet + v10c token classifier)
Replaced the older 3-specialist (line + hash + number) UNet path. Current
production:
- **One 4-channel UNet** (`models/unet_unified_v8_yardside_recover/best.pth`)
  emits per-pixel scores for yard / side / hash / number masks.
- **Tokenizer** (`src/pipeline/cc_tokenizer_v3.py`) converts the mask into
  per-region tokens.
- **Three-phase token classifier:**
  - **Phase 1** — encoder (`src/pipeline/model_token_v10.py`,
    weights `models/token_only_v10_phase1_pseudo/best.pth`) attends across
    all tokens.
  - **Phase 2** — Refinement for Number tokens (`src/pipeline/train_rf_b.py`,
    weights `models/rf_b_phase2_pseudo/best.pth`) takes the encoder feature
    plus the number crop classifier's logits
    (`models/dsresnet10ww_round3_128x32/best.pth`) and predicts each number's
    NGS-x class.
  - **Phase 3** — Cross-attention head (`src/pipeline/model_token_v10b.py`
    used by `train_token_v10c_stage2`, weights
    `models/v10c_phase3_pseudo/best.pth`) refines every token using the
    Phase-2-resolved number anchors and yields each token's NGS-x label
    (yardlines, hashes, sidelines) at 99.66% val accuracy.
- **Keypoint extraction** (`src/homography/keypoints_from_tokens.py`)
  emits image↔NGS correspondences from the labeled tokens, including
  number-edge tangents and hash×yardline crossings.
- **Per-frame H solver** (`src/homography/h_tracker.py::HomographyTrackerLite`)
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
`src/tracker.py::PlayerTracker`. Tracks in NGS yards (not pixels), so
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

Post-tracking team classification via `src/team_classifier.py::classify_teams_color_pca`
— per-track HSV signature, baseline-subtract, PCA → PC1, median-split.
Forces 11/11 by construction (NFL formation prior).

### 5. NGS validation — DONE on play_065
`scripts/aux/compare/compare_ngs_play_065.py` runs the pipeline +
loads the NGS TSV + sweeps a snap-frame offset (NGS `ball_snap` event
marks t=0, our clip starts ~120 frames before snap) + Hungarian
position-only match (22 NGS × N tracks) + scores position / speed /
accel.

play_065 result (2026-05-14):
- Best snap offset: frame 114 (= 3.80 s into clip; user estimated ~4 s)
- All 22 NGS players matched
- **Position RMSE median: 0.437 yd** (mean 0.884 — a couple outliers)
- **Speed RMSE: 0.890 yd/s, correlation 0.91**
- Accel correlation: 0.26 (noisy, as expected for second derivatives
  without stronger position smoothing)
- Outputs at `output/ngs_compare/`.

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
3. Homography (unified UNet + v10c, auto-anchor, LOO red-flag filter).
4. Player tracking (custom field-coord Kalman, multi-cue association).
5. Team classification (color PCA + median split, 11/11 forced).
6. NGS validation on play_065 (sub-yard position, 0.91 speed correlation).

**Next:**
1. **Run NGS validation on the other 3 in-scope plays** (2019102712/p011,
   p046, p118). Confirm position + speed metrics generalize.
2. **Trajectory smoothing pass** before differentiating, to recover
   useful accel correlations.
3. **Batch-process the remaining ~1650 clips** — straightforward once
   validation cleared. Need a runner that captures per-clip quality
   flags + writes per-play tracking CSVs.
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

End-to-end production pipeline today is invoked by aux scripts:

- **Single-clip 4-panel viz** (source+corrs / tracker boxes /
  rectified / tracking dots, team-colored): `bash
  scripts/aux/runpod/play_065_4panel_runpod.sh` (RunPod). Locally:
  `python scripts/aux/viz/make_play_065_4panel.py --device mps`.
- **NGS comparison**: `bash scripts/aux/runpod/compare_ngs_play_065_runpod.sh`.
- **Filter diagnostics** (LOO vs old): `bash
  scripts/aux/runpod/diagnose_h_filters_runpod.sh`.
- **Homography-only on a single clip** (legacy 3-specialist path, still
  works): `python -m src.homography.rectify --clip <path> --out <path>`.
- **RunPod training** (active types): `python
  scripts/aux/runpod/launch_runpod.py --training-type
  {rfdetr,number-classifier}`. The `unet-unified` and phase{1,2,3}_pseudo
  trainers run via their own scripts in `scripts/aux/training/` (not
  wired into launch_runpod.py).

## File Structure

**Production code lives in `src/` and is fully importable.** `scripts/aux/`
contains every entry-point / utility / runpod launcher we still use.

- `src/pipeline/` — model defs + helpers used at inference time:
  `cc_tokenizer{,_v2,_v3}.py`, `model_token_v10{,b}.py`,
  `train_rf_a.py` (encoder features + number-crop classifier loader),
  `train_rf_b.py` (RFB phase 2), `train_token_v10c_stage2.py`
  (cross-attention forward + PAINTED_TO_21), `train_token_v6.py`
  (N_NGS_X_CLASSES, build_targets), `train_token_v8.py`
  (AugmentedHSetDataset), `train_dense_regression.py`,
  `train_h_set_regressor.py`, `train_scene_refiner.py`, plus the
  H-solver helpers `h_pnl_dlt.py`, `h_pnl_set_regressor.py`.
- `src/homography/` — production homography modules: `field_model.py`,
  `apply_homography.py`, `keypoints_from_tokens.py`,
  `keypoint_track_bank.py`, `h_tracker.py` (HomographyTrackerLite +
  smooth_hs + **loo_filter_and_replace** + **detect_bad_runs**),
  `rectify.py` (legacy 3-specialist entry point, still works),
  `specialists.py`, plus `painted_numbers.py`, `line_fit.py`,
  `yardline_tracker.py`, `grid_solver_v2.py`, `distortion.py`.
- `src/detector.py` — RF-DETR wrapper + detection cache.
- `src/tracker.py` — `PlayerTracker` (field-coord Kalman, multi-cue
  association, graveyard, color signatures).
- `src/team_classifier.py` — post-tracking team labels
  (`classify_teams_color_pca` is the canonical method; others kept for
  ablation), `select_long_tracks`, `compute_color_signature` (used both
  by tracker and team_classifier).
- `src/smoothing.py` — Sav-Gol helpers for trajectory smoothing.
- `src/pipeline.py` — high-level entry point (older; superseded in
  practice by aux scripts that orchestrate the same flow).
- `scripts/aux/viz/make_play_065_4panel.py` — single 1920×1080 viz:
  source+corrs / tracker boxes / rectified / tracking dots, with BAD
  RUN overlay if any.
- `scripts/aux/compare/compare_ngs_play_065.py` — NGS validation
  (snap-sweep + Hungarian match + per-player stats + side-by-side viz).
- `scripts/aux/diagnostics/analyze_h_residuals.py` — single-clip LOO
  residual report.
- `scripts/aux/diagnostics/diagnose_h_filters.py` — multi-clip
  old-vs-new filter confusion matrix.
- `scripts/aux/runpod/` — RunPod launchers + the .sh wrappers that
  bundle, upload, run, download per task. `launch_runpod.py` is the
  general-purpose pod lifecycle tool (with `--min-upload` / `--min-download`
  / `--country-code` / `--data-center-id` filters to avoid slow hosts).
- `scripts/aux/training/` — training entry points for the 6 active
  models: `train_phase1_pseudo.py`, `train_phase2_pseudo.py`,
  `train_phase3_pseudo.py`, `train_unified_mask.py`,
  `train_number_classifier.py`, `train_rfdetr.py`.
- `scripts/aux/data_prep/` — 12 scripts that built the kept training
  data: `segment_plays.py`, `convert_clips_30fps.py`,
  `inject_intrinsics_into_manifest.py`, `build_smart_pool.py`,
  `build_dense_field_training_pool.py`, `build_unified_mask_dataset.py`,
  `build_qc_unified_mask_dataset.py`, `farm_pseudo_labels.py`,
  `farm_pseudo_labels_all_clips.py`, `qc_pseudo_labels.py`,
  `extract_crop_logits.py`, `build_round3_dataset.py`.

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
