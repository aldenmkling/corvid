# corvid — All-22 → NGS player tracking

Named for crows because the pipeline manufactures a bird's-eye view of
the field from a single ground-level sideline camera, recovering top-down
NGS-yard coordinates frame by frame.

End-to-end computer-vision pipeline that turns NFL All-22 sideline-wide
video into NGS-style player tracking data: per-frame `(x, y)` in NGS yards
for every player on the field.

The intended use is building tape-based athletic profiles (speed,
acceleration, change-of-direction) from broadcast / All-22 footage by
combining CV-derived tracking with existing highlight-play analysis.

## Status — V1 shipped (2026-05-14)

Validated against NGS ground truth on the 4 in-scope plays:

| Clip | Pos RMSE (median) | Speed correlation | Notes |
|---|---|---|---|
| 2019092204 / play_065 | **0.437 yd** | 0.907 | Hardman 83-yd TD |
| 2019102712 / play_011 | **0.422 yd** | 0.914 | Kumerow 34-yd |
| 2019102712 / play_046 | **0.472 yd** | 0.904 | Kelce 29-yd TD |
| 2019102712 / play_118 | 0.765 yd | 0.743 | goal-line scrum (hardest) |

3/4 plays clear the sub-yard position-RMSE target. Speed correlation ≥ 0.90
on three of four. Acceleration correlation is still weak (0.18-0.26) —
better trajectory smoothing is the top V2 item.

## Quick start

```bash
# End-to-end inference: clip → per-frame per-player NGS-yard CSV.
python -m src.pipeline --clip videos/clips/2019092204/play_065/sideline.mp4 \
                       --out output/play_065_tracking.csv

# Score the pipeline against NGS ground truth (4-panel viz + per-player CSV).
python scripts/aux/compare/compare_ngs.py \
    --clip videos/clips/2019092204/play_065/sideline.mp4 \
    --ngs-tsv data/ngs/2019_KC_2019092204_1643.tsv

# Single-clip 4-panel debug viz.
python scripts/aux/viz/four_panel_clip_overview.py \
    --clip videos/clips/2019092204/play_065/sideline.mp4
```

Output:

```csv
frame_idx,track_id,x_yd,y_yd,team,in_bad_run
0,1,42.13,23.45,A,False
0,2,41.02,28.10,A,False
...
```

## Pipeline

```
clip
 ├─► UNet (4-channel mask: yard / side / hash / number)
 ├─► tokenizer (mask → per-region tokens)
 ├─► TokenEncoder         (phase 1)
 ├─► CropClassifier + NumberRefiner   (phase 2, number tokens)
 ├─► TokenLabeler         (phase 3, cross-attention)
 ├─► keypoints → HomographySolver (per frame H)
 ├─► LOO-residual filter + polynomial bridge + Sav-Gol smoothing on H
 ├─► RF-DETR-Large player detection (per frame)
 ├─► PlayerTracker         (field-coord Kalman + multi-cue association)
 ├─► classify_teams_color_pca   (post-tracking)
 └─► per-track Sav-Gol → NGS-yard CSV
```

Eight production checkpoints (~310 MB). See `models/README.md`.

## Repo layout

```
src/         production pipeline       (see src/README.md)
scripts/     training + tooling        (see scripts/README.md)
models/      weights (gitignored)      (see models/README.md)
data/        datasets + manifests      (see data/README.md)
videos/      raw video (gitignored except manifests)
output/      generated artifacts       (mostly gitignored)
priors/      research notes
CLAUDE.md    pipeline notes + design decisions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # see CLAUDE.md "Tech Stack" for deps
```

Weights are not checked in; pull them out-of-band into `models/`. Video
clips and the bulk training datasets under `data/training/` are also
gitignored — regenerate from the labels under `data/labels/` and the
manifest at `data/manifests/h_pool_and_intrinsics.json`.

## Validation data

Four NGS-labeled plays under `data/ngs/`. The play↔TSV mapping plus snap
frames are documented in `data/README.md`.

## V2 roadmap

In priority order:

1. Retrain models on the larger pseudo-label pool (self-distillation; the
   phase-3 labels are cleaner than the original training data).
2. Better trajectory smoothing (full-trajectory Kalman / RTS smoother) so
   acceleration correlation becomes useful.
3. Player identity (jersey-number detection at snap) — current tracker
   outputs are anonymous `track_id 1..N`.
4. Multi-frame architecture (replace frame-by-frame inference with a
   temporal model).
5. Two-camera fusion (sideline + endzone).

See `CLAUDE.md` for the long-form design notes.
