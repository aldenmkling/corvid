# `src/` — production pipeline

End-to-end inference code. A clip goes in, per-frame per-player NGS-yard
positions come out.

## Entry point

```bash
python -m src.pipeline --clip <path/to/sideline.mp4> --out <path/to/tracking.csv>
```

Outputs a CSV with columns `frame_idx, track_id, x_yd, y_yd, team, in_bad_run`.

`src/` is fully self-contained — it imports only from itself plus pip
packages and reads weights from `models/` + the manifest at
`data/manifests/h_pool_and_intrinsics.json`. Nothing in `src/` depends on
`scripts/`.

## Layout

```
src/
├── pipeline.py                          end-to-end orchestrator (clip → CSV)
│
├── field_mapping/                       stage 1: per-frame homography
│   ├── pipeline.py                      FieldMappingPipeline
│   ├── tokenizer.py                     mask → tokens (tokenize_frame)
│   ├── encoder.py                       TokenEncoder (phase-1 transformer)
│   ├── crop_classifier.py               CropClassifier (DSResNet10ww-style)
│   ├── number_refiner.py                NumberRefiner (phase-2 RFB)
│   ├── token_labeler.py                 TokenLabeler (phase-3 cross-attn)
│   ├── keypoints.py                     tokens → image↔NGS correspondences
│   ├── homography.py                    HomographyTrackerLite + LOO filter
│   ├── keypoint_bank.py                 bank validation for H candidates
│   ├── apply_homography.py              pixel ↔ field projection helpers
│   ├── field_model.py                   NFL field constants
│   └── classes.py                       NGS-x quantization + painted-number map
│
├── player_detection/                    stage 2
│   └── detector.py                      RF-DETR-Large + cache helpers
│
└── player_tracking/                     stages 3-5
    ├── tracker.py                       field-coord Kalman + multi-cue assoc
    ├── color_signature.py               24-d chromatic signature
    ├── team_classifier.py               PCA + median-split (forces 11/11)
    └── trajectory_smoothing.py          per-track Savitzky-Golay
```

## Pipeline flow

```
clip
 ├─► UNet (4-ch mask)                            field_mapping
 ├─► tokenizer
 ├─► TokenEncoder (phase 1)
 ├─► CropClassifier + NumberRefiner (phase 2)
 ├─► TokenLabeler (phase 3, cross-attn)
 ├─► keypoints → HomographySolver (per frame)
 ├─► LOO filter + polynomial bridge + Sav-Gol smoothing on H
 ├─► RF-DETR detector (per frame)               player_detection
 ├─► PlayerTracker (field-coord Kalman)         player_tracking
 ├─► classify_teams_color_pca
 └─► per-track Sav-Gol → NGS-yard CSV
```
