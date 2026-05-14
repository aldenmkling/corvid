# `scripts/` — auxiliary tooling

Everything outside the production pipeline: training, data prep,
diagnostics, visualization, NGS comparison, RunPod orchestration.

Dependencies are one-way: `scripts/ → src/`. `src/` never imports from
`scripts/`.

## Layout (`scripts/aux/`)

```
scripts/aux/
├── compare/
│   └── compare_ngs.py                run the pipeline on a clip + score it
│                                     against an NGS TSV (position / speed /
│                                     accel + side-by-side viz)
│
├── viz/
│   └── four_panel_clip_overview.py   1920×1080 4-panel viz of a single clip
│                                     (source / corrs / rectified / tracking)
│
├── diagnostics/
│   ├── h_residual_report.py          single-clip LOO residual stats
│   └── compare_h_filters.py          old-vs-new H-filter confusion matrix
│
├── training/                          training scripts (RunPod targets)
│   ├── train_unet.py                  4-channel unified UNet
│   ├── train_encoder.py               TokenEncoder (phase 1)
│   ├── train_number_refiner.py        NumberRefiner (phase 2)
│   ├── train_token_labeler.py         TokenLabeler (phase 3)
│   ├── train_crop_classifier.py       CropClassifier (production)
│   ├── train_crop_classifier_mit.py   mit_b0 alternate (not in production)
│   └── train_player_detector.py       RF-DETR
│
├── data_prep/                         dataset building / pseudo-labeling
│   ├── segment_plays.py               game video → per-play clips
│   ├── convert_clips_30fps.py         normalize 60 fps → 30 fps
│   ├── farm_pseudo_labels.py          run pipeline → save token labels
│   ├── farm_pseudo_labels_all_clips.py
│   ├── qc_pseudo_labels.py
│   ├── build_unified_mask_dataset.py  pseudo labels → 4-ch UNet training set
│   ├── build_qc_unified_mask_dataset.py
│   ├── build_round3_dataset.py        crop classifier round-3 dataset
│   ├── build_smart_pool.py            stratified clip-pool sampler
│   ├── build_dense_field_training_pool.py
│   ├── extract_crop_logits.py
│   └── inject_intrinsics_into_manifest.py
│
└── runpod/                            pod lifecycle + bash launchers
    ├── launch_runpod.py               provision pod, upload, train, terminate
    ├── compare_ngs_runpod.sh
    ├── compare_ngs_3plays_runpod.sh
    ├── four_panel_clip_overview_runpod.sh
    ├── h_residual_report_runpod.sh
    ├── compare_h_filters_runpod.sh
    ├── train_crop_classifier_runpod.sh
    └── requirements_farm_runpod.txt
```

## Common operations

```bash
# Run pipeline + compare against NGS ground truth for a single play.
python scripts/aux/compare/compare_ngs.py \
    --clip videos/clips/2019092204/play_065/sideline.mp4 \
    --ngs-tsv data/ngs/2019_KC_2019092204_1643.tsv

# 4-panel debug viz of a single clip.
python scripts/aux/viz/four_panel_clip_overview.py \
    --clip videos/clips/2019092204/play_065/sideline.mp4

# Launch a training job on RunPod (auto-uploads, trains, terminates).
python scripts/aux/runpod/launch_runpod.py \
    --training-type rfdetr --epochs 50
```

## Conventions

- Output paths default to `output/<subdir>/<game>_<play>_*` so that NGS
  comparison artifacts and 4-panel videos sit alongside each other with
  consistent naming.
- RunPod shell launchers all package the local repo, upload it, run the
  Python entry point, download artifacts, and terminate the pod.
- Long-running operations are always launched with `nohup ... &` (Claude
  sessions time out and kill child processes otherwise).
