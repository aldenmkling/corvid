# `data/` — datasets, manifests, ground truth

Grouped into four sections: small text manifests, NGS ground-truth TSVs,
hand-labeled inputs, and generated training datasets. Most bulk content is
gitignored; see `.gitignore`.

```
data/
├── manifests/                  small JSON/txt files (tracked in git)
├── ngs/                        NGS ground-truth TSVs (tracked)
├── labels/                     hand-labeled data (labels tracked, images ignored)
└── training/                   generated training datasets (all ignored)
```

## `manifests/` — small tracked artifacts

Pipeline manifests, blacklists, QC ledgers — all small JSONs / one txt.

| File | Contents |
|---|---|
| `h_pool_and_intrinsics.json` | Stratified clip pool + per-clip camera intrinsics (K, dist). **The central manifest.** Referenced from `src/pipeline.py` and most aux scripts. |
| `hard_train_samples.json` | Clips/frames flagged as hard during phase-3 training. |
| `hash_excluded_frames.txt` | Bad-GT frame IDs to skip when building hash mask training sets. |
| `phase1_blacklist.json` | Clips known to break phase 1. |
| `phase1_failures.json` | Phase-1 farming failure log. |
| `phase1_failures_qc.json` | QC review of phase-1 failures. |
| `pseudo_labels_qc.json` | QC review of pseudo-label outputs. |

## `ngs/` — NGS ground-truth TSVs

Per-play NGS tracking exports at 10 Hz (`x, y, s, a, dir, o, event, ...`).
Used by `scripts/aux/compare/compare_ngs.py` for validation.

```
2019_KC_2019092204_1643.tsv   (BAL@KC, play_065)
2019_GB_2019102712_282.tsv    (GB@KC,  play_011)
2019_GB_2019102712_3067.tsv   (GB@KC,  play_046)
2019_KC_2019102712_1205.tsv   (GB@KC,  play_118)
```

## `labels/` — hand-labeled inputs

| Subdir | Contents | Tracked? |
|---|---|---|
| `player_detection/` | YOLO-format labels for RF-DETR. 967 frames, 8 games. Bulk JPEGs gitignored; only `export_final/labels/*.txt` + `classes.txt` are tracked. |
| `field_keypoints/` | Per-image keypoint JSONs from Label Studio. 892 frames after AL round 2. Labels tracked; raw annotation images gitignored. |
| `line_masks/` | Hand-painted yardline/sideline masks (`train/`, `valid/`, `al_round3/`). Bulk PNGs gitignored. |
| `view_classifier/` | Sideline-vs-endzone classifier training set. Fully gitignored (bulk JPEGs). |

## `training/` — generated training datasets (all gitignored)

These are reproducible from the labels above + the pipeline; regenerate
rather than copy them around.

| Subdir | Contents | Built by |
|---|---|---|
| `crop_classifier/` | Per-class 128×32 crops of painted numbers (`round1/`, `round2/`, `round3/`, `round3_128x32/`). Round 3_128x32 is the live training set for `CropClassifier`. | `scripts/aux/data_prep/build_round3_dataset.py` |
| `player_detection_coco/` | COCO-format RF-DETR dataset built from `labels/player_detection/`. | `scripts/aux/data_prep/` (build_coco helper) |
| `pseudo_labels/` | One `.npz` per clip with auto-labeled tokens (for phase-1/2/3 training). | `scripts/aux/data_prep/farm_pseudo_labels.py` |
| `pseudo_labels_crops/` | Corresponding crop arrays for each pseudo-labeled clip. | `farm_pseudo_labels.py` |
| `unified_masks/` | 4-channel UNet training masks (`v8_yardside_recover/` is the live set). | `scripts/aux/data_prep/build_unified_mask_dataset.py` |

## Conventions

- One game per play: a clip is keyed by `<gameId>/play_<NNN>/{sideline,endzone}.mp4`.
- Manifests use clip-relative paths (`2019092204/play_065/sideline.mp4`).
- Anything regeneratable lives under `training/` and is gitignored.
  Anything hand-labeled lives under `labels/` and is tracked (labels only).
