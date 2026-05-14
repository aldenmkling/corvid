# `models/` — production weights

Eight checkpoints, ~310 MB total. Weights are not tracked in git (see
`.gitignore`). Architectures live in `src/`.

## Pipeline stages

### Field mapping (stage 1)

| File | Class | Role | Notes |
|---|---|---|---|
| `unet_unified_v8_yardside_recover/best.pth` | UNet (smp mit_b0) | 4-channel mask: yard / side / hash / number | 43 MB |
| `token_only_v10_phase1_pseudo/best.pth` | `TokenEncoder` | Per-token transformer (phase 1) | 3.2 MB |
| `rf_b_phase2_pseudo/best.pth` | `NumberRefiner` | Per-number-token row-feature refiner (phase 2) | 0.4 MB |
| `v10c_phase3_pseudo/best.pth` | `TokenLabeler` | Cross-attention labeler (phase 3) | 3.2 MB |
| `dsresnet10ww_round3_128x32/best.pth` | `CropClassifier` | Painted-number digit classifier (9-class) | 0.6 MB |

### Player detection (stage 2)

| File | Class | Role | Notes |
|---|---|---|---|
| `rfdetr_best_ema.pth` | RF-DETR-Large (EMA weights) | Per-frame player boxes | 128 MB — use this for inference |
| `rfdetr_best_regular.pth` | RF-DETR-Large (regular weights) | Backup | 128 MB |

### Auxiliary

| File | Class | Role | Notes |
|---|---|---|---|
| `view_classifier.pt` | YOLO-cls | Sideline vs endzone (for play segmentation) | 2.8 MB |

## Conventions

- Folder names are training-context artifacts (`v10_phase1_pseudo`,
  `round3_128x32`). The class names in `src/` are pipeline-role names.
  See `src/field_mapping/*.py` for the loaders.
- Each folder has `best.pth` (lowest val loss) and `last.pth` (last epoch);
  production uses `best.pth`. The bare `.pth` files at the top of `models/`
  are RF-DETR + the view classifier.
- Loading helpers (e.g. `load_crop_classifier`) live alongside the
  matching class in `src/field_mapping/`.

## Provenance

- `dsresnet10ww_round3_128x32` was retrained from scratch on 2026-05-14
  after the original training script was lost. Val acc 96.76% on
  `data/training/crop_classifier/round3_128x32` (12,142 samples, 9 classes).
  Matches the previous training within noise (96.71%).
- All other checkpoints came from RunPod RTX-5090 training runs; see
  `scripts/aux/training/` for the training entry points.
