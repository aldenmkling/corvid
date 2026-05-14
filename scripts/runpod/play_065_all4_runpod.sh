#!/usr/bin/env bash
# Run BOTH play_065 viz scripts on a fresh RunPod pod and pull all 4 outputs.
# Outputs:
#   output/play_065_visuals/source_with_corrs.mp4
#   output/play_065_visuals/rectified.mp4
#   output/play_065_visuals/tracker.mp4         (team-colored boxes)
#   output/play_065_visuals/tracking_dots.mp4   (team-colored dots)
set -e
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

echo "=== creating pod ==="
.venv/bin/python scripts/runpod/launch_runpod.py --create-only --training-type unet --gpu-type "NVIDIA GeForce RTX 5090" --cloud-type ALL --disk-size 40 --create-retries 6 --min-upload 5
SSH_HOST=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_host'])")
SSH_PORT=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_port'])")
TARGET="root@$SSH_HOST"
SSH="ssh -n -o StrictHostKeyChecking=no -p $SSH_PORT $TARGET"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== packing payload ==="
TAR=/tmp/play_065_all4_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src \
    scripts/training \
    scripts/data_prep \
    scripts/testing/make_play_065_visuals.py \
    scripts/testing/make_play_065_tracking_dots.py \
    scripts/runpod/requirements_farm_runpod.txt \
    models/unet_unified_v8_yardside_recover/best.pth \
    models/token_only_v10_phase1_pseudo/best.pth \
    models/rf_b_phase2_pseudo/best.pth \
    models/v10c_phase3_pseudo/best.pth \
    models/dsresnet10ww_round3_128x32/best.pth \
    models/rfdetr_best_ema.pth \
    videos/clips/2019092204/play_065/sideline.mp4 \
    data/h_pool_and_intrinsics.json

echo "=== uploading ($(du -sh $TAR | cut -f1)) ==="
$SCP "$TAR" "$TARGET:/workspace/payload.tar.gz"
rm "$TAR"

echo "=== extracting + installing deps ==="
$SSH "cd /workspace && tar -xzf payload.tar.gz && rm payload.tar.gz \
    && python -m venv venv --system-site-packages \
    && source venv/bin/activate \
    && pip install -q -r scripts/runpod/requirements_farm_runpod.txt rfdetr scikit-learn"

echo "=== running viz 1/2: source/rectified/tracker ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python scripts/testing/make_play_065_visuals.py --device cuda 2>&1"

echo "=== running viz 2/2: tracking dots ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python scripts/testing/make_play_065_tracking_dots.py --device cuda 2>&1"

echo "=== downloading outputs ==="
mkdir -p "$ROOT/output/play_065_visuals"
$SCP "$TARGET:/workspace/output/play_065_visuals/*.mp4" "$ROOT/output/play_065_visuals/"

echo "=== terminating pod ==="
.venv/bin/python scripts/runpod/launch_runpod.py --terminate

echo "=== done ==="
ls -la "$ROOT/output/play_065_visuals/"
