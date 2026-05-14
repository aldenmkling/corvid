#!/usr/bin/env bash
# Run the single-file 4-panel viz on a fresh RunPod pod.
# Output: output/play_065_visuals/4panel.mp4
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
TAR=/tmp/play_065_4panel_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src \
    scripts/training \
    scripts/data_prep \
    scripts/testing/make_play_065_4panel.py \
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

echo "=== running 4-panel viz ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python scripts/testing/make_play_065_4panel.py --device cuda 2>&1"

echo "=== downloading output ==="
mkdir -p "$ROOT/output/play_065_visuals"
$SCP "$TARGET:/workspace/output/play_065_visuals/4panel.mp4" "$ROOT/output/play_065_visuals/"

echo "=== terminating pod ==="
.venv/bin/python scripts/runpod/launch_runpod.py --terminate
echo "=== done ==="
ls -la "$ROOT/output/play_065_visuals/4panel.mp4"
