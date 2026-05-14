#!/usr/bin/env bash
# Run analyze_h_residuals.py on a fresh RunPod pod.
# Usage: bash scripts/aux/runpod/h_residual_report_runpod.sh
set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

echo "=== creating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --create-only --training-type unet --gpu-type "NVIDIA GeForce RTX 5090" --cloud-type ALL --disk-size 40 --create-retries 6
SSH_HOST=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_host'])")
SSH_PORT=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_port'])")
TARGET="root@$SSH_HOST"
SSH="ssh -n -o StrictHostKeyChecking=no -p $SSH_PORT $TARGET"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== packing payload ==="
TAR=/tmp/h_resid_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src \
    src/field_mapping \
    src/player_detection \
    src/player_tracking \
    scripts/aux/data_prep \
    scripts/aux/diagnostics/h_residual_report.py \
    scripts/aux/runpod/requirements_farm_runpod.txt \
    models/unet_unified_v8_yardside_recover/best.pth \
    models/token_only_v10_phase1_pseudo/best.pth \
    models/rf_b_phase2_pseudo/best.pth \
    models/v10c_phase3_pseudo/best.pth \
    models/dsresnet10ww_round3_128x32/best.pth \
    videos/clips/2019092204/play_065/sideline.mp4 \
    data/manifests/h_pool_and_intrinsics.json

echo "=== uploading ($(du -sh $TAR | cut -f1)) ==="
$SCP "$TAR" "$TARGET:/workspace/payload.tar.gz"
rm "$TAR"

echo "=== extracting + installing deps ==="
$SSH "cd /workspace && tar -xzf payload.tar.gz && rm payload.tar.gz \
    && python -m venv venv --system-site-packages \
    && source venv/bin/activate \
    && pip install -q -r scripts/aux/runpod/requirements_farm_runpod.txt scikit-learn"

echo "=== running analysis ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python scripts/aux/diagnostics/h_residual_report.py --device cuda 2>&1"

echo "=== downloading output ==="
mkdir -p "$ROOT/output"
$SCP "$TARGET:/workspace/output/h_residual_analysis.json" "$ROOT/output/" || true

echo "=== terminating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --terminate

echo "=== done ==="
