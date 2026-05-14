#!/usr/bin/env bash
# Compare our play_065 tracker output against NGS ground truth.
# Output: output/ngs_compare/{play_065_per_player.csv, play_065_summary.txt,
#                              play_065_compare.mp4}
set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

echo "=== creating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --create-only --training-type unet --gpu-type "NVIDIA GeForce RTX 5090" --cloud-type ALL --disk-size 40 --create-retries 6 --min-upload 50
SSH_HOST=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_host'])")
SSH_PORT=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_port'])")
TARGET="root@$SSH_HOST"
SSH="ssh -n -o StrictHostKeyChecking=no -p $SSH_PORT $TARGET"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== prepping NGS TSV into data/ngs (pod expects it there) ==="
mkdir -p data/ngs
cp "/Users/aldenkling/Desktop/Personal Research/ngs_highlights-master/play_data/2019_KC_2019092204_1643.tsv" data/ngs/

echo "=== packing payload ==="
TAR=/tmp/ngs_compare_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src \
    src/pipeline \
    scripts/aux/data_prep \
    scripts/aux/compare/compare_ngs_play_065.py \
    scripts/aux/runpod/requirements_farm_runpod.txt \
    models/unet_unified_v8_yardside_recover/best.pth \
    models/token_only_v10_phase1_pseudo/best.pth \
    models/rf_b_phase2_pseudo/best.pth \
    models/v10c_phase3_pseudo/best.pth \
    models/dsresnet10ww_round3_128x32/best.pth \
    models/rfdetr_best_ema.pth \
    videos/clips/2019092204/play_065/sideline.mp4 \
    data/h_pool_and_intrinsics.json \
    data/ngs/2019_KC_2019092204_1643.tsv

echo "=== uploading ($(du -sh $TAR | cut -f1)) ==="
$SCP "$TAR" "$TARGET:/workspace/payload.tar.gz"
rm "$TAR"

echo "=== extracting + installing deps ==="
$SSH "cd /workspace && tar -xzf payload.tar.gz && rm payload.tar.gz \
    && python -m venv venv --system-site-packages \
    && source venv/bin/activate \
    && pip install -q -r scripts/aux/runpod/requirements_farm_runpod.txt rfdetr scikit-learn pandas"

echo "=== running NGS compare ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python scripts/aux/compare/compare_ngs_play_065.py --device cuda 2>&1"

echo "=== downloading outputs ==="
mkdir -p "$ROOT/output/ngs_compare"
$SCP "$TARGET:/workspace/output/ngs_compare/*" "$ROOT/output/ngs_compare/"

echo "=== terminating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --terminate
echo "=== done ==="
ls -la "$ROOT/output/ngs_compare/"
