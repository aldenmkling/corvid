#!/usr/bin/env bash
# Train the DSResNet10ww number-crop classifier on a fresh RunPod pod.
# Output: models/dsresnet10ww_round3_128x32/{best,last}.pth + logs.
set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

echo "=== creating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --create-only --training-type unet --gpu-type "NVIDIA GeForce RTX 5090" --cloud-type ALL --disk-size 30 --create-retries 6 --min-upload 25
SSH_HOST=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_host'])")
SSH_PORT=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_port'])")
TARGET="root@$SSH_HOST"
SSH="ssh -n -o StrictHostKeyChecking=no -p $SSH_PORT $TARGET"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== packing payload ==="
TAR=/tmp/train_crop_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src/field_mapping/crop_classifier.py \
    src/field_mapping/classes.py \
    scripts/aux/training/train_crop_classifier.py \
    scripts/aux/runpod/requirements_farm_runpod.txt \
    data/number_classifier/round3_128x32

echo "=== uploading ($(du -sh $TAR | cut -f1)) ==="
$SCP "$TAR" "$TARGET:/workspace/payload.tar.gz"
rm "$TAR"

echo "=== extracting + installing deps ==="
$SSH "cd /workspace && tar -xzf payload.tar.gz && rm payload.tar.gz \
    && python -m venv venv --system-site-packages \
    && source venv/bin/activate \
    && pip install -q -r scripts/aux/runpod/requirements_farm_runpod.txt"

echo "=== training ==="
$SSH "cd /workspace && source venv/bin/activate \
    && python -u scripts/aux/training/train_crop_classifier.py --device cuda 2>&1"

echo "=== downloading outputs ==="
mkdir -p "$ROOT/models/dsresnet10ww_round3_128x32"
$SCP "$TARGET:/workspace/models/dsresnet10ww_round3_128x32/*" "$ROOT/models/dsresnet10ww_round3_128x32/"

echo "=== terminating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --terminate

echo "=== done ==="
ls -la "$ROOT/models/dsresnet10ww_round3_128x32/"
