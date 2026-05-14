#!/usr/bin/env bash
# NGS comparison on the 3 remaining in-scope plays (p011, p046, p118).
# Each runs the full pipeline + NGS sweep + per-player scoring + side-by-side viz.
set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

echo "=== creating pod ==="
.venv/bin/python scripts/aux/runpod/launch_runpod.py --create-only --training-type unet --gpu-type "NVIDIA GeForce RTX 5090" --cloud-type ALL --disk-size 40 --create-retries 6 --min-upload 25
SSH_HOST=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_host'])")
SSH_PORT=$(.venv/bin/python -c "import json;d=json.load(open('.runpod_pod.json'));print(d['ssh_port'])")
TARGET="root@$SSH_HOST"
SSH="ssh -n -o StrictHostKeyChecking=no -p $SSH_PORT $TARGET"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "=== prepping NGS TSVs (copy 3 into data/ngs) ==="
mkdir -p data/ngs
NGS_DIR="/Users/aldenkling/Desktop/Personal Research/ngs_highlights-master/play_data"
cp "$NGS_DIR/2019_GB_2019102712_282.tsv"   data/ngs/
cp "$NGS_DIR/2019_KC_2019102712_1205.tsv"  data/ngs/
cp "$NGS_DIR/2019_GB_2019102712_3067.tsv"  data/ngs/

echo "=== packing payload ==="
TAR=/tmp/ngs_3plays_payload.tar.gz
tar -czf "$TAR" \
    --exclude=__pycache__ --exclude=*.pyc \
    -C "$ROOT" \
    src \
    src/field_mapping \
    src/player_detection \
    src/player_tracking \
    scripts/aux/data_prep \
    scripts/aux/compare/compare_ngs.py \
    scripts/aux/runpod/requirements_farm_runpod.txt \
    models/unet_unified_v8_yardside_recover/best.pth \
    models/token_only_v10_phase1_pseudo/best.pth \
    models/rf_b_phase2_pseudo/best.pth \
    models/v10c_phase3_pseudo/best.pth \
    models/dsresnet10ww_round3_128x32/best.pth \
    models/rfdetr_best_ema.pth \
    videos/clips/2019102712/play_011/sideline.mp4 \
    videos/clips/2019102712/play_046/sideline.mp4 \
    videos/clips/2019102712/play_118/sideline.mp4 \
    data/h_pool_and_intrinsics.json \
    data/ngs/2019_GB_2019102712_282.tsv \
    data/ngs/2019_KC_2019102712_1205.tsv \
    data/ngs/2019_GB_2019102712_3067.tsv

echo "=== uploading ($(du -sh $TAR | cut -f1)) ==="
$SCP "$TAR" "$TARGET:/workspace/payload.tar.gz"
rm "$TAR"

echo "=== extracting + installing deps ==="
$SSH "cd /workspace && tar -xzf payload.tar.gz && rm payload.tar.gz \
    && python -m venv venv --system-site-packages \
    && source venv/bin/activate \
    && pip install -q -r scripts/aux/runpod/requirements_farm_runpod.txt rfdetr scikit-learn pandas"

echo "=== launching 3 NGS comparisons in background ==="
$SSH "cd /workspace && source venv/bin/activate && nohup bash -c '
python scripts/aux/compare/compare_ngs.py --device cuda \
    --clip videos/clips/2019102712/play_011/sideline.mp4 \
    --ngs-tsv data/ngs/2019_GB_2019102712_282.tsv \
    --snap-center 300

python scripts/aux/compare/compare_ngs.py --device cuda \
    --clip videos/clips/2019102712/play_046/sideline.mp4 \
    --ngs-tsv data/ngs/2019_KC_2019102712_1205.tsv \
    --snap-center 120

python scripts/aux/compare/compare_ngs.py --device cuda \
    --clip videos/clips/2019102712/play_118/sideline.mp4 \
    --ngs-tsv data/ngs/2019_GB_2019102712_3067.tsv \
    --snap-center 120

echo === ALL DONE ===
' > /workspace/ngs_3plays.log 2>&1 &"

echo "=== launched. Tail /workspace/ngs_3plays.log on the pod for progress. ==="
echo "Pod will need manual download + termination via:"
echo "  bash scripts/aux/runpod/_finalize_ngs_3plays.sh"
