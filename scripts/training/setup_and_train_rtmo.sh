#!/bin/bash
# RunPod entrypoint for RTMO-s hash detection training.
# Clones mmpose, installs the stack, fine-tunes RTMO-s on /workspace/data_rtmo.

set -e

cd /workspace

echo "─── Setting up venv + dependencies ───"
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -q -U pip
pip install -q openmim
mim install -q "mmengine>=0.10.0"
mim install -q "mmcv>=2.0.1,<2.2.0"
mim install -q "mmdet>=3.0.0,<3.3.0"

echo "─── Cloning mmpose ───"
if [ ! -d /workspace/mmpose ]; then
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git
fi
cd mmpose
pip install -q -v -e .

echo "─── Verifying installation ───"
python -c "import mmpose; print('mmpose', mmpose.__version__)"
python -c "import mmcv; print('mmcv', mmcv.__version__)"
python -c "import mmdet; print('mmdet', mmdet.__version__)"

echo "─── Starting training ───"
cd /workspace
python /workspace/mmpose/tools/train.py /workspace/rtmo_hash_config.py \
    --work-dir /workspace/output_rtmo_hash 2>&1 | tee /workspace/train_rtmo.log

echo "─── Done. Best ckpt:"
ls -la /workspace/output_rtmo_hash/best_*.pth || true
