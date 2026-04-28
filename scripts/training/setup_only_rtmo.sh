#!/bin/bash
# Stage 1: install MMPose stack, parse config, build model. No training.
# Exits 0 if everything is wired up; non-zero if config fails to parse or
# model fails to build.

set -e
cd /workspace

echo "─── Setting up venv + deps ───"
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -q -U pip
# mmcv's setup.py uses pkg_resources (removed in setuptools >=70).
pip install -q "setuptools<70" "wheel" "packaging" "numpy" "torch"
pip install -q "mmengine>=0.10.0"
# Full mmcv with CUDA ops. Build from source (~30 min) — no prebuilt
# wheel exists for our PyTorch 2.8 + CUDA 12.8 combo, but EDPoseHead in
# mmpose unconditionally imports mmcv.ops so we need the full build.
echo "─── Building full mmcv with CUDA ops (~30 min) ───"
pip install -q "numpy<2"   # xtcocotools ABI compat
pip install -q --no-build-isolation "mmcv>=2.0.1,<2.2.0"
pip install -q "mmdet>=3.0.0,<3.3.0"

echo "─── Cloning mmpose ───"
if [ ! -d /workspace/mmpose ]; then
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git
fi
cd mmpose
# chumpy (legacy 3D body dep) has broken setup.py on Python 3.11.
# RTMO is 2D pose — doesn't need it. Strip it from requirements.
for f in requirements/runtime.txt requirements.txt; do
    [ -f "$f" ] && sed -i '/chumpy/d' "$f"
done
pip install -q --no-build-isolation -e .

echo "─── Verifying versions ───"
python -c "import mmpose, mmcv, mmdet, mmengine; \
print('mmpose', mmpose.__version__, '| mmcv', mmcv.__version__, \
'| mmdet', mmdet.__version__, '| mmengine', mmengine.__version__)"

echo "─── Parsing config ───"
cd /workspace
python -c "
from mmengine.config import Config
cfg = Config.fromfile('/workspace/rtmo_hash_config.py')
print('config OK. model.head.num_keypoints =', cfg.model.head.num_keypoints)
print('train data_root =', cfg.train_dataloader.dataset.data_root)
print('load_from =', cfg.load_from)
"

echo "─── Trying to build the model (no weights, no data) ───"
python -c "
from mmengine.config import Config
from mmengine.registry import MODELS
import mmpose  # triggers registry population
cfg = Config.fromfile('/workspace/rtmo_hash_config.py')
# Strip load_from so we don't try to fetch weights yet.
cfg.model.pop('init_cfg', None)
model = MODELS.build(cfg.model)
n_params = sum(p.numel() for p in model.parameters())
print(f'model built OK. params = {n_params/1e6:.1f} M')
"

echo "─── Stage 1 SETUP OK ───"
