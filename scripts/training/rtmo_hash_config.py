"""MMPose RTMO-s config for hash-only keypoint detection.

Inherits from MMPose's stock RTMO-s COCO config, then overrides:
  - dataset_info: 1 keypoint class (hash) instead of 17 body joints
  - dataset path: /workspace/data/{train,valid}
  - head: num_keypoints=1
  - schedule: shorter (100 epochs, smaller LR for fine-tuning)
  - load_from: COCO-pretrained RTMO-s checkpoint
"""

_base_ = [
    "/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/"
    "rtmo-s_8xb32-600e_body7-640x640.py"
]

# Override the base config's `metafile` (used by oks_calculator + loss
# metainfo paths). Base default is 'configs/_base_/datasets/coco.py'
# which is a relative path that breaks. Use our 1-keypoint metainfo.
metafile = "/workspace/hash_metainfo.py"
_meta = "/workspace/hash_metainfo.py"

# ── Dataset metadata ─────────────────────────────────────────────────────
dataset_info = dict(
    dataset_name="hash_keypoints",
    paper_info=dict(
        author="rebuild pipeline",
        title="Field hash keypoint detection",
        container="custom",
        year="2026",
        homepage="",
    ),
    keypoint_info={
        0: dict(name="hash", id=0, color=[51, 153, 255], type="", swap=""),
    },
    skeleton_info={},
    joint_weights=[1.0],
    sigmas=[0.026],
)

# ── Data ─────────────────────────────────────────────────────────────────
# Override the base's CombinedDataset (7-dataset multi-source) with just
# our single hash dataset. Keep base's data_mode='bottomup' and inherit
# its train_pipeline_stage1 + val_pipeline so 'inputs' gets produced.
data_root = "/workspace/data_rtmo/"

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type="CocoDataset",   # mmpose's CocoDataset (in mmpose scope)
        data_root=data_root,
        data_mode="bottomup",
        ann_file="train/annotations.json",
        data_prefix=dict(img="train/images/"),
        metainfo=dict(from_file=_meta),
        pipeline={{_base_.train_pipeline_stage1}},
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type="CocoDataset",
        data_root=data_root,
        data_mode="bottomup",
        ann_file="valid/annotations.json",
        data_prefix=dict(img="valid/images/"),
        metainfo=dict(from_file=_meta),
        test_mode=True,
        pipeline={{_base_.val_pipeline}},
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "valid/annotations.json",
    score_mode="bbox",
    nms_mode="none",
)
test_evaluator = val_evaluator

# ── Model ────────────────────────────────────────────────────────────────
# Override every metainfo path that base config picked up from `metafile`.
model = dict(
    head=dict(
        num_keypoints=1,
        assigner=dict(
            oks_calculator=dict(metainfo=_meta),
        ),
        loss_oks=dict(metainfo=_meta),
    ),
)

# ── Schedule ─────────────────────────────────────────────────────────────
max_epochs = 100
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - 10, 1)],
)

base_lr = 1e-4
optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=5,
        start_factor=0.1, by_epoch=True, convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=5, end=max_epochs, T_max=max_epochs - 5,
        by_epoch=True, eta_min_ratio=0.05,
    ),
]

# ── Pretrained ──
load_from = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/"
    "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth"
)

# Output
work_dir = "/workspace/output_rtmo_hash"

# Default hooks
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=10,
        save_best="coco/AP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    logger=dict(type="LoggerHook", interval=20),
)
