_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/ctw1500.py',
]

load_from = '/home/yzy/mmocr/workdir_det/dbnet/dbnet/best_icdar_hmean_epoch_140.pth'

model = dict(
    backbone=dict(
        frozen_stages=2,  # Freeze stem + stage 1 + stage 2
        init_cfg=None  # Disable ImageNet init since we load from checkpoint
    )
)

# Dataset settings
ctw1500_textdet_train = _base_.ctw1500_textdet_train
ctw1500_textdet_train.pipeline = _base_.train_pipeline
ctw1500_textdet_test = _base_.ctw1500_textdet_test
ctw1500_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ctw1500_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw1500_textdet_test)

test_dataloader = val_dataloader

# Optimizer and Schedule
# Original DBNet uses LR=0.007 for BS=16.
# We use BS=4, so base LR should be ~0.00175.
# For fine-tuning, we reduce it further to 0.001 or 0.0005.
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='SGD', momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Learning rate schedule
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=100),
]

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='icdar/hmean',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=5),
)

work_dir = 'workdir_det/dbnet_ft_ctw1500_gentle'
