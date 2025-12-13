_base_ = [
    '_base_fcenet_resnet50-dcnv2_fpn.py',
    '../_base_/datasets/textdet_art_rctw_rects_finetune.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
train_cfg = dict(max_epochs=1500)
param_scheduler = [dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500)]

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset',
        data_root='data/textdet_finetune_art_rctw_rects',
        ann_file='instances_train.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/textdet_finetune_art_rctw_rects',
        ann_file='instances_val.json',
        test_mode=True,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)

# 修改验证间隔为每3个epoch验证一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1500, val_interval=3)

# 配置权重保存策略：保留最新3个权重 + 保留最优权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # 每3个epoch保存一次
        max_keep_ckpts=3,  # 保留最新的3个权重
        save_best='icdar/hmean',  # 明确监控icdar/hmean指标
        rule='greater'  # hmean越大越好
    )
)

# 早停机制：连续6次验证无提升自动停止训练
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='icdar/hmean',  # 监控icdar/hmean指标
        patience=6,  # 连续6次验证无提升则停止
        rule='greater'  # hmean越大越好
    )
]
