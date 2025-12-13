_base_ = [
    '_base_dbnetpp_resnet18_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_lsvt_ctw_pretrain.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = None

train_list = [_base_.textdet_lsvt_ctw_train]
test_list = [_base_.textdet_lsvt_ctw_test]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

# 修改验证间隔为每3个epoch验证一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_interval=3)

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
