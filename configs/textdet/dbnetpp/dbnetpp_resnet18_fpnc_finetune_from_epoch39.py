_base_ = [
    '_base_dbnetpp_resnet18_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_lsvt_ctw_pretrain.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# 加载预训练权重（epoch 39，hmean=0.5476）
# 注意：使用 --resume 时需要注释掉这一行，否则会从 epoch 39 而不是 last_checkpoint 继续
# load_from = 'work_dirs/dbnetpp_r18_pretrain_lsvt_ctw/best_final.pth'

train_list = [_base_.textdet_lsvt_ctw_train]
test_list = [_base_.textdet_lsvt_ctw_test]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)

# 降低学习率并缩短训练周期
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0014, momentum=0.9, weight_decay=0.0001))  # 原始 0.007 → 0.0014 (1/5)

# 只训练 30 个 epoch（从 epoch 39 开始，到 epoch 69 结束）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=69, val_interval=3)

# 学习率调度器：使用 PolyLR，终点为 69
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=69),
]

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
