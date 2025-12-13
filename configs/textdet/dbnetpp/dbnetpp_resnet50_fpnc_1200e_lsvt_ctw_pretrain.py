_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_lsvt_ctw_pretrain.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = '/home/yzy/mmocr/work_dirs/dbnetpp_r50_pretrain_lsvt_ctw/epoch_6.pth'

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

train_list = [_base_.textdet_lsvt_ctw_train]
test_list = [_base_.textdet_lsvt_ctw_test]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=2)

# 训练稳定性增强：减小初始学习率、增加线性热身、加入梯度裁剪
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2))

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
    # dict(
    #     type='EarlyStoppingHook',
    #     monitor='icdar/hmean',  # 监控icdar/hmean指标
    #     patience=6,  # 连续6次验证无提升则停止
    #     rule='greater'  # hmean越大越好
    # )
]

# 学习率策略：缩短热身到 2 个 epoch，快速进入正常学习率避免长时间几乎不学习
param_scheduler = [
    dict(type='LinearLR', begin=0, end=2, start_factor=0.1, by_epoch=True),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=2, end=1200, by_epoch=True),
]
