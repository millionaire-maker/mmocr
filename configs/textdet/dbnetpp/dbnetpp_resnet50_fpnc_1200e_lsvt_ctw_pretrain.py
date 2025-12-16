_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_lsvt_ctw_pretrain.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# 指定work_dir便于在云服务器上统一管理训练产物，也方便 --resume
work_dir = 'work_dirs/dbnetpp_r50_pretrain_lsvt_ctw'

# 如需从已有checkpoint开始训练，填写本机路径或使用命令行覆盖：
#   --cfg-options load_from=/path/to/xxx.pth
load_from = None

# RTX 3090(24GB) + 固定训练分辨率(640x640)场景下，开启 cudnn benchmark 可显著提升吞吐
env_cfg = dict(cudnn_benchmark=True)

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

# 使用当前服务器上已有的数据目录（避免依赖不存在的 data/textdet_pretrain_lsvt_ctw）
textdet_lsvt_data_root = 'data/lsvt_mmocr'
textdet_ctw_data_root = 'data/ctw_mmocr'

textdet_lsvt_train = dict(
    type='OCRDataset',
    data_root=textdet_lsvt_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
textdet_ctw_train = dict(
    type='OCRDataset',
    data_root=textdet_ctw_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

textdet_lsvt_test = dict(
    type='OCRDataset',
    data_root=textdet_lsvt_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)
textdet_ctw_test = dict(
    type='OCRDataset',
    data_root=textdet_ctw_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)

train_list = [textdet_lsvt_train, textdet_ctw_train]
test_list = [textdet_lsvt_test, textdet_ctw_test]

train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(enable=True, base_batch_size=16)

# AMP + 梯度裁剪：提升吞吐、降低显存并增强训练稳定性
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    loss_scale='dynamic')

# 数据集较大，验证过于频繁会明显拖慢训练；保持更合理的验证间隔
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=5)

# 配置权重保存策略：保留最新3个权重 + 保留最优权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,  # 与val_interval对齐，减少I/O开销
        max_keep_ckpts=3,  # 保留最新的3个权重
        save_best='icdar/hmean',  # 明确监控icdar/hmean指标
        rule='greater'  # hmean越大越好
    )
)

# 早停机制：连续6次验证无提升自动停止训练
custom_hooks = [
    # 预训练阶段通常更关注“学习到更通用的特征”，如需节省时间可启用早停
    # dict(
    #     type='EarlyStoppingHook',
    #     monitor='icdar/hmean',
    #     patience=10,
    #     min_delta=0.001,
    #     rule='greater',
    # ),
]

# 学习率策略：缩短热身到 2 个 epoch，快速进入正常学习率避免长时间几乎不学习
param_scheduler = [
    dict(type='LinearLR', begin=0, end=2, start_factor=0.1, by_epoch=True),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=2, end=1200, by_epoch=True),
]
