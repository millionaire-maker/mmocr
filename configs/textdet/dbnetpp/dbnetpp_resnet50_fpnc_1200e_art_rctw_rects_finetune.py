_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_art_rctw_rects_finetune.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# 指定work_dir便于在云服务器上统一管理训练产物，也方便 --resume
work_dir = 'work_dirs/dbnetpp_r50_finetune_art_rctw_rects'

# 微调阶段强烈建议从 LSVT+CTW 预训练权重开始：
# - 首次启动：直接用 load_from（只加载模型权重，不恢复优化器/调度器状态）
# - 如需中断后继续：请使用 `--resume` 并确保 `load_from=None`，否则会从该权重“错误地resume”
load_from = 'work_dirs/dbnetpp_r50_pretrain_lsvt_ctw/best_icdar_hmean_epoch_60.pth'

# RTX 3090(24GB) + 固定训练分辨率(640x640)场景下，开启 cudnn benchmark 可显著提升吞吐
env_cfg = dict(cudnn_benchmark=True)

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

# 使用当前服务器上已有的数据目录（避免依赖不存在的 data/textdet_finetune_art_rctw_rects）
textdet_art_data_root = 'data/art_mmocr'
textdet_rctw_data_root = 'data/rctw17_mmocr'
textdet_rects_data_root = 'data/rects_mmocr'

textdet_art_train = dict(
    type='OCRDataset',
    data_root=textdet_art_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
textdet_rctw_train = dict(
    type='OCRDataset',
    data_root=textdet_rctw_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
textdet_rects_train = dict(
    type='OCRDataset',
    data_root=textdet_rects_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

textdet_art_test = dict(
    type='OCRDataset',
    data_root=textdet_art_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)
textdet_rctw_test = dict(
    type='OCRDataset',
    data_root=textdet_rctw_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)
textdet_rects_test = dict(
    type='OCRDataset',
    data_root=textdet_rects_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)

train_list = [textdet_art_train, textdet_rctw_train, textdet_rects_train]
test_list = [textdet_art_test, textdet_rctw_test, textdet_rects_test]

train_dataloader = dict(
    # 3090(24GB) 上在 640x640 训练分辨率下，batch=32 基本可跑满显存但仍留少量余量
    # 若出现 OOM，可优先降到 28/24
    batch_size=32,
    # 该机器 CPU 核数较少（7c），num_workers 过大容易造成调度抖动；8 相对更稳
    num_workers=8,
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

# 分别输出 ART / RCTW / RECTS 的验证集指标（同时保留原始 icdar/hmean 作为总分）
val_evaluator = dict(
    type='MultiDatasetHmeanIOUMetric',
    dataset_prefixes=dict(
        art=textdet_art_data_root,
        rctw=textdet_rctw_data_root,
        rects=textdet_rects_data_root,
    ))
test_evaluator = val_evaluator

auto_scale_lr = dict(enable=True, base_batch_size=32)

# 微调优化器配置：降低学习率避免灾难性遗忘；同时启用 AMP 以提升吞吐并节省显存
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    loss_scale='dynamic')

# 学习率策略：短热身 + Poly，适配微调阶段较短训练周期
param_scheduler = [
    dict(type='LinearLR', begin=0, end=2, start_factor=0.1, by_epoch=True),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=2, end=200, by_epoch=True),
]

# 微调阶段一般不需要 1200 epochs；配合早停，优先追求收敛与泛化
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=3)

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
        patience=12,  # 连续12次验证无提升则停止（避免因短期波动过早停止）
        min_delta=0.001,  # 改善幅度>0.1%即视为有效提升
        rule='greater'  # hmean越大越好
    )
]
