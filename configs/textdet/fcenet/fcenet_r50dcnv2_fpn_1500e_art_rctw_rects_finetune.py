_base_ = [
    '_base_fcenet_resnet50-dcnv2_fpn.py',
    '../_base_/datasets/textdet_art_rctw_rects_finetune.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

work_dir = 'work_dirs/fcenet_r50dcnv2_finetune_art_rctw_rects'

# 基线：从零开始在 finetune 数据上训练（如需对比“pretrain->finetune”，用命令行覆盖 load_from）
load_from = None

# 固定输入分辨率/尺度增强场景下，开启 cudnn benchmark 可提升吞吐
env_cfg = dict(cudnn_benchmark=True)

max_epochs = 100

# AMP + 梯度裁剪：提升吞吐、降低显存并增强训练稳定性
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=5, norm_type=2),
    loss_scale='dynamic')

# 学习率策略：短热身 + Poly，适配 150~200 epoch 训练
param_scheduler = [
    dict(type='LinearLR', begin=0, end=2, start_factor=0.1, by_epoch=True),
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=1e-7,
        begin=2,
        end=max_epochs,
        by_epoch=True),
]

# 使用当前服务器上已有的数据目录
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

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    # ReCTS 数据中存在少量退化/非法 polygon（如全为 -1），需先过滤避免 FCENet 生成 target 时崩溃
    dict(type='FixInvalidPolygon', fix_from_bbox=False),
    dict(
        type='RandomResize',
        scale=(800, 800),
        ratio_range=(0.75, 2.5),
        keep_ratio=True),
    dict(
        type='TextDetRandomCropFlip',
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2),
    dict(
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.8),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='RandomRotate',
                max_angle=30,
                pad_with_fixed_color=False,
                use_canvas=True)
        ],
        prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(type='Resize', scale=800, keep_ratio=True),
            dict(type='SourceImagePad', target_scale=800)
        ],
                    dict(type='Resize', scale=800, keep_ratio=False)],
        prob=[0.6, 0.4]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(type='FixInvalidPolygon', fix_from_bbox=False),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ConcatDataset 下额外输出每个数据集指标（同时保留 icdar/hmean 作为总分）
val_evaluator = dict(
    type='MultiDatasetHmeanIOUMetric',
    dataset_prefixes=dict(
        art=textdet_art_data_root,
        rctw=textdet_rctw_data_root,
        rects=textdet_rects_data_root,
    ))
test_evaluator = val_evaluator

auto_scale_lr = dict(enable=True, base_batch_size=32)

# 修改验证间隔为每3个epoch验证一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)

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

# 早停机制：连续多次验证无提升自动停止训练（避免无人值守时长时间无效跑）
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='icdar/hmean',  # 监控icdar/hmean指标
        patience=12,  # 连续12次验证无提升则停止
        min_delta=0.001,  # 改善幅度>0.1%即视为有效提升
        rule='greater'  # hmean越大越好
    )
]
