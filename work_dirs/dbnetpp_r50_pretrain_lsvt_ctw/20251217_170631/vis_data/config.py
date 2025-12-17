auto_scale_lr = dict(base_batch_size=16, enable=True)
custom_hooks = []
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3,
        rule='greater',
        save_best='icdar/hmean',
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/dbnetpp_r50_pretrain_lsvt_ctw/epoch_5.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(
            epsilon_ratio=0.002, text_repr_type='quad',
            type='DBPostprocessor'),
        type='DBHead'),
    neck=dict(
        asf_cfg=dict(attention_type='ScaleChannelSpatial'),
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        lateral_channels=256,
        type='FPNC'),
    type='DBNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=5, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.003, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=2, start_factor=0.1, type='LinearLR'),
    dict(
        begin=2,
        by_epoch=True,
        end=1200,
        eta_min=1e-07,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=None)
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='instances_val.json',
                data_root='data/lsvt_mmocr',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            dict(
                ann_file='instances_val.json',
                data_root='data/ctw_mmocr',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                4068,
                1024,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'instances',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_list = [
    dict(
        ann_file='instances_val.json',
        data_root='data/lsvt_mmocr',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='instances_val.json',
        data_root='data/ctw_mmocr',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        4068,
        1024,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'instances',
        ),
        type='PackTextDetInputs'),
]
textdet_ctw_data_root = 'data/ctw_mmocr'
textdet_ctw_test = dict(
    ann_file='instances_val.json',
    data_root='data/ctw_mmocr',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
textdet_ctw_train = dict(
    ann_file='instances_train.json',
    data_root='data/ctw_mmocr',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
textdet_lsvt_ctw_data_root = 'data/textdet_pretrain_lsvt_ctw'
textdet_lsvt_ctw_test = dict(
    ann_file='instances_val.json',
    data_root='data/textdet_pretrain_lsvt_ctw',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
textdet_lsvt_ctw_train = dict(
    ann_file='instances_train.json',
    data_root='data/textdet_pretrain_lsvt_ctw',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
textdet_lsvt_data_root = 'data/lsvt_mmocr'
textdet_lsvt_test = dict(
    ann_file='instances_val.json',
    data_root='data/lsvt_mmocr',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
textdet_lsvt_train = dict(
    ann_file='instances_train.json',
    data_root='data/lsvt_mmocr',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='instances_train.json',
                data_root='data/lsvt_mmocr',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='instances_train.json',
                data_root='data/ctw_mmocr',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                brightness=0.12549019607843137,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
            dict(
                args=[
                    [
                        'Fliplr',
                        0.5,
                    ],
                    dict(cls='Affine', rotate=[
                        -10,
                        10,
                    ]),
                    [
                        'Resize',
                        [
                            0.5,
                            3.0,
                        ],
                    ],
                ],
                type='ImgAugWrapper'),
            dict(min_side_ratio=0.1, type='RandomCrop'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(size=(
                640,
                640,
            ), type='Pad'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='instances_train.json',
        data_root='data/lsvt_mmocr',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='instances_train.json',
        data_root='data/ctw_mmocr',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        brightness=0.12549019607843137,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        args=[
            [
                'Fliplr',
                0.5,
            ],
            dict(cls='Affine', rotate=[
                -10,
                10,
            ]),
            [
                'Resize',
                [
                    0.5,
                    3.0,
                ],
            ],
        ],
        type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='instances_val.json',
                data_root='data/lsvt_mmocr',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            dict(
                ann_file='instances_val.json',
                data_root='data/ctw_mmocr',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                4068,
                1024,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'instances',
                ),
                type='PackTextDetInputs'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/dbnetpp_r50_pretrain_lsvt_ctw'
