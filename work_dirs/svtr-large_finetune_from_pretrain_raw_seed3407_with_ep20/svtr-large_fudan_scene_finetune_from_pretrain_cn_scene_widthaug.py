auto_scale_lr = dict(base_batch_size=256, enable=True)
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='Fudan/recog/word_acc',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
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
dictionary = dict(
    dict_file='data/charset/charset_rec_cn_en.txt',
    type='Dictionary',
    with_padding=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fudan_scene_rec_root = 'data/fudan/scene'
fudan_scene_test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='scene_test',
        data_root='data/fudan/scene',
        pipeline=None,
        type='RecogLMDBDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
fudan_scene_textrecog_test = dict(
    ann_file='scene_test',
    data_root='data/fudan/scene',
    pipeline=None,
    type='RecogLMDBDataset')
fudan_scene_textrecog_train = dict(
    ann_file='scene_train',
    data_root='data/fudan/scene',
    pipeline=None,
    type='RecogLMDBDataset')
fudan_scene_textrecog_val = dict(
    ann_file='scene_val',
    data_root='data/fudan/scene',
    pipeline=None,
    type='RecogLMDBDataset')
fudan_scene_train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='scene_train',
        data_root='data/fudan/scene',
        pipeline=None,
        type='RecogLMDBDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
fudan_scene_val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='scene_val',
        data_root='data/fudan/scene',
        pipeline=None,
        type='RecogLMDBDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
launcher = 'pytorch'
load_from = 'work_dirs/svtr-large_pretrain_plus_gapfix_seed3407/best_PretrainCN_holdout_recog_word_acc_epoch_20.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    data_preprocessor=dict(
        mean=[
            127.5,
        ], std=[
            127.5,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file='data/charset/charset_rec_cn_en.txt',
            type='Dictionary',
            with_padding=True,
            with_unknown=True),
        in_channels=384,
        max_seq_len=40,
        module_loss=dict(
            letter_case='lower', type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        type='SVTRDecoder'),
    encoder=dict(
        depth=[
            3,
            9,
            9,
        ],
        embed_dims=[
            192,
            256,
            512,
        ],
        img_size=[
            48,
            160,
        ],
        in_channels=3,
        max_seq_len=40,
        merging_types='Conv',
        mixer_types=[
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
        ],
        num_heads=[
            6,
            8,
            16,
        ],
        out_channels=384,
        prenorm=False,
        type='SVTREncoder',
        window_size=[
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
        ]),
    preprocessor=dict(
        in_channels=3,
        margins=[
            0.05,
            0.05,
        ],
        num_control_points=20,
        output_image_size=(
            48,
            160,
        ),
        resized_image_size=(
            32,
            64,
        ),
        type='STN'),
    type='SVTR')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.99,
        ),
        eps=8e-08,
        lr=0.00025,
        type='AdamW',
        weight_decay=0.05),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        convert_to_iter_based=True,
        end=2,
        end_factor=1.0,
        start_factor=0.5,
        type='LinearLR',
        verbose=False),
    dict(
        T_max=29,
        begin=2,
        convert_to_iter_based=True,
        end=30,
        type='CosineAnnealingLR',
        verbose=False),
]
pretrain_epoch = 20
pretrain_work_dir = 'work_dirs/svtr-large_pretrain_cn_scene_widthaug'
randomness = dict(deterministic=False, seed=3407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='scene_test',
        data_root='data/fudan/scene',
        pipeline=[
            dict(type='LoadImageFromNDArray'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='RecogLMDBDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_prefixes=[
        'Fudan',
    ],
    metrics=[
        dict(mode=[
            'exact',
        ], type='WordMetric'),
        dict(type='OneMinusNEDMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='scene_train',
        data_root='data/fudan/scene',
        pipeline=[
            dict(ignore_empty=True, min_size=5, type='LoadImageFromNDArray'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='TextRecogGeneralAug'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='CropHeight'),
                ],
                type='RandomApply'),
            dict(
                condition='min(results["img_shape"])>10',
                true_transforms=dict(
                    prob=0.4,
                    transforms=[
                        dict(
                            kernel_size=5,
                            op='GaussianBlur',
                            sigma=1,
                            type='TorchVisionWrapper'),
                    ],
                    type='RandomApply'),
                type='ConditionApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        brightness=0.5,
                        contrast=0.5,
                        hue=0.1,
                        op='ColorJitter',
                        saturation=0.5,
                        type='TorchVisionWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ImageContentJitter'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        args=[
                            dict(
                                cls='AdditiveGaussianNoise',
                                scale=0.31622776601683794),
                        ],
                        type='ImgAugWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ReversePixels'),
                ],
                type='RandomApply'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='RecogLMDBDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(ignore_empty=True, min_size=5, type='LoadImageFromNDArray'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug'),
        ],
        type='RandomApply'),
    dict(prob=0.4, transforms=[
        dict(type='CropHeight'),
    ], type='RandomApply'),
    dict(
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            prob=0.4,
            transforms=[
                dict(
                    kernel_size=5,
                    op='GaussianBlur',
                    sigma=1,
                    type='TorchVisionWrapper'),
            ],
            type='RandomApply'),
        type='ConditionApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                brightness=0.5,
                contrast=0.5,
                hue=0.1,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                args=[
                    dict(
                        cls='AdditiveGaussianNoise',
                        scale=0.31622776601683794),
                ],
                type='ImgAugWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ReversePixels'),
        ],
        type='RandomApply'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    256,
                    64,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='scene_val',
        data_root='data/fudan/scene',
        pipeline=[
            dict(type='LoadImageFromNDArray'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='RecogLMDBDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'Fudan',
    ],
    metrics=[
        dict(mode=[
            'exact',
        ], type='WordMetric'),
        dict(type='OneMinusNEDMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/svtr-large_finetune_from_pretrain_raw_seed3407_with_ep20'
