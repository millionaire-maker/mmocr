from copy import deepcopy

_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/default_runtime.py',
    '../_base_/rec_datasets/fudan_scene_lmdb.py',
]

train_pipeline = deepcopy(_base_.train_pipeline)
train_pipeline[0]['type'] = 'LoadImageFromNDArray'
test_pipeline = deepcopy(_base_.test_pipeline)
test_pipeline[0]['type'] = 'LoadImageFromNDArray'
tta_pipeline = deepcopy(_base_.tta_pipeline)
tta_pipeline[0]['type'] = 'LoadImageFromNDArray'

dictionary = dict(
    type='Dictionary',
    dict_file='data/charset/charset_rec_cn_en.txt',
    with_padding=True,
    with_unknown=True,
)

model = dict(
    preprocessor=dict(output_image_size=(48, 160)),
    encoder=dict(
        img_size=[48, 160],
        max_seq_len=40,
        out_channels=384,
        embed_dims=[192, 256, 512],
        depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        mixer_types=['Local'] * 10 + ['Global'] * 11,
    ),
    decoder=dict(in_channels=384, max_seq_len=40, dictionary=dictionary),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2.5e-4,
        betas=(0.9, 0.99),
        eps=8e-8,
        weight_decay=0.05,
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        end_factor=1.0,
        end=2,
        verbose=False,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=29,
        begin=2,
        end=30,
        verbose=False,
        convert_to_iter_based=True,
    ),
]

train_dataloader = deepcopy(_base_.fudan_scene_train_dataloader)
train_dataloader['batch_size'] = 128
train_dataloader['num_workers'] = 2
train_dataloader['prefetch_factor'] = 2
train_dataloader['persistent_workers'] = True
train_dataloader['dataset']['pipeline'] = train_pipeline

val_dataloader = deepcopy(_base_.fudan_scene_val_dataloader)
val_dataloader['num_workers'] = 2
val_dataloader['prefetch_factor'] = 2
val_dataloader['persistent_workers'] = True
val_dataloader['dataset']['pipeline'] = test_pipeline

test_dataloader = deepcopy(_base_.fudan_scene_test_dataloader)
test_dataloader['num_workers'] = 2
test_dataloader['prefetch_factor'] = 2
test_dataloader['persistent_workers'] = True
test_dataloader['dataset']['pipeline'] = test_pipeline

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(type='WordMetric', mode=['exact']),
        dict(type='OneMinusNEDMetric'),
    ],
    dataset_prefixes=['Fudan'],
)
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='Fudan/recog/word_acc',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

env_cfg = dict(cudnn_benchmark=True)

auto_scale_lr = dict(base_batch_size=train_dataloader['batch_size'] * 2)
