from copy import deepcopy

_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/default_runtime.py',
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
    encoder=dict(
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=['Local'] * 8 + ['Global'] * 7,
        max_seq_len=40,
    ),
    decoder=dict(dictionary=dictionary),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
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
        T_max=19,
        begin=2,
        end=20,
        verbose=False,
        convert_to_iter_based=True,
    ),
]

synth_lmdb_root = 'data/recog_lmdb'
synth_train = dict(
    type='RecogLMDBDataset',
    data_root=synth_lmdb_root,
    ann_file='synth_rec_ch_train.lmdb',
    pipeline=train_pipeline,
    test_mode=False,
)
synth_val = dict(
    type='RecogLMDBDataset',
    data_root=synth_lmdb_root,
    ann_file='synth_rec_ch_train.lmdb',
    pipeline=test_pipeline,
    test_mode=True,
)

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synth_train,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=synth_val,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(type='WordMetric', mode=['exact']),
        dict(type='OneMinusNEDMetric'),
    ],
    dataset_prefixes=['SynthCH'],
)
test_evaluator = val_evaluator

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(base_batch_size=train_dataloader['batch_size'])
