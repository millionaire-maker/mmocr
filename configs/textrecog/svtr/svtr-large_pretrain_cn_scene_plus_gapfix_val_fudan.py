from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene_val_fudan.py']

# Pretrain on (original synth LMDB) + (gap-fix low-res LMDB),
# and still validate/save_best on real Fudan val.
#
# Required LMDBs:
# - data/pretrain_cn_scene
# - data/pretrain_cn_scene_gapfix_h24

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader['dataset'] = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[
        dict(
            type='RecogLMDBDataset',
            data_root='data',
            ann_file='pretrain_cn_scene',
            pipeline=deepcopy(_base_.train_pipeline),
            test_mode=False,
        ),
        dict(
            type='RecogLMDBDataset',
            data_root='data',
            ann_file='pretrain_cn_scene_gapfix_h24',
            pipeline=deepcopy(_base_.train_pipeline),
            test_mode=False,
        ),
    ],
)
