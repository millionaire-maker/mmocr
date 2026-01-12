from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene.py']

model = dict(
    preprocessor=dict(
        _delete_=True,
        type='TPSPP',
        in_channels=3,
        resized_image_size=(32, 128),
        output_image_size=(48, 160),
        # num_img_channel=64,
        # point_size=(2, 16),
        # p_stride=2,
    ))

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader['dataset'] = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[
        dict(
            type='RecogLMDBDataset',
            data_root='data',
            ann_file='pretrain_cn_scene',
            indices=_base_.pretrain_train_size,
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
