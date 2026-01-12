from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene.py']

model = dict(preprocessor=None)

_target_scale = (160, 48)

train_pipeline = deepcopy(_base_.train_pipeline)
test_pipeline = deepcopy(_base_.test_pipeline)
tta_pipeline = deepcopy(_base_.tta_pipeline)

for _t in train_pipeline:
    if _t.get('type') == 'Resize':
        _t['scale'] = _target_scale

for _t in test_pipeline:
    if _t.get('type') == 'Resize':
        _t['scale'] = _target_scale

for _t in tta_pipeline:
    if _t.get('type') == 'TestTimeAug':
        for _aug in _t['transforms']:
            for _sub_t in _aug:
                if isinstance(_sub_t, dict) and _sub_t.get('type') == 'Resize':
                    _sub_t['scale'] = _target_scale

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader['dataset'] = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[
        dict(
            type='RecogLMDBDataset',
            data_root='data',
            ann_file='pretrain_cn_scene',
            pipeline=train_pipeline,
            test_mode=False,
        ),
        dict(
            type='RecogLMDBDataset',
            data_root='data',
            ann_file='pretrain_cn_scene_gapfix_h24',
            pipeline=train_pipeline,
            test_mode=False,
        ),
    ],
)

val_dataloader = deepcopy(_base_.val_dataloader)
val_dataloader['dataset']['pipeline'] = test_pipeline

test_dataloader = val_dataloader
