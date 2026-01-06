from copy import deepcopy

_base_ = ['svtr-large_fudan_scene_finetune_from_pretrain_cn_scene.py']

pretrain_work_dir = 'work_dirs/svtr-large_pretrain_cn_scene_notps'
pretrain_epoch = 20
load_from = f'{pretrain_work_dir}/epoch_{pretrain_epoch}.pth'

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
train_dataloader['dataset']['pipeline'] = train_pipeline

val_dataloader = deepcopy(_base_.val_dataloader)
val_dataloader['dataset']['pipeline'] = test_pipeline

test_dataloader = deepcopy(_base_.test_dataloader)
test_dataloader['dataset']['pipeline'] = test_pipeline
