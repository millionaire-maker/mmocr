from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene.py']

# Validate directly on real Fudan val during synthetic pretraining so that
# the reported scores reflect real-domain transfer instead of synthetic hold-out.
fudan_val = dict(
    # IMPORTANT: override the whole dataset dict, otherwise base config's
    # `indices` (hold-out indices for synthetic LMDB) will be merged in and
    # cause IndexError on Fudan.
    _delete_=True,
    type='RecogLMDBDataset',
    data_root='data/fudan/scene',
    ann_file='scene_val',
    pipeline=deepcopy(_base_.test_pipeline),
    test_mode=True,
)

val_dataloader = deepcopy(_base_.val_dataloader)
val_dataloader['dataset'] = fudan_val

test_dataloader = val_dataloader

val_evaluator = deepcopy(_base_.val_evaluator)
val_evaluator['dataset_prefixes'] = ['Fudan']
test_evaluator = val_evaluator

default_hooks = deepcopy(_base_.default_hooks)
default_hooks['checkpoint']['save_best'] = 'Fudan/recog/word_acc'
