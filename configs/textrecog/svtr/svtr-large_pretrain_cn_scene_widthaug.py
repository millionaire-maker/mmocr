from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene.py']

# Synthetic LMDB is generated with a fixed raw width (e.g. 256), while real
# word crops (e.g. Fudan) have a huge raw aspect-ratio range.
# Adding a width-randomization step before the final fixed resize helps the
# pretrain features transfer to real datasets.
train_pipeline = deepcopy(_base_.train_pipeline)
train_pipeline.insert(
    1,
    dict(type='RandomResizeWidth', min_width=16, max_width=512, prob=1.0),
)

pretrain_train = deepcopy(_base_.pretrain_train)
pretrain_train['pipeline'] = train_pipeline

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader['dataset'] = pretrain_train

