from copy import deepcopy

_base_ = ['svtr-large_pretrain_cn_scene.py']

# Validate on real Fudan val during synthetic pretraining so that the reported
# scores reflect real-domain transfer instead of synthetic hold-out.
#
# Additionally, evaluate on a small synthetic hold-out split for sanity check,
# and save best checkpoints for both domains.
fudan_val = dict(
    # IMPORTANT: define from scratch to avoid inheriting base config's
    # `indices` (hold-out indices for synthetic LMDB), which will cause
    # IndexError on Fudan LMDB.
    type='RecogLMDBDataset',
    data_root='data/fudan/scene',
    ann_file='scene_val',
    pipeline=deepcopy(_base_.test_pipeline),
    test_mode=True,
)

val_dataloader = deepcopy(_base_.val_dataloader)
pretrain_holdout = deepcopy(_base_.val_dataloader['dataset'])
val_dataloader['dataset'] = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[fudan_val, pretrain_holdout],
    verify_meta=False,
)

test_dataloader = val_dataloader

val_evaluator = deepcopy(_base_.val_evaluator)
val_evaluator['dataset_prefixes'] = ['Fudan', 'PretrainCN_holdout']
test_evaluator = val_evaluator

default_hooks = deepcopy(_base_.default_hooks)
default_hooks['checkpoint']['save_best'] = [
    'Fudan/recog/word_acc',
    'PretrainCN_holdout/recog/word_acc',
]
default_hooks['checkpoint']['rule'] = 'greater'
default_hooks['checkpoint']['max_keep_ckpts'] = 5

# Use EMA weights for evaluation & checkpoint saving to reduce metric noise.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        momentum=0.0002,
        # IMPORTANT:
        # Do NOT EMA-average buffers. Some buffers (e.g. attention masks with
        # +/-inf) will become NaN after lerp_ (inf - inf), making validation
        # collapse to constant predictions (often the first charset token '!').
        # Keep buffers synced from the source model instead.
        update_buffers=False,
    )
]

# IMPORTANT:
# Do NOT enable deterministic algorithms for SVTR+CTC on CUDA.
# It will trigger CuBLAS determinism requirements (CUBLAS_WORKSPACE_CONFIG)
# and may still fail later because CTC backward is non-deterministic on GPU.
randomness = dict(seed=None, deterministic=False)
