textdet_art_rctw_rects_data_root = 'data/textdet_finetune_art_rctw_rects'

textdet_art_rctw_rects_train = dict(
    type='OCRDataset',
    data_root=textdet_art_rctw_rects_data_root,
    ann_file='instances_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

textdet_art_rctw_rects_test = dict(
    type='OCRDataset',
    data_root=textdet_art_rctw_rects_data_root,
    ann_file='instances_val.json',
    test_mode=True,
    pipeline=None)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=[textdet_art_rctw_rects_train],
        pipeline=None))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=[textdet_art_rctw_rects_test],
        pipeline=None))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)
