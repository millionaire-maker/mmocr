fudan_scene_rec_root = 'data/recog_lmdb'

fudan_scene_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=fudan_scene_rec_root,
    ann_file='fudan_scene_train.lmdb',
    pipeline=None)

fudan_scene_textrecog_val = dict(
    type='RecogLMDBDataset',
    data_root=fudan_scene_rec_root,
    ann_file='fudan_scene_val.lmdb',
    pipeline=None)

fudan_scene_textrecog_test = dict(
    type='RecogLMDBDataset',
    data_root=fudan_scene_rec_root,
    ann_file='fudan_scene_test.lmdb',
    pipeline=None)

fudan_scene_train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=fudan_scene_textrecog_train)

fudan_scene_val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=fudan_scene_textrecog_val)

fudan_scene_test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=fudan_scene_textrecog_test)

auto_scale_lr = dict(base_batch_size=64)
