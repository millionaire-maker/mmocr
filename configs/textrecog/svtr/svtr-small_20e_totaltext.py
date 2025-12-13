_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/default_runtime.py',
]

# 模型：在 tiny 基础上放大为 small 尺度
model = dict(
    encoder=dict(
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=['Local'] * 8 + ['Global'] * 7,
    )
)

# 训练策略与优化器（与官方 tiny 配置保持一致，可按需再调）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5 / (10**4) * 2048 / 2048,  # 约 5e-4
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

# 数据集：使用你已准备好的 Total-Text 文本识别标注与图像
totaltext_root = '/home/yzy/mmocr/datasets/totaltext'

totaltext_textrecog_train = dict(
    type='OCRDataset',
    data_root=totaltext_root,
    ann_file='textrecog_train.json',
    test_mode=False,
    # 你的 JSON 中的 img_path 已包含 'textrecog_imgs/train/...'
    # 因此这里无需设置 data_prefix（默认等价于 dict(img_path='')）
    pipeline=None,
)

totaltext_textrecog_test = dict(
    type='OCRDataset',
    data_root=totaltext_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None,
)

train_list = [totaltext_textrecog_train]
test_list = [totaltext_textrecog_test]

val_evaluator = dict(dataset_prefixes=['TotalText'])
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=128,           # 如显存不足可降至 64 或 32
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline,
    ),
)

test_dataloader = val_dataloader

# 自动按 batch_size 等比缩放学习率的基准 batch（参考 tiny 配置）
auto_scale_lr = dict(base_batch_size=512)


