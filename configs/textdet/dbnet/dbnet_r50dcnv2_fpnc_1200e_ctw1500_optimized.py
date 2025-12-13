_base_ = [
    '../_base_/datasets/ctw1500.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
]

# 优化核心：根据 Grid Search 结果，将 unclip_ratio 调整为 2.2
# 这对弯曲文本检测至关重要
model = dict(
    postprocessor=dict(type='DBPostprocessor', text_repr_type='poly', unclip_ratio=2.2)
)

# 数据集配置 (复用 _base_ 中的 pipeline)
ctw1500_textdet_train = _base_.ctw1500_textdet_train
ctw1500_textdet_test = _base_.ctw1500_textdet_test

ctw1500_textdet_train.pipeline = _base_.train_pipeline
ctw1500_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ctw1500_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ctw1500_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
