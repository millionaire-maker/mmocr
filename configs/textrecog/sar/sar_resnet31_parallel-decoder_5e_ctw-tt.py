_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_sar_resnet31_parallel-decoder.py',
    # 这两个基础数据集配置由 prepare_dataset.py 生成/提供
    '../_base_/datasets/totaltext.py',
    '../_base_/datasets/ctw1500.py',
]

# dataset settings
# 训练集：CTW1500 + Total-Text 识别裁剪后的数据
train_list = [
    _base_.ctw1500_textrecog_train,
    _base_.totaltext_textrecog_train,
]

# 测试/验证集：分别评估两数据集
test_list = [
    _base_.ctw1500_textrecog_test,
    _base_.totaltext_textrecog_test,
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=10))

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline,
    ),
)

test_dataloader = val_dataloader

# 在多数据集评估时，给每个数据集加上前缀，便于区分指标
val_evaluator = dict(dataset_prefixes=['CTW1500', 'TotalText'])
test_evaluator = val_evaluator


