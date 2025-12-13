_base_ = [
    '_base_dbnetpp_resnet18_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/textdet_art_rctw_rects_finetune.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# 加载LSVT+CTW预训练后微调的最佳权重 (epoch 3, hmean ≈0.55)
# 作为ArT+RCTW+RECTS数据集微调的基线模型
# 注意:使用 --resume 继续训练时必须注释掉这一行,否则会重新加载预训练模型而不是从checkpoint恢复
# load_from = 'work_dirs/dbnetpp_r18_finetune_from_epoch39/best_icdar_hmean_epoch_3.pth'

# 指定work_dir为之前的训练目录,确保--resume能找到checkpoint
work_dir = 'work_dirs/dbnetpp_r18_finetune_art_rctw_rects'

train_list = [_base_.textdet_art_rctw_rects_train]
test_list = [_base_.textdet_art_rctw_rects_test]

train_dataloader = dict(
    batch_size=4,  # 降低到4以进一步减少显存压力
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=1,  # 降低从2到1,减少验证时显存波动
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)  # 同步修改base_batch_size


# 微调优化器配置：降低学习率避免灾难性遗忘
# 从预训练权重微调,使用较低学习率(原始0.007的1/7)以保护已学习特征
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
)

# 缩短训练周期为150 epochs进行初步测试,避免过度训练
# 配合早停机制(patience=6),预计实际训练epochs会更少
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=3)

# 配置权重保存策略：保留最新3个权重 + 保留最优权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # 每3个epoch保存一次
        max_keep_ckpts=3,  # 保留最新的3个权重
        save_best='icdar/hmean',  # 明确监控icdar/hmean指标
        rule='greater'  # hmean越大越好
    )
)

# 早停机制:优化后的配置,给予模型更多训练机会
# patience=12: 连续12次验证(36个epoch)无提升才停止
# min_delta=0.001: 改善幅度>0.1%即视为有效提升,避免因微小波动提前停止
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='icdar/hmean',  # 监控icdar/hmean指标
        patience=12,  # 连续12次验证无提升则停止(从6增加到12)
        min_delta=0.001,  # 最小有效改善阈值0.1%(新增,原默认值0.1过大)
        rule='greater'  # hmean越大越好
    )
]
