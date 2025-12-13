_base_ = ['fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py']

_base_.model.backbone = dict(
    type='ResNetWithAttention',
    depth=50,
    num_stages=4,
    out_indices=(1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
    stage_with_dcn=(False, True, True, True),
    attention_type='se',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

# 训练配置覆盖（与基础配置保持一致）
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
auto_scale_lr = dict(base_batch_size=4)

# 修改验证间隔为每3个epoch验证一次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1500, val_interval=3)

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

# 早停机制：连续6次验证无提升自动停止训练
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='icdar/hmean',  # 监控icdar/hmean指标
        patience=6,  # 连续6次验证无提升则停止
        rule='greater'  # hmean越大越好
    )
]
