from copy import deepcopy

_base_ = [
    'svtr-large_pretrain_cn_scene.py',
]

# -----------------------------
# SVTRv2 Ablation Switches
# -----------------------------
use_svtrv2_backbone = True
use_msr = True
use_frm = True
use_sgm = True  # pretrain建议开启；推理不增加开销

rectifier = None

msr_base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]  # [w, h]
msr_base_h = 32
msr_max_ratio = 12
msr_pad_to_max = True
msr_max_size = None

if msr_pad_to_max and msr_max_size is None:
    _max_w = max([w for w, _ in msr_base_shape] + [msr_base_h * msr_max_ratio])
    _max_h = max([h for _, h in msr_base_shape] + [msr_base_h])
    msr_max_size = (_max_w, _max_h)  # (w, h)

train_pipeline = deepcopy(_base_.train_pipeline)
test_pipeline = deepcopy(_base_.test_pipeline)
tta_pipeline = deepcopy(_base_.tta_pipeline)

for _t in train_pipeline:
    if _t.get('type') == 'Resize':
        _t.clear()
        _t.update(
            dict(
                type='SVTRv2AdaptiveResize',
                use_msr=use_msr,
                scale=(256, 64),
                base_shape=msr_base_shape,
                base_h=msr_base_h,
                max_ratio=msr_max_ratio,
                padding=False,
                pad_to_max=msr_pad_to_max,
                max_size=msr_max_size,
                random_resize_factor=True,
                padding_doub=True,
                interpolation='bicubic',
            ))

for _t in test_pipeline:
    if _t.get('type') == 'Resize':
        _t.clear()
        _t.update(
            dict(
                type='SVTRv2AdaptiveResize',
                use_msr=use_msr,
                scale=(256, 64),
                base_shape=msr_base_shape,
                base_h=msr_base_h,
                max_ratio=msr_max_ratio,
                padding=False,
                pad_to_max=msr_pad_to_max,
                max_size=msr_max_size,
                random_resize_factor=False,
                interpolation='bicubic',
            ))

for _t in tta_pipeline:
    if _t.get('type') == 'Resize':
        _t.clear()
        _t.update(
            dict(
                type='SVTRv2AdaptiveResize',
                use_msr=use_msr,
                scale=(256, 64),
                base_shape=msr_base_shape,
                base_h=msr_base_h,
                max_ratio=msr_max_ratio,
                padding=False,
                pad_to_max=msr_pad_to_max,
                max_size=msr_max_size,
                random_resize_factor=False,
                interpolation='bicubic',
            ))

dictionary = deepcopy(_base_.dictionary)

svtr_encoder = deepcopy(_base_.model.encoder)
if msr_pad_to_max and msr_max_size is not None:
    svtr_encoder['img_size'] = [msr_max_size[1], msr_max_size[0]]  # (h, w)
svtrv2_encoder = dict(
    type='SVTRv2Backbone',
    in_channels=3,
    max_sz=[64, 384],
    dims=[128, 256, 384],
    depths=[6, 6, 6],
    num_heads=[4, 8, 12],
    mixer=[
        ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
        ['Conv', 'Conv', 'FGlobal', 'Global', 'Global', 'Global'],
        ['Global', 'Global', 'Global', 'Global', 'Global', 'Global'],
    ],
    kernel_sizes=[
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
    ],
    num_convs=[
        [2, 2, 2, 2, 2, 2],
        [2, 2, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
    ],
    sub_k=[[1, 1], [2, 1], [1, 1]],
    use_pos_embed=False,
    last_stage=False,
    feat2d=True,
)

model = deepcopy(_base_.model)
model['preprocessor'] = rectifier
model['encoder'] = dict(
    _delete_=True,
    type='SVTRFlexibleEncoder',
    use_svtrv2_backbone=use_svtrv2_backbone,
    svtr_encoder=svtr_encoder,
    svtrv2_encoder=svtrv2_encoder,
)
model['decoder'] = dict(
    _delete_=True,
    type='SVTRv2CTCDecoder',
    in_channels=384,
    max_seq_len=40,
    module_loss=dict(
        type='CTCModuleLoss',
        letter_case='lower',
        zero_infinity=True,
    ),
    postprocessor=dict(type='CTCPostProcessor'),
    dictionary=dictionary,
    frm=dict(type='FeatureRearrangementModule', enabled=use_frm),
    sgm=dict(
        type='SemanticGuidanceModule',
        enabled=use_sgm,
        loss_weight=0.1,
        sub_str_len=5,
        num_layer=1,
        drop_path_rate=0.1,
        detach_visual=False,
    ),
)

# Make sure dataloaders use the updated pipelines.
pretrain_train = deepcopy(_base_.pretrain_train)
pretrain_train['pipeline'] = train_pipeline
pretrain_val = deepcopy(_base_.pretrain_val)
pretrain_val['pipeline'] = test_pipeline

train_dataloader = deepcopy(_base_.train_dataloader)
train_dataloader['dataset'] = pretrain_train
val_dataloader = deepcopy(_base_.val_dataloader)
val_dataloader['dataset'] = pretrain_val
test_dataloader = val_dataloader
