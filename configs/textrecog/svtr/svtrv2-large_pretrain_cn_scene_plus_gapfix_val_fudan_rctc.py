_base_ = ['svtrv2-large_pretrain_cn_scene_plus_gapfix_val_fudan.py']

# SVTRv2 stage-1 pretraining ablation: MSR + FRM, without SGM.
use_sgm = False
model = dict(decoder=dict(sgm=dict(enabled=False)))
