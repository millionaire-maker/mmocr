_base_ = [
    '../fcenet_r50dcnv2_fpn_direct_finetune_art_rctw_rects/'
    'fcenet_r50dcnv2_fpn_1500e_art_rctw_rects_finetune.py'
]

# 仅评测 RCTW（加速定位问题、做后处理阈值对照）
val_dataloader = dict(dataset=dict(datasets=[_base_.textdet_rctw_test]))
test_dataloader = dict(dataset=dict(datasets=[_base_.textdet_rctw_test]))

val_evaluator = dict(
    dataset_prefixes=dict(_delete_=True, rctw='data/rctw17_mmocr'))
test_evaluator = dict(
    dataset_prefixes=dict(_delete_=True, rctw='data/rctw17_mmocr'))
