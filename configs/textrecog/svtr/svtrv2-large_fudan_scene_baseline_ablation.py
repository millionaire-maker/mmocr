from copy import deepcopy

_base_ = ['svtrv2-large_fudan_scene_baseline.py']

# A "full-on" SVTRv2 example for ablations via `--cfg-options`.
# Suggested toggles:
# - MSR:   train_pipeline.*.use_msr
# - FRM:   model.decoder.frm.enabled
# - SGM:   model.decoder.sgm.enabled
# - SVTRv2 backbone: model.encoder.use_svtrv2_backbone

train_pipeline = deepcopy(_base_.train_pipeline)
test_pipeline = deepcopy(_base_.test_pipeline)
tta_pipeline = deepcopy(_base_.tta_pipeline)

for _t in train_pipeline:
    if _t.get('type') == 'SVTRv2AdaptiveResize':
        _t['use_msr'] = True
        _t['random_resize_factor'] = True
        _t['padding_doub'] = True

for _t in test_pipeline:
    if _t.get('type') == 'SVTRv2AdaptiveResize':
        _t['use_msr'] = True

for _t in tta_pipeline:
    if _t.get('type') == 'SVTRv2AdaptiveResize':
        _t['use_msr'] = True

model = deepcopy(_base_.model)
model['encoder']['use_svtrv2_backbone'] = True
model['decoder']['frm']['enabled'] = True
model['decoder']['sgm']['enabled'] = False
