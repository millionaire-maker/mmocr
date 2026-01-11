from copy import deepcopy

_base_ = [
    'svtr-large_fudan_scene_baseline.py',
]

# This config keeps the legacy SVTR backbone (encoder) and rectifier (STN/TPS)
# unchanged, and only replaces the decoder with SVTRv2CTCDecoder to enable
# FRM/SGM ablations without introducing MSR or SVTRv2 backbone changes.
#
# Toggle examples (recommended):
# - FRM: `--cfg-options model.decoder.frm.enabled=False`
# - SGM: `--cfg-options model.decoder.sgm.enabled=False`
use_frm = True
use_sgm = True

dictionary = deepcopy(_base_.dictionary)

model = deepcopy(_base_.model)
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
    frm=dict(
        type='FeatureRearrangementModule',
        enabled=use_frm,
    ),
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

