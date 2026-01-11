_base_ = ['svtrv2-large_fudan_scene_baseline.py']

# SVTRv2 stage-1 style ablation: MSR + FRM, without SGM.
# This mirrors OpenOCR's first-stage training recipe (`svtrv2_rctc.yml`),
# where SGM is introduced in the second stage.
use_sgm = False
model = dict(decoder=dict(sgm=dict(enabled=False)))
