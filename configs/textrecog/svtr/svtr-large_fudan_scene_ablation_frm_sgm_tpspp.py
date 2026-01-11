_base_ = ['svtr-large_fudan_scene_ablation_frm_sgm.py']

# Replace the legacy STN(TPS) rectifier with TPS++.
model = dict(
    preprocessor=dict(
        _delete_=True,
        type='TPSPP',
        in_channels=3,
        resized_image_size=(32, 128),
        output_image_size=(48, 160),
    ))

