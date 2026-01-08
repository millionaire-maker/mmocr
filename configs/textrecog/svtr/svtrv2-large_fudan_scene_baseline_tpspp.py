_base_ = ['svtrv2-large_fudan_scene_baseline.py']

# Enable TPS++ rectifier (compatible with SVTRv2 backbone/FRM/SGM/MSR).
model = dict(
    preprocessor=dict(
        _delete_=True,
        type='TPSPP',
        in_channels=3,
        resized_image_size=(32, 128),
        output_image_size=(48, 160),
    ))

