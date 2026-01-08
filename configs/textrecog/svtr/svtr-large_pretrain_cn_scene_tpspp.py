_base_ = ['svtr-large_pretrain_cn_scene.py']

model = dict(
    preprocessor=dict(
        _delete_=True,
        type='TPSPP',
        in_channels=3,
        resized_image_size=(32, 128),
        output_image_size=(48, 160),
        # num_img_channel=64,
        # point_size=(2, 16),
        # p_stride=2,
    ))
