_base_ = ['svtr-large_fudan_scene_baseline.py']

# 是否启用 TPS++（IJCAI 2023）
use_tpspp = True

tpspp_preprocessor = dict(
    _delete_=True,
    type='TPSPP',
    in_channels=3,
    resized_image_size=(32, 128),
    output_image_size=(48, 160),
    # num_img_channel=64,
    # point_size=(2, 16),
    # p_stride=2,
)

model = dict(preprocessor=tpspp_preprocessor if use_tpspp else None)
