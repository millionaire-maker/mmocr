_base_ = ['svtr-large_fudan_scene_finetune_from_pretrain_cn_scene.py']

pretrain_work_dir = 'work_dirs/svtr-large_pretrain_cn_scene_tpspp'
pretrain_epoch = 20
load_from = f'{pretrain_work_dir}/epoch_{pretrain_epoch}.pth'

model = dict(
    preprocessor=dict(
        _delete_=True,
        type='TPSPP',
        in_channels=3,
        resized_image_size=(32, 128),
        output_image_size=(48, 160),
    ))
