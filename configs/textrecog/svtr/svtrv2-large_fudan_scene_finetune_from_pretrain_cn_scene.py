_base_ = ['svtrv2-large_fudan_scene_baseline.py']

pretrain_work_dir = 'work_dirs/svtrv2-large_pretrain_cn_scene'
pretrain_epoch = 20
load_from = f'{pretrain_work_dir}/epoch_{pretrain_epoch}.pth'

