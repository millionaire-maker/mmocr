_base_ = ['svtr-large_fudan_scene_baseline_tpspp.py']

pretrain_work_dir = 'work_dirs/svtr-large_pretrain_cn_scene_tpspp'
pretrain_epoch = 20
load_from = f'{pretrain_work_dir}/epoch_{pretrain_epoch}.pth'

