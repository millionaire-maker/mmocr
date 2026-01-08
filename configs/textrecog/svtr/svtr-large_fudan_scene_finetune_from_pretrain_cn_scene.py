_base_ = ['svtr-large_fudan_scene_baseline.py']

pretrain_work_dir = 'work_dirs/svtr-large_new_pretrain'
pretrain_epoch = 13
load_from = f'{pretrain_work_dir}/best_Fudan_recog_word_acc_epoch_{pretrain_epoch}.pth'
