_base_ = ['svtr-large_fudan_scene_baseline.py']

pretrain_work_dir = 'work_dirs/svtr-large_pretrain'
pretrain_epoch = 20
load_from = f'{pretrain_work_dir}/best_PretrainCN_holdout_recog_word_acc_epoch_{pretrain_epoch}.pth'
