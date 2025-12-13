_textspotting_data_root = '/home/yzy/mmocr/data/ctw1500'

_textspotting_train = dict(
    type='OCRDataset',
    data_root=_textspotting_data_root,
    ann_file='textspotting_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

_textspotting_test = dict(
    type='OCRDataset',
    data_root=_textspotting_data_root,
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=None)
