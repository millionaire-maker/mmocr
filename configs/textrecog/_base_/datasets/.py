_textrecog_data_root = 'data/cute80'

_textrecog_test = dict(
    type='OCRDataset',
    data_root=_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
