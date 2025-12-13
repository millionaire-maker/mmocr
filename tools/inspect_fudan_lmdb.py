import argparse
import io
import lmdb
import random
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect Fudan Scene LMDB samples')
    parser.add_argument('--lmdb', required=True, help='LMDB 路径，例如 data/fudan/scene/scene_train')
    parser.add_argument('--num', type=int, default=8, help='随机抽样数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def load_sample(txn, idx):
    img_key = f'image-{idx:09d}'
    label_key = f'label-{idx:09d}'
    imgbuf = txn.get(img_key.encode())
    label_buf = txn.get(label_key.encode())
    if imgbuf is None or label_buf is None:
        return None
    img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
    label = label_buf.decode('utf-8')
    return img_key, img.size, label  # PIL size: (width, height)


def main():
    args = parse_args()
    env = lmdb.open(
        args.lmdb,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        total = int(txn.get('num-samples'.encode()))
        print(f'LMDB {args.lmdb} 总样本数：{total}')
        rng = random.Random(args.seed)
        count = min(args.num, total)
        indices = sorted(rng.sample(range(1, total + 1), count))
        for idx in indices:
            sample = load_sample(txn, idx)
            if sample is None:
                print(f'- idx {idx}: missing data')
                continue
            key, (w, h), label = sample
            print(f'- {key}: size={h}x{w}x3 label={label}')

    env.close()


if __name__ == '__main__':
    main()
