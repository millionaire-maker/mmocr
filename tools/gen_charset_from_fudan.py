import argparse
import json
import lmdb
import os
import unicodedata
from collections import Counter

EXTRA_CHARS = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
    '0123456789'
    '，。、：“”‘’：；·—…？！%-,.()（）『』《》〈〉﹣﹔；￥¥@ '  # Added ￥, ¥, @, and space
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate charset from Fudan Scene LMDB')
    parser.add_argument('--lmdb', default='data/fudan/scene/scene_train', help='训练集 LMDB 路径')
    parser.add_argument('--charset-file', default='dicts/custom_charset.txt')
    parser.add_argument('--stats-file', default='reports/charset_stats.json')
    return parser.parse_args()


def iter_labels(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        total = int(txn.get('num-samples'.encode()))
        for idx in range(1, total + 1):
            label_key = f'label-{idx:09d}'.encode()
            label_buf = txn.get(label_key)
            if label_buf is None:
                continue
            text = label_buf.decode('utf-8')
            text = unicodedata.normalize('NFKC', text)
            yield text
    env.close()


def main():
    args = parse_args()
    counter = Counter()
    for text in iter_labels(args.lmdb):
        for ch in text:
            counter[ch] += 1
    for ch in EXTRA_CHARS:
        counter.setdefault(ch, 0)

    # 排序：频次降序，频次相同时按字符 Unicode 升序
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], ord(kv[0])))

    os.makedirs(os.path.dirname(args.charset_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)

    with open(args.stats_file, 'w', encoding='utf-8') as f:
        json.dump({ch: freq for ch, freq in sorted_items}, f, ensure_ascii=False, indent=2)

    with open(args.charset_file, 'w', encoding='utf-8') as f:
        for ch, _ in sorted_items:
            if ch == '\n':
                f.write('\\n\n')
            elif ch == '\r':
                f.write('\\r\n')
            else:
                f.write(f'{ch}\n')

    print(f'写入 {len(sorted_items)} 个字符到 {args.charset_file}')
    print(f'统计信息写入 {args.stats_file}')


if __name__ == '__main__':
    main()
