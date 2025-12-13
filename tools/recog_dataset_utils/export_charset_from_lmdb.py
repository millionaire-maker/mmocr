import argparse
import json
import os
from collections import Counter

import lmdb


def iter_lmdb_labels(path: str):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        num_bytes = txn.get(b"num-samples")
        num = int(num_bytes.decode("utf-8")) if num_bytes else 0
        for idx in range(1, num + 1):
            key = f"label-{idx:09d}".encode("utf-8")
            label_bytes = txn.get(key)
            if label_bytes is None:
                continue
            yield label_bytes.decode("utf-8", errors="replace")
    env.close()


def iter_txt_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[1]


def collect_labels(sources):
    counter = Counter()
    per_source = {}
    for src in sources:
        if not os.path.exists(src):
            print(f"[WARN] skip missing {src}")
            continue
        if os.path.isdir(src) and os.path.exists(os.path.join(src, "data.mdb")):
            iterator = iter_lmdb_labels(src)
            src_type = "lmdb"
        else:
            iterator = iter_txt_labels(src)
            src_type = "txt"
        label_count = 0
        total_chars = 0
        for label in iterator:
            label_count += 1
            total_chars += len(label)
            for ch in label:
                counter[ch] += 1
        per_source[src] = {"type": src_type, "num_labels": label_count, "total_chars": total_chars}
    return counter, per_source


def write_outputs(counter: Counter, per_source, output_txt: str, stats_path: str):
    charset = sorted(counter.keys(), key=lambda c: ord(c))
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(f"{ch}\n")
    stats = {
        "num_unique_chars": len(charset),
        "total_char_count": sum(counter.values()),
        "top50": counter.most_common(50),
        "sources": per_source,
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved charset to {output_txt} ({len(charset)} chars)")
    print(f"Saved stats to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export charset from LMDB/text datasets.")
    parser.add_argument(
        "--lmdbs",
        nargs="+",
        default=[
            "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_train.lmdb",
            "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_val.lmdb",
            "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_test.lmdb",
        ],
        help="LMDB directories to scan.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra LMDB or txt datasets to include.",
    )
    parser.add_argument(
        "--output",
        default="/home/yzy/mmocr/data/charset/charset_rec_cn_en.txt",
        help="Path to save charset txt (one char per line).",
    )
    parser.add_argument(
        "--stats",
        default="/home/yzy/mmocr/data/charset/charset_rec_cn_en_stats.json",
        help="Path to save charset stats json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sources = list(dict.fromkeys(args.lmdbs + args.extra))
    counter, per_source = collect_labels(sources)
    if not counter:
        raise RuntimeError("No labels collected; please check input paths.")
    write_outputs(counter, per_source, args.output, args.stats)


if __name__ == "__main__":
    main()
