import argparse
import json
import os
from collections import Counter


def iter_labels(txt_files):
    for path in txt_files:
        if not os.path.exists(path):
            print(f"[WARN] skip missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                yield parts[1]


def build_charset(txt_files, output_txt, stats_path):
    counter = Counter()
    for label in iter_labels(txt_files):
        for ch in label:
            counter[ch] += 1
    charset = sorted(counter.keys())
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(f"{ch}\n")
    stats = {
        "num_chars": len(charset),
        "total_count": sum(counter.values()),
        "top20": counter.most_common(20),
        "inputs": txt_files,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Exported charset of {len(charset)} chars to {output_txt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export charset from recog label txt files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/home/yzy/mmocr/data/recog_mega/mega_train_from_raw.txt",
            "/home/yzy/mmocr/data/synth_rec_ch/train.txt",
        ],
        help="Label txt files to scan (img_path<TAB>label).",
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
    build_charset(args.inputs, args.output, args.stats)


if __name__ == "__main__":
    main()
