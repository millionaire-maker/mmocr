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


def build_default_sources():
    base = [
        "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_train.lmdb",
        "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_val.lmdb",
        "/home/yzy/mmocr/data/recog_lmdb/fudan_scene_test.lmdb",
    ]
    synth_main = "/home/yzy/mmocr/data/synth_rec_ch/train.txt"
    synth_debug = "/home/yzy/mmocr/data/synth_rec_ch_debug/train.txt"
    if os.path.exists(synth_main):
        base.append(synth_main)
    elif os.path.exists(synth_debug):
        print("[INFO] synth_rec_ch/train.txt missing, fallback to debug txt.")
        base.append(synth_debug)
    else:
        print("[WARN] No synth_rec_ch txt found; synth coverage will be skipped.")
    return base


def check_source(path: str, charset_set):
    if not os.path.exists(path):
        print(f"[WARN] skip missing {path}")
        return None
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "data.mdb")):
        iterator = iter_lmdb_labels(path)
        src_type = "lmdb"
    else:
        iterator = iter_txt_labels(path)
        src_type = "txt"
    missing_counter = Counter()
    total_chars = 0
    for label in iterator:
        for ch in label:
            total_chars += 1
            if ch not in charset_set:
                missing_counter[ch] += 1
    missing_total = sum(missing_counter.values())
    covered = total_chars - missing_total
    coverage = covered / total_chars if total_chars else 1.0
    result = {
        "source": path,
        "type": src_type,
        "total_chars": total_chars,
        "missing_total": missing_total,
        "unique_missing": len(missing_counter),
        "missing_top10": missing_counter.most_common(10),
        "coverage": coverage,
    }
    return result, missing_counter


def parse_args():
    parser = argparse.ArgumentParser(description="Check charset coverage for multiple datasets.")
    parser.add_argument(
        "--charset",
        default="/home/yzy/mmocr/data/charset/charset_rec_cn_en.txt",
        help="Charset txt file (one char per line).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Datasets to scan (lmdb dir or txt). Default uses fudan_scene and synth txt.",
    )
    parser.add_argument(
        "--output",
        default="/home/yzy/mmocr/data/charset/charset_coverage_report.json",
        help="Path to save coverage report.",
    )
    return parser.parse_args()


def load_charset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    return set(chars)


def main():
    args = parse_args()
    charset_set = load_charset(args.charset)
    sources = args.sources if args.sources else build_default_sources()
    sources = list(dict.fromkeys(sources))
    results = []
    missing_agg = Counter()
    for src in sources:
        res = check_source(src, charset_set)
        if res is None:
            continue
        result, missing_counter = res
        results.append(result)
        missing_agg.update(missing_counter)
        status = (
            f"{result['coverage']*100:.2f}% coverage, "
            f"missing {result['missing_total']} chars "
            f"({result['unique_missing']} unique)"
        )
        print(f"[{result['type'].upper()}] {result['source']}: {status}")
        if result["missing_top10"]:
            tops = ", ".join([f"{ch}:{cnt}" for ch, cnt in result["missing_top10"]])
            print(f"  top_missing: {tops}")
    report = {
        "charset": args.charset,
        "num_chars": len(charset_set),
        "sources": results,
        "missing_agg_top20": missing_agg.most_common(20),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved coverage report to {args.output}")


if __name__ == "__main__":
    main()
