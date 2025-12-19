import argparse
import json
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build recognition LMDB from a TSV label file."
    )
    parser.add_argument(
        "--label-tsv",
        required=True,
        help="TSV file: <img_relpath>\\t<label> per line.",
    )
    parser.add_argument(
        "--img-root",
        required=True,
        help="Image root path for lmdb_converter.",
    )
    parser.add_argument(
        "--out-lmdb",
        required=True,
        help="Output LMDB directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing LMDB before building.",
    )
    return parser.parse_args()


def tsv_to_jsonl(tsv_path: str, jsonl_path: str) -> int:
    count = 0
    with open(tsv_path, "r", encoding="utf-8") as f_in, open(
        jsonl_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            record = {"filename": img_path, "text": label}
            f_out.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def ensure_out_dir(out_lmdb: str, overwrite: bool) -> None:
    if os.path.exists(out_lmdb):
        if overwrite:
            if os.path.islink(out_lmdb):
                os.unlink(out_lmdb)
            else:
                shutil.rmtree(out_lmdb)
        else:
            raise FileExistsError(f"{out_lmdb} already exists.")
    os.makedirs(out_lmdb, exist_ok=True)


def run_lmdb_converter(repo_root: str, jsonl_path: str, out_lmdb: str, img_root: str):
    converter = os.path.join(
        repo_root, "tools/dataset_converters/textrecog/lmdb_converter.py"
    )
    cmd = [
        sys.executable,
        converter,
        jsonl_path,
        out_lmdb,
        "--img-root",
        img_root,
        "--label-format",
        "jsonl",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    if not os.path.exists(args.label_tsv):
        raise FileNotFoundError(f"Label TSV not found: {args.label_tsv}")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    jsonl_path = os.path.splitext(args.label_tsv)[0] + ".jsonl"
    tsv_to_jsonl(args.label_tsv, jsonl_path)
    ensure_out_dir(args.out_lmdb, args.overwrite)
    run_lmdb_converter(repo_root, jsonl_path, args.out_lmdb, args.img_root)


if __name__ == "__main__":
    main()
